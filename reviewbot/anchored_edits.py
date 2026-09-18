"""Anchored edits: an answer format the model does not have to transcribe.

A unified diff asks the model for three things at once — *what* to change, the
*surrounding context* byte for byte, and the *geometry* (`@@ -463,9 +463,9 @@`)
that has to agree with that context. Two of the three are bookkeeping, and both
are where the model's patches died. Of the three prod tasks that failed to apply
on 2026-09-16/17 (`pvt_v2`, `qwen3_omni_moe`, `zamba`), one lost on a single
trailing comma inside a 40-element tensor it had to re-type, and one on hunk
geometry that did not match its own hunk body.

An anchored edit is ``{"path", "old", "new"}``: replace ``old`` with ``new`` in
``path``, where ``old`` must occur **exactly once**. No line numbers, no hunk
counts, no context lines that are not already part of the text being replaced.
What is left is the one thing only the model can supply.

Two properties matter as much as the smaller ask:

* **The rejection is precise.** "your anchor does not occur" / "occurs 3 times"
  names the defect, where `git apply`'s "patch does not apply" does not — and
  when the anchor is merely close, :func:`apply_edits` can quote the line that
  is really there, which is the whole of what the correction turn needs.
* **Nothing is guessed.** Exactly-once or refuse. There is no fuzz, no offset
  search and no "closest match" application: an ambiguous anchor is a rejection,
  never a silent edit in the wrong place.

Measured against the deployed model replaying the two real rejections, asking
for edits instead of a diff got 3 of 4 `zamba` edits and 2 of 2 `pvt_v2` edits
exactly right, and on one run the model correctly returned an EMPTY edit list
rather than invent an anchor for a line it could not find.

The module is pure: callers pass a reader, and get back the new contents to
write. That keeps it testable without a checkout, and keeps the decision to
touch the worktree in one place (``tasks._apply_anchored_edits``).
"""

from __future__ import annotations

import difflib
import posixpath
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "Edit",
    "ParsedEdits",
    "apply_edits",
    "parse_edits",
    "rejection_feedback",
    "summarize",
]

# A fix is a focused change; a hundred anchors is a rewrite, and each one is an
# unbounded string in the reply. The cap is on the schema, not on the fix.
MAX_EDITS = 40
# Lines of real file content shown around a candidate for a missing anchor.
_WINDOW_PAD = 4
# How many places to quote back for one missing anchor. More than this and the
# feedback stops being a pointer and becomes another thing to read.
_MAX_CANDIDATES = 2
# Fraction of the anchor that must appear, in order, in a line before it is
# worth quoting back as "did you mean". 0.6 keeps the real zamba near-miss
# (0.96) and drops unrelated code.
_CLOSE_ENOUGH = 0.6
# A run of shared characters this long is what makes two lines candidates at
# all: it prefilters the file, and it is the floor on the longest common block.
_SHINGLE = 12
# Below this an anchor line carries no signal — `)` or `else:` is in every file,
# and offering a coincidence is worse than offering nothing.
_MIN_ANCHOR_CHARS = 12


@dataclass(frozen=True)
class Edit:
    """One anchored replacement. ``old`` must occur exactly once in ``path``."""

    path: str
    old: str
    new: str


@dataclass(frozen=True)
class ParsedEdits:
    """What :func:`parse_edits` made of the model's ``edits`` value.

    ``problems`` is non-empty when the value is not the schema — a *shape*
    complaint, distinct from an anchor that does not match, so the model is told
    which of the two it got wrong.
    """

    edits: list[Edit]
    problems: list[str]


def parse_edits(raw: Any) -> ParsedEdits:
    """Validate the ``edits`` value from the answer JSON.

    Accepts only a list of objects with string ``path``/``old``/``new``. Paths
    are normalised and checked: repo-relative, no ``..``, no absolute path, no
    leading ``a/``-style prefix. A path escape is the one problem here that is a
    *security* boundary rather than a quality one — the applier writes files, so
    the path the model chose is checked before anything is read or written.
    """
    if not isinstance(raw, list):
        return ParsedEdits([], ["`edits` must be a JSON list of edit objects."])
    if len(raw) > MAX_EDITS:
        return ParsedEdits(
            [],
            [
                f"`edits` has {len(raw)} entries, more than the {MAX_EDITS} allowed. "
                "Make the smallest change that fixes the failure."
            ],
        )

    edits: list[Edit] = []
    problems: list[str] = []
    for index, item in enumerate(raw, start=1):
        label = f"edit {index}"
        if not isinstance(item, dict):
            problems.append(f"{label}: not a JSON object.")
            continue
        path = item.get("path")
        old = item.get("old")
        new = item.get("new")
        if not isinstance(path, str) or not path.strip():
            problems.append(f'{label}: missing a string "path".')
            continue
        if not isinstance(old, str) or not old:
            problems.append(
                f'{label} ({path}): "old" must be a non-empty string copied from '
                "the file."
            )
            continue
        if not isinstance(new, str):
            problems.append(f'{label} ({path}): "new" must be a string.')
            continue
        clean = _safe_path(path)
        if clean is None:
            problems.append(
                f"{label} ({path}): not a repository-relative path. Use the path "
                "as it appears in the repository, with no leading `/`, `a/` or "
                "`b/` and no `..`."
            )
            continue
        if old == new:
            problems.append(
                f'{label} ({clean}): "old" and "new" are identical, so the edit '
                "changes nothing."
            )
            continue
        edits.append(Edit(path=clean, old=old, new=new))

    return ParsedEdits(edits, problems)


def _safe_path(path: str) -> Optional[str]:
    """``path`` as a repo-relative path, or ``None`` if it escapes the repo.

    Only the textual part of the check lives here; the caller is still expected
    to resolve symlinks against the worktree root before writing.
    """
    candidate = path.strip().replace("\\", "/")
    if candidate.startswith("/") or (len(candidate) > 1 and candidate[1] == ":"):
        return None
    for prefix in ("a/", "b/", "./"):
        while candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
    normalised = posixpath.normpath(candidate)
    if not normalised or normalised == "." or normalised.startswith("../"):
        return None
    return normalised


def apply_edits(
    edits: list[Edit], *, read: Callable[[str], Optional[str]]
) -> tuple[dict[str, str], list[str]]:
    """Apply ``edits`` in memory. Returns ``(contents, problems)``.

    ``contents`` maps each edited path to its new text, and is empty when
    ``problems`` is non-empty: the edits are **all or nothing**, so a worktree is
    never left holding half of a rejected answer. Render ``problems`` for the
    model with :func:`rejection_feedback`, and for a one-line error with
    :func:`summarize`.

    ``read`` maps a repo-relative path to its text, or ``None`` when there is no
    such file. Anchored edits only ever modify a file that already exists — a new
    file has nothing to anchor to, and `patch` remains the format for that.

    Edits are applied in order against the accumulated text, so several edits to
    one file compose, and an anchor may legitimately be created by an earlier
    edit. Every edit is still evaluated even after one fails, so the correction
    turn is told about all of them at once instead of one per round trip.
    """
    contents: dict[str, str] = {}
    original: dict[str, str] = {}
    problems: list[str] = []
    for index, edit in enumerate(edits, start=1):
        label = f"edit {index} ({edit.path})"
        if edit.path not in contents:
            text = read(edit.path)
            if text is None:
                problems.append(
                    f"{label}: there is no such file in the checkout. Anchored "
                    "edits can only change a file that already exists."
                )
                continue
            contents[edit.path] = text
            original[edit.path] = text
        text = contents[edit.path]
        count = text.count(edit.old)
        if count == 1:
            contents[edit.path] = text.replace(edit.old, edit.new, 1)
            continue
        if count == 0:
            problems.append(f"{label}: `old` does not occur in the file.")
            hint = _did_you_mean(text, edit.old)
            if hint:
                problems.append(hint)
        else:
            lines = _occurrence_lines(text, edit.old)
            problems.append(
                f"{label}: `old` occurs {count} times (starting at "
                f"{_join_lines(lines)}), so it does not identify one place. "
                "Extend it with the lines above and below until it is unique."
            )

    if problems:
        return {}, problems
    # A file whose edits cancelled out is not a change; leaving it out keeps
    # the synthesized diff to what the model actually did.
    return {p: t for p, t in contents.items() if t != original[p]}, []


def summarize(problems: list[str]) -> str:
    """The problems as one line, for a job error and an operator's eye.

    Indented entries are the quoted file content that goes with the entry above
    them, which belongs in the model's feedback and not in an error string.
    """
    reasons = [p for p in problems if not p.startswith(" ")]
    head = "; ".join(reasons[:3])
    if len(reasons) > 3:
        head += f"; and {len(reasons) - 3} more"
    return head


def rejection_feedback(problems: list[str]) -> str:
    """The correction turn's feedback for a rejected set of edits.

    Deliberately free of any echo of what the model wrote: the rejected answer
    is already in the conversation, and repeating a fabricated anchor next to the
    real content is what kept the fabrication alive when `git apply -v`'s search
    block was sent back (see ``tasks.apply_error_for_model``).
    """
    body = "\n".join(problems)
    return (
        "Your edits were rejected. Nothing was changed — an edit is applied only "
        "when its `old` text occurs EXACTLY ONCE in the file it names.\n\n"
        f"{body}\n\n"
        "Re-read the file with the browse tools and copy `old` from what you see, "
        "character for character, including trailing commas and whitespace. Do "
        "not reconstruct it from memory or from a previous attempt. If the text "
        "you meant to change is not in the file, then the change itself is wrong: "
        "say so in `body` and return an empty `edits` list rather than inventing "
        "an anchor."
    )


def _did_you_mean(text: str, old: str) -> str:
    """The real lines that come closest to a missing anchor's first line.

    An anchor that does not occur is either aimed at the wrong file or is a
    near-miss of a real line — one comma, one prefix. The near-miss is the
    common case and the one the model cannot resolve on its own, because what it
    believes the line says is exactly what it will write again. Quoting the line
    that is actually there settles it; quoting nothing leaves it guessing.

    "Close" is measured as *how much of the anchor appears in the line, in
    order* — not ``difflib``'s ratio, whose denominator counts the other line's
    length too and so scores a near-miss down for the very prefix that is
    missing. The real `zamba` pair (``"<s><s> Tell me…"`` against
    ``"[PAD]×6<s> Tell me…"``) rates 0.57 by ratio and 0.96 by this measure.
    """
    # Probe with the LONGEST line of the anchor, not the first. A multi-line
    # anchor usually opens on boilerplate — `self.assertEqual(` matches every
    # assertion in the file — while the line that disagrees with the file is the
    # long one carrying the value. Probing the first line quotes back two
    # unrelated assertions; probing the longest lands on the real text.
    anchor = max((line.strip() for line in old.splitlines()), key=len, default="")
    if len(anchor) < _MIN_ANCHOR_CHARS:
        # Too short to be distinctive: every guess would be a coincidence.
        return ""
    lines = text.splitlines()
    stripped = [line.strip() for line in lines]
    hits = [i for i, line in enumerate(stripped) if line == anchor][:_MAX_CANDIDATES]
    if not hits:
        hits = _closest(anchor, stripped)
    if not hits:
        return ""
    blocks = [_window(lines, i) for i in sorted(hits)]
    lead = (
        "  The file does contain this, which is close to a line of your anchor "
        "— it is the source of truth, your `old` must match it byte for byte:"
    )
    return lead + "\n" + "\n\n".join(blocks)


def _closest(anchor: str, stripped: list[str]) -> list[int]:
    """Indexes of the lines most of ``anchor`` appears in, best first.

    The shingle pass is a prefilter, not the measure: without it this would run
    a ``SequenceMatcher`` against every line of the file for every failed
    anchor, which on a 3,000-line test module is seconds of CPU inside a
    rejection path. With it, only lines sharing a run of ``_SHINGLE`` characters
    are scored at all.
    """
    shingles = {
        anchor[i : i + _SHINGLE]
        for i in range(0, len(anchor) - _SHINGLE + 1, max(1, len(anchor) // 40))
    }
    scored: list[tuple[float, int, int]] = []
    for index, line in enumerate(stripped):
        if not any(shingle in line for shingle in shingles):
            continue
        matcher = difflib.SequenceMatcher(None, anchor, line, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        longest = matcher.find_longest_match(0, len(anchor), 0, len(line)).size
        share = matched / len(anchor)
        if share >= _CLOSE_ENOUGH and longest >= _SHINGLE:
            scored.append((share, longest, index))
    scored.sort(reverse=True)
    return [index for _, _, index in scored[:_MAX_CANDIDATES]]


def _window(lines: list[str], index: int) -> str:
    """``lines`` around ``index``, numbered the way ``tools.read_file`` numbers
    its output so the model is not asked to reconcile two renderings."""
    lo = max(0, index - _WINDOW_PAD)
    hi = min(len(lines), index + _WINDOW_PAD + 1)
    return "\n".join(f"{i + 1:>6}\t{lines[i]}" for i in range(lo, hi))


def _occurrence_lines(text: str, needle: str) -> list[int]:
    """1-based line numbers where ``needle`` starts."""
    found: list[int] = []
    start = 0
    while True:
        at = text.find(needle, start)
        if at < 0:
            return found
        found.append(text.count("\n", 0, at) + 1)
        start = at + 1


def _join_lines(lines: list[int]) -> str:
    shown = [f"line {n}" for n in lines[:6]]
    if len(lines) > 6:
        shown.append("…")
    return ", ".join(shown)
