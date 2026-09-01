"""Detect a patch that makes its own test pass by changing what "passing" means.

The GPU verify gate re-runs the targeted tests against the candidate patch. That
is strong evidence for a *code* fix and **no evidence at all** for an
*expectation* fix: if the patch rewrote the assertion, re-running it passes by
construction. serge nevertheless stamped every published PR with "✅ Verified on
GPU … opened this PR only after they passed with this patch", including
huggingface/transformers#48437, which rewrote a fill-mask assertion to expect
``<unk>`` — a masked-LM asserting it predicts ``<unk>`` at its own mask.

`prompts.py` already tells a *reviewer* never to accept that reasoning
(the CHANGED EXPECTATIONS section). This module is the other half: it decides
mechanically, before the PR body is written, whether the claim may be made.

The rule
--------
A change is **expectation-only** when:

1. every file it touches is a test file, **and**
2. removing string and number literals from the changed lines leaves the two
   sides *identical*.

Rule 2 is what separates the two kinds of test edit, and it was designed against
five real serge PRs (see ``tests/test_expectation_guard.py``):

* ``#48437`` ``assertEqual(pred_token, "happiness")`` → ``"<unk>"`` — the
  skeleton is untouched, only literals move. **Expectation-only.**
* ``#48440`` ``generate(**inputs, max_new_tokens=20, do_sample=False)`` →
  ``…, tokenizer=tokenizer)`` — a new identifier appears, so the call itself
  changed. **A real fix**, and it was merged as one.

Deliberately **not** a blocker. Re-recording a genuinely changed model output is
ordinary maintenance (``#48439``), and this module cannot tell that from
``#48437``. It removes the false evidence and labels the change; a human still
decides. What it must never do is let "the tests pass now" stand as an argument
for a value the patch itself chose.

Scope note: rule 2 also catches a loosened ``atol``/``rtol`` and a bumped
``max_new_tokens``. That is intended — those also make the test pass by
redefining passing, and they deserve the same "no automated confidence" label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["PatchClassification", "classify_patch", "is_test_path"]


# A quoted literal, honouring backslash escapes, in either quote style. Ordered
# longest-first so a triple-quoted block is consumed whole rather than as three
# empty strings.
_STRING_LITERAL = re.compile(
    r'"""(?:\\.|[^\\])*?"""'
    r"|'''(?:\\.|[^\\])*?'''"
    r'|"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)
# A number literal. The leading \b keeps it from biting into an identifier —
# `float16` and `Sam2VisionModel` must survive intact, or every dtype change
# would read as a literal edit.
_NUMBER_LITERAL = re.compile(r"\b\d+\.?\d*(?:[eE][-+]?\d+)?\b")
# Punctuation that only ever reflects *formatting*. `#48439` reflowed one
# `Expectations` entry across four lines; dropping these makes that a no-op,
# which it is.
_LAYOUT = re.compile(r"[\s,()\[\]{}:]+")

_COMMENT = re.compile(r"^\s*#")

# Values that are not a new baseline but a broken one. Kept short and literal on
# purpose: this list is quoted at a human, so a false positive costs a sentence
# in a PR body, and a miss costs nothing that the label itself does not already
# carry.
_DEGENERATE = {
    "<unk>",
    "<pad>",
    "<empty>",
    "",
    "nan",
    "inf",
    "-inf",
    "none",
    "null",
}


def is_test_path(path: str) -> bool:
    """Whether a repo-relative path is a test file.

    Both halves are needed: transformers keeps its integration tests under
    ``tests/``, but a ``conftest.py`` or a ``test_*.py`` elsewhere is still test
    code, and a fixture under ``tests/`` that a source module imports is still
    the thing being changed.
    """
    parts = path.split("/")
    if "tests" in parts or "test" in parts:
        return True
    name = parts[-1]
    return (
        name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    )


def _residue(lines: list[str]) -> str:
    """What a set of changed lines says once every literal value is removed.

    Two sides with the same residue differ only in the *values* they mention —
    which is the whole question this module answers.
    """
    joined = "\n".join(lines)
    joined = _STRING_LITERAL.sub(" ", joined)
    joined = _NUMBER_LITERAL.sub(" ", joined)
    # Operators are kept (`==` vs `!=` is a behaviour change, not a value one);
    # only layout is dropped.
    return _LAYOUT.sub(" ", joined).strip()


def _literals(lines: list[str]) -> list[str]:
    out: list[str] = []
    for m in _STRING_LITERAL.finditer("\n".join(lines)):
        text = m.group(0)
        for q in ('"""', "'''", '"', "'"):
            if text.startswith(q) and text.endswith(q) and len(text) >= 2 * len(q):
                text = text[len(q) : -len(q)]
                break
        out.append(text)
    return out


def _is_degenerate(value: str) -> bool:
    stripped = value.strip()
    if stripped.lower() in _DEGENERATE:
        return True
    # An all-zero tensor/list slice: only zeros, separators and decimal points.
    if stripped and re.fullmatch(r"[0.,\s\[\]-]+", stripped) and "0" in stripped:
        return True
    return False


@dataclass
class PatchClassification:
    """What :func:`classify_patch` decided, and the evidence for it."""

    expectation_only: bool = False
    changed_files: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    #: Added literal values that read as broken rather than merely new.
    degenerate_values: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "expectation_only": self.expectation_only,
            "source_files": self.source_files,
            "test_files": self.test_files,
            "degenerate_values": self.degenerate_values,
        }

    def reason(self) -> str:
        """One sentence for a PR body or a log line."""
        if not self.expectation_only:
            return ""
        base = (
            "This patch changes only expected values in test files — the "
            "assertions were rewritten, not the code under test."
        )
        if self.degenerate_values:
            shown = ", ".join(f"`{v}`" for v in self.degenerate_values[:3])
            base += (
                f" One or more of the new values looks degenerate ({shown}): a "
                "result like that is usually a symptom, not a new baseline."
            )
        return base


def _iter_file_hunks(patch: str):
    """Yield ``(path, removed_lines, added_lines)`` per file in a unified diff."""
    path: str | None = None
    removed: list[str] = []
    added: list[str] = []
    for raw in patch.splitlines():
        if raw.startswith("diff --git "):
            if path is not None:
                yield path, removed, added
            path, removed, added = None, [], []
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            # /dev/null on a deletion; the a/ side already named the file.
            if target != "/dev/null":
                path = target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("--- "):
            source = raw[4:].strip()
            if path is None and source != "/dev/null":
                path = source[2:] if source.startswith("a/") else source
            continue
        if raw.startswith("@@"):
            continue
        if raw.startswith("+"):
            added.append(raw[1:])
        elif raw.startswith("-"):
            removed.append(raw[1:])
    if path is not None:
        yield path, removed, added


def _meaningful(lines: list[str]) -> list[str]:
    """Changed lines that carry code. Blank lines and comments are dropped: a
    reflowed comment must not read as a behaviour change (``#48437`` moved
    ``# [MASK] is token at 6th position`` along with the assertion), and a blank
    line never is one."""
    return [ln for ln in lines if ln.strip() and not _COMMENT.match(ln)]


def classify_patch(
    patch: str, changed_files: list[str] | None = None
) -> PatchClassification:
    """Classify a unified diff.

    Returns a :class:`PatchClassification` whose ``expectation_only`` is True
    only when every touched file is a test file *and* the change is confined to
    literal values. An empty or unparseable patch classifies as not
    expectation-only — the safe direction, since the flag only ever *removes* a
    claim serge would otherwise make.

    ``changed_files`` is the set of paths actually being committed, and it is
    **authoritative for the file split**. Pass it: ``patch`` is the LLM's
    proposal, but transformers' modular normalizer regenerates a
    ``modeling_*.py`` that the proposal never mentions, so a patch that looks
    test-only can commit a source file. Judging on the diff alone would call
    that an expectation change and strip a verification claim that was earned.
    """
    result = PatchClassification()
    literal_only = True
    saw_change = False

    for path, removed, added in _iter_file_hunks(patch):
        result.changed_files.append(path)

        old, new = _meaningful(removed), _meaningful(added)
        if not old and not new:
            continue
        saw_change = True
        if _residue(old) != _residue(new):
            literal_only = False

    for path in changed_files if changed_files is not None else result.changed_files:
        if path not in result.changed_files:
            result.changed_files.append(path)
        if is_test_path(path):
            if path not in result.test_files:
                result.test_files.append(path)
        elif path not in result.source_files:
            result.source_files.append(path)

    result.expectation_only = bool(
        saw_change and literal_only and not result.source_files
    )

    if result.expectation_only:
        before = set(
            _literals(
                _meaningful([ln for _, r, _ in _iter_file_hunks(patch) for ln in r])
            )
        )
        for value in _literals(
            _meaningful([ln for _, _, a in _iter_file_hunks(patch) for ln in a])
        ):
            if value not in before and _is_degenerate(value):
                result.degenerate_values.append(value)

    return result
