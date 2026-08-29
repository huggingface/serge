"""Guard against an agent re-issuing the same tool call forever.

The failure mode this exists for, measured on prod task 433e8274 (glm_ocr
integration-failure triage): the model asked for

    grep(pattern="GlmOcrProcessor|Glm46VProcessor", path="src/transformers/models/glm_ocr")

**21 times**, and another grep **44 times**, each returning the identical "no
matches" — and hit the 2M cumulative-input-token cap after ~55 turns and only
60s of LLM wall time. It never reasoned about the fix at all: the budget was
gone, so the loop forced a tool-less final answer and the patch that came out
failed GPU verification. All three verify rounds died the same way.

The tool calls themselves are cheap. What is expensive is the *turn* around each
one: every repeat resends the whole conversation, ~40k input tokens a time. So
the cure is not to memoize the tool — it is to tell the model it is looping and,
if it keeps going, to stop the loop early and spend the remaining budget on an
answer instead of on the 50th identical grep.

There is a second shape the exact-match counter cannot see, and it is the one
that dominates. Measured on prod task 9d210794 (blt): 153 tool calls, of which
**137 were repeat visits to a path it had already opened** — ``modular_blt.py``
53 times, at a different line range each time. Every one of those is a distinct
signature, so the counter above stays at zero while the budget drains. So the
guard counts a second thing: visits per opened *path*, with its own thresholds,
and a nudge that names the file and the ranges already served rather than just
saying stop. A model that is told "you already have lines 1-200, 180-420 and
400-650 of this file" can act on that; "you are repeating" it cannot.

Deliberately **not** a cache, in either direction. A repeated ``read_file``
after an ``apply_patch`` must return the *new* content, so every call still
executes for real; the guard only appends a correction to the result and counts
how often it has had to.
"""

from __future__ import annotations

import json


# Appended to a repeated call's result. Addressed at the model: it names what
# happened, why continuing is pointless, and the two ways out (change the
# arguments, or answer). Kept short — it rides along on every repeat.
_NUDGE = (
    "\n\n[serge] You have already made this exact {name} call {count} times in "
    "this session and it returned the same thing every time. Repeating it cannot "
    "tell you anything new, and each attempt consumes a large share of your "
    "remaining token budget. Either search for something different (a different "
    "pattern, a different path) or stop searching and give your final answer now."
)

_TRIPPED_NOTE = (
    "\n\n[serge] This is repeat #{repeats}. You are in a loop. Give your final "
    "answer from what you already know — no further tool calls will be answered."
)


# Appended when a call re-opens a path already opened in this session. Unlike
# the exact-repeat nudge this one is *actionable*: it names the file and what
# was already served, so the model can decide it has the bytes rather than
# guess. Ranges are listed because "you read this file 12 times" invites a 13th
# read of a part it has not seen — "you have lines 1-200, 180-420" does not.
_PATH_NUDGE = (
    "\n\n[serge] You have already {verb} `{path}` {count} times in this session "
    "({ranges}). {advice} If you need a part you have not seen yet, ask for those "
    "line numbers specifically; otherwise work from what you already have."
)

_PATH_TRIPPED_NOTE = (
    "\n\n[serge] {revisits} of your tool calls have now re-opened a path you had "
    "already opened. You are not learning anything new by browsing. Give your "
    "final answer from what you have — no further tool calls will be answered."
)

# How many ranges to name before eliding. Enough to show the overlap, short
# enough that the nudge stays cheap on every repeat.
_MAX_LISTED_RANGES = 4


def _range_label(parsed: dict) -> str:
    """Human-readable span for one ``read_file`` call, e.g. ``lines 400-650``."""
    start = parsed.get("start_line")
    end = parsed.get("end_line")
    start = start if isinstance(start, int) else None
    end = end if isinstance(end, int) else None
    if start is None and end is None:
        return "the whole file"
    if start is not None and end is not None:
        return f"lines {start}-{end}"
    if start is not None:
        return f"from line {start}"
    return f"up to line {end}"


def normalize_arguments(arguments: str) -> str:
    """Canonical form of a tool call's arguments, for identity comparison.

    Parsed and re-dumped with sorted keys so that key order and whitespace don't
    disguise the same call as a new one; falls back to the stripped raw string
    when the arguments aren't JSON (some providers emit partial or non-JSON
    arguments, and two identical malformed strings are still a repeat)."""
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return (arguments or "").strip()
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return (arguments or "").strip()


# Tools whose ``path`` argument names something the agent has *opened*. A second
# ``read_file`` of the same file is a re-visit even when the line range differs —
# that is the shape that dominated the measured prod window (137 of 153 calls in
# job 9d210794 were repeat visits to an already-opened path, ``modular_blt.py``
# 53 times). ``grep`` deliberately is not here: the same path with a different
# pattern is a genuinely new search, not a re-read.
_PATH_TOOLS = frozenset({"read_file", "list_dir"})


def path_visit(name: str, arguments: str) -> tuple[str, str] | None:
    """The ``(path, range_label)`` a call opened, or ``None`` for a tool that
    opens nothing.

    The path is normalized only enough to make two spellings of the same file
    compare equal (``./x.py`` and ``x.py``); no filesystem access, so this stays
    usable in the metrics path where the checkout may already be gone."""
    if name not in _PATH_TOOLS:
        return None
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("path")
    if not isinstance(raw, str):
        # list_dir's path is optional and defaults to the repo root; a missing
        # path is still a visit to a real location, so name it.
        if name != "list_dir":
            return None
        return ".", "the directory"
    path = raw.strip().lstrip("/")
    while path.startswith("./"):
        path = path[2:]
    path = path.rstrip("/") or "."
    label = "the directory" if name == "list_dir" else _range_label(parsed)
    return path, label


def path_argument(name: str, arguments: str) -> str | None:
    """Just the path :func:`path_visit` would report. Kept for callers that do
    not care which part of the file was asked for."""
    visit = path_visit(name, arguments)
    return None if visit is None else visit[0]


class ToolRepeatGuard:
    """Count identical tool calls and produce a correction for the repeats.

    ``trip_after`` is the total number of *repeat* calls (across all signatures)
    tolerated before :attr:`tripped` goes true and the caller should break out of
    the agent loop and force a final answer. ``0`` disables the guard entirely.

    Counting repeats globally rather than per-signature is what catches the real
    prod shape: the stuck model alternated between two identical greps, so a
    per-signature limit of 6 would have allowed 12 wasted turns and a limit low
    enough to stop that would misfire on legitimate re-reads.
    """

    def __init__(
        self,
        trip_after: int = 6,
        *,
        path_revisit_limit: int = 3,
        path_trip_after: int = 40,
    ):
        self.trip_after = trip_after
        # Visits to one path tolerated before every further visit is nudged.
        # Higher than the exact-repeat allowance on purpose: re-opening a file
        # at a genuinely new range is normal investigation, and the healthy
        # sessions in the measured window did it a handful of times each.
        self.path_revisit_limit = path_revisit_limit
        # Total re-opens across all paths before the loop is cut off. Set well
        # above any healthy session (the worst measured one spent 137 calls
        # this way, the one that produced a PR about 10) so this only catches
        # the pathological shape. 0 disables the cut-off, leaving the nudges.
        self.path_trip_after = path_trip_after
        self.counts: dict[str, int] = {}
        self.repeats = 0
        # Visits per opened path, counted whether or not the guard is enabled:
        # this is the session's observability record (see :meth:`stats`), and it
        # has to be there even for a deployment that turned the nudge off.
        self.path_visits: dict[str, int] = {}
        # Range labels already served per path, in order, for the nudge text.
        self.path_ranges: dict[str, list[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.trip_after > 0

    @property
    def tripped(self) -> bool:
        """True when the loop should stop and spend what is left on an answer.

        Either counter can trip it: byte-identical repeats, or sheer volume of
        re-opening paths already read. They are separate budgets because they
        are separate failure modes — the first is a model stuck on one call, the
        second is a model browsing in circles without noticing."""
        return self._repeats_tripped or self._paths_tripped

    @property
    def _repeats_tripped(self) -> bool:
        return self.enabled and self.repeats >= self.trip_after

    @property
    def _paths_tripped(self) -> bool:
        return self.path_trip_after > 0 and self.path_revisits >= self.path_trip_after

    def observe(self, name: str, arguments: str) -> str | None:
        """Record one tool call. Returns text to append to its result when the
        call is a repeat, else ``None``.

        The first occurrence of a call is always free — legitimate investigation
        re-lists directories and re-reads files. Only an *exact* re-run of a call
        already made in this session is treated as a repeat.

        A call that re-opens a path already opened gets the path nudge instead —
        it is the more useful correction, and the two would otherwise stack on
        the same result."""
        path_note = self._observe_path(name, arguments)
        if not self.enabled:
            return path_note
        signature = f"{name}\x00{normalize_arguments(arguments)}"
        count = self.counts.get(signature, 0) + 1
        self.counts[signature] = count
        if count == 1:
            return path_note
        self.repeats += 1
        note = _NUDGE.format(name=name, count=count)
        if self._repeats_tripped:
            note += _TRIPPED_NOTE.format(repeats=self.repeats)
        return note

    def _observe_path(self, name: str, arguments: str) -> str | None:
        """Record one path visit; return the nudge when it is a re-open past the
        allowance. Counting happens whether or not the exact-repeat guard is
        enabled — the session's browse record must not depend on that setting."""
        visit = path_visit(name, arguments)
        if visit is None:
            return None
        path, label = visit
        count = self.path_visits.get(path, 0) + 1
        self.path_visits[path] = count
        self.path_ranges.setdefault(path, []).append(label)
        if self.path_revisit_limit <= 0 or count <= self.path_revisit_limit:
            return None
        note = _PATH_NUDGE.format(
            verb="listed" if name == "list_dir" else "read",
            path=path,
            count=count,
            ranges=self._describe_ranges(path),
            advice=(
                "Listing a directory you have not changed cannot tell you anything new."
                if name == "list_dir"
                else "Re-reading a file you have not modified cannot tell you "
                "anything new."
            ),
        )
        if self._paths_tripped:
            note += _PATH_TRIPPED_NOTE.format(revisits=self.path_revisits)
        return note

    def _describe_ranges(self, path: str) -> str:
        """The spans already served for ``path``, deduplicated and elided."""
        seen: list[str] = []
        for label in self.path_ranges.get(path, [])[:-1]:  # exclude this call
            if label not in seen:
                seen.append(label)
        if not seen:
            return "the same range each time"
        shown = ", ".join(seen[:_MAX_LISTED_RANGES])
        return shown + (", …" if len(seen) > _MAX_LISTED_RANGES else "")

    def path_summary(self) -> str:
        """One-line description of the re-opened paths, for the run log."""
        worst = ", ".join(f"{path}×{n}" for path, n in self.worst_paths())
        return f"{self.path_revisits} re-open(s) of an already-opened path" + (
            f" (most re-opened: {worst})" if worst else ""
        )

    def trip_summary(self) -> str:
        """Why the loop is being cut off, naming the counter that did it.

        The two are separate budgets, so reporting the wrong one is actively
        confusing: a path trip described as "0 repeated tool call(s)" reads like
        a bug in the guard rather than a model browsing in circles."""
        if self._repeats_tripped and self._paths_tripped:
            return f"{self.summary()}; {self.path_summary()}"
        if self._paths_tripped:
            return self.path_summary()
        return self.summary()

    def summary(self) -> str:
        """One-line description of the worst offenders, for the run log."""
        worst = sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)
        parts = [
            f"{sig.split(chr(0), 1)[0]}×{count}"
            for sig, count in worst[:3]
            if count > 1
        ]
        return f"{self.repeats} repeated tool call(s)" + (
            f" (most repeated: {', '.join(parts)})" if parts else ""
        )

    # ------------------------------------------------------------------
    # observability
    # ------------------------------------------------------------------
    @property
    def distinct_paths(self) -> int:
        """How many separate files/directories the session opened."""
        return len(self.path_visits)

    @property
    def path_revisits(self) -> int:
        """Calls spent re-opening a path already opened in this session.

        The headline waste number: with 153 calls over 12 distinct paths, 137 of
        them were this. Counted as ``visits - 1`` per path, so a file read once
        contributes nothing.
        """
        return sum(count - 1 for count in self.path_visits.values() if count > 1)

    def worst_paths(self, limit: int = 3) -> list[tuple[str, int]]:
        """The most re-opened paths, highest first — for logs and nudges."""
        ranked = sorted(self.path_visits.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(path, count) for path, count in ranked[:limit] if count > 1]

    def stats(self) -> dict[str, int]:
        """Flat counters for the per-job metrics record."""
        return {
            "repeats": self.repeats,
            "distinct_paths": self.distinct_paths,
            "path_revisits": self.path_revisits,
        }
