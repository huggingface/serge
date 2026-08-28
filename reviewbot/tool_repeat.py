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

Deliberately **not** a cache. A repeated ``read_file`` after an ``apply_patch``
must return the *new* content, so every call still executes for real; the guard
only appends a correction to the result and counts how often it has had to.
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


def path_argument(name: str, arguments: str) -> str | None:
    """The path a call opened, or ``None`` when the tool does not open one.

    Normalized only enough to make two spellings of the same file compare equal
    (``./x.py`` and ``x.py``); no filesystem access, so it stays usable in the
    metrics path where the checkout may already be gone."""
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
        return "." if name == "list_dir" else None
    path = raw.strip().lstrip("/")
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/") or "."


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

    def __init__(self, trip_after: int = 6):
        self.trip_after = trip_after
        self.counts: dict[str, int] = {}
        self.repeats = 0
        # Visits per opened path, counted whether or not the guard is enabled:
        # this is the session's observability record (see :meth:`stats`), and it
        # has to be there even for a deployment that turned the nudge off.
        self.path_visits: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.trip_after > 0

    @property
    def tripped(self) -> bool:
        return self.enabled and self.repeats >= self.trip_after

    def observe(self, name: str, arguments: str) -> str | None:
        """Record one tool call. Returns text to append to its result when the
        call is a repeat, else ``None``.

        The first occurrence of a call is always free — legitimate investigation
        re-lists directories and re-reads files. Only an *exact* re-run of a call
        already made in this session is treated as a repeat."""
        path = path_argument(name, arguments)
        if path is not None:
            self.path_visits[path] = self.path_visits.get(path, 0) + 1
        if not self.enabled:
            return None
        signature = f"{name}\x00{normalize_arguments(arguments)}"
        count = self.counts.get(signature, 0) + 1
        self.counts[signature] = count
        if count == 1:
            return None
        self.repeats += 1
        note = _NUDGE.format(name=name, count=count)
        if self.tripped:
            note += _TRIPPED_NOTE.format(repeats=self.repeats)
        return note

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
