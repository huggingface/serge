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
