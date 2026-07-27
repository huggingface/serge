"""Watchdogs for agentic loops that get stuck after deciding what to do."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WatchdogTrigger:
    reason: str
    pattern: str
    occurrences: int
    tool_calls: int
    min_tool_calls: int
    repeated_intent_limit: int
    post_intent_tool_calls: int
    post_intent_reasoning_chars: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "reason": self.reason,
            "pattern": self.pattern,
            "occurrences": self.occurrences,
            "tool_calls": self.tool_calls,
            "min_tool_calls": self.min_tool_calls,
            "repeated_intent_limit": self.repeated_intent_limit,
            "post_intent_tool_calls": self.post_intent_tool_calls,
            "post_intent_reasoning_chars": self.post_intent_reasoning_chars,
        }


class WatchdogEarlyExit(BaseException):
    """Raised by local replay callers that need to break out of callbacks."""

    def __init__(self, trigger: WatchdogTrigger) -> None:
        payload = trigger.as_dict()
        super().__init__(str(payload.get("reason") or "watchdog early exit"))
        self.payload = payload
        self.trigger = trigger


class RepeatedIntentWatchdog:
    """Detect loops that keep deciding without acting or finishing.

    The agent can legitimately spend many turns investigating. This catches a
    narrower failure mode: decisive language such as "the fix should be..." or
    "let me write the patch" followed by more browsing/reasoning instead of an
    edit tool or final answer. It also catches local triage runs that conclude
    "no patch" or "final decision" and then continue browsing.
    """

    INTENT_PATTERNS = (
        re.compile(r"\blet me write the patch\b", re.IGNORECASE),
        re.compile(r"\blet me write the diff\b", re.IGNORECASE),
        re.compile(r"\blet me implement (?:it|this|the fix)\b", re.IGNORECASE),
        re.compile(
            r"\bi(?:'|’)ll (?:now )?(?:add|write|make) "
            r"(?:the )?(?:patch|change|fix)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bthe fix (?:is|should be) to\b", re.IGNORECASE),
        re.compile(r"\bi think the fix (?:is|should be)\b", re.IGNORECASE),
        re.compile(r"\bfinal decision:\s*(?:no patch|no change)\b", re.IGNORECASE),
        re.compile(r"\b(?:no patch|no change) (?:is needed|needed)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:i think|i believe) (?:the )?slowness is expected\b", re.IGNORECASE
        ),
        re.compile(r"\bthis is expected framework overhead\b", re.IGNORECASE),
        re.compile(r"\bthe slowness is expected\b", re.IGNORECASE),
        re.compile(r"\bthe answer is no patch\b", re.IGNORECASE),
        re.compile(r"\bi should just .*return no patch\b", re.IGNORECASE),
        re.compile(r"\bproduce (?:the )?no-patch result\b", re.IGNORECASE),
        re.compile(r"\breturn (?:a )?no-patch result\b", re.IGNORECASE),
        re.compile(r"\bemit final json with an empty patch\b", re.IGNORECASE),
    )
    EDIT_TOOL_NAMES = {
        "apply_patch",
        "edit_file",
        "replace_file",
        "write_file",
    }

    def __init__(
        self,
        repeated_intent_limit: int,
        min_tool_calls: int,
        post_intent_tool_calls: int,
        post_intent_reasoning_chars: int,
    ) -> None:
        self.repeated_intent_limit = repeated_intent_limit
        self.min_tool_calls = min_tool_calls
        self.post_intent_tool_calls = post_intent_tool_calls
        self.post_intent_reasoning_chars = post_intent_reasoning_chars
        self.reasoning = ""
        self.intent_counts: dict[str, int] = {}
        self.tool_calls = 0
        self.edit_tool_seen = False
        self.first_intent_tool_calls: int | None = None
        self.first_intent_reasoning_chars: int | None = None
        self.first_intent_pattern: str | None = None
        self.total_reasoning_chars = 0
        self.trigger: WatchdogTrigger | None = None

    @property
    def enabled(self) -> bool:
        return self.repeated_intent_limit > 0

    def observe(self, kind: str, text: str) -> WatchdogTrigger | None:
        if not self.enabled or self.trigger is not None:
            return self.trigger

        if kind == "tool":
            self.tool_calls += 1
            tool_name = text.split("(", 1)[0].strip()
            if tool_name in self.EDIT_TOOL_NAMES:
                self.edit_tool_seen = True
            if (
                self.first_intent_tool_calls is not None
                and not self.edit_tool_seen
                and self.post_intent_tool_calls > 0
                and self.tool_calls - self.first_intent_tool_calls
                >= self.post_intent_tool_calls
            ):
                return self._trigger(
                    reason="Decisive intent was followed by more non-edit tool calls",
                    pattern=self.first_intent_pattern or "",
                    occurrences=max(self.intent_counts.values(), default=1),
                )
            return None

        if kind != "reasoning" or self.edit_tool_seen:
            return None

        chunk = text.lower()
        self.total_reasoning_chars += len(text)
        if self.reasoning and re.match(r"^[a-z0-9_']+$", chunk):
            self.reasoning += chunk
        else:
            self.reasoning += " " + chunk
        # Keep enough context for split streamed tokens while bounding memory.
        if len(self.reasoning) > 20000:
            self.reasoning = self.reasoning[-12000:]

        if self.tool_calls < self.min_tool_calls:
            return None

        normalized_reasoning = re.sub(r"\s+", " ", self.reasoning)
        for pattern in self.INTENT_PATTERNS:
            count = len(pattern.findall(normalized_reasoning))
            key = pattern.pattern
            previous = self.intent_counts.get(key, 0)
            if count <= previous:
                continue
            self.intent_counts[key] = count
            if self.first_intent_tool_calls is None:
                self.first_intent_tool_calls = self.tool_calls
                self.first_intent_reasoning_chars = self.total_reasoning_chars
                self.first_intent_pattern = pattern.pattern
            if count >= self.repeated_intent_limit:
                return self._trigger(
                    reason=(
                        "Repeated decisive intent detected without an edit/final result"
                    ),
                    pattern=pattern.pattern,
                    occurrences=count,
                )

        if (
            self.first_intent_reasoning_chars is not None
            and self.post_intent_reasoning_chars > 0
            and self.total_reasoning_chars - self.first_intent_reasoning_chars
            >= self.post_intent_reasoning_chars
        ):
            return self._trigger(
                reason=(
                    "Decisive intent was followed by extended reasoning "
                    "without an edit/final result"
                ),
                pattern=self.first_intent_pattern or "",
                occurrences=max(self.intent_counts.values(), default=1),
            )

        return None

    def _trigger(self, reason: str, pattern: str, occurrences: int) -> WatchdogTrigger:
        self.trigger = WatchdogTrigger(
            reason=reason,
            pattern=pattern,
            occurrences=occurrences,
            tool_calls=self.tool_calls,
            min_tool_calls=self.min_tool_calls,
            repeated_intent_limit=self.repeated_intent_limit,
            post_intent_tool_calls=self.post_intent_tool_calls,
            post_intent_reasoning_chars=self.post_intent_reasoning_chars,
        )
        return self.trigger
