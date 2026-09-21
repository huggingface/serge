"""Derive a readable task report from a job's persisted event history.

The task page used to be a live console and nothing else: every fact about a run
— did the GPU reproduce confirm the failure, did the normalizer pass, what did
the agent loop spend, which tools it leaned on — was in there, interleaved with
a few hundred lines of transcript, and only findable by scrolling. This module
turns that same history into three structured views the page renders *outside*
the stream:

* :func:`build_steps` — the phases the run actually went through, in order,
  each with a status, the lines that justify it, and the tokens spent inside it.
* :func:`build_tool_usage` — per-tool call counts and how much text each tool
  put into (and pulled out of) the conversation.
* :func:`split_instruction` — the task text cut into sections so the page can
  keep the boilerplate collapsed and show the part that differs per group.

Everything here is a pure function over ``job.history``, so it works on jobs
already on disk — no re-run, no new event, no migration. That is the whole
reason it parses log strings rather than reading structured fields: the strings
are this repo's own, emitted a few dozen lines away in :mod:`reviewbot.tasks`
and :mod:`reviewbot.verify`, and parsing them is what lets the page explain a
run that finished before this module existed.

Token figures come in two flavours and the page must not blur them:

* Per-step ``tokens_in``/``tokens_out`` are **real** provider counts, differenced
  out of the cumulative ``metrics`` events the agent loop already emits.
* Per-tool figures are **character counts**, and the token column beside them is
  ``chars / 4``. serge does not ship the tokenizer of the model it is pointed at
  (the provider is configurable), so an exact per-tool token count does not
  exist here — see the same reasoning at ``tasks._prompt_prefix_log``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
#
# A phase is either an explicit ``step`` event or one synthesised from a log
# line, because the two GPU gates run *between* steps and emit no step of their
# own: `gpu_reproduce` sits between `preflight` and `llm`, and `gpu_verify`
# after `commit`. Ordering is by event sequence, never by this table.

# ``step`` text -> (phase key, label). ``llm:3/80`` is a turn marker, not a
# phase, and folds into the `llm` phase it belongs to.
_STEP_PHASES: dict[str, tuple[str, str]] = {
    "launch": ("launch", "Launch runner"),
    "clone": ("clone", "Checkout"),
    "preflight": ("preflight", "Preflight"),
    # Emitted by tasks._history_notes: serge's own history lookups, which run
    # between the reproduce gate and the first turn. One step covers both the
    # prior-art search and, on a regression cluster, the culprit PR's thread.
    "history": ("history", "Project history"),
    "llm": ("llm", "Agent loop"),
    "apply": ("apply", "Apply patch"),
    "normalize": ("normalize", "Normalize"),
    "commit": ("commit", "Commit"),
    "done": ("done", "Finish"),
}

# A log line starting with one of these opens a phase of its own.
_LOG_PHASES: tuple[tuple[str, str, str], ...] = (
    ("GPU reproduce: dispatching", "gpu_reproduce", "GPU reproduce"),
    ("GPU verify: dispatching", "gpu_verify", "GPU verify"),
    ("Opened PR", "pr", "Pull request"),
)

_OK = "ok"
_FAILED = "failed"
_WARN = "warn"
_RUNNING = "running"
_INFO = "info"

# Per-phase verdict rules, most specific first: (status, substring). Matched
# against every log line the phase collected, in the order listed, so a failure
# marker beats a success marker emitted earlier in the same phase.
_PHASE_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "launch": ((_OK, "Launching task runner"),),
    "clone": ((_OK, "Checkout ready in"),),
    "preflight": (
        (_FAILED, "Preflight fails on the pristine checkout"),
        (_WARN, "Preflight unavailable"),
        (_OK, "Preflight passed."),
    ),
    "gpu_reproduce": (
        (_FAILED, "not reproducible"),
        (_FAILED, "did NOT fail at the base commit"),
        (_FAILED, "already PASS on"),
        (_WARN, "environment issue"),
        (_OK, "reproduced ✓"),
    ),
    "gpu_verify": (
        (_FAILED, "did not confirm the fix"),
        (_WARN, "could not run"),
        (_WARN, "not accepting"),
        (_WARN, "skipping"),
        (_OK, "fixed ✓"),
    ),
    "normalize": (
        (_FAILED, "Patch validation failed"),
        (_OK, "normalizer is clean"),
    ),
    "commit": ((_OK, "Created branch"),),
    # A guard cutting the loop off is not a failure — the run can and often does
    # still produce a good patch — but it is the single most useful thing to see
    # at a glance, because it means the model never decided it was done.
    "llm": (
        (_WARN, "Stuck in a tool-call loop"),
        (_WARN, "Stuck re-opening"),
        (_OK, "LLM done:"),
    ),
    "apply": ((_FAILED, "patch did not apply"),),
    "history": (
        (_WARN, "lookup failed"),
        (_WARN, "relore did not answer"),
        (_OK, "already discuss these tests"),
        # A search that ran and matched nothing is a real answer, not a miss.
        (_OK, "no earlier thread matched"),
        # A regression cluster only: the blamed PR's discussion was fetched.
        (_OK, "its discussion is in the"),
    ),
    "pr": ((_OK, "Opened PR"),),
    "done": ((_OK, "Opened PR"),),
}

# Log lines that are pure progress noise inside a phase: one per LLM turn, and
# the turn-by-turn "calling" chatter. Kept out of the step detail so a 22-turn
# loop does not render 22 identical rows.
_NOISE_PREFIXES = (
    "LLM turn (blind=",
    "Runner pod ",
)


def _is_noise(text: str) -> bool:
    return any(text.startswith(p) for p in _NOISE_PREFIXES)


def _step_phase(text: str) -> Optional[tuple[str, str]]:
    """The phase an ``llm:3/80``-style step belongs to, or None if unknown."""
    base = text.split(":", 1)[0].strip()
    return _STEP_PHASES.get(base)


def _log_phase(text: str) -> Optional[tuple[str, str]]:
    for prefix, key, label in _LOG_PHASES:
        if text.startswith(prefix):
            return key, label
    return None


class _Phase:
    def __init__(self, key: str, label: str, index: int, ts: Optional[float]):
        self.key = key
        self.label = label
        self.index = index
        self.started_at = ts
        self.ended_at = ts
        self.lines: list[str] = []
        self.failed = False
        self.tool_calls = 0
        self.metrics_before: Optional[dict[str, Any]] = None
        self.metrics_last: Optional[dict[str, Any]] = None
        self.tests: list[dict[str, Any]] = []
        self.run_url: Optional[str] = None

    def add_line(self, text: str) -> None:
        """Append unless an earlier line in this phase already said it.

        Several facts are logged twice — once where they happen and once by the
        caller reporting the outcome — and the shorter of the pair is a prefix
        of the longer ("Opened PR #48945." against "Opened PR #48945
        (draft->ready): <url>"). Rendering both reads as two separate events.
        """
        trimmed = text.rstrip(" .…")
        for i, existing in enumerate(self.lines):
            if existing.rstrip(" .…").startswith(trimmed):
                return
            if trimmed.startswith(existing.rstrip(" .…")):
                self.lines[i] = text
                return
        self.lines.append(text)

    def status(self) -> str:
        if self.failed:
            return _FAILED
        for status, needle in _PHASE_RULES.get(self.key, ()):
            if any(needle in line for line in self.lines):
                return status
        return _INFO


_RUN_URL_RE = re.compile(r"https://github\.com/\S+/actions/runs/\d+")


def _num(raw: Any) -> Optional[float]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def _metrics_delta(phase: _Phase, key: str) -> Optional[int]:
    """One cumulative counter's growth across ``phase``.

    The loop emits ``metrics`` twice over: a char-estimated overlay every ~0.75s
    while a turn streams, and the authoritative provider counts once the turn
    lands. Both are cumulative and the authoritative one is last, so taking the
    final reading inside the phase minus the final reading before it is right
    for either — and ``max(0, …)`` guards the one case it is not, a streaming
    estimate that overshot the turn it was estimating.
    """
    if phase.metrics_last is None:
        return None
    end = _num(phase.metrics_last.get(key))
    if end is None:
        return None
    start = 0.0
    if phase.metrics_before is not None:
        start = _num(phase.metrics_before.get(key)) or 0.0
    return int(max(0.0, end - start))


def build_steps(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The phases a task run went through, in order.

    Each entry carries ``key``, ``label``, ``status`` (``ok``/``failed``/
    ``warn``/``info``/``running``), the ``lines`` that justify the status, the
    wall-clock window, and the real token counters spent inside it.
    """
    phases: list[_Phase] = []
    current: Optional[_Phase] = None
    # Cumulative metrics as of the moment the current phase opened.
    carried: Optional[dict[str, Any]] = None

    def open_phase(key: str, label: str, ts: Optional[float]) -> _Phase:
        nonlocal current, carried
        phase = _Phase(key, label, len(phases), ts)
        phase.metrics_before = carried
        phases.append(phase)
        current = phase
        return phase

    for event in history:
        kind = event.get("kind")
        text = str(event.get("text") or "")
        ts = _num(event.get("ts"))

        if kind == "step":
            found = _step_phase(text)
            if found is None:
                continue
            key, label = found
            # `llm:3/80` after `llm` is the same phase, one turn later.
            if current is not None and current.key == key:
                if ts is not None:
                    current.ended_at = ts
                continue
            open_phase(key, label, ts)
            continue

        if current is None:
            # Events before the first step (there normally are none) get an
            # anonymous bucket rather than being dropped.
            open_phase("start", "Start", ts)
            assert current is not None

        if ts is not None:
            current.ended_at = ts

        if kind == "log":
            found = _log_phase(text)
            if found is not None:
                key, label = found
                # Already inside that phase: the line belongs to it, it does not
                # start a second one. Two of these openers fire more than once
                # per run ("Opened PR …" is logged twice), and without this the
                # page showed the same phase twice in a row.
                phase = current if current.key == key else open_phase(key, label, ts)
                phase.add_line(text)
                if ts is not None:
                    phase.ended_at = ts
                continue
            if not _is_noise(text):
                current.add_line(text)
            url = _RUN_URL_RE.search(text)
            if url is not None:
                current.run_url = url.group(0)
            continue

        if kind in ("error", "normalize_error", "patch_apply_error"):
            current.failed = True
            current.add_line(text)
            continue

        if kind == "tool":
            current.tool_calls += 1
            continue

        if kind == "verify_result":
            # Emitted by reviewbot.tasks alongside the GPU gates: the verdict
            # artifact's per-node-id outcomes. Absent on jobs that ran before
            # that event existed, which is why the phase status never depends
            # on it.
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                continue
            targeted = payload.get("targeted")
            if isinstance(targeted, list):
                current.tests = [t for t in targeted if isinstance(t, dict)]
            continue

        if kind == "metrics":
            try:
                payload = json.loads(text)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            current.metrics_last = payload
            carried = payload
            continue

    out: list[dict[str, Any]] = []
    for phase in phases:
        if not phase.lines and not phase.tests and phase.metrics_last is None:
            continue
        entry: dict[str, Any] = {
            "key": phase.key,
            "label": phase.label,
            "status": phase.status(),
            "lines": phase.lines,
            "started_at": phase.started_at,
            "ended_at": phase.ended_at,
            "tool_calls": phase.tool_calls,
        }
        if phase.started_at is not None and phase.ended_at is not None:
            entry["seconds"] = round(max(0.0, phase.ended_at - phase.started_at), 1)
        turns = _metrics_delta(phase, "turns")
        if turns:
            entry["turns"] = turns
        if phase.run_url:
            entry["run_url"] = phase.run_url
        if phase.tests:
            entry["tests"] = phase.tests
        tokens_in = _metrics_delta(phase, "in")
        tokens_out = _metrics_delta(phase, "out")
        if tokens_in:
            entry["tokens_in"] = tokens_in
        if tokens_out:
            entry["tokens_out"] = tokens_out
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Tool usage
# ---------------------------------------------------------------------------

# `_emit_chat_message` truncates a tool result at 2,000 chars and appends the
# count it dropped, so the TRUE size is recoverable from a stored event. Without
# this the table would report every large grep as exactly 2,000 chars and the
# one number that says what the loop actually paid for would be a constant.
_TRUNCATED_RE = re.compile(r"… \[\+(\d+) chars truncated\]$")

# Rough chars-per-token. Only ever used for the per-tool column, which is
# labelled an estimate in the UI for this reason.
_CHARS_PER_TOKEN = 4


def _true_len(content: Any, recorded: Any = None) -> int:
    """Length of a chat payload's text before the event log truncated it.

    ``recorded`` is the exact pre-truncation length when the emitter stored one
    (newer jobs); otherwise it is recovered from the truncation marker.
    """
    exact = _num(recorded)
    if exact is not None:
        return int(exact)
    text = str(content or "")
    match = _TRUNCATED_RE.search(text)
    if match is None:
        return len(text)
    return len(text) - len(match.group(0)) + int(match.group(1))


def build_tool_usage(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-tool call counts and text volume, busiest first.

    ``chars_in`` is what the model wrote to call the tool, ``chars_out`` what
    came back — the half that matters, because every tool result stays in the
    transcript and is re-sent on every later turn.
    """
    stats: dict[str, dict[str, int]] = {}

    def bucket(name: str) -> dict[str, int]:
        return stats.setdefault(name, {"calls": 0, "chars_in": 0, "chars_out": 0})

    for event in history:
        if event.get("kind") != "chat":
            continue
        try:
            payload = json.loads(str(event.get("text") or ""))
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        role = payload.get("role")
        if role == "assistant":
            for call in payload.get("tool_calls") or ():
                if not isinstance(call, dict):
                    continue
                entry = bucket(str(call.get("name") or "?"))
                entry["calls"] += 1
                entry["chars_in"] += _true_len(
                    call.get("arguments"), call.get("argument_chars")
                )
        elif role == "tool":
            name = payload.get("name")
            if not name:
                continue
            bucket(str(name))["chars_out"] += _true_len(
                payload.get("content"), payload.get("content_chars")
            )

    rows = []
    for name, entry in stats.items():
        rows.append(
            {
                "name": name,
                "calls": entry["calls"],
                "chars_in": entry["chars_in"],
                "chars_out": entry["chars_out"],
                "est_tokens_in": entry["chars_in"] // _CHARS_PER_TOKEN,
                "est_tokens_out": entry["chars_out"] // _CHARS_PER_TOKEN,
            }
        )
    rows.sort(key=lambda r: (-r["calls"], -r["chars_out"], r["name"]))
    return rows


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

# The nightly's task text is a fixed trunk plus a per-category addendum, and the
# addendum is the only part that differs between one task and the next (see
# `instruction_addendum` in transformers-ci's integration_failure_triage.py).
# It announces itself with a box-drawing rule, which is the split point: keep
# everything after the LAST such rule open and let the page fold the rest.
_RULE_RE = re.compile(r"^──\s*(.*?)\s*─*\s*$")


def split_instruction(text: Optional[str]) -> list[dict[str, Any]]:
    """The task text as titled sections, in order.

    Sections before the last box-drawing rule are the trunk every task carries;
    the last one is what makes this task different. ``primary`` marks the
    latter so the page can open it and collapse the rest — a 4,000-character
    wall of text where 3,000 characters are identical on every run is the
    reason the page was unreadable.
    """
    body = (text or "").strip()
    if not body:
        return []
    sections: list[dict[str, Any]] = []
    title: Optional[str] = None
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content or title:
            sections.append({"title": title, "body": content})

    for line in body.split("\n"):
        match = _RULE_RE.match(line.strip())
        if match is not None and match.group(1):
            flush()
            title = match.group(1).strip()
            buf = []
            continue
        buf.append(line)
    flush()

    for i, section in enumerate(sections):
        section["primary"] = i == len(sections) - 1 and len(sections) > 1
    return sections
