import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from . import __version__
from .compression import MessageCompressor
from .config import Config
from .context_script import run_context_script
from .github_client import GitHubClient
from .llm_client import (
    ChatCompletionClient,
    ChatResult,
    ToolCall,
    _parse_text_tool_calls,
)
from .patch import DiffSnippetLine, ParsedFile, extract_hunk_snippet, parse_patch
from .prompts import (
    build_followup_system_prompt,
    build_followup_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from .tool_repeat import ToolRepeatGuard
from .tools import (
    RepoHelperTool,
    ToolEnv,
    build_tool_specs,
    install_helper_tools,
    load_repo_helper_tools,
    run_tool,
)

log = logging.getLogger(__name__)


@dataclass
class InlineCommentContext:
    """Metadata captured from a ``pull_request_review_comment`` event so
    the follow-up flow can answer the question in-thread on the exact
    line the commenter was looking at.

    ``in_reply_to_id`` is set when the trigger comment was itself a reply
    inside an existing thread; the reply endpoint accepts any comment_id
    in the thread so we always use ``comment_id`` to post the reply.
    """

    comment_id: int
    path: str
    side: str
    line: int
    diff_hunk: str
    in_reply_to_id: Optional[int] = None


@dataclass
class ReviewRequest:
    owner: str
    repo: str
    number: int
    trigger_comment_id: int
    trigger_comment_body: str
    commenter: str
    # Populated only for pull_request_review_comment events. When set,
    # the runner dispatches to run_followup instead of run_review.
    inline: Optional[InlineCommentContext] = None


@dataclass
class DraftComment:
    """One validated inline comment, with a stable id the web UI can
    address when applying edits (override body, discard). ``diff_hunk``
    is a GitHub-style snippet around the commented line (empty if the
    patch wasn't available)."""

    id: str
    path: str
    side: str
    line: int
    body: str
    diff_hunk: list[DiffSnippetLine] = field(default_factory=list)


@dataclass
class ReviewDraft:
    """The output of prepare_review: everything needed to publish a
    review, but not yet posted. Tweakable from the web UI via edits."""

    owner: str
    repo: str
    number: int
    head_sha: str
    summary: str
    event: str
    comments: list[DraftComment] = field(default_factory=list)
    rejected_count: int = 0
    metrics_line: str = ""
    # Number of diff chunks that were skipped because the cumulative input
    # token budget (llm_max_input_tokens) was hit mid-review. Zero in the
    # normal case; non-zero means the review didn't cover every hunk.
    truncated_chunks: int = 0
    # Cumulative input/output token counts across every LLM call this
    # review made (all chunks + tool turns + synthesis). Stored as
    # separate fields so the journal can query them without parsing the
    # metrics_line string.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Resolved LLM model id that produced this review (after endpoint
    # auto-discovery, if llm_model was unset). Surfaced in the published
    # review footer; None when no LLM call was made.
    model: Optional[str] = None
    # Per-job agent-loop counters (:func:`session_record`). Like the token
    # counts above, this is a carrier field: it is NOT part of the draft JSON
    # a review pod ships back, it travels beside it on the terminal callback.
    session: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewEdits:
    """User-supplied tweaks applied at publish time. Only fields that are
    set override the draft; missing keys mean "use the draft value"."""

    summary: Optional[str] = None
    event: Optional[str] = None
    # Map of DraftComment.id -> override body (None = keep original).
    comment_overrides: dict[str, str] = field(default_factory=dict)
    # Set of DraftComment.id to discard (drop from the published review).
    discarded_comment_ids: set[str] = field(default_factory=set)


_FENCED_BLOCK_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```")
# Opening fence of a JSON block at the very start of a reply, and the fence
# that closes it. Used to peel the wrapper off a stub-JSON reply — see
# `_prose_outside_json`.
_LEADING_FENCE_RE = re.compile(r"\A```[ \t]*(?:json|JSON)?[ \t]*\r?\n?")
_CLOSING_FENCE_RE = re.compile(r"\A\s*```[ \t]*\r?\n?")
_TAGGED_DIFF_LINE_RE = re.compile(r"^\[(R|L)\s*(\d+)\] ")
_PARSE_PREVIEW_CHARS = 500


def _content_preview(text: str, limit: int = _PARSE_PREVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars truncated]"


def _extract_json(
    content: Optional[str], require_any_key: Optional[tuple[str, ...]] = None
) -> dict[str, Any]:
    """Forgiving JSON extraction. Tries, in order:

    1. Direct parse of the stripped content.
    2. Each fenced ``` block (with or without a `json` language tag).
    3. ``raw_decode`` starting at every ``{`` position, picking the first
       attempt that yields a JSON object.

    The third pass means trailing prose after the JSON ("Hope this helps!")
    or surrounding chatter ("Sure, here you go: {...}") doesn't break us.
    Raises ValueError with a length-and-preview diagnostic when nothing parses.

    ``require_any_key`` narrows what counts as a hit to objects carrying at
    least one of those keys. Without it, pass 3 will happily return *any*
    ``{...}`` in the reply — including a leaked tool call's own argument
    object, which is how ``{"path": ..., "start_line": ...}`` was once
    mistaken for a review and published as an empty summary (serge#79).
    Callers that know their JSON contract should always pass it.
    """
    if not content:
        raise ValueError("LLM response was empty")
    text = content.strip()
    if not text:
        raise ValueError("LLM response was whitespace only")

    decoder = json.JSONDecoder()

    def accept(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        return require_any_key is None or any(k in obj for k in require_any_key)

    try:
        result = decoder.decode(text)
        if accept(result):
            return result
    except json.JSONDecodeError:
        pass

    for match in _FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if accept(result):
            return result

    for idx in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            result, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if accept(result):
            return result

    expected = ""
    if require_any_key is not None:
        expected = f" with any of the keys {list(require_any_key)}"
    raise ValueError(
        f"LLM response did not contain a JSON object{expected} "
        f"(length={len(content)} chars, preview={_content_preview(text)!r})"
    )


# The review JSON contract from prompts.py — at least one of these must be
# present for a decoded object to be a review rather than incidental JSON.
_REVIEW_JSON_KEYS = ("summary", "comments", "event")

# Chat-template special tokens (`<|im_end|>`, `<|tool_call_begin|>`, …) that a
# model can leak into its content. `llm_client` recovers leaked tool calls, so
# reaching here means the leak wasn't a parseable tool call — treat text made of
# nothing but these as no review at all rather than publishing the markup.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _is_model_markup_only(text: str) -> bool:
    """True when ``text`` is non-empty but says nothing — it is entirely a
    leaked tool call, or entirely model special tokens.

    The first check reuses the tool-call parser rather than stripping tokens,
    because a leaked call leaves its own id and arguments between the tokens
    (``functions.read_file:6``) and token-stripping alone would read that as
    substance. Deliberately narrow: a legitimate review may quote
    ``<|endoftext|>`` while discussing a tokenizer, so we only reject text
    with *no* substance left — we never rewrite text that has some.
    """
    if not text.strip():
        return False
    calls, remainder = _parse_text_tool_calls(text)
    if calls and not remainder.strip():
        return True
    return not _SPECIAL_TOKEN_RE.sub("", text).strip()


def _prose_outside_json(content: Optional[str]) -> str:
    """The markdown left over once the JSON object `_extract_json` picked up
    (and any fence wrapping it) is removed.

    Models occasionally answer with a *stub* JSON object — empty summary, no
    comments — and then write the actual review as prose underneath, often
    forgetting the closing ``` of the fence they opened. Handing that whole
    reply to the reader publishes the fence and the stub verbatim, and an
    unterminated fence swallows the entire review into one code block.

    Returns "" when there is no prose to salvage, so callers can keep their
    own fallback.
    """
    text = (content or "").strip()
    if not text:
        return ""
    text = _LEADING_FENCE_RE.sub("", text, count=1)

    decoder = json.JSONDecoder()
    for idx in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            _, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        head = text[:idx]
        # Drop only the fence that closes the JSON block — a global strip
        # would eat legitimate code fences inside the prose review.
        tail = _CLOSING_FENCE_RE.sub("", text[idx + end :], count=1)
        return "\n\n".join(part for part in (head.strip(), tail.strip()) if part)
    return ""


@dataclass
class _DiffChunk:
    text: str
    parsed_by_path: dict[str, ParsedFile]
    visible_positions: dict[str, set[tuple[str, int]]]


def _copy_positions_map(
    positions: dict[str, set[tuple[str, int]]],
) -> dict[str, set[tuple[str, int]]]:
    return {path: set(vals) for path, vals in positions.items()}


def _extract_visible_positions(text: str) -> set[tuple[str, int]]:
    visible: set[tuple[str, int]] = set()
    for line in text.splitlines():
        m = _TAGGED_DIFF_LINE_RE.match(line)
        if not m:
            continue
        side = "RIGHT" if m.group(1) == "R" else "LEFT"
        visible.add((side, int(m.group(2))))
    return visible


def _split_annotated_block(
    path: str, parsed: ParsedFile, max_chars: int
) -> list[tuple[str, set[tuple[str, int]]]]:
    """Split a file diff into one or more prompt-sized blocks.

    We prefer hunk-aligned splits; if a single hunk is still too large,
    we fall back to line-based splits inside that hunk and repeat the
    hunk header in each fragment so the model keeps local context.
    """
    header = f"--- a/{path}\n+++ b/{path}\n"
    full = f"{header}{parsed.annotated}\n"
    if len(full) <= max_chars:
        return [(full, _extract_visible_positions(parsed.annotated))]

    budget = max(1, max_chars - len(header) - 1)
    raw_lines = parsed.annotated.splitlines(keepends=True)
    sections: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        if line.startswith("@@") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    units: list[str] = []
    for section_lines in sections:
        section_text = "".join(section_lines)
        if len(section_text) <= budget:
            units.append(section_text)
            continue

        prefix = (
            section_lines[0]
            if section_lines and section_lines[0].startswith("@@")
            else ""
        )
        remainder = section_lines[1:] if prefix else section_lines
        chunk_lines: list[str] = []
        chunk_len = len(prefix)
        for line in remainder:
            if chunk_lines and chunk_len + len(line) > budget:
                units.append(prefix + "".join(chunk_lines))
                chunk_lines = []
                chunk_len = len(prefix)
            chunk_lines.append(line)
            chunk_len += len(line)
        if chunk_lines or prefix:
            units.append(prefix + "".join(chunk_lines))

    blocks: list[tuple[str, set[tuple[str, int]]]] = []
    current_units: list[str] = []
    current_len = 0
    for unit in units:
        if current_units and current_len + len(unit) > budget:
            body = "".join(current_units)
            block = f"{header}{body}"
            if not block.endswith("\n"):
                block += "\n"
            blocks.append((block, _extract_visible_positions(body)))
            current_units = [unit]
            current_len = len(unit)
        else:
            current_units.append(unit)
            current_len += len(unit)
    if current_units:
        body = "".join(current_units)
        block = f"{header}{body}"
        if not block.endswith("\n"):
            block += "\n"
        blocks.append((block, _extract_visible_positions(body)))
    return blocks


def _build_annotated_diff_chunks(
    files: list[dict],
    max_chars: int,
    skip_paths: set[str],
) -> tuple[list[_DiffChunk], list[str]]:
    """Build one or more review chunks without dropping diff content.

    `max_chars` is now a per-chunk budget, not a whole-PR truncation
    threshold. Large PRs are split across multiple prompts so every
    changed hunk remains reviewable.
    """
    chunks: list[_DiffChunk] = []
    skipped: list[str] = []
    current_parts: list[str] = []
    current_parsed: dict[str, ParsedFile] = {}
    current_visible: dict[str, set[tuple[str, int]]] = {}
    current_len = 0

    for f in files:
        path = f.get("filename")
        patch = f.get("patch")
        if not path or not patch:
            continue
        if path in skip_paths:
            skipped.append(path)
            continue

        parsed = parse_patch(path, patch)
        for block, visible in _split_annotated_block(path, parsed, max_chars):
            if current_parts and current_len + len(block) > max_chars:
                chunks.append(
                    _DiffChunk(
                        text="".join(current_parts),
                        parsed_by_path=dict(current_parsed),
                        visible_positions=_copy_positions_map(current_visible),
                    )
                )
                current_parts = []
                current_parsed = {}
                current_visible = {}
                current_len = 0

            current_parts.append(block)
            current_len += len(block)
            current_parsed[path] = parsed
            current_visible.setdefault(path, set()).update(visible)

    if current_parts:
        chunks.append(
            _DiffChunk(
                text="".join(current_parts),
                parsed_by_path=dict(current_parsed),
                visible_positions=_copy_positions_map(current_visible),
            )
        )

    return chunks, skipped


def _validate_comments(
    raw_comments: list[dict[str, Any]],
    visible_positions: dict[str, set[tuple[str, int]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for c in raw_comments:
        path = c.get("path")
        side = c.get("side", "RIGHT")
        line = c.get("line")
        body = c.get("body")
        if not (
            isinstance(path, str) and isinstance(line, int) and isinstance(body, str)
        ):
            rejected.append(c)
            continue
        side = side if side in ("RIGHT", "LEFT") else "RIGHT"
        positions = visible_positions.get(path)
        if not positions or (side, line) not in positions:
            rejected.append(c)
            continue
        valid.append({"path": path, "side": side, "line": line, "body": body})
    return valid, rejected


@dataclass
class _AggregateMetrics:
    """Accumulated stats across all LLM turns in one agentic loop."""

    turns: int = 0
    tool_calls: int = 0
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Why the loop stopped. ``STOP_ANSWERED`` is the *only* value meaning the
    # model decided it was done; every other value is a guard cutting it off and
    # forcing an answer out of whatever budget was left. Worth recording per job:
    # in the one prod window we could still read, all seven sessions that ran LLM
    # turns ended on a guard and none on the model's own terms, which is not
    # visible anywhere today.
    stop_reason: str = "answered"
    # Copied off the ToolRepeatGuard at every exit (see :func:`_run_agentic_loop`).
    repeats: int = 0
    distinct_paths: int = 0
    path_revisits: int = 0
    # Times the answer was rejected and re-asked: the normalize/patch gate and
    # the truncated-final-answer salvage respectively.
    validation_retries: int = 0
    truncation_retries: int = 0


# Loop-exit reasons, recorded on _AggregateMetrics.stop_reason. Only ANSWERED
# means the model finished on its own terms.
STOP_ANSWERED = "answered"
STOP_INPUT_TOKEN_CAP = "input_token_cap"
STOP_REPEAT_GUARD = "repeat_guard"
# The path counter tripping, not the exact-argument one. Reported separately
# because they mean different things: `repeat_guard` is a model stuck re-issuing
# one call, `path_revisit_guard` is a model browsing the same files in circles.
STOP_PATH_REVISIT_GUARD = "path_revisit_guard"
STOP_BLIND_TURN_CAP = "blind_turn_cap"
STOP_STRICT_TOOL_CAP = "strict_tool_cap"
STOP_ABSOLUTE_CEILING = "absolute_ceiling"
STOP_CHUNK_BUDGET = "chunk_input_token_cap"
# Not a loop exit: the job finished (or failed) without ever running one. The
# reproduce-first gate classifying a group ENVIRONMENT is the common case — 3 of
# the 10 tasks in the measured window — and it has to be countable, otherwise
# "serge did nothing for 0 turns" and "serge never ran" look identical.
STOP_NO_LLM_TURNS = "no_llm_turns"
# The runner died before reporting anything, so what it spent is unknown. This
# must NOT be recorded as `no_llm_turns`: that value means "the job legitimately
# never reached the loop" and is a cheap outcome, and conflating the two makes an
# expensive job look free. Job `b228e033` was killed by TASK_RUNNER_TIMEOUT while
# polling for a GPU runner, after 28 LLM turns and 879,010 input tokens, and was
# recorded as `no_llm_turns` with turns=0 — 1 of the 7 `no_llm_turns` rows in the
# store was this, i.e. 14% of the "free bail" population was not free at all.
STOP_RUNNER_LOST = "runner_lost"


def no_llm_session_record(stop_reason: str = STOP_NO_LLM_TURNS) -> dict[str, Any]:
    """A zeroed session for a job that never reported one.

    ``stop_reason`` distinguishes *why* the counters are zero: the default means
    the job really did skip the agent loop, while :data:`STOP_RUNNER_LOST` means
    the counters are unknown rather than zero.
    """
    return {
        "turns": 0,
        "tool_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "seconds": 0.0,
        "stop_reason": stop_reason,
        "repeats": 0,
        "distinct_paths": 0,
        "path_revisits": 0,
        "validation_retries": 0,
        "truncation_retries": 0,
        "rounds": 0,
    }


def _format_aggregated_metrics(m: "_AggregateMetrics") -> str:
    parts = [
        f"{m.turns} LLM turn{'s' if m.turns != 1 else ''}",
        f"{m.tool_calls} tool call{'s' if m.tool_calls != 1 else ''}",
        f"{m.latency_seconds:.1f}s",
    ]
    if m.prompt_tokens or m.completion_tokens:
        parts.append(
            f"{m.prompt_tokens or '?'} in / {m.completion_tokens or '?'} out tokens"
        )
    return " · ".join(parts)


def session_record(m: "_AggregateMetrics") -> dict[str, Any]:
    """The per-job counters worth keeping after the job row is evicted.

    ``WEB_JOB_RETENTION`` is 25 jobs — about two days of traffic — so anything
    only visible in the job row cannot be used to show that a change to the
    agent loop helped. These are the numbers that answer "did the model finish,
    or did a guard cut it off, and what did it spend the budget on?".
    """
    return {
        "turns": m.turns,
        "tool_calls": m.tool_calls,
        "prompt_tokens": m.prompt_tokens,
        "completion_tokens": m.completion_tokens,
        "seconds": round(m.latency_seconds, 1),
        "stop_reason": m.stop_reason,
        "repeats": m.repeats,
        "distinct_paths": m.distinct_paths,
        "path_revisits": m.path_revisits,
        "validation_retries": m.validation_retries,
        "truncation_retries": m.truncation_retries,
        "rounds": 1,
    }


# Counters that add up when a job runs the agent loop more than once — a task
# with several candidate groups, or a GPU-verify retry round.
_SESSION_SUMS = (
    "turns",
    "tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "repeats",
    "path_revisits",
    "validation_retries",
    "truncation_retries",
)


def _rounds(record: dict[str, Any]) -> int:
    """A record's loop count, defaulting to one.

    Not ``or 1``: ``rounds: 0`` is a real value — a job that never reached the
    agent loop — and reading it as 1 would invent a loop that never ran.
    """
    value = record.get("rounds")
    return 1 if value is None else int(value)


def merge_session_records(
    total: Optional[dict[str, Any]], part: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Fold one agent-loop session into a job-level total.

    A single task job can run the loop several times (one per candidate group,
    plus a round per GPU-verify retry), and the thing worth reporting is what the
    *job* spent — the 2M input-token cap is per loop, but the bill is per job.
    ``rounds`` counts how many loops were folded in; ``stop_reason`` reports the
    last loop that a guard cut off, since one cut-off round is what starves the
    patch that gets published."""
    if not part:
        return dict(total or {})
    if not total:
        merged = dict(part)
        # ``part`` may itself already be an accumulation (a candidate that ran
        # several verify rounds); don't reset its count to one.
        merged["rounds"] = _rounds(part)
        return merged
    merged = dict(total)
    for key in _SESSION_SUMS:
        merged[key] = int(merged.get(key, 0) or 0) + int(part.get(key, 0) or 0)
    merged["seconds"] = round(
        float(merged.get("seconds", 0.0) or 0.0)
        + float(part.get("seconds", 0.0) or 0.0),
        1,
    )
    merged["distinct_paths"] = max(
        int(merged.get("distinct_paths", 0) or 0),
        int(part.get("distinct_paths", 0) or 0),
    )
    merged["rounds"] = _rounds(merged) + _rounds(part)
    if part.get("stop_reason", STOP_ANSWERED) != STOP_ANSWERED:
        merged["stop_reason"] = part["stop_reason"]
    return merged


def _merge_metrics(total: "_AggregateMetrics", part: "_AggregateMetrics") -> None:
    total.turns += part.turns
    total.tool_calls += part.tool_calls
    total.latency_seconds += part.latency_seconds
    total.prompt_tokens += part.prompt_tokens
    total.completion_tokens += part.completion_tokens
    total.repeats += part.repeats
    total.path_revisits += part.path_revisits
    total.validation_retries += part.validation_retries
    total.truncation_retries += part.truncation_retries
    # Each chunk browses with a fresh guard, so distinct paths are per-chunk and
    # summing them double-counts a file two chunks both opened. An upper bound is
    # the honest reading available without keeping every path around.
    total.distinct_paths = max(total.distinct_paths, part.distinct_paths)
    # A cut-off chunk means the session was cut off, whatever later chunks did.
    if part.stop_reason != STOP_ANSWERED:
        total.stop_reason = part.stop_reason


def _make_tool_env(
    cfg: Config, helper_tools: list[RepoHelperTool] | None = None
) -> Optional[ToolEnv]:
    if not cfg.repo_checkout_path:
        if helper_tools:
            log.info(
                "Repo helper tools are configured, but repo_checkout_path is empty; "
                "running without tools"
            )
        return None
    try:
        env = ToolEnv(
            repo_root=cfg.repo_checkout_path,
            helper_tools={tool.name: tool for tool in helper_tools or []},
            sandbox_mode=cfg.helper_sandbox,
        )
    except Exception:
        log.exception("repo checkout path invalid; running without browse tools")
        return None
    log.info("Browse tools enabled, rooted at %s", env.repo_root)
    return env


def _load_helper_tools(
    gh: GitHubClient, owner: str, repo: str, pr: dict, cfg: Config
) -> list[RepoHelperTool]:
    if not cfg.helper_tools_path:
        return []
    default_branch = pr.get("base", {}).get("repo", {}).get("default_branch") or "main"
    try:
        content = gh.get_file_contents(
            owner,
            repo,
            cfg.helper_tools_path,
            ref=default_branch,
        )
    except Exception:
        log.exception("failed to fetch helper tools config")
        return []
    if not content:
        return []
    try:
        helpers = load_repo_helper_tools(content)
    except ValueError:
        log.exception("failed to parse helper tools config")
        return []
    if helpers:
        log.info(
            "Loaded %d repo helper tool(s) from %s: %s",
            len(helpers),
            cfg.helper_tools_path,
            ", ".join(tool.name for tool in helpers),
        )
    return helpers


def _install_helper_tools_with_emit(
    helpers: list[RepoHelperTool],
    emit: Callable[[str, str], None],
) -> None:
    """Run any declared install hooks before the agent loop starts.

    Failures are reported via logs + the streaming UI but don't abort
    the review — if the helper really isn't on PATH, its first tool
    call will surface that to the model.
    """
    pending = [h for h in helpers if h.install]
    if not pending:
        return
    emit("step", "install")
    emit(
        "log",
        f"Installing {len(pending)} helper tool(s): "
        + ", ".join(h.name for h in pending),
    )
    for result in install_helper_tools(pending):
        if result.ok:
            log.info("helper install ok: %s", result.message)
            emit("log", result.message)
        else:
            log.warning("helper install failed: %s: %s", result.name, result.message)
            emit("log", f"{result.name}: install FAILED — {result.message}")


def _build_runner_context(
    *,
    all_files: list[dict],
    skipped: list[str],
    chunk_index: int,
    chunk_total: int,
) -> Optional[str]:
    notes: list[str] = []
    if chunk_total > 1:
        notes.append(
            f"This PR diff was split into {chunk_total} chunks because the full diff exceeded "
            f"the per-call budget. You are reviewing chunk {chunk_index} of {chunk_total}."
        )
        notes.append(
            "Only place inline comments on lines shown in this chunk's diff below."
        )
        changed_paths = [f.get("filename") for f in all_files if f.get("filename")]
        if changed_paths:
            notes.append(
                "Changed files in the full PR:\n- " + "\n- ".join(changed_paths)
            )
    if skipped:
        notes.append(
            "The following files were excluded from this review by the target repo's "
            "`.ai/context-script`. Do NOT review their contents and do NOT place inline "
            "comments on them. Refer to REPO-PROVIDED CONTEXT for the reason and any related "
            "guidance:\n- " + "\n- ".join(skipped)
        )
    return "\n\n".join(notes) if notes else None


def _merge_chunk_summaries(summaries: list[tuple[int, str]], chunk_total: int) -> str:
    """Fallback merge used when the synthesis LLM call is unavailable or
    fails. Joins per-chunk summaries with blank lines and never mentions
    chunking — that is an internal implementation detail and would
    confuse anyone reading the published review on GitHub."""
    clean = [
        summary.strip()
        for _, summary in summaries
        if isinstance(summary, str) and summary.strip()
    ]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return "\n\n".join(clean)


_SYNTHESIS_SYSTEM_PROMPT = """You are merging several partial code-review
summaries into ONE coherent pull-request review.

Inputs you receive:
- The PR title (for grounding).
- A numbered list of partial summaries; each was written after looking
  at a different section of the same PR's diff.

Your job: produce a single summary that reads as if one reviewer wrote
it after seeing the whole PR. The output must:

1. NEVER mention chunks, sections, parts, "the first summary", "the
   second summary", or the merge process itself. Write as a peer
   engineer leaving one review on the PR page.
2. Be GitHub-flavored markdown. Open with a one-sentence verdict, then
   group findings under a few `**bold**` or `##` headings
   (Correctness, Security, Style, Tests, etc.) — skip headings with
   nothing to say. Use bullet lists for individual points. Use
   backticks for paths, function names, and short code references.
3. Deduplicate overlapping observations. If two partials flagged the
   same issue, mention it once.
4. Preserve concrete file/function references from the partials.
5. Stay tight — a few paragraphs or short bulleted sections, not a
   wall of text.

Output ONLY the markdown summary. No preamble, no JSON, no code fences
around the whole thing.
"""


def _synthesize_merged_summary(
    llm: ChatCompletionClient,
    summaries: list[tuple[int, str]],
    *,
    pr_title: str,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
    emit: Optional[Callable[[str, str], None]] = None,
) -> tuple[Optional[str], Optional["_AggregateMetrics"]]:
    """Run a small synthesis LLM call to merge per-chunk summaries into
    a single PR-level review. Returns (text, metrics) on success or
    (None, None) on any failure — the caller falls back to a plain join.

    Tools are off here on purpose: we already have the per-chunk
    findings in hand, the synthesis call only needs to rewrite them into
    one cohesive markdown summary."""
    clean = [
        (idx, summary.strip())
        for idx, summary in summaries
        if isinstance(summary, str) and summary.strip()
    ]
    if len(clean) < 2:
        return None, None

    parts = [f"PR title: {pr_title or '(no title)'}", "", "Partial summaries:"]
    for chunk_idx, summary in clean:
        parts.append(f"\n[{chunk_idx}]\n{summary}")
    user_prompt = "\n".join(parts)

    if emit:
        emit("step", "llm")
        emit("log", "Merging per-chunk summaries into one PR review…")

    metrics = _AggregateMetrics()
    chunk_cb = _wrap_chunk_cb(emit, metrics)
    try:
        chat = llm.complete(
            [
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            chunk_callback=chunk_cb,
            extra={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("synthesis merge failed: %s", exc)
        return None, None

    metrics.turns += 1
    metrics.latency_seconds += chat.latency_seconds
    if chat.prompt_tokens is not None:
        metrics.prompt_tokens += chat.prompt_tokens
    if chat.completion_tokens is not None:
        metrics.completion_tokens += chat.completion_tokens
    _emit_metrics(emit, metrics)

    text = (chat.content or "").strip()
    if not text:
        return None, None
    return text, metrics


def _merge_chunk_event(events: list[str], comments_count: int) -> str:
    normalized = [e for e in events if e in ("COMMENT", "REQUEST_CHANGES", "APPROVE")]
    if any(e == "REQUEST_CHANGES" for e in normalized):
        return "REQUEST_CHANGES"
    if normalized and all(e == "APPROVE" for e in normalized) and comments_count == 0:
        return "APPROVE"
    return "COMMENT"


_DEFAULT_FORCE_FINAL_MESSAGE = (
    "You have used all available tool calls. Based on what you have "
    "already gathered, produce the final review now as a single JSON "
    "object with EXACTLY these keys:\n"
    '  - "summary": a non-empty markdown string with your overall review\n'
    '  - "event": one of "COMMENT", "REQUEST_CHANGES", or "APPROVE"\n'
    '  - "comments": an array of inline comments (may be empty)\n'
    "Reply with the JSON object only — no surrounding prose, no code "
    "fences, no extra commentary. Do not request any more tools."
)

# A final answer that stops with finish_reason="length" ran out of output
# budget mid-JSON — almost always because reasoning consumed it (the model
# provider caps completion tokens, so raising max_tokens does not help). Rather
# than fail the whole task, re-ask for the JSON only with reasoning minimised so
# the answer fits. Bounded so a pathological model can't loop forever.
_MAX_TRUNCATION_RETRIES = 2
_TRUNCATION_RECOVERY_MESSAGE = (
    "Your previous reply was cut off at the model's output-token limit before "
    "you produced the complete JSON object — the reasoning used up the budget. "
    "Reply again with ONLY the final JSON object the task requires: no "
    "analysis, no explanation, no <think> block, no markdown fences. Keep it "
    "minimal so the whole answer fits within the output limit."
)
# An empty final answer (blank content, usually finish_reason=None) means the
# model returned nothing parseable — often the provider truncated the stream on
# a very large context. Re-ask once, tool-free, for just the JSON — same
# recovery path as the length-truncation case, bounded by the same retry cap.
_EMPTY_ANSWER_RECOVERY_MESSAGE = (
    "Your previous reply was empty — no JSON object came through. Reply now "
    "with ONLY the final JSON object the task requires: no analysis, no "
    "explanation, no <think> block, no markdown fences."
)
# A reply that is non-empty but is *only* chat-template markup. Says what was
# wrong explicitly, because unlike the other two shapes this one is a mistake
# the model can avoid rather than a budget it ran out of.
_MARKUP_ANSWER_RECOVERY_MESSAGE = (
    "Your previous reply contained only tool-call markup — no JSON object came "
    "through. Do not write tool calls as text; you have no tools left on this "
    "turn. Reply now with ONLY the final JSON object the task requires: no "
    "analysis, no explanation, no <think> block, no markdown fences."
)

_RECOVERY_MESSAGES = {
    "empty": _EMPTY_ANSWER_RECOVERY_MESSAGE,
    "truncated": _TRUNCATION_RECOVERY_MESSAGE,
    "markup": _MARKUP_ANSWER_RECOVERY_MESSAGE,
}
# How each defect is named in the operator-facing log line.
_DEFECT_LABELS = {
    "empty": "empty",
    "truncated": "truncated",
    "markup": "nothing but tool-call markup",
}


def _final_answer_defect(chat: ChatResult) -> Optional[str]:
    """Which unusable shape the final answer has, or ``None`` when it is worth
    parsing. One classifier because all three callers need the same verdict:
    the loop (salvage or not), the log line (what was wrong) and the recovery
    prompt (what to tell the model).

    - ``"empty"`` — blank content, commonly ``finish_reason=None`` when the
      provider truncates the stream on a very large context.
    - ``"truncated"`` — hit the output-token limit (``finish_reason="length"``).
    - ``"markup"`` — non-empty but says nothing: the model wrote its tool calls
      out as chat-template special tokens instead of calling them (serge#81).

    ``"markup"`` is the one that used to escape. Its content is non-empty, so
    the older emptiness test said "parse it"; on the task path ``_extract_json``
    then raised ``_UnparseableLLMOutput`` with **zero** recovery attempts. The
    review path had a backstop for it after parsing, the task path had none —
    which is why 3 of 10 task jobs in the 2026-08-24→26 window died
    "unparseable". Checked last because it is the only test that has to walk the
    content.
    """
    if not (chat.content or "").strip():
        return "empty"
    if chat.finish_reason == "length":
        return "truncated"
    if _is_model_markup_only(chat.content or ""):
        return "markup"
    return None


def _needs_final_salvage(chat: ChatResult) -> bool:
    """True when a tool-free final answer should be re-asked rather than
    parsed."""
    return _final_answer_defect(chat) is not None


def _final_recovery_message(chat: ChatResult) -> str:
    return _RECOVERY_MESSAGES.get(
        _final_answer_defect(chat) or "", _TRUNCATION_RECOVERY_MESSAGE
    )


def _salvage_decision(
    defect: Optional[str], last_defect: Optional[str], attempts: int
) -> str:
    """Whether to re-ask for a defective final answer: ``"reask"``,
    ``"repeated"`` (the recovery already failed on this same shape) or
    ``"stop"``.

    One function because both salvage sites — the normal no-tool-calls turn and
    the forced final turn after a guard trip — need the identical rule, and the
    forced one is where the expensive case actually happens.

    Re-asking is worth one attempt: `9f20cf8b` returned an empty forced-final
    answer and recovery 1/2 produced the JSON. It is *not* worth a second
    attempt at the same shape: `36798024` and `d0dca820` each spent both
    re-asks going empty -> empty -> empty, ~39.5k input tokens apiece for
    ~75-100 output tokens of nothing, and neither recovered. A recovery that
    returns the same defect has demonstrably not worked, so the signal to stop
    is repetition, not an attempt count.
    """
    if defect is None:
        return "stop"
    if defect == last_defect:
        return "repeated"
    if attempts >= _MAX_TRUNCATION_RETRIES:
        return "stop"
    return "reask"


def _emit_salvage_giveup(
    emit: Optional[Callable[[str, str], None]], defect: str, attempts: int
) -> None:
    if emit is None:
        return
    emit(
        "log",
        f"Final answer was {_DEFECT_LABELS.get(defect, defect)} again after "
        f"{attempts} re-ask(s); the recovery is not working, not re-asking.",
    )


def _emit_final_salvage(
    emit: Optional[Callable[[str, str], None]], chat: ChatResult, attempt: int
) -> None:
    if emit is None:
        return
    what = _DEFECT_LABELS.get(_final_answer_defect(chat) or "", "unparseable")
    emit(
        "log",
        f"Final answer was {what} (finish_reason={chat.finish_reason}); re-asking "
        f"for the JSON only (recovery {attempt}/{_MAX_TRUNCATION_RETRIES})",
    )


def _run_agentic_loop(
    llm: ChatCompletionClient,
    initial_messages: list[dict[str, Any]],
    *,
    cfg: Config,
    tool_env: Optional[ToolEnv],
    emit: Optional[Callable[[str, str], None]] = None,
    prior_prompt_tokens: int = 0,
    final_force_message: Optional[str] = None,
    validate: Optional[Callable[[ChatResult], Optional[str]]] = None,
    max_validation_retries: int = 0,
) -> tuple[ChatResult, _AggregateMetrics]:
    """Run a tool-augmented chat loop until the model emits a final
    (non-tool) response, falling back to a final non-tool turn if the
    iteration budget is exhausted.

    Returns the *last* ChatResult (whose ``content`` carries the JSON
    review) and an aggregate-metrics struct.

    ``validate`` (optional) turns the final answer into a verification gate:
    when the model emits a non-tool response, ``validate(chat)`` is called and,
    if it returns a non-empty string, that string is appended to the *same*
    conversation as a user turn and the loop continues so the model can correct
    its answer (used by /tasks to feed the repo normalizer's rejection back to
    the model). ``validate`` returning ``None`` accepts the answer. The model
    gets at most ``max_validation_retries`` such corrective re-prompts; after
    that the last answer is returned regardless. Reviews pass no ``validate``
    and are unaffected."""
    messages = list(initial_messages)
    metrics = _AggregateMetrics()
    tools_arg = build_tool_specs(tool_env) if tool_env is not None else None

    # llm_client emits ("token", ...) for response content,
    # ("reasoning", ...) for chain-of-thought deltas, and
    # ("stream_metrics", ...) with a char-based estimate of the current
    # turn's output every ~0.75s. We wrap `emit` so stream_metrics is
    # overlaid on prior-turn cumulative totals and surfaced as the
    # regular "metrics" event the UI already consumes.
    chunk_cb: Optional[Callable[[str, str], None]] = _wrap_chunk_cb(emit, metrics)

    # We never pass `response_format` here. Several inference stacks
    # reject it alongside tools (Kimi-K2 on HF Router, vLLM with some
    # tool parsers) and Anthropic's OpenAI shim rejects
    # `{"type": "json_object"}` outright (only `"json_schema"` is
    # accepted). The system prompt's "output ONLY a single JSON object"
    # instruction plus _extract_json's forgiving parsing handle the
    # final answer just as well across every provider we target.

    # ``tool_max_iterations <= 0`` means "no cap". By default, the cap
    # counts only *blind* tool turns: the model emitted tool calls
    # without any reasoning OR content. Productive turns — where the
    # model either thought (reasoning_chars > 0), said something
    # (content), or returned a final answer (no tool calls) — don't
    # burn the budget. /tasks can opt into a stricter mode where the cap
    # counts total tool calls, preserving budget for the final patch JSON.
    iter_cap: Optional[int] = (
        cfg.tool_max_iterations if cfg.tool_max_iterations > 0 else None
    )
    # Hard cap on cumulative *input* tokens for the whole review. Once
    # the running prompt-token total (prior chunks + this chunk) crosses
    # this threshold we stop spinning the tool loop and force a final
    # answer. ``llm_max_input_tokens <= 0`` disables the cap.
    input_tokens_cap: Optional[int] = (
        cfg.llm_max_input_tokens if cfg.llm_max_input_tokens > 0 else None
    )
    # Absolute backstop: at least twice the configured blind-turn cap,
    # but never below 60. Prevents a runaway model from chaining tool
    # calls indefinitely while still leaving real investigations room
    # to grow when the operator raises tool_max_iterations.
    ABSOLUTE_ITER_CEILING = max(60, (iter_cap or 0) * 2)
    # Stops a model that has started re-issuing the *same* tool call from eating
    # the whole input-token budget on it. Each repeat is told it is repeating;
    # past the limit we break out and spend what's left on an answer.
    repeat_guard = ToolRepeatGuard(
        getattr(cfg, "tool_repeat_limit", 0) or 0,
        path_revisit_limit=getattr(cfg, "tool_path_revisit_limit", 0) or 0,
        path_trip_after=getattr(cfg, "tool_path_trip_after", 0) or 0,
    )
    iteration = 0
    blind_tool_turns = 0
    validation_retries = 0
    truncation_retries = 0
    # The defect the last re-ask was issued for. A re-ask that produces the same
    # shape again has demonstrably not worked; see the loop below.
    last_defect: Optional[str] = None

    def _finalize(chat: ChatResult) -> tuple[ChatResult, "_AggregateMetrics"]:
        """Attach the session's browse/retry record to the metrics on the way
        out. Every ``return`` from this function goes through here so a job's
        counters are complete no matter which exit it took."""
        metrics.repeats = repeat_guard.repeats
        metrics.distinct_paths = repeat_guard.distinct_paths
        metrics.path_revisits = repeat_guard.path_revisits
        metrics.validation_retries = validation_retries
        metrics.truncation_retries = truncation_retries
        return chat, metrics

    # Set for one turn to recover a truncated final answer: disable tools and
    # force minimal reasoning so the whole output budget goes to the JSON.
    force_json_only = False
    while True:
        iteration += 1
        if iteration > ABSOLUTE_ITER_CEILING:
            log.warning(
                "Agent loop hit absolute ceiling of %d iterations; bailing out",
                ABSOLUTE_ITER_CEILING,
            )
            metrics.stop_reason = STOP_ABSOLUTE_CEILING
            break
        if iter_cap is not None and blind_tool_turns >= iter_cap:
            metrics.stop_reason = STOP_BLIND_TURN_CAP
            break
        if (
            iter_cap is not None
            and getattr(cfg, "tool_max_iterations_strict", False)
            and metrics.tool_calls >= iter_cap
        ):
            log.warning(
                "Strict tool-call budget hit (%d >= %d); bailing out",
                metrics.tool_calls,
                iter_cap,
            )
            metrics.stop_reason = STOP_STRICT_TOOL_CAP
            break
        if (
            input_tokens_cap is not None
            and prior_prompt_tokens + metrics.prompt_tokens >= input_tokens_cap
        ):
            cumulative = prior_prompt_tokens + metrics.prompt_tokens
            log.warning(
                "Input token budget hit (%d >= %d); bailing out for final answer",
                cumulative,
                input_tokens_cap,
            )
            if emit is not None:
                emit(
                    "log",
                    f"Input token budget hit ({cumulative} >= {input_tokens_cap}); "
                    "asking for a final review without tools",
                )
            metrics.stop_reason = STOP_INPUT_TOKEN_CAP
            break
        if iter_cap is not None:
            if getattr(cfg, "tool_max_iterations_strict", False):
                label = f"{metrics.tool_calls}/{iter_cap}"
            else:
                label = f"{blind_tool_turns}/{iter_cap}"
        else:
            label = f"{iteration}"
        log.info(
            "Agent loop iteration raw=%d blind_tool_turns=%s",
            iteration,
            label,
        )
        if emit is not None:
            emit("step", f"llm:{label}")
            emit("log", f"LLM turn (blind={label})")
        turn_tools = None if force_json_only else tools_arg
        turn_effort = "low" if force_json_only else cfg.llm_reasoning_effort
        chat = llm.complete(
            messages,
            max_tokens=cfg.llm_max_tokens,
            tools=turn_tools,
            tool_choice="auto" if turn_tools else None,
            chunk_callback=chunk_cb,
            extra={"reasoning_effort": turn_effort} if turn_effort else None,
        )
        force_json_only = False
        metrics.turns += 1
        metrics.latency_seconds += chat.latency_seconds
        if chat.prompt_tokens is not None:
            metrics.prompt_tokens += chat.prompt_tokens
        if chat.completion_tokens is not None:
            metrics.completion_tokens += chat.completion_tokens
        _emit_metrics(emit, metrics)
        # Log what the model emitted this turn (content + finish_reason +
        # tool-call names). Captures the empty final turn behind
        # "unparseable output" failures.
        _emit_chat_message(
            emit,
            "assistant",
            content=chat.content,
            reasoning_chars=chat.reasoning_chars,
            finish_reason=chat.finish_reason,
            tool_calls=chat.tool_calls,
        )

        if not chat.tool_calls:
            # Salvage a truncated OR empty final answer before anything else:
            # the model either ran out of output budget mid-JSON
            # (finish_reason="length", reasoning ate it) or returned nothing
            # parseable at all (blank content — commonly finish_reason=None when
            # the provider truncates a huge-context stream). Re-ask for the JSON
            # only, tool-less and low-reasoning, instead of returning content
            # that just fails the parse.
            defect = _final_answer_defect(chat)
            decision = _salvage_decision(defect, last_defect, truncation_retries)
            if decision == "repeated":
                _emit_salvage_giveup(emit, defect or "", truncation_retries)
            elif decision == "reask":
                truncation_retries += 1
                last_defect = defect
                _emit_final_salvage(emit, chat, truncation_retries)
                messages.append({"role": "assistant", "content": chat.content or None})
                messages.append(
                    {"role": "user", "content": _final_recovery_message(chat)}
                )
                force_json_only = True
                continue
            if validate is None:
                return _finalize(chat)
            # Verification gate: let the caller check the final answer (for
            # /tasks: apply the patch + run the repo normalizer). A non-empty
            # string is a rejection to feed back; None accepts.
            feedback = validate(chat)
            if feedback is None or validation_retries >= max_validation_retries:
                return _finalize(chat)
            validation_retries += 1
            if emit is not None:
                emit(
                    "log",
                    f"Patch validation failed (correction "
                    f"{validation_retries}/{max_validation_retries}); "
                    "asking the model to fix it",
                )
            # Continue the SAME conversation: append the rejected answer and
            # the feedback, then loop so the model can browse + correct.
            messages.append({"role": "assistant", "content": chat.content or None})
            messages.append({"role": "user", "content": feedback})
            continue

        # A "blind" tool turn is one where the model fired tool calls
        # without reasoning or content — i.e. chaining tools without
        # thinking between them. Those are the turns we want to limit;
        # tool calls preceded by reasoning are healthy investigation.
        thought = bool(chat.reasoning_chars) or bool((chat.content or "").strip())
        if not thought:
            blind_tool_turns += 1
            log.info(
                "Blind tool turn (no reasoning/content); blind_tool_turns=%d",
                blind_tool_turns,
            )

        if tool_env is None:
            # Model emitted tool calls even though we didn't pass tools —
            # ignore them and treat the textual content as the answer.
            log.warning(
                "Model emitted %d tool_call(s) but tools are disabled; using "
                "content as final answer",
                len(chat.tool_calls),
            )
            return _finalize(chat)

        # Append the assistant's tool_calls turn so the next request has
        # the full conversation, then execute each call and append the
        # results as `tool` messages.
        messages.append(
            {
                "role": "assistant",
                "content": chat.content or None,
                "tool_calls": [_assistant_tool_call_dict(tc) for tc in chat.tool_calls],
            }
        )
        for tc in chat.tool_calls:
            metrics.tool_calls += 1
            if emit is not None:
                emit("tool", f"{tc.name}({_summarize_args_str(tc.arguments)})")
                _emit_metrics(emit, metrics)
            result = _execute_tool_call(tool_env, tc)
            # An identical re-run still executes (a re-read after an edit must
            # see the new content) — but the model is told it is repeating, so
            # it can break out before the repeat budget runs down.
            repeat_note = repeat_guard.observe(tc.name, tc.arguments)
            if repeat_note is not None:
                result = f"{result}{repeat_note}"
                log.info(
                    "Corrected tool call %s (%s); %s | %s",
                    tc.name,
                    _summarize_args_str(tc.arguments),
                    repeat_guard.summary(),
                    repeat_guard.path_summary(),
                )
            # Kimi-K2 (and some other engines) require ``name`` on tool
            # replies; OpenAI's spec ignores it. Always sending it is
            # the safer cross-provider choice.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result,
                }
            )
            _emit_chat_message(emit, "tool", content=result, tool_name=tc.name)

        if repeat_guard.tripped:
            log.warning(
                "Tool-repeat guard tripped (%s); asking for a final "
                "answer instead of spending the rest of the budget looping",
                repeat_guard.trip_summary(),
            )
            if emit is not None:
                emit(
                    "log",
                    f"Stuck in a tool-call loop ({repeat_guard.trip_summary()}); "
                    "asking for a final answer without tools",
                )
            metrics.stop_reason = (
                STOP_REPEAT_GUARD
                if repeat_guard._repeats_tripped
                else STOP_PATH_REVISIT_GUARD
            )
            break

    # Iteration budget hit — force a final answer with tools disabled.
    # Only reachable when ``tool_max_iterations > 0`` and the model
    # used up its content-turn allowance, or when the absolute ceiling
    # tripped (runaway tool calling).
    log.warning(
        "Agent budget exhausted (blind_tool_turns=%d, tool_calls=%d, raw_iter=%d, cap=%d); "
        "asking model for a final review without tools",
        blind_tool_turns,
        metrics.tool_calls,
        iteration,
        cfg.tool_max_iterations,
    )
    if emit is not None and not repeat_guard.tripped:
        # The repeat guard already emitted its own, more specific reason.
        emit("log", "Agent budget exhausted; asking for a final review without tools")
    messages.append(
        {
            "role": "user",
            "content": final_force_message or _DEFAULT_FORCE_FINAL_MESSAGE,
        }
    )
    final_extra = {"reasoning_effort": cfg.llm_reasoning_effort or "low"}
    # Do NOT pass `response_format` here — same reason as the main loop
    # above (Anthropic's OpenAI shim and others reject
    # `{"type": "json_object"}`, accepting only `"json_schema"`). The
    # force message already instructs "single JSON object only", and
    # _extract_json parses the result forgivingly.
    while True:
        chat = llm.complete(
            messages,
            max_tokens=cfg.llm_max_tokens,
            chunk_callback=chunk_cb,
            extra=final_extra,
        )
        metrics.turns += 1
        metrics.latency_seconds += chat.latency_seconds
        if chat.prompt_tokens is not None:
            metrics.prompt_tokens += chat.prompt_tokens
        if chat.completion_tokens is not None:
            metrics.completion_tokens += chat.completion_tokens
        _emit_metrics(emit, metrics)
        _emit_chat_message(
            emit,
            "assistant",
            content=chat.content,
            reasoning_chars=chat.reasoning_chars,
            finish_reason=chat.finish_reason,
        )

        # Salvage an empty/truncated forced-final answer before validating or
        # returning it. This is the exact failure the budget-exhausted path used
        # to die on: an empty completion (finish_reason=None) went straight to
        # the parser and surfaced as "LLM returned unparseable output". Re-ask
        # for the JSON only instead (bounded by _MAX_TRUNCATION_RETRIES).
        defect = _final_answer_defect(chat)
        decision = _salvage_decision(defect, last_defect, truncation_retries)
        if decision == "repeated":
            # This is the site the expensive case actually hits: both measured
            # jobs were already guard-terminated before this turn.
            _emit_salvage_giveup(emit, defect or "", truncation_retries)
        elif decision == "reask":
            truncation_retries += 1
            last_defect = defect
            _emit_final_salvage(emit, chat, truncation_retries)
            messages.append({"role": "assistant", "content": chat.content or None})
            messages.append({"role": "user", "content": _final_recovery_message(chat)})
            continue

        # The verification gate must run on the forced final answer too —
        # exhausting the tool budget must not silently bypass validation (for
        # /tasks this is the normalize gate; skipping it here is how
        # un-normalized patches reach an opened PR). Corrections on this path
        # are tool-less: the model fixes its diff from the normalizer's
        # feedback, which needs no browsing.
        if validate is None:
            return _finalize(chat)
        feedback = validate(chat)
        if feedback is None or validation_retries >= max_validation_retries:
            return _finalize(chat)
        validation_retries += 1
        if emit is not None:
            emit(
                "log",
                f"Patch validation failed (correction "
                f"{validation_retries}/{max_validation_retries}); "
                "asking the model to fix it (no tools)",
            )
        messages.append({"role": "assistant", "content": chat.content or None})
        messages.append({"role": "user", "content": feedback})


def _wrap_chunk_cb(
    emit: Optional[Callable[[str, str], None]],
    metrics: "_AggregateMetrics",
) -> Optional[Callable[[str, str], None]]:
    """Forward token/reasoning chunks unchanged, but turn the
    stream-side ``stream_metrics`` estimates into full ``metrics``
    payloads with prior-turn totals overlaid. This way the UI counter
    grows monotonically across turns instead of resetting each time a
    new stream starts."""
    if emit is None:
        return None

    def cb(kind: str, text: str) -> None:
        if kind != "stream_metrics":
            emit(kind, text)
            return
        try:
            live = json.loads(text)
            this_in = int(live.get("in", 0))
            this_out = int(live.get("out", 0))
            this_seconds = float(live.get("seconds", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        cum_in = metrics.prompt_tokens + this_in
        cum_out = metrics.completion_tokens + this_out
        cum_seconds = metrics.latency_seconds + this_seconds
        rate = cum_out / cum_seconds if cum_seconds > 0 else 0.0
        emit(
            "metrics",
            json.dumps(
                {
                    "in": cum_in,
                    "out": cum_out,
                    "rate": round(rate, 1),
                    "seconds": round(cum_seconds, 1),
                    "turns": metrics.turns + 1,
                    "tools": metrics.tool_calls,
                }
            ),
        )

    return cb


def _emit_metrics(
    emit: Optional[Callable[[str, str], None]], metrics: "_AggregateMetrics"
) -> None:
    if emit is None:
        return
    rate = (
        metrics.completion_tokens / metrics.latency_seconds
        if metrics.latency_seconds > 0
        else 0.0
    )
    payload = json.dumps(
        {
            "in": metrics.prompt_tokens,
            "out": metrics.completion_tokens,
            "rate": round(rate, 1),
            "seconds": round(metrics.latency_seconds, 1),
            "turns": metrics.turns,
            "tools": metrics.tool_calls,
        }
    )
    emit("metrics", payload)


# Per-message log cap. Assistant turns are small (the model's own output),
# but tool results can be large (a grepped file); truncate so the run log
# stays debuggable without bloating the SQLite job store with the full
# 500k-token context that already lives in the prompt.
_LOG_MSG_MAX_CHARS = 2000


def _emit_chat_message(
    emit: Optional[Callable[[str, str], None]],
    role: str,
    *,
    content: Optional[str] = None,
    reasoning_chars: int = 0,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[list[ToolCall]] = None,
    tool_name: Optional[str] = None,
) -> None:
    """Record one chat turn in the run log as a ``message`` event.

    Captures what the model actually emitted each turn — content,
    reasoning length, finish_reason, tool-call names/args — plus truncated
    tool results. Deliberately does NOT re-store the giant diff/user
    context (it is the same every turn and already accounted for in the
    metrics). This is what makes "LLM returned unparseable output"
    failures diagnosable: the empty final turn (content="",
    finish_reason=None) shows up here. Best-effort — never raises into the
    agent loop."""
    if emit is None:
        return
    payload: dict[str, Any] = {"role": role}
    if content:
        payload["content"] = (
            content
            if len(content) <= _LOG_MSG_MAX_CHARS
            else content[:_LOG_MSG_MAX_CHARS]
            + f"… [+{len(content) - _LOG_MSG_MAX_CHARS} chars truncated]"
        )
    if reasoning_chars:
        payload["reasoning_chars"] = reasoning_chars
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    if tool_calls:
        payload["tool_calls"] = [
            {"name": tc.name, "arguments": _summarize_args_str(tc.arguments)}
            for tc in tool_calls
        ]
    if tool_name:
        payload["name"] = tool_name
    try:
        # kind "chat" (not "message"): "message" is the SSE default event type,
        # which would double-fire the browser's onmessage handler.
        emit("chat", json.dumps(payload))
    except Exception:  # never let logging break a run
        log.debug("failed to emit chat message", exc_info=True)


def _assistant_tool_call_dict(tc: ToolCall) -> dict[str, Any]:
    """Serialize a ToolCall back into the OpenAI-compat assistant message.
    Re-attaches Gemini 3's thought_signature at the exact path the API
    expects it, and only when one is present — so OpenAI / HF Router / Gemini
    2.5 turns stay byte-for-byte what they were before."""
    call: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments},
    }
    if tc.thought_signature:
        call["extra_content"] = {"google": {"thought_signature": tc.thought_signature}}
    return call


def _execute_tool_call(env: ToolEnv, tc: ToolCall) -> str:
    """Parse the model's tool arguments and dispatch. Always returns a
    string — errors are surfaced to the model rather than raised."""
    try:
        args = json.loads(tc.arguments) if tc.arguments else {}
    except json.JSONDecodeError as exc:
        log.warning("tool %s emitted unparseable arguments: %s", tc.name, exc)
        return f"error: arguments were not valid JSON: {exc}"
    if not isinstance(args, dict):
        return f"error: arguments must be a JSON object, got {type(args).__name__}"
    log.info("tool call: %s(%s)", tc.name, _summarize_args(args))
    output = run_tool(env, tc.name, args)
    log.info("tool call %s returned %d chars", tc.name, len(output))
    return output


def _summarize_args(args: dict[str, Any], limit: int = 200) -> str:
    s = json.dumps(args, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "..."


def _summarize_args_str(raw: str, limit: int = 200) -> str:
    """Like _summarize_args but takes the raw arguments string from the
    model — handy for emitting a breadcrumb before JSON-parsing."""
    s = (raw or "").replace("\n", " ").strip()
    return s if len(s) <= limit else s[:limit] + "..."


def _summarize_rejected_comments(
    rejected: list[dict[str, Any]], max_items: int = 5
) -> str:
    """Render the first few rejected comments as `path:line` refs so the
    log line stays short even if the model emitted dozens of bogus inline
    comments. Without this, the full payloads bloat the action log."""
    refs: list[str] = []
    for c in rejected[:max_items]:
        path = c.get("path", "?")
        line = c.get("line", "?")
        refs.append(f"{path}:{line}")
    if len(rejected) > max_items:
        refs.append(f"...(+{len(rejected) - max_items} more)")
    return ", ".join(refs)


def _load_review_rules(
    gh: GitHubClient, owner: str, repo: str, pr: dict, cfg: Config
) -> str:
    default_branch = pr.get("base", {}).get("repo", {}).get("default_branch") or "main"
    try:
        content = gh.get_file_contents(
            owner, repo, cfg.review_rules_path, ref=default_branch
        )
    except Exception:
        log.exception("failed to fetch review rules")
        content = None
    return content or cfg.default_review_rules


class EmptyReviewError(Exception):
    """``publish_review`` was handed a review with nothing in it — no summary
    and no inline comments. Public (unlike ``_UnparseableLLMOutput``) because
    the web app and the webhook publisher both need to catch it and report the
    failure instead of posting an empty review."""

    def user_message(self) -> str:
        return (
            "The model did not produce a usable review (no summary and no "
            "inline comments), so nothing was posted. Re-run the review, or "
            "try a different model."
        )


class _UnparseableLLMOutput(Exception):
    """The LLM returned content we couldn't parse as JSON. Carries the raw
    content + finish_reason + metrics line so the caller can render an
    error to whatever surface they own (GitHub comment, web UI, etc.)."""

    def __init__(
        self,
        content: str,
        finish_reason: Optional[str],
        metrics_line: str,
        session: Optional[dict[str, Any]] = None,
        salvage_attempts: int = 0,
    ):
        super().__init__("unparseable LLM output")
        self.content = content
        self.finish_reason = finish_reason
        self.metrics_line = metrics_line
        # These are the sessions we most want counters for: an unparseable
        # answer is what a starved final turn produces, and the job row records
        # only the error string.
        self.session = session or {}
        # How many tool-free re-asks the loop already spent trying to recover
        # this answer. 0 and >0 are different bugs wearing the same error
        # string: 0 means the salvage never recognised the shape (what the
        # task path did with leaked markup), >0 means it recognised it and the
        # model still could not produce JSON. Without this the two are
        # indistinguishable from the job row, which is how the task path's
        # missing markup case stayed hidden.
        self.salvage_attempts = salvage_attempts

    def _salvage_note(self) -> str:
        if self.salvage_attempts:
            return f", {self.salvage_attempts} salvage re-ask(s) did not recover it"
        return ", no salvage was attempted for this shape"

    def user_message(self) -> str:
        if self.finish_reason == "length":
            return (
                "LLM response was truncated before it produced valid review JSON "
                f"(finish_reason=length, {self.metrics_line}"
                f"{self._salvage_note()}). Increase "
                "LLM_MAX_TOKENS for this provider, or narrow the review scope / "
                "reduce TOOL_MAX_ITERATIONS so the final answer has enough output "
                "budget."
            )
        return (
            "LLM returned unparseable output "
            f"(finish_reason={self.finish_reason}, {self.metrics_line}"
            f"{self._salvage_note()})"
        )


def prepare_review(
    cfg: Config,
    gh: GitHubClient,
    req: ReviewRequest,
    *,
    chunk_callback: Optional[Callable[[str, str], None]] = None,
) -> Optional[ReviewDraft]:
    """Run the LLM review pipeline and return a ReviewDraft ready to be
    published (or edited then published).

    `chunk_callback(kind, text)` is invoked as work progresses so callers
    (e.g. the web UI) can surface live activity. Kinds:
      - "log": a human-readable status line
      - "token": a slice of the assistant's streamed content
      - "tool": a tool-call breadcrumb

    Returns None when the PR has no reviewable diff (in which case a
    notice is posted to the PR, matching the original behavior).

    Raises _UnparseableLLMOutput when the model's final reply can't be
    JSON-parsed; the caller decides how to surface that.
    """

    def _emit(kind: str, text: str) -> None:
        if chunk_callback is not None:
            try:
                chunk_callback(kind, text)
            except Exception:
                log.debug("chunk_callback raised; suppressing", exc_info=True)

    log.info(
        "Starting review of %s/%s#%d (triggered by @%s)",
        req.owner,
        req.repo,
        req.number,
        req.commenter,
    )
    _emit("log", f"Starting review of {req.owner}/{req.repo}#{req.number}")

    if req.trigger_comment_id:
        try:
            gh.add_reaction_to_issue_comment(
                req.owner, req.repo, req.trigger_comment_id, "eyes"
            )
        except Exception:
            log.debug("reaction failed (non-fatal)", exc_info=True)

    _emit("step", "fetch")
    pr = gh.get_pr(req.owner, req.repo, req.number)
    files = gh.get_pr_files(req.owner, req.repo, req.number)
    _emit("log", f"Fetched PR with {len(files)} changed file(s)")

    _emit("step", "context")
    ctx_result = run_context_script(
        cfg.context_script_path,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        files=files,
        timeout_seconds=cfg.context_script_timeout,
        cwd=cfg.repo_checkout_path or None,
        sandbox_mode=cfg.helper_sandbox,
    )
    skip_paths: set[str] = set(ctx_result.skip_files) if ctx_result else set()
    extra_context = ctx_result.context if ctx_result else None
    if ctx_result:
        log.info(
            "context script: context=%d chars, skip_files=%d",
            len(extra_context or ""),
            len(skip_paths),
        )
        _emit(
            "log",
            f"Context script: {len(extra_context or '')} chars context, {len(skip_paths)} skip(s)",
        )

    diff_chunks, skipped = _build_annotated_diff_chunks(
        files, cfg.max_diff_chars, skip_paths
    )
    if skipped:
        log.info(
            "Excluded %d file(s) per .ai/context-script: %s", len(skipped), skipped
        )
    if not diff_chunks:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            "No reviewable diff hunks were found (binary files, empty patches, or all files excluded by .ai/context-script).",
        )
        _emit("log", "No reviewable diff hunks; posted notice and stopped")
        return None
    if len(diff_chunks) > 1:
        log.info(
            "Split oversized PR diff into %d review chunks (budget=%d chars/chunk)",
            len(diff_chunks),
            cfg.max_diff_chars,
        )
        _emit(
            "log",
            f"Split oversized PR diff into {len(diff_chunks)} review chunk(s)",
        )

    head_sha = (pr.get("head") or {}).get("sha")
    if not head_sha:
        raise RuntimeError("PR payload missing head.sha")

    review_rules = _load_review_rules(gh, req.owner, req.repo, pr, cfg)
    helper_tools = _load_helper_tools(gh, req.owner, req.repo, pr, cfg)
    _install_helper_tools_with_emit(helper_tools, _emit)
    tool_env = _make_tool_env(cfg, helper_tools)

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
        compressor=MessageCompressor.from_env(),
    )
    system_prompt = build_system_prompt(
        review_rules, tools_enabled=tool_env is not None
    )
    total_metrics = _AggregateMetrics()
    all_valid: list[dict[str, Any]] = []
    all_events: list[str] = []
    all_summaries: list[tuple[int, str]] = []
    rejected_count = 0
    skipped_chunks_for_budget = 0

    for idx, chunk in enumerate(diff_chunks, start=1):
        if (
            cfg.llm_max_input_tokens > 0
            and total_metrics.prompt_tokens >= cfg.llm_max_input_tokens
        ):
            remaining = len(diff_chunks) - idx + 1
            log.warning(
                "Input token budget hit before chunk %d/%d (%d >= %d); "
                "skipping %d remaining chunk(s)",
                idx,
                len(diff_chunks),
                total_metrics.prompt_tokens,
                cfg.llm_max_input_tokens,
                remaining,
            )
            _emit(
                "log",
                f"Input token budget hit ({total_metrics.prompt_tokens} >= "
                f"{cfg.llm_max_input_tokens}); skipping {remaining} remaining "
                "chunk(s) and finishing early",
            )
            skipped_chunks_for_budget = remaining
            total_metrics.stop_reason = STOP_CHUNK_BUDGET
            break
        runner_context = _build_runner_context(
            all_files=files,
            skipped=skipped,
            chunk_index=idx,
            chunk_total=len(diff_chunks),
        )
        user_prompt = build_user_prompt(
            repo_full_name=f"{req.owner}/{req.repo}",
            number=req.number,
            title=pr.get("title") or "",
            body=pr.get("body") or "",
            author=(pr.get("user") or {}).get("login") or "unknown",
            commenter=req.commenter,
            trigger_comment=req.trigger_comment_body,
            diff=chunk.text,
            extra_context=extra_context,
            runner_context=runner_context,
        )

        _emit("step", "llm")
        if len(diff_chunks) > 1:
            _emit("log", f"Calling LLM for diff chunk {idx}/{len(diff_chunks)}…")
        else:
            _emit("log", "Calling LLM…")
        chat, chunk_metrics = _run_agentic_loop(
            llm,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            cfg=cfg,
            tool_env=tool_env,
            emit=_emit,
            prior_prompt_tokens=total_metrics.prompt_tokens,
        )
        _merge_metrics(total_metrics, chunk_metrics)

        try:
            result = _extract_json(chat.content, _REVIEW_JSON_KEYS)
        except ValueError as exc:
            metrics_line = _format_aggregated_metrics(total_metrics)
            log.error(
                "could not parse LLM output as JSON: %s "
                "(content_chars=%d, finish_reason=%s, prompt_tokens=%s, completion_tokens=%s)",
                exc,
                len(chat.content or ""),
                chat.finish_reason,
                chat.prompt_tokens,
                chat.completion_tokens,
            )
            raise _UnparseableLLMOutput(
                content=chat.content or "",
                finish_reason=chat.finish_reason,
                metrics_line=metrics_line,
                session=session_record(total_metrics),
                salvage_attempts=total_metrics.truncation_retries,
            ) from exc

        summary = (result.get("summary") or "").strip()
        event = result.get("event") or cfg.review_event
        # Fallback: forced-final turns sometimes return a stub JSON
        # object alongside the actual review written as prose. Since
        # `_extract_json` accepts the first decodable `{...}`, we can
        # end up with an empty `summary` while the model's real
        # write-up sits in `chat.content`. Salvage the prose rather
        # than publishing an empty "(no overall summary provided)" —
        # but peel off the JSON stub and its ``` fence first, or the
        # published summary opens with a fence the model never closed
        # and renders as one giant code block.
        if (
            not summary
            and not (result.get("comments") or [])
            and chat.content
            and chat.content.strip()
        ):
            salvaged = _prose_outside_json(chat.content) or chat.content.strip()
            if _is_model_markup_only(salvaged):
                # Nothing but leaked special tokens — publishing this is how
                # serge#79 got a review body of raw `<|tool_call_begin|>`
                # markup. Leave the summary empty so the publish gate refuses.
                log.warning(
                    "Discarding salvaged summary: content was only model "
                    "special-token markup (%d chars)",
                    len(salvaged),
                )
            else:
                summary = salvaged
                log.warning(
                    "Parsed JSON yielded empty summary/comments; using "
                    "raw content (%d chars) as summary",
                    len(summary),
                )
        if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
            event = cfg.review_event
        if event == "APPROVE" and not cfg.allow_approve:
            log.info(
                "Downgrading APPROVE to COMMENT (Actions tokens cannot approve; set ALLOW_APPROVE=1 in App mode to permit)"
            )
            event = "COMMENT"
        all_events.append(event)
        if summary:
            all_summaries.append((idx, summary))

        valid, rejected = _validate_comments(
            result.get("comments") or [], chunk.visible_positions
        )
        rejected_count += len(rejected)
        if rejected:
            log.warning(
                "Dropped %d invalid comment(s) from chunk %d/%d (referenced lines not in visible diff or malformed): %s",
                len(rejected),
                idx,
                len(diff_chunks),
                _summarize_rejected_comments(rejected),
            )
        for c in valid:
            c["_parsed"] = chunk.parsed_by_path.get(c["path"])
        all_valid.extend(valid)

    # If the PR was reviewed in multiple chunks, the per-chunk summaries
    # each describe their slice in isolation. Run one extra LLM call to
    # rewrite them into a single PR-level review — otherwise the
    # published summary would read as N disjoint notes referring to
    # "chunk N", which leaks an implementation detail.
    if len(diff_chunks) > 1 and sum(1 for _, s in all_summaries if s.strip()) > 1:
        synth_text, synth_metrics = _synthesize_merged_summary(
            llm,
            all_summaries,
            pr_title=pr.get("title") or "",
            max_tokens=cfg.llm_max_tokens,
            reasoning_effort=cfg.llm_reasoning_effort,
            emit=_emit,
        )
        if synth_metrics is not None:
            _merge_metrics(total_metrics, synth_metrics)
        if synth_text:
            summary = synth_text
        else:
            summary = _merge_chunk_summaries(all_summaries, len(diff_chunks))
    else:
        summary = _merge_chunk_summaries(all_summaries, len(diff_chunks))

    metrics_line = _format_aggregated_metrics(total_metrics)
    _emit("log", f"LLM done: {metrics_line}")

    event = _merge_chunk_event(all_events, len(all_valid))

    draft_comments: list[DraftComment] = []
    seen_comments: set[tuple[str, str, int, str]] = set()
    for i, c in enumerate(all_valid):
        dedupe_key = (c["path"], c["side"], c["line"], c["body"])
        if dedupe_key in seen_comments:
            continue
        seen_comments.add(dedupe_key)
        parsed = c.get("_parsed")
        hunk = (
            extract_hunk_snippet(parsed.raw_patch, c["side"], c["line"])
            if isinstance(parsed, ParsedFile)
            else []
        )
        draft_comments.append(
            DraftComment(
                id=f"c{i}",
                path=c["path"],
                side=c["side"],
                line=c["line"],
                body=c["body"],
                diff_hunk=hunk,
            )
        )

    _emit("step", "done")
    return ReviewDraft(
        owner=req.owner,
        repo=req.repo,
        number=req.number,
        head_sha=head_sha,
        summary=summary,
        event=event,
        comments=draft_comments,
        rejected_count=rejected_count,
        metrics_line=metrics_line,
        truncated_chunks=skipped_chunks_for_budget,
        prompt_tokens=total_metrics.prompt_tokens,
        completion_tokens=total_metrics.completion_tokens,
        model=llm.model,
        session=session_record(total_metrics),
    )


def effective_draft(
    draft: ReviewDraft,
    edits: Optional[ReviewEdits] = None,
    *,
    allow_approve: bool,
) -> ReviewDraft:
    """Resolve a generated draft plus optional user edits into the exact
    review that will be posted to GitHub: summary/event overrides applied,
    invalid events ignored, ``APPROVE`` downgraded to ``COMMENT`` unless
    ``allow_approve``, and overridden/discarded comments materialized.

    This is the single source of truth shared by ``publish_review`` (what
    it posts) and the web app's audit log (what it records as the published
    draft), so the two cannot drift. In particular the audit log must see
    the same ``APPROVE`` -> ``COMMENT`` downgrade GitHub actually receives."""
    edits = edits or ReviewEdits()

    summary = edits.summary if edits.summary is not None else draft.summary

    event = edits.event or draft.event
    if event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
        event = draft.event
    if event == "APPROVE" and not allow_approve:
        log.info(
            "Downgrading APPROVE to COMMENT (Actions tokens cannot approve; set ALLOW_APPROVE=1 in App mode to permit)"
        )
        event = "COMMENT"

    comments: list[DraftComment] = []
    for c in draft.comments:
        if c.id in edits.discarded_comment_ids:
            continue
        body = edits.comment_overrides.get(c.id, c.body)
        if not isinstance(body, str) or not body.strip():
            continue
        comments.append(replace(c, body=body))

    return replace(draft, summary=summary, event=event, comments=comments)


def publish_review(
    cfg: Config,
    gh: GitHubClient,
    draft: ReviewDraft,
    *,
    edits: Optional[ReviewEdits] = None,
) -> ReviewDraft:
    """Apply optional user edits to a ReviewDraft and post it via the
    GitHub reviews API. Mirrors the body-formatting rules previously
    inlined in run_review (persona header, dropped-comments note,
    metrics footer).

    Returns the *effective* ReviewDraft that was actually posted (edits
    applied, event downgraded). Callers persist this so the audit log
    records exactly what GitHub received rather than the raw draft.

    Raises :class:`EmptyReviewError` when there is nothing to say — no summary
    and no inline comments. Such a review is never useful to a human, and
    posting it is what turned a model malfunction into a public "(no overall
    summary provided)" (or worse, raw special-token markup) on the PR."""
    effective = effective_draft(draft, edits, allow_approve=cfg.allow_approve)

    summary_text = (effective.summary or "").strip()
    if _is_model_markup_only(summary_text):
        summary_text = ""
    if not summary_text and not effective.comments:
        raise EmptyReviewError(
            f"refusing to publish an empty review on {effective.owner}/"
            f"{effective.repo}#{effective.number}: no summary and no inline "
            "comments"
        )

    comments_payload: list[dict[str, Any]] = [
        {"path": c.path, "side": c.side, "line": c.line, "body": c.body}
        for c in effective.comments
    ]

    # `summary_text` (not `effective.summary`) so a markup-only summary that
    # rode along with real inline comments renders the fallback line rather
    # than the leaked tokens.
    body = summary_text or "(no overall summary provided)"
    if cfg.persona_header:
        body = f"{cfg.persona_header}\n\n{body}"
    if effective.rejected_count:
        body += (
            f"\n\n_Note: {effective.rejected_count} suggested inline comment(s) "
            "were dropped because they referenced lines not present in the diff._"
        )
    if effective.truncated_chunks:
        body += (
            f"\n\n_Note: review finished early after hitting the input-token "
            f"budget; {effective.truncated_chunks} remaining diff chunk(s) were "
            "not reviewed._"
        )
    if cfg.is_staging:
        body += "\n\n_Note: posted from a staging deployment._"
    footer_parts = [f"serge `v{__version__}`"]
    if effective.model:
        footer_parts.append(f"model: `{effective.model}`")
    if effective.metrics_line:
        footer_parts.append(effective.metrics_line)
    if footer_parts:
        body += f"\n\n_{' · '.join(footer_parts)}_"

    gh.create_review(
        effective.owner,
        effective.repo,
        effective.number,
        commit_id=effective.head_sha,
        body=body,
        comments=comments_payload,
        event=effective.event,
    )
    log.info(
        "Posted review on %s/%s#%d (%d inline, event=%s, %s)",
        effective.owner,
        effective.repo,
        effective.number,
        len(comments_payload),
        effective.event,
        effective.metrics_line,
    )
    return effective


def run_review(
    cfg: Config,
    gh: GitHubClient,
    req: ReviewRequest,
    *,
    force_comment_event: bool = False,
) -> None:
    """Webhook + Action entry point. Unchanged behavior: prepares the
    review, then immediately publishes it. Renders a fallback issue
    comment if the LLM output is unparseable."""
    try:
        draft = prepare_review(cfg, gh, req)
    except _UnparseableLLMOutput as exc:
        gh.post_issue_comment(
            req.owner,
            req.repo,
            req.number,
            f"{exc.user_message()}\n\n```\n{exc.content[:3000]}\n```",
        )
        return
    if draft is None:
        return
    edits = ReviewEdits(event="COMMENT") if force_comment_event else None
    try:
        publish_review(cfg, gh, draft, edits=edits)
    except EmptyReviewError as exc:
        # Say so in the thread the reviewer is watching; silently posting
        # nothing looks identical to serge never having run.
        log.warning("empty review not published: %s", exc)
        gh.post_issue_comment(req.owner, req.repo, req.number, exc.user_message())


_FOLLOWUP_FORCE_FINAL_MESSAGE = (
    "You have used all available tool calls. Based on what you have "
    "already gathered, write the final reply now as plain markdown "
    "(no JSON, no tool calls)."
)


def run_followup(
    cfg: Config,
    gh: GitHubClient,
    req: ReviewRequest,
    *,
    chunk_callback: Optional[Callable[[str, str], None]] = None,
) -> None:
    """Answer a single inline follow-up question and post the reply in
    the same comment thread. Triggered from pull_request_review_comment
    events; ``req.inline`` carries the anchor (path/line/side/diff_hunk).

    ``chunk_callback(kind, text)`` mirrors ``prepare_review``: when given,
    progress is streamed so the web UI can follow the reply live (the
    follow-up has no draft, so the page just shows the console + metrics).
    """
    assert req.inline is not None, "run_followup requires req.inline"
    inline = req.inline

    def _emit(kind: str, text: str) -> None:
        if chunk_callback is not None:
            try:
                chunk_callback(kind, text)
            except Exception:
                log.debug("chunk_callback raised; suppressing", exc_info=True)

    log.info(
        "Starting follow-up on %s/%s#%d %s:%d (triggered by @%s)",
        req.owner,
        req.repo,
        req.number,
        inline.path,
        inline.line,
        req.commenter,
    )
    _emit(
        "log",
        f"Answering inline follow-up on {req.owner}/{req.repo}#{req.number} "
        f"({inline.path}:{inline.line})",
    )

    try:
        gh.add_reaction_to_review_comment(
            req.owner, req.repo, inline.comment_id, "eyes"
        )
    except Exception:
        log.debug("reaction failed (non-fatal)", exc_info=True)

    _emit("step", "fetch")
    pr = gh.get_pr(req.owner, req.repo, req.number)
    review_rules = _load_review_rules(gh, req.owner, req.repo, pr, cfg)
    helper_tools = _load_helper_tools(gh, req.owner, req.repo, pr, cfg)
    _install_helper_tools_with_emit(helper_tools, _emit)
    tool_env = _make_tool_env(cfg, helper_tools)

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
        compressor=MessageCompressor.from_env(),
    )

    system_prompt = build_followup_system_prompt(
        review_rules, tools_enabled=tool_env is not None
    )
    user_prompt = build_followup_user_prompt(
        repo_full_name=f"{req.owner}/{req.repo}",
        number=req.number,
        title=pr.get("title") or "",
        body=pr.get("body") or "",
        author=(pr.get("user") or {}).get("login") or "unknown",
        commenter=req.commenter,
        trigger_comment=req.trigger_comment_body,
        path=inline.path,
        side=inline.side,
        line=inline.line,
        diff_hunk=inline.diff_hunk,
    )

    _emit("step", "llm")
    _emit("log", "Calling LLM…")
    chat, metrics = _run_agentic_loop(
        llm,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        cfg=cfg,
        tool_env=tool_env,
        emit=_emit,
        final_force_message=_FOLLOWUP_FORCE_FINAL_MESSAGE,
    )
    _emit("step", "done")

    reply = (chat.content or "").strip()
    if not reply:
        reply = (
            "_Could not produce a reply (the model returned an empty "
            f"response after {metrics.turns} turn(s))._"
        )

    body = reply
    if cfg.persona_header:
        body = f"{cfg.persona_header}\n\n{body}"
    metrics_line = _format_aggregated_metrics(metrics)
    if metrics_line:
        body += f"\n\n_{metrics_line}_"

    gh.reply_to_review_comment(req.owner, req.repo, req.number, inline.comment_id, body)
    log.info(
        "Posted follow-up reply on %s/%s#%d comment %d (%s)",
        req.owner,
        req.repo,
        req.number,
        inline.comment_id,
        metrics_line,
    )
