import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import requests

from .compression import MessageCompressor

log = logging.getLogger(__name__)


class _ToolsUnsupported(Exception):
    """Raised when the upstream endpoint rejects a tool-augmented request
    with 400, so the caller can retry once without tools. Carries the
    response body preview for logging."""


class _ParameterUnsupported(Exception):
    """Raised when the upstream endpoint rejects a specific payload
    parameter with a 400 (e.g. newer Claude models deprecate
    ``temperature``), so the caller can strip that one field and retry
    once. Carries the offending parameter name and the response body
    preview for logging."""

    def __init__(self, param: str, body_preview: str):
        self.param = param
        self.body_preview = body_preview
        super().__init__(f"{param}: {body_preview}")


def _is_parameter_rejection(param: str, body_preview: str) -> bool:
    """True when a 400 body indicates ``param`` itself is not accepted by
    the model — as opposed to a value-range complaint. Provider-neutral:
    matches the OpenAI-compat error text emitted by Anthropic's shim
    ("`temperature` is deprecated for this model.") and similar
    unsupported-parameter wording from other endpoints."""
    text = body_preview.lower()
    if param.lower() not in text:
        return False
    return any(
        signal in text
        for signal in (
            "deprecat",
            "unsupported",
            "not supported",
            "unexpected",
            "unknown",
            "not permitted",
            "not allowed",
            "cannot be used",
        )
    )


# Anthropic requires this header on its native routes (notably /v1/models,
# which backs the model dropdown); the OpenAI-compat chat route ignores it.
_ANTHROPIC_VERSION = "2023-06-01"


def _is_anthropic_base(api_base: str) -> bool:
    return "api.anthropic.com" in api_base.lower()


class LLMResponseError(requests.HTTPError):
    """Non-OK HTTP response from the chat-completions endpoint that
    exhausted retries (or wasn't retryable to begin with). Carries the
    status code, upstream reason phrase, and a short preview of the
    response body so the web UI / action log can show *why* the request
    failed without re-fetching it.
    """

    def __init__(self, status_code: int, reason: str, url: str, body_preview: str):
        self.status_code = status_code
        self.reason = reason
        self.url = url
        self.body_preview = body_preview
        super().__init__(f"{status_code} {reason} for {url}: {body_preview}")


@dataclass
class ToolCall:
    """One tool/function call emitted by the assistant. ``arguments`` is
    the raw string the model produced — the caller is responsible for
    JSON-parsing it (and for handling models that emit malformed JSON).

    ``thought_signature`` is Gemini 3's encrypted reasoning token, carried
    over the OpenAI-compat endpoint at ``tool_call.extra_content.google.
    thought_signature``. It MUST be echoed back unchanged on the next
    request or Gemini 3 rejects the follow-up turn with a 400. ``None`` for
    every other provider (OpenAI, HF Router, Gemini 2.5), which don't send
    it — so it round-trips invisibly for them."""

    id: str
    name: str
    arguments: str
    thought_signature: Optional[str] = None


@dataclass
class ChatResult:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    # Number of chain-of-thought characters the model emitted during
    # this turn (sum of `reasoning_content`/`reasoning`/`thinking`
    # delta fields). Used by the agent loop to distinguish "thoughtful
    # tool-using turn" from "blind tool chaining". 0 for non-reasoning
    # models, which is fine — they're expected to emit content.
    reasoning_chars: int = 0

    @property
    def prompt_tokens(self) -> Optional[int]:
        v = self.usage.get("prompt_tokens")
        return v if isinstance(v, int) else None

    @property
    def completion_tokens(self) -> Optional[int]:
        v = self.usage.get("completion_tokens")
        return v if isinstance(v, int) else None


# Rate-limit headers, in the spellings the OpenAI-compatible endpoints serge
# talks to actually use. Request budgets only: a token budget resets on a
# different clock and pacing on it would stall the loop for a limit it is not
# hitting.
_REMAINING_HEADERS = (
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining",
    "ratelimit-remaining",
)
_RESET_HEADERS = (
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset",
    "ratelimit-reset",
)
# "6m0s", "1.5s", "300ms" — the duration form OpenAI-style resets use.
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
# A reset given as an absolute epoch rather than a duration. Anything past this
# is a timestamp, not "wait 1.7 billion seconds".
_EPOCH_THRESHOLD = 1_000_000_000


def _header(response: Optional["requests.Response"], names: tuple[str, ...]) -> str:
    """First present value among ``names``, or ``""``. Tolerates a Mock
    ``headers`` (tests) and a response that never arrived."""
    headers = getattr(response, "headers", None)
    if not hasattr(headers, "get"):
        return ""
    for name in names:
        value = headers.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_duration(raw: str) -> Optional[float]:
    """Seconds from a rate-limit reset value: ``"20"``, ``"6m0s"``, ``"300ms"``,
    or an absolute epoch. ``None`` when it is not any of those."""
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        pass
    else:
        if value >= _EPOCH_THRESHOLD:  # absolute timestamp
            return max(0.0, value - time.time())
        return max(0.0, value)
    total = 0.0
    matched = False
    for amount, unit in _DURATION_RE.findall(raw):
        matched = True
        total += float(amount) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total if matched else None


def _retry_after_seconds(response: Optional["requests.Response"]) -> Optional[float]:
    """``Retry-After`` in seconds, from either the delta or HTTP-date form."""
    raw = _header(response, ("Retry-After", "retry-after"))
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
    except Exception:  # noqa: BLE001 - a malformed date is simply no answer
        return None


def _reset_seconds(response: Optional["requests.Response"]) -> Optional[float]:
    """Seconds until the request budget resets, per the server's own headers."""
    return _parse_duration(_header(response, _RESET_HEADERS))


def _budget_interval(response: Optional["requests.Response"]) -> Optional[float]:
    """Spacing that spends the *reported* remaining requests evenly over the
    window they reset in — the provider's own numbers, not a guess.

    ``None`` when the endpoint publishes no budget (most do not), or when there
    is enough headroom that pacing would only slow the loop down for nothing."""
    remaining_raw = _header(response, _REMAINING_HEADERS)
    if not remaining_raw:
        return None
    try:
        remaining = int(float(remaining_raw))
    except ValueError:
        return None
    reset = _reset_seconds(response)
    if reset is None or reset <= 0:
        return None
    if remaining <= 0:
        # Budget spent: the whole window is the wait.
        return reset
    if remaining > _PACE_WHEN_REMAINING_BELOW:
        return None
    return reset / remaining


# Only pace when the reported headroom is this tight. Above it the loop runs at
# full speed: a limit we are nowhere near is not worth a single second.
_PACE_WHEN_REMAINING_BELOW = 10


class ChatCompletionClient:
    """Minimal OpenAI-compatible /v1/chat/completions client.

    Works with any endpoint that speaks the OpenAI chat-completions protocol:
    OpenAI, vLLM, TGI's OpenAI route, HF Router, Anthropic's OpenAI shim,
    LM Studio, llama.cpp server, etc.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: Optional[str] = None,
        bill_to: Optional[str] = None,
        stream: bool = False,
        compressor: Optional["MessageCompressor"] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.bill_to = bill_to or None
        self.stream = stream
        self.compressor = compressor
        # Adaptive pacing, learned from 429s (see :meth:`_throttle_wait`). One
        # client serves one agentic loop, so this is per-task state: the loop
        # that tripped a rate limit is the loop that must slow down.
        self._min_interval = 0.0
        self._next_earliest = 0.0

    def _api_base_v1(self) -> str:
        if self.api_base.endswith("/v1"):
            return self.api_base
        return f"{self.api_base}/v1"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.bill_to:
            # HF Router: route inference billing to an org the token has access to.
            headers["X-HF-Bill-To"] = self.bill_to
        if _is_anthropic_base(self.api_base):
            # /v1/chat/completions is Anthropic's OpenAI shim, but /v1/models is
            # the native route and rejects requests without anthropic-version.
            # The shim ignores the extra header, so send it unconditionally.
            headers["anthropic-version"] = _ANTHROPIC_VERSION
        return headers

    def _discover_model(self) -> str:
        models_url = f"{self._api_base_v1()}/models"
        response = requests.get(
            models_url,
            headers=self._headers(),
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(
                f"Failed to discover model from {models_url} (status {response.status_code}). "
                "Provide llm_model explicitly."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to parse {models_url} response as JSON."
            ) from exc

        data = payload.get("data")
        first_model = data[0] if isinstance(data, list) and data else None
        model_id = first_model.get("id") if isinstance(first_model, dict) else None
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeError(
                f"No models found at {models_url}. Check the endpoint URL or provide llm_model explicitly."
            )
        log.info("Discovered LLM model %s from %s", model_id, models_url)
        self.model = model_id
        return model_id

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        return self._discover_model()

    def list_models(self, timeout: int = 15) -> list[str]:
        """Model ids the endpoint's ``/models`` route advertises, sorted
        case-insensitively and de-duplicated. Works against any OpenAI-compatible
        base (OpenAI, Anthropic's shim, HF Router, vLLM, …). Raises RuntimeError
        on a non-OK response or unparseable body so callers can surface why."""
        models_url = f"{self._api_base_v1()}/models"
        response = requests.get(models_url, headers=self._headers(), timeout=timeout)
        if not response.ok:
            raise RuntimeError(
                f"Failed to list models from {models_url} (status {response.status_code})."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Failed to parse {models_url} response as JSON."
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        ids = {
            entry["id"]
            for entry in data
            if isinstance(entry, dict)
            and isinstance(entry.get("id"), str)
            and entry["id"]
        }
        return sorted(ids, key=str.lower)

    # Optional sampling parameters a model may reject/deprecate outright.
    # Each is safe to drop on a 400 — the server falls back to its own
    # default — so retrying without it keeps the review alive. Ordered
    # most- to least-commonly rejected.
    _REMOVABLE_PARAMS = (
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
    )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: Optional[dict[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        extra: Optional[dict[str, Any]] = None,
        chunk_callback: Optional[Callable[[str, str], None]] = None,
    ) -> ChatResult:
        model = self._resolve_model()
        if self.compressor is not None:
            messages = self.compressor.compress(messages, model=model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if self.stream:
            payload["stream"] = True
            # Ask servers that support it to deliver a final usage chunk.
            # Servers that don't honor this still stream content correctly.
            payload["stream_options"] = {"include_usage": True}
        if extra:
            payload.update(extra)

        url = f"{self._api_base_v1()}/chat/completions"
        attempts = 3
        retryable = (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
        # Pre-compute an input-token estimate so the web UI's "in"
        # counter shows a value while the stream is still in flight —
        # authoritative ``usage.prompt_tokens`` only arrives at
        # end-of-stream. Cheap char-based heuristic; overwritten by
        # the authoritative value once it lands.
        est_input_tokens = self._estimate_input_tokens(messages, tools)
        started = time.monotonic()
        # The endpoint may reject specific payload fields with a 400: some
        # models deprecate a sampling parameter (e.g. newer Claude reject
        # `temperature`), others don't support function-calling. Strip the
        # offending field and retry, at most once per removable field, so
        # the review still completes.
        while True:
            try:
                return self._post_with_retries(
                    url,
                    payload,
                    attempts,
                    retryable,
                    started,
                    tools_in_use="tools" in payload,
                    chunk_callback=chunk_callback,
                    est_input_tokens=est_input_tokens,
                )
            except _ParameterUnsupported as exc:
                if exc.param not in payload:
                    raise
                payload.pop(exc.param, None)
            except _ToolsUnsupported:
                # Strip the function-calling fields and try once more
                # (degraded: no browse tools).
                if "tools" not in payload:
                    raise
                payload.pop("tools", None)
                payload.pop("tool_choice", None)

    # Cap any single retry wait, even if the server hands us a huge
    # Retry-After. 120s is plenty for dynamic-rate-limit recovery
    # without letting a misbehaving upstream pin a worker forever.
    _RETRY_WAIT_CEILING_SECONDS = 120.0

    # A 429 gets its OWN, much larger budget than the 3 attempts a 5xx or a
    # dropped connection gets, because it is a different kind of failure. A
    # server error may never clear, so failing fast is right. A rate limit
    # clears by *waiting* — and what we throw away by giving up is the whole
    # agentic loop that came before it: one 429 ended the `deepseek_vl` task on
    # 2026-08-18 after it had already spent 1.13M input tokens (transformers#48050).
    # 6 attempts of backed-off waiting is minutes of patience against an hour of
    # wasted work.
    _RATE_LIMIT_ATTEMPTS = 6
    # Pacing after a rate limit. Retrying the one rejected call is not enough:
    # the loop is about to issue the same burst of tool turns that tripped the
    # limit, so the next calls have to be spaced out too. The interval doubles
    # per 429 and decays on success, which converges on whatever rate the
    # provider is actually willing to serve without needing to know its limit.
    _THROTTLE_FIRST_SECONDS = 2.0
    _THROTTLE_CEILING_SECONDS = 30.0
    _THROTTLE_DECAY = 0.5
    # Below this the pacing is noise; drop it entirely so a loop that hit one
    # transient 429 returns to full speed instead of limping forever.
    _THROTTLE_FLOOR_SECONDS = 0.25

    def _throttle_wait(self) -> None:
        """Block until this client is allowed to issue its next request.

        No-op until a 429 has been seen — an endpoint that never rate-limits us
        is never slowed down."""
        if self._min_interval <= 0.0:
            return
        remaining = self._next_earliest - time.monotonic()
        if remaining > 0:
            log.info("pacing LLM call: waiting %.1fs after a rate limit", remaining)
            time.sleep(remaining)

    def _note_rate_limited(self, response: "requests.Response") -> None:
        """Widen the spacing between requests after a 429.

        Prefer what the server said. ``Retry-After`` (or an ``x-ratelimit-reset``
        family header) is the window it wants us to sit out, which beats any
        number we could invent; the doubling heuristic is only the fallback for
        providers that send neither."""
        told = _retry_after_seconds(response)
        if told is None:
            told = _reset_seconds(response)
        if told is not None:
            self._min_interval = min(
                max(told, self._THROTTLE_FLOOR_SECONDS), self._THROTTLE_CEILING_SECONDS
            )
            return
        widened = max(self._min_interval * 2, self._THROTTLE_FIRST_SECONDS)
        self._min_interval = min(widened, self._THROTTLE_CEILING_SECONDS)

    def _note_request_done(
        self, response: Optional["requests.Response"] = None, *, rate_limited: bool
    ) -> None:
        """Record when the next request may go out.

        When the provider publishes its budget (the ``x-ratelimit-*`` family),
        pace from it directly: spreading the remaining allowance over the window
        it resets in keeps the loop UNDER the limit instead of discovering the
        limit by being rejected. Only headroom the server reports as tight
        actually slows anything down.

        Otherwise a successful call decays the spacing, so the loop speeds back
        up as the provider lets it — driven by real successes rather than
        wall-clock optimism."""
        paced = None if rate_limited else _budget_interval(response)
        if paced is not None:
            self._min_interval = min(paced, self._THROTTLE_CEILING_SECONDS)
        elif not rate_limited and self._min_interval > 0.0:
            self._min_interval *= self._THROTTLE_DECAY
            if self._min_interval < self._THROTTLE_FLOOR_SECONDS:
                self._min_interval = 0.0
        self._next_earliest = time.monotonic() + self._min_interval

    @staticmethod
    def _retry_delay(attempt: int, response: "requests.Response") -> float:
        """How long to sleep before retrying.

        Ask the server first: ``Retry-After`` (delta or HTTP-date), then the
        ``x-ratelimit-reset`` family, which says when the budget refills and is
        the honest answer when the endpoint sends no ``Retry-After``. Exponential
        backoff (2, 4, 8, …) is the last resort, for a server that says nothing.

        Always bounded by the per-wait ceiling — a server is allowed to be wrong
        or hostile, and no single wait should pin a worker."""
        told = _retry_after_seconds(response)
        if told is None:
            told = _reset_seconds(response)
        if told is not None and told > 0:
            return min(told, ChatCompletionClient._RETRY_WAIT_CEILING_SECONDS)
        return float(2**attempt)

    @staticmethod
    def _estimate_input_tokens(
        messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]]
    ) -> int:
        """Char-based estimate of the prompt's token count (~4 chars/tok).
        Counts every string in messages — content, tool-call arguments,
        tool replies — and the tool schema JSON when tools are passed."""
        chars = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                chars += len(content)
            for tc in m.get("tool_calls") or ():
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = fn.get("arguments") or ""
                if isinstance(name, str):
                    chars += len(name)
                if isinstance(args, str):
                    chars += len(args)
                sig = _extract_thought_signature(tc)
                if sig:
                    chars += len(sig)
        if tools:
            chars += len(json.dumps(tools))
        return chars // 4

    def _post_with_retries(
        self,
        url: str,
        payload: dict[str, Any],
        attempts: int,
        retryable: tuple,
        started: float,
        *,
        tools_in_use: bool,
        chunk_callback: Optional[Callable[[str, str], None]] = None,
        est_input_tokens: int = 0,
    ) -> ChatResult:
        body = json.dumps(payload)
        attempt = 0  # 5xx / transport failures, budget `attempts`
        rate_limits = 0  # 429s, budget `_RATE_LIMIT_ATTEMPTS` — see below
        while True:
            attempt += 1
            self._throttle_wait()
            try:
                r = requests.post(
                    url,
                    headers=self._headers(),
                    data=body,
                    timeout=300,
                    stream=self.stream,
                )
                # 429 (rate limit) gets hit during agentic tool loops on
                # providers with bursty dynamic limits (e.g. Together.ai via HF
                # Router, where a few quick tool turns can exceed a 1 RPM cap).
                # It spends its own budget, NOT the error budget: a rate limit
                # is a "come back later", so waiting it out is the correct
                # response, and a persistent 5xx must still fail fast.
                if r.status_code == 429 and rate_limits < self._RATE_LIMIT_ATTEMPTS:
                    rate_limits += 1
                    attempt -= 1
                    # Widen the spacing for the calls that FOLLOW, but let the
                    # explicit delay below govern this retry — pacing on top of
                    # Retry-After would just wait twice for the same rejection.
                    self._note_rate_limited(r)
                    delay = self._retry_delay(rate_limits, r)
                    log.warning(
                        "LLM call rate-limited (429) %d/%d; retrying in %.1fs, "
                        "then pacing calls %.1fs apart",
                        rate_limits,
                        self._RATE_LIMIT_ATTEMPTS,
                        delay,
                        self._min_interval,
                    )
                    time.sleep(delay)
                    continue
                if r.status_code >= 500 and attempt < attempts:
                    delay = self._retry_delay(attempt, r)
                    log.warning(
                        "LLM call attempt %d/%d returned %d; retrying in %.1fs",
                        attempt,
                        attempts,
                        r.status_code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                # Pace from this response's own budget headers when it has them.
                self._note_request_done(r, rate_limited=False)
                if not r.ok:
                    # Surface the upstream error body so the action log
                    # explains *why* the request was rejected (e.g. "tools
                    # not supported by this model"). Without this, the
                    # caller only sees "400 Bad Request" and has to guess.
                    body_preview = (r.text or "")[:2000]
                    log.error(
                        "LLM call returned %d %s for %s; body=%s",
                        r.status_code,
                        r.reason,
                        url,
                        body_preview,
                    )
                    if r.status_code == 400:
                        # Check for a rejected sampling parameter first:
                        # stripping tools wouldn't fix it, and some models
                        # (e.g. newer Claude) reject `temperature` outright.
                        for param in self._REMOVABLE_PARAMS:
                            if param in payload and _is_parameter_rejection(
                                param, body_preview
                            ):
                                log.warning(
                                    "Retrying without `%s` (the model appears "
                                    "to reject the parameter)",
                                    param,
                                )
                                raise _ParameterUnsupported(param, body_preview)
                        if tools_in_use:
                            log.warning(
                                "Retrying once without tools (the endpoint may "
                                "not support function-calling for this model)"
                            )
                            raise _ToolsUnsupported(body_preview)
                    raise LLMResponseError(
                        r.status_code, r.reason or "", url, body_preview
                    )
                if self.stream:
                    (
                        content,
                        usage,
                        tool_calls,
                        finish_reason,
                        reasoning_chars,
                    ) = self._consume_stream(
                        r,
                        chunk_callback=chunk_callback,
                        est_input_tokens=est_input_tokens,
                    )
                else:
                    data = r.json()
                    choice = data["choices"][0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    # Same vendor-placeholder defense as the streaming
                    # path: drop a "None"/"null" filler that some
                    # inference stacks emit when tool calls are present.
                    if isinstance(content, str) and content.strip() in ("None", "null"):
                        content = ""
                    usage = data.get("usage") or {}
                    tool_calls = _parse_tool_calls_from_message(
                        message.get("tool_calls")
                    )
                    finish_reason = choice.get("finish_reason")
                    # Non-streaming endpoints sometimes include
                    # `reasoning_content` directly on the message. Best-
                    # effort capture so the agent loop's "did the model
                    # think this turn?" check works either way.
                    reasoning_text = (
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    )
                    reasoning_chars = (
                        len(reasoning_text) if isinstance(reasoning_text, str) else 0
                    )
                    if chunk_callback is not None and content:
                        # Non-streaming path: still emit the full content
                        # in one piece so callers don't need a separate
                        # code path for the buffered case.
                        try:
                            chunk_callback("token", content)
                        except Exception:
                            log.debug(
                                "chunk_callback raised; suppressing", exc_info=True
                            )
            except retryable as exc:
                self._note_request_done(rate_limited=False)
                if attempt >= attempts:
                    log.error(
                        "LLM call attempt %d/%d failed during %s: %s; giving up",
                        attempt,
                        attempts,
                        "stream" if self.stream else "request",
                        exc,
                    )
                    raise
                log.warning(
                    "LLM call attempt %d/%d failed during %s: %s; retrying",
                    attempt,
                    attempts,
                    "stream" if self.stream else "request",
                    exc,
                )
                time.sleep(2**attempt)
                continue
            # Structured `tool_calls` always wins; only fall back to scraping
            # the content when the provider sent none. Covers both the
            # streaming and buffered paths, which converge here.
            if not tool_calls and content:
                recovered, content = _parse_text_tool_calls(content)
                if recovered:
                    tool_calls = recovered
                    log.warning(
                        "Model wrote %d tool call(s) into content as special "
                        "tokens instead of the tool_calls field (model=%s, "
                        "finish=%s); recovered %s as a tool turn",
                        len(recovered),
                        self.model,
                        finish_reason,
                        ", ".join(tc.name for tc in recovered),
                    )
            latency = time.monotonic() - started
            log.info(
                "LLM call ok in %.1fs (prompt=%s, completion=%s, stream=%s, "
                "tool_calls=%d, finish=%s)",
                latency,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                self.stream,
                len(tool_calls),
                finish_reason,
            )
            return ChatResult(
                content=content,
                usage=usage,
                latency_seconds=latency,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning_chars=reasoning_chars,
            )
        raise RuntimeError("unreachable")  # loop always returns or raises

    # Emit a heartbeat log line every PROGRESS_INTERVAL_SECONDS while a stream
    # is in flight, so the action's console output makes clear that bytes are
    # still arriving from the LLM (and lets us spot a hang vs a slow stream).
    PROGRESS_INTERVAL_SECONDS = 10.0

    # How often to push an estimated-tokens "metrics" event to the chunk
    # callback while a stream is in flight. The OpenAI streaming protocol
    # only delivers authoritative `usage` at end-of-stream, so we estimate
    # from byte counts (~4 chars per token) to give the UI a live counter.
    LIVE_METRICS_INTERVAL_SECONDS = 0.75
    LIVE_METRICS_CHARS_PER_TOKEN = 4

    # Delta keys that reasoning/thinking models stream their chain-of-thought
    # into (instead of `content`). We buffer these so we can periodically dump
    # the latest chunk into the action log — useful for watching what the
    # model is actually doing during a long stream.
    REASONING_DELTA_KEYS = ("reasoning", "reasoning_content", "thinking")

    # Flush a slice of buffered reasoning to the log once this many new chars
    # have arrived since the last flush.
    REASONING_FLUSH_CHARS = 400

    @classmethod
    def _consume_stream(
        cls,
        r: "requests.Response",
        *,
        chunk_callback: Optional[Callable[[str, str], None]] = None,
        est_input_tokens: int = 0,
    ) -> tuple[str, dict[str, Any], list[ToolCall], Optional[str], int]:
        """Parse an OpenAI-style SSE chat-completions stream.

        Each event is a `data: {json}` line; the terminal event is `data: [DONE]`.
        We accumulate `choices[0].delta.content` and capture the trailing `usage`
        block when the server emits one (requires stream_options.include_usage).

        Logs a periodic progress line so long-running streams visibly make
        progress in the action's console output. Also tracks per-field char
        counts on ``delta`` so reasoning/thinking models (which stream
        content into ``delta.reasoning_content`` or similar instead of
        ``delta.content``) are easy to spot in the action log.

        Raises ChunkedEncodingError / ConnectionError if the upstream cuts the
        connection mid-stream — the outer retry loop in ``complete`` handles
        these by re-issuing the request.
        """
        parts: list[str] = []
        usage: dict[str, Any] = {}
        chars = 0
        events = 0
        # Buffer the head of the content stream so we can detect a
        # vendor placeholder like "None" / "null" that arrives split
        # across multiple deltas (e.g. "No" + "ne"). Once we've seen
        # enough chars to be certain it's not a placeholder, we flush
        # the buffer to the UI and switch to direct forwarding.
        # At end of stream, if the entire accumulated content matched
        # a placeholder we drop it from chat.content as well.
        content_head_buffer = ""
        content_head_flushed = False
        CONTENT_HEAD_FLUSH_THRESHOLD = 8
        PLACEHOLDER_CONTENTS = ("None", "null", "")
        # Per-field char counts across all observed delta keys, so it's
        # obvious in the log if the model streams into a non-standard field.
        delta_field_chars: dict[str, int] = {}
        # Buffer reasoning chunks so we can periodically flush a tail of them
        # to the log; tracks the offset already logged.
        reasoning_buffer: list[str] = []
        reasoning_logged_chars = 0
        first_delta_logged = False
        # Tool calls stream as a list of partial dicts addressed by index;
        # each chunk may bring an id, function.name, or a slice of
        # function.arguments that we concatenate.
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        stream_started = time.monotonic()
        last_progress = stream_started
        last_live_metrics = stream_started
        # SSE is defined as UTF-8; force the decode regardless of what
        # (if anything) the server declared in Content-Type. Without
        # this, ``requests`` falls back to ISO-8859-1 for text/* and
        # any non-ASCII char (em dashes, smart quotes, accented names)
        # arrives as mojibake on the client side and gets re-encoded
        # into the published review body.
        r.encoding = "utf-8"
        try:
            for raw in r.iter_lines(decode_unicode=True):
                now = time.monotonic()
                if now - last_progress >= cls.PROGRESS_INTERVAL_SECONDS:
                    log.info(
                        "LLM stream progress: %.1fs elapsed, %d events, "
                        "%d content chars, delta fields=%s",
                        now - stream_started,
                        events,
                        chars,
                        cls._format_field_counts(delta_field_chars),
                    )
                    last_progress = now
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                chunk = raw[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    event = json.loads(chunk)
                except json.JSONDecodeError:
                    log.debug("skipping unparseable stream chunk: %r", chunk[:200])
                    continue
                events += 1
                chunk_usage = event.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    streamed_tool_calls = delta.get("tool_calls")
                    if isinstance(streamed_tool_calls, list):
                        for tc in streamed_tool_calls:
                            cls._merge_tool_call_chunk(tool_call_parts, tc)
                    if not first_delta_logged and delta:
                        log.info(
                            "LLM first delta keys=%s sample=%s",
                            sorted(delta.keys()),
                            cls._truncate_repr(delta, 300),
                        )
                        first_delta_logged = True
                    for key, value in delta.items():
                        if isinstance(value, str) and value:
                            delta_field_chars[key] = delta_field_chars.get(
                                key, 0
                            ) + len(value)
                            if key in cls.REASONING_DELTA_KEYS:
                                reasoning_buffer.append(value)
                                # Forward reasoning live so the web UI can
                                # show the model's chain-of-thought as it
                                # arrives. Skip non-string / empty values
                                # so a null delta doesn't render as "null"
                                # or "None" in the console; reasoning
                                # tokens are sometimes interleaved with
                                # nulls in vendor stream formats.
                                if (
                                    chunk_callback is not None
                                    and isinstance(value, str)
                                    and value
                                ):
                                    try:
                                        chunk_callback("reasoning", value)
                                    except Exception:
                                        log.debug(
                                            "chunk_callback raised; suppressing",
                                            exc_info=True,
                                        )
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        # Some inference stacks (vLLM with certain tool
                        # parsers, Kimi-K2 on HF Router) emit a literal
                        # "None" / "null" content chunk as filler when
                        # the model is firing tool calls without real
                        # text. The chunks can arrive split across
                        # multiple deltas ("No" + "ne"), so we buffer
                        # the head of the stream and only commit it
                        # once we're sure it isn't a placeholder.
                        chars += len(piece)
                        if content_head_flushed:
                            # Past the threshold — append straight
                            # through, both to parts and UI.
                            parts.append(piece)
                            if chunk_callback is not None:
                                try:
                                    chunk_callback("token", piece)
                                except Exception:
                                    log.debug(
                                        "chunk_callback raised; suppressing",
                                        exc_info=True,
                                    )
                        else:
                            content_head_buffer += piece
                            if len(content_head_buffer) >= CONTENT_HEAD_FLUSH_THRESHOLD:
                                content_head_flushed = True
                                if (
                                    content_head_buffer.strip()
                                    not in PLACEHOLDER_CONTENTS
                                ):
                                    parts.append(content_head_buffer)
                                    if chunk_callback is not None:
                                        try:
                                            chunk_callback("token", content_head_buffer)
                                        except Exception:
                                            log.debug(
                                                "chunk_callback raised; suppressing",
                                                exc_info=True,
                                            )
                                content_head_buffer = ""
                # Periodic live-metrics estimate so the UI counter ticks
                # during long reasoning streams. Authoritative `usage`
                # arrives at end-of-stream and the caller will overwrite
                # this estimate then.
                if (
                    chunk_callback is not None
                    and now - last_live_metrics >= cls.LIVE_METRICS_INTERVAL_SECONDS
                ):
                    total_out_chars = chars + sum(len(s) for s in reasoning_buffer)
                    est_out = total_out_chars // cls.LIVE_METRICS_CHARS_PER_TOKEN
                    elapsed = now - stream_started
                    try:
                        chunk_callback(
                            "stream_metrics",
                            json.dumps(
                                {
                                    "in": est_input_tokens,
                                    "out": est_out,
                                    "seconds": round(elapsed, 1),
                                }
                            ),
                        )
                    except Exception:
                        log.debug(
                            "chunk_callback raised; suppressing",
                            exc_info=True,
                        )
                    last_live_metrics = now
                    total_reasoning = sum(len(s) for s in reasoning_buffer)
                    if (
                        total_reasoning - reasoning_logged_chars
                        >= cls.REASONING_FLUSH_CHARS
                    ):
                        joined = "".join(reasoning_buffer)
                        new_slice = joined[reasoning_logged_chars:]
                        log.info("LLM reasoning >> %s", cls._compact(new_slice))
                        reasoning_logged_chars = len(joined)
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.ConnectionError,
        ) as exc:
            log.warning(
                "LLM stream interrupted after %.1fs, %d events, %d content "
                "chars, delta fields=%s: %s",
                time.monotonic() - stream_started,
                events,
                chars,
                cls._format_field_counts(delta_field_chars),
                exc,
            )
            raise
        joined_reasoning = "".join(reasoning_buffer)
        if len(joined_reasoning) > reasoning_logged_chars:
            tail = joined_reasoning[reasoning_logged_chars:]
            log.info("LLM reasoning >> %s", cls._compact(tail))
        # End-of-stream: the head buffer either holds non-placeholder
        # content (flush it) or a placeholder we silently dropped.
        if not content_head_flushed and content_head_buffer:
            if content_head_buffer.strip() not in PLACEHOLDER_CONTENTS:
                parts.append(content_head_buffer)
                if chunk_callback is not None:
                    try:
                        chunk_callback("token", content_head_buffer)
                    except Exception:
                        log.debug(
                            "chunk_callback raised; suppressing",
                            exc_info=True,
                        )
        tool_calls = cls._finalize_tool_calls(tool_call_parts)
        log.info(
            "LLM stream complete: %.1fs elapsed, %d events, %d content chars, "
            "tool_calls=%d, finish=%s, delta fields=%s",
            time.monotonic() - stream_started,
            events,
            chars,
            len(tool_calls),
            finish_reason,
            cls._format_field_counts(delta_field_chars),
        )
        return "".join(parts), usage, tool_calls, finish_reason, len(joined_reasoning)

    @staticmethod
    def _format_field_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "{}"
        return "{" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "}"

    @staticmethod
    def _truncate_repr(obj: Any, limit: int) -> str:
        s = json.dumps(obj, default=str, ensure_ascii=False)
        return s if len(s) <= limit else s[:limit] + "..."

    @staticmethod
    def _compact(text: str) -> str:
        """Collapse whitespace/newlines so a multi-line reasoning chunk fits
        on a single log line, keeping the action's console output readable."""
        return " ".join(text.split())

    @staticmethod
    def _merge_tool_call_chunk(
        parts: dict[int, dict[str, Any]], chunk: dict[str, Any]
    ) -> None:
        """Accumulate a streamed tool_call delta into ``parts``.

        OpenAI streams tool calls one chunk at a time keyed by ``index``;
        each chunk may set ``id`` and the ``function.name`` once, and
        contributes a slice of ``function.arguments`` that we concatenate.
        Defensive against missing fields — some providers omit ``index``
        on the very first chunk."""
        idx = chunk.get("index")
        if not isinstance(idx, int):
            idx = len(parts)
        slot = parts.setdefault(
            idx, {"id": None, "name": None, "arguments": "", "thought_signature": None}
        )
        if chunk.get("id"):
            slot["id"] = chunk["id"]
        fn = chunk.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        args_piece = fn.get("arguments")
        if isinstance(args_piece, str):
            slot["arguments"] += args_piece
        sig = _extract_thought_signature(chunk)
        if sig:
            slot["thought_signature"] = sig

    @staticmethod
    def _finalize_tool_calls(parts: dict[int, dict[str, Any]]) -> list[ToolCall]:
        out: list[ToolCall] = []
        for idx in sorted(parts):
            slot = parts[idx]
            name = slot.get("name") or ""
            if not name:
                # Stream produced a tool_calls slot with no function name —
                # nothing useful we can do with it; drop quietly.
                continue
            out.append(
                ToolCall(
                    id=slot.get("id") or f"call_{idx}",
                    name=name,
                    arguments=slot.get("arguments") or "",
                    thought_signature=slot.get("thought_signature"),
                )
            )
        return out


def _extract_thought_signature(tc: dict[str, Any]) -> Optional[str]:
    """Pull Gemini 3's ``extra_content.google.thought_signature`` off a raw
    tool-call dict, returning ``None`` for any other provider's shape."""
    extra = tc.get("extra_content")
    if not isinstance(extra, dict):
        return None
    google = extra.get("google")
    if not isinstance(google, dict):
        return None
    sig = google.get("thought_signature")
    return sig if isinstance(sig, str) and sig else None


# Kimi (Moonshot) models served through the HF Router sometimes serialize a
# tool call into `message.content` as their raw chat-template special tokens
# and leave the structured `tool_calls` field empty, with
# `finish_reason="stop"`. The agent loop reads that as "no tool calls, so this
# is the final answer" and publishes the markup verbatim — that is how a review
# body of nothing but `<|tool_calls_section_begin|>...` reached serge#79. Parse
# the markup back into real ToolCall objects so the turn stays a tool turn.
_TEXT_TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_TEXT_TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
_TEXT_TOOL_CALL_BEGIN = "<|tool_call_begin|>"
_TEXT_TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
_TEXT_TOOL_CALL_END = "<|tool_call_end|>"

# The id between the begin and argument markers, e.g. `functions.read_file:6`.
# Both the `functions.` namespace prefix and the `:index` suffix are optional,
# so a bare `read_file` still parses.
_TEXT_TOOL_CALL_ID_RE = re.compile(
    r"\A(?:functions?[.:])?(?P<name>[A-Za-z0-9_.\-]+?)(?::(?P<index>\d+))?\Z"
)


def _split_text_tool_call_id(raw: str) -> str:
    """The function name out of a Kimi-style text tool-call id, or "" when the
    id doesn't look like one (so the caller drops the call rather than
    inventing a tool that doesn't exist)."""
    match = _TEXT_TOOL_CALL_ID_RE.match(raw.strip())
    return match.group("name") if match else ""


def _parse_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Recover tool calls a model wrote into ``content`` as special tokens.

    Returns ``(calls, content_without_the_markup)``. Returns ``([], content)``
    unchanged when no marker is present, which is every well-behaved
    provider — so the common path costs one substring check.

    Tolerant of a truncated tail: a trailing call whose ``<|tool_call_end|>``
    never arrived still yields a ToolCall from the rest of the content, since
    the arguments are usually complete by then and re-asking would throw the
    turn away.
    """
    if _TEXT_TOOL_CALL_BEGIN not in content:
        return [], content

    calls: list[ToolCall] = []
    kept: list[str] = []
    pos = 0
    while True:
        begin = content.find(_TEXT_TOOL_CALL_BEGIN, pos)
        if begin < 0:
            break
        kept.append(content[pos:begin])
        after_begin = begin + len(_TEXT_TOOL_CALL_BEGIN)
        arg_sep = content.find(_TEXT_TOOL_CALL_ARGUMENT_BEGIN, after_begin)
        if arg_sep < 0:
            # No argument marker after this begin marker — nothing parseable
            # is left, and whatever follows is not review prose either.
            pos = len(content)
            break
        end = content.find(_TEXT_TOOL_CALL_END, arg_sep)
        args_start = arg_sep + len(_TEXT_TOOL_CALL_ARGUMENT_BEGIN)
        arguments = content[args_start : end if end >= 0 else len(content)].strip()
        raw_id = content[after_begin:arg_sep].strip()
        name = _split_text_tool_call_id(raw_id)
        if name:
            calls.append(
                ToolCall(
                    id=raw_id or f"call_{len(calls)}",
                    name=name,
                    arguments=arguments,
                )
            )
        if end < 0:
            pos = len(content)
            break
        pos = end + len(_TEXT_TOOL_CALL_END)
    kept.append(content[pos:])

    cleaned = "".join(kept)
    for marker in (_TEXT_TOOL_CALLS_SECTION_BEGIN, _TEXT_TOOL_CALLS_SECTION_END):
        cleaned = cleaned.replace(marker, "")
    return calls, cleaned.strip()


def _parse_tool_calls_from_message(raw: Any) -> list[ToolCall]:
    """Build ToolCall objects from the non-streaming ``message.tool_calls``."""
    if not isinstance(raw, list):
        return []
    out: list[ToolCall] = []
    for i, tc in enumerate(raw):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args) if args is not None else ""
        out.append(
            ToolCall(
                id=tc.get("id") or f"call_{i}",
                name=name,
                arguments=args,
                thought_signature=_extract_thought_signature(tc),
            )
        )
    return out
