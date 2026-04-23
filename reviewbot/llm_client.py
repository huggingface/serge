import json
import logging
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
    JSON-parsing it (and for handling models that emit malformed JSON)."""

    id: str
    name: str
    arguments: str


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


class AddingSomethingUseless:
    pass


p = AddingSomethingUseless()


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
        try:
            return self._post_with_retries(
                url,
                payload,
                attempts,
                retryable,
                started,
                tools_in_use=bool(tools),
                chunk_callback=chunk_callback,
                est_input_tokens=est_input_tokens,
            )
        except _ToolsUnsupported:
            # Endpoint rejected the tool-augmented payload. Strip the
            # function-calling fields and try once more so the review can
            # still complete (degraded: no browse tools).
            for k in ("tools", "tool_choice"):
                payload.pop(k, None)
            return self._post_with_retries(
                url,
                payload,
                attempts,
                retryable,
                started,
                tools_in_use=False,
                chunk_callback=chunk_callback,
                est_input_tokens=est_input_tokens,
            )

    # Cap any single retry wait, even if the server hands us a huge
    # Retry-After. 120s is plenty for dynamic-rate-limit recovery
    # without letting a misbehaving upstream pin a worker forever.
    _RETRY_WAIT_CEILING_SECONDS = 120.0

    @staticmethod
    def _retry_delay(attempt: int, response: "requests.Response") -> float:
        """Compute how long to sleep before retrying. Honors ``Retry-After``
        (seconds or HTTP-date form) when the server provides one; otherwise
        falls back to exponential backoff (2,4,8,...)."""
        ra = response.headers.get("Retry-After")
        # Guard against non-string values (e.g. Mock objects in tests).
        # Real ``requests.Response`` only ever returns ``str | None`` here.
        if isinstance(ra, str) and ra:
            try:
                return min(float(ra), ChatCompletionClient._RETRY_WAIT_CEILING_SECONDS)
            except ValueError:
                # HTTP-date form (rare on LLM endpoints) — parse it.
                try:
                    from email.utils import parsedate_to_datetime

                    target = parsedate_to_datetime(ra)
                    secs = target.timestamp() - time.time()
                    if secs > 0:
                        return min(
                            secs, ChatCompletionClient._RETRY_WAIT_CEILING_SECONDS
                        )
                except Exception:  # noqa: BLE001
                    pass
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
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(
                    url,
                    headers=self._headers(),
                    data=body,
                    timeout=300,
                    stream=self.stream,
                )
                # 429 (rate limit) and 5xx are retryable. 429 in particular
                # gets hit during agentic tool loops on providers with
                # bursty-traffic dynamic limits (e.g. Together.ai via HF
                # Router, where a few quick tool turns can exceed a 1 RPM
                # cap). Honor Retry-After when the server provides it.
                if (
                    r.status_code == 429 or r.status_code >= 500
                ) and attempt < attempts:
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
                    if r.status_code == 400 and tools_in_use:
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
                if attempt == attempts:
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
        slot = parts.setdefault(idx, {"id": None, "name": None, "arguments": ""})
        if chunk.get("id"):
            slot["id"] = chunk["id"]
        fn = chunk.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        args_piece = fn.get("arguments")
        if isinstance(args_piece, str):
            slot["arguments"] += args_piece

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
                )
            )
        return out


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
        out.append(ToolCall(id=tc.get("id") or f"call_{i}", name=name, arguments=args))
    return out
