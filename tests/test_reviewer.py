import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from reviewbot.llm_client import ChatCompletionClient, ChatResult, ToolCall
from reviewbot.patch import parse_patch
from reviewbot.tools import ToolEnv
from reviewbot.reviewer import (
    _LOG_MSG_MAX_CHARS,
    _MAX_TRUNCATION_RETRIES,
    _UnparseableLLMOutput,
    _assistant_tool_call_dict,
    _build_annotated_diff_chunks,
    _content_preview,
    _emit_chat_message,
    _final_recovery_message,
    _is_model_markup_only,
    _needs_final_salvage,
    _extract_json,
    _REVIEW_JSON_KEYS,
    _merge_chunk_event,
    _merge_chunk_summaries,
    _prose_outside_json,
    _run_agentic_loop,
    _summarize_rejected_comments,
    STOP_ANSWERED,
    STOP_BLIND_TURN_CAP,
    STOP_INPUT_TOKEN_CAP,
    STOP_NO_LLM_TURNS,
    STOP_REPEAT_GUARD,
    merge_session_records,
    no_llm_session_record,
    session_record,
)


class EmitChatMessageTests(unittest.TestCase):
    def _capture(self):
        events: list[tuple[str, str]] = []
        return events, (lambda kind, text: events.append((kind, text)))

    def test_assistant_turn_records_content_and_tool_calls(self) -> None:
        events, emit = self._capture()
        _emit_chat_message(
            emit,
            "assistant",
            content="looking at the diff",
            reasoning_chars=42,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(id="t0", name="read_file", arguments='{"path":"a.py"}')
            ],
        )
        self.assertEqual(len(events), 1)
        kind, text = events[0]
        # "chat", NOT "message": "message" is the SSE default event type.
        self.assertEqual(kind, "chat")
        payload = json.loads(text)
        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["content"], "looking at the diff")
        self.assertEqual(payload["reasoning_chars"], 42)
        self.assertEqual(payload["finish_reason"], "tool_calls")
        self.assertEqual(payload["tool_calls"][0]["name"], "read_file")

    def test_empty_final_turn_is_still_logged(self) -> None:
        # The exact failure mode we need visible: an empty completion with
        # finish_reason=None must produce a record (with no content key).
        events, emit = self._capture()
        _emit_chat_message(emit, "assistant", content="", finish_reason=None)
        self.assertEqual(len(events), 1)
        payload = json.loads(events[0][1])
        self.assertEqual(payload["role"], "assistant")
        self.assertNotIn("content", payload)
        self.assertNotIn("finish_reason", payload)

    def test_long_tool_result_is_truncated(self) -> None:
        events, emit = self._capture()
        big = "x" * (_LOG_MSG_MAX_CHARS + 500)
        _emit_chat_message(emit, "tool", content=big, tool_name="grep")
        payload = json.loads(events[0][1])
        self.assertEqual(payload["name"], "grep")
        self.assertLess(len(payload["content"]), len(big))
        self.assertIn("truncated", payload["content"])

    def test_none_emit_is_a_noop(self) -> None:
        _emit_chat_message(None, "assistant", content="hi")  # must not raise


class FinalSalvageTests(unittest.TestCase):
    def test_empty_content_needs_salvage(self) -> None:
        # The production failure: empty completion, finish_reason=None.
        self.assertTrue(
            _needs_final_salvage(ChatResult(content="", finish_reason=None))
        )
        self.assertTrue(
            _needs_final_salvage(ChatResult(content="   \n", finish_reason=None))
        )

    def test_length_truncation_needs_salvage(self) -> None:
        self.assertTrue(
            _needs_final_salvage(
                ChatResult(content='{"partial', finish_reason="length")
            )
        )

    def test_good_answer_does_not_need_salvage(self) -> None:
        self.assertFalse(
            _needs_final_salvage(
                ChatResult(content='{"ok": true}', finish_reason="stop")
            )
        )

    def test_recovery_message_varies_by_cause(self) -> None:
        empty = _final_recovery_message(ChatResult(content="", finish_reason=None))
        truncated = _final_recovery_message(
            ChatResult(content='{"partial', finish_reason="length")
        )
        self.assertIn("empty", empty)
        self.assertIn("cut off", truncated)
        self.assertNotEqual(empty, truncated)


class AssistantToolCallDictTests(unittest.TestCase):
    def test_omits_extra_content_without_signature(self) -> None:
        tc = ToolCall(id="t0", name="read_file", arguments='{"path":"a.py"}')
        self.assertEqual(
            _assistant_tool_call_dict(tc),
            {
                "id": "t0",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            },
        )

    def test_reattaches_thought_signature_at_gemini_path(self) -> None:
        tc = ToolCall(
            id="t0",
            name="read_file",
            arguments='{"path":"a.py"}',
            thought_signature="sig-abc",
        )
        self.assertEqual(
            _assistant_tool_call_dict(tc)["extra_content"],
            {"google": {"thought_signature": "sig-abc"}},
        )


class ExtractJsonTests(unittest.TestCase):
    def test_raw_json_object(self) -> None:
        result = _extract_json('{"summary": "ok", "comments": []}')
        self.assertEqual(result, {"summary": "ok", "comments": []})

    def test_strips_surrounding_whitespace(self) -> None:
        result = _extract_json('   \n  {"summary": "ok"}  \n\n')
        self.assertEqual(result, {"summary": "ok"})

    def test_fenced_block_with_json_tag(self) -> None:
        content = 'Here you go:\n```json\n{"summary": "ok"}\n```\nThanks!'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_fenced_block_without_language_tag(self) -> None:
        content = '```\n{"summary": "ok"}\n```'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_fenced_block_uppercase_tag(self) -> None:
        content = '```JSON\n{"a": 1}\n```'
        self.assertEqual(_extract_json(content), {"a": 1})

    def test_skips_empty_fenced_block_then_recovers(self) -> None:
        content = '```\n\n```\nOr maybe:\n```json\n{"summary": "ok"}\n```'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_json_embedded_in_prose_no_fences(self) -> None:
        content = 'Sure: {"summary": "ok", "event": "COMMENT"} — let me know!'
        self.assertEqual(
            _extract_json(content),
            {"summary": "ok", "event": "COMMENT"},
        )

    def test_json_with_braces_in_prose_before_and_after(self) -> None:
        # Stray braces in surrounding prose used to break the naive
        # find('{') / rfind('}') slicing; raw_decode at every '{' recovers.
        content = 'Note: use { for sets.\n{"summary": "ok"}\nUse } to close.'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_first_object_wins_when_multiple_candidates(self) -> None:
        content = '{"summary": "first"}\n\nAlso: {"summary": "second"}'
        # Direct parse fails because of trailing data; first raw_decode wins.
        self.assertEqual(_extract_json(content), {"summary": "first"})

    def test_top_level_array_unwraps_to_inner_object(self) -> None:
        # If the model wraps the review in an array (against the contract),
        # the raw_decode pass still recovers the inner object — pragmatic
        # over strict, since downstream code only needs a dict.
        self.assertEqual(_extract_json('[{"summary": "ok"}]'), {"summary": "ok"})

    def test_top_level_array_with_no_inner_object_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("[1, 2, 3]")
        self.assertIn("did not contain a JSON object", str(ctx.exception))

    def test_empty_string_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_none_content_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json(None)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_only_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("   \n\t  \n")
        self.assertIn("whitespace", str(ctx.exception).lower())

    def test_failure_message_includes_length_and_preview(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("I cannot help with this request.")
        msg = str(ctx.exception)
        self.assertIn("length=", msg)
        self.assertIn("preview=", msg)
        self.assertIn("cannot help", msg)

    def test_failure_preview_truncates_long_content(self) -> None:
        long = "x" * 5000
        with self.assertRaises(ValueError) as ctx:
            _extract_json(long)
        self.assertIn("length=5000", str(ctx.exception))
        # Full 5000 chars must NOT be in the preview.
        self.assertLess(len(str(ctx.exception)), 1500)

    def test_nested_object_with_braces_inside_strings(self) -> None:
        content = '{"summary": "use { and } carefully", "comments": []}'
        self.assertEqual(
            _extract_json(content),
            {"summary": "use { and } carefully", "comments": []},
        )


class ExtractJsonRequiredKeysTests(unittest.TestCase):
    """`require_any_key` stops the forgiving raw_decode pass from returning
    incidental JSON. Without it a leaked tool call's own argument object was
    accepted as a review, yielding an empty summary that got published as raw
    special-token markup (serge#79)."""

    # The tool-call arguments Kimi leaked into content on the serge#79 run.
    LEAKED_ARGS = (
        '{"path": "reviewbot/reviewer.py", "start_line": 248, "end_line": 280}'
    )

    def test_leaked_tool_arguments_are_rejected_as_a_review(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json(self.LEAKED_ARGS, _REVIEW_JSON_KEYS)
        msg = str(ctx.exception)
        self.assertIn("did not contain a JSON object", msg)
        self.assertIn("summary", msg)

    def test_same_content_is_still_accepted_without_the_filter(self) -> None:
        # Default behaviour is unchanged for callers that don't opt in.
        self.assertEqual(
            _extract_json(self.LEAKED_ARGS)["path"], "reviewbot/reviewer.py"
        )

    def test_review_object_passes_the_filter(self) -> None:
        content = '{"summary": "ok", "comments": [], "event": "COMMENT"}'
        self.assertEqual(_extract_json(content, _REVIEW_JSON_KEYS)["summary"], "ok")

    def test_any_single_contract_key_is_enough(self) -> None:
        self.assertEqual(
            _extract_json('{"event": "APPROVE"}', _REVIEW_JSON_KEYS),
            {"event": "APPROVE"},
        )

    def test_skips_a_non_review_object_and_finds_the_review_after_it(self) -> None:
        content = f'Reading: {self.LEAKED_ARGS}\n\n{{"summary": "the real review"}}'
        self.assertEqual(
            _extract_json(content, _REVIEW_JSON_KEYS),
            {"summary": "the real review"},
        )

    def test_filter_also_applies_to_fenced_blocks(self) -> None:
        content = (
            f'```json\n{self.LEAKED_ARGS}\n```\n```json\n{{"summary": "real"}}\n```'
        )
        self.assertEqual(_extract_json(content, _REVIEW_JSON_KEYS), {"summary": "real"})


class ModelMarkupOnlyTests(unittest.TestCase):
    def test_leaked_tool_call_markup_is_markup_only(self) -> None:
        # What actually got published on serge#79, once `_prose_outside_json`
        # had removed the JSON arguments from the middle of it.
        self.assertTrue(
            _is_model_markup_only(
                "<|tool_calls_section_begin|><|tool_call_begin|>"
                "functions.read_file:6<|tool_call_argument_begin|>\n\n"
                "<|tool_call_end|><|tool_calls_section_end|>"
            )
        )

    def test_real_review_is_not_markup_only(self) -> None:
        self.assertFalse(_is_model_markup_only("This patch looks correct."))

    def test_review_quoting_a_special_token_is_not_markup_only(self) -> None:
        # A tokenizer review may legitimately mention these tokens; we must
        # not treat such a review as garbage.
        self.assertFalse(
            _is_model_markup_only(
                "The test asserts `<|endoftext|>` is appended, which is right."
            )
        )

    def test_empty_and_whitespace_are_not_markup_only(self) -> None:
        # "" means "no summary", which the publish gate handles separately.
        self.assertFalse(_is_model_markup_only(""))
        self.assertFalse(_is_model_markup_only("   \n\t "))


class ProseOutsideJsonTests(unittest.TestCase):
    """The stub-JSON-plus-prose reply: `_extract_json` takes the stub, this
    takes the review the model actually wrote."""

    STUB = '{"summary": "", "event": "COMMENT", "comments": []}'
    PROSE = (
        "### Correctness\n- `src/foo.py` drops the return value.\n\n### Style\n- nit"
    )

    def test_unclosed_json_fence_yields_prose_only(self) -> None:
        # The peft#3354 case: the model opened ```json and never closed it,
        # so the whole review rendered inside one code block.
        content = f"```json\n{self.STUB}\n\n{self.PROSE}\n"
        out = _prose_outside_json(content)
        self.assertEqual(out, self.PROSE)
        self.assertNotIn("```json", out)
        self.assertNotIn('"summary"', out)

    def test_closed_json_fence_yields_prose_only(self) -> None:
        content = f"```json\n{self.STUB}\n```\n\n{self.PROSE}\n"
        self.assertEqual(_prose_outside_json(content), self.PROSE)

    def test_unfenced_stub_yields_prose_only(self) -> None:
        content = f"{self.STUB}\n\n{self.PROSE}"
        self.assertEqual(_prose_outside_json(content), self.PROSE)

    def test_prose_before_the_json_is_kept(self) -> None:
        content = f"Here is my review:\n\n{self.STUB}\n\n{self.PROSE}"
        self.assertEqual(
            _prose_outside_json(content),
            f"Here is my review:\n\n{self.PROSE}",
        )

    def test_code_fences_inside_the_prose_survive(self) -> None:
        prose = "### Correctness\n\n```python\nx = 1\n```\n\nLooks wrong."
        content = f"```json\n{self.STUB}\n```\n\n{prose}"
        self.assertEqual(_prose_outside_json(content), prose)

    def test_json_only_reply_has_no_prose(self) -> None:
        self.assertEqual(_prose_outside_json(f"```json\n{self.STUB}\n```"), "")
        self.assertEqual(_prose_outside_json(self.STUB), "")

    def test_no_json_object_returns_empty(self) -> None:
        self.assertEqual(_prose_outside_json("just prose, no object"), "")

    def test_empty_and_none_content(self) -> None:
        self.assertEqual(_prose_outside_json(""), "")
        self.assertEqual(_prose_outside_json("   \n\t "), "")
        self.assertEqual(_prose_outside_json(None), "")

    def test_matches_what_extract_json_consumed(self) -> None:
        # The two halves of the same reply: the dict and the leftover prose.
        content = f"```json\n{self.STUB}\n\n{self.PROSE}\n"
        self.assertEqual(_extract_json(content), json.loads(self.STUB))
        self.assertEqual(_prose_outside_json(content), self.PROSE)


class ContentPreviewTests(unittest.TestCase):
    def test_short_content_returned_verbatim(self) -> None:
        self.assertEqual(_content_preview("hello"), "hello")

    def test_long_content_truncated_with_marker(self) -> None:
        out = _content_preview("x" * 1000, limit=100)
        self.assertTrue(out.startswith("x" * 100))
        self.assertIn("+900 chars truncated", out)


class UnparseableLLMOutputTests(unittest.TestCase):
    def test_length_finish_reason_gets_actionable_message(self) -> None:
        exc = _UnparseableLLMOutput(
            content='{"summary":',
            finish_reason="length",
            metrics_line="56 LLM turns · 62 tool calls",
        )

        msg = exc.user_message()
        self.assertIn("truncated", msg)
        self.assertIn("finish_reason=length", msg)
        self.assertIn("Increase LLM_MAX_TOKENS", msg)
        self.assertIn("reduce TOOL_MAX_ITERATIONS", msg)

    def test_other_finish_reason_keeps_generic_message(self) -> None:
        exc = _UnparseableLLMOutput(
            content="oops",
            finish_reason="stop",
            metrics_line="1 LLM turn",
        )

        self.assertIn("unparseable output", exc.user_message())


class DiffChunkingTests(unittest.TestCase):
    def test_large_single_file_is_split_without_losing_positions(self) -> None:
        patch = "@@ -0,0 +1,18 @@\n" + "\n".join(
            f"+line_{i}_{'x' * 20}" for i in range(1, 19)
        )
        files = [{"filename": "src/big.py", "patch": patch}]

        chunks, skipped = _build_annotated_diff_chunks(
            files, max_chars=220, skip_paths=set()
        )

        self.assertEqual(skipped, [])
        self.assertGreater(len(chunks), 1)
        parsed = parse_patch("src/big.py", patch)
        visible: set[tuple[str, int]] = set()
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 220)
            self.assertIn("--- a/src/big.py", chunk.text)
            visible.update(chunk.visible_positions.get("src/big.py", set()))
        self.assertEqual(visible, parsed.valid_positions)

    def test_skip_paths_are_omitted_and_reported(self) -> None:
        files = [
            {"filename": "kept.py", "patch": "@@ -0,0 +1 @@\n+ok"},
            {"filename": "skip.py", "patch": "@@ -0,0 +1 @@\n+nope"},
        ]

        chunks, skipped = _build_annotated_diff_chunks(
            files, max_chars=500, skip_paths={"skip.py"}
        )

        self.assertEqual(skipped, ["skip.py"])
        self.assertEqual(len(chunks), 1)
        self.assertIn("kept.py", chunks[0].text)
        self.assertNotIn("skip.py", chunks[0].text)


class ChunkMergeTests(unittest.TestCase):
    def test_merge_chunk_summaries_does_not_mention_chunks(self) -> None:
        # The fallback merge is what the published review falls back to
        # when the synthesis LLM call is unavailable; it must NOT leak
        # the chunking implementation detail to GitHub readers.
        out = _merge_chunk_summaries([(1, "first"), (2, "second")], 2)
        self.assertNotIn("chunk", out.lower())
        self.assertIn("first", out)
        self.assertIn("second", out)

    def test_merge_chunk_summaries_single_passes_through(self) -> None:
        out = _merge_chunk_summaries([(1, "only summary")], 1)
        self.assertEqual(out, "only summary")

    def test_merge_chunk_summaries_skips_empty(self) -> None:
        out = _merge_chunk_summaries([(1, "kept"), (2, "   ")], 2)
        self.assertEqual(out, "kept")

    def test_merge_chunk_event_escalates_request_changes(self) -> None:
        self.assertEqual(
            _merge_chunk_event(
                ["COMMENT", "REQUEST_CHANGES", "APPROVE"], comments_count=1
            ),
            "REQUEST_CHANGES",
        )

    def test_merge_chunk_event_keeps_approve_only_when_clean(self) -> None:
        self.assertEqual(
            _merge_chunk_event(["APPROVE", "APPROVE"], comments_count=0),
            "APPROVE",
        )
        self.assertEqual(
            _merge_chunk_event(["APPROVE", "APPROVE"], comments_count=1),
            "COMMENT",
        )


class SummarizeRejectedCommentsTests(unittest.TestCase):
    def test_empty_list_renders_empty_string(self) -> None:
        self.assertEqual(_summarize_rejected_comments([]), "")

    def test_renders_path_line_refs(self) -> None:
        out = _summarize_rejected_comments(
            [{"path": "foo.py", "line": 10}, {"path": "bar.py", "line": 20}]
        )
        self.assertEqual(out, "foo.py:10, bar.py:20")

    def test_truncates_after_max_items(self) -> None:
        rejected = [{"path": f"f{i}.py", "line": i} for i in range(10)]
        out = _summarize_rejected_comments(rejected, max_items=3)
        self.assertIn("f0.py:0", out)
        self.assertIn("f2.py:2", out)
        self.assertIn("+7 more", out)
        self.assertNotIn("f9.py:9", out)

    def test_handles_missing_fields_gracefully(self) -> None:
        out = _summarize_rejected_comments([{}, {"path": "foo.py"}])
        self.assertEqual(out, "?:?, foo.py:?")


class _CfgStub:
    """Lean Config stand-in for _run_agentic_loop, which only reads a
    handful of fields. Avoids the full Config(**kwargs) dance."""

    def __init__(
        self,
        *,
        llm_max_tokens: int = 1024,
        tool_max_iterations: int = 30,
        llm_max_input_tokens: int = 0,
        llm_reasoning_effort: str | None = None,
        tool_repeat_limit: int = 0,
    ) -> None:
        self.llm_max_tokens = llm_max_tokens
        self.tool_max_iterations = tool_max_iterations
        self.llm_max_input_tokens = llm_max_input_tokens
        self.llm_reasoning_effort = llm_reasoning_effort
        self.tool_repeat_limit = tool_repeat_limit


class _FakeLLM:
    """Returns a queue of ChatResult objects, one per .complete() call.
    Final entry is reused if the loop calls beyond the queue (so the
    "force final answer" tail can always satisfy itself)."""

    def __init__(self, results: list[ChatResult]) -> None:
        self._results = list(results)
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs) -> ChatResult:
        self.calls.append({"messages": list(messages), **kwargs})
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


class InputTokenBudgetTests(unittest.TestCase):
    """Cumulative input-token cap should short-circuit the agentic loop
    and trigger the existing 'force final answer' tail."""

    def test_cap_breaks_loop_and_forces_final_answer(self) -> None:
        cfg = _CfgStub(llm_max_input_tokens=1_500_000)
        # Turn 1: tool call, reports 1.2M prompt tokens.
        # Turn 2 (forced final): returns the answer JSON. Loop should
        # never run a 3rd turn because the cap fires before it.
        results = [
            ChatResult(
                content="",
                usage={"prompt_tokens": 1_200_000, "completion_tokens": 50},
                tool_calls=[ToolCall(id="t0", name="noop", arguments="{}")],
            ),
            ChatResult(
                content='{"summary": "done", "comments": []}',
                usage={"prompt_tokens": 400_000, "completion_tokens": 30},
            ),
        ]
        llm = _FakeLLM(results)
        # Loop needs a real ToolEnv so tool_calls aren't short-circuited
        # by the "tools disabled" branch. /tmp is a real dir on every
        # platform we run tests on.
        tool_env = ToolEnv(repo_root="/tmp")
        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=tool_env,
            prior_prompt_tokens=400_000,  # prior chunks already used 0.4M
        )
        # We expect exactly two complete() calls: the first turn, then
        # the forced final-answer turn after the cap fires.
        self.assertEqual(len(llm.calls), 2)
        # Final-answer call must run without tools.
        self.assertNotIn("tools", llm.calls[1])
        self.assertEqual(chat.content, '{"summary": "done", "comments": []}')
        self.assertEqual(metrics.turns, 2)

    def test_disabled_cap_does_not_short_circuit(self) -> None:
        cfg = _CfgStub(llm_max_input_tokens=0, tool_max_iterations=2)
        # No tool calls => loop returns on the first turn naturally.
        results = [
            ChatResult(
                content='{"summary": "ok", "comments": []}',
                usage={"prompt_tokens": 5_000_000, "completion_tokens": 10},
            ),
        ]
        llm = _FakeLLM(results)
        _, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
            prior_prompt_tokens=10_000_000,
        )
        self.assertEqual(metrics.turns, 1)
        self.assertEqual(len(llm.calls), 1)


class ToolRepeatGuardLoopTests(unittest.TestCase):
    """The repeat guard has to cut the loop off, not just annotate results.

    Prod task 433e8274 spent ~55 turns and its whole 2M input-token budget on two
    greps it kept re-issuing verbatim, then was forced into a tool-less answer
    with nothing learned. Here the same shape stops after a handful of turns.
    """

    def _repeating_llm(self, tool_turns: int = 4) -> _FakeLLM:
        # Each of the first ``tool_turns`` turns re-issues the identical grep;
        # the trailing entry answers the forced final turn and is reused for any
        # turn beyond it (see _FakeLLM).
        repeat = ChatResult(
            content="",
            usage={"prompt_tokens": 40_000, "completion_tokens": 40},
            tool_calls=[
                ToolCall(
                    id="t", name="grep", arguments='{"pattern": "Foo", "path": "."}'
                )
            ],
        )
        final = ChatResult(
            content='{"summary": "stuck", "comments": []}',
            usage={"prompt_tokens": 40_000, "completion_tokens": 20},
        )
        return _FakeLLM([repeat] * tool_turns + [final])

    def test_repeated_calls_break_the_loop_early(self) -> None:
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=3)
        llm = self._repeating_llm()
        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        # 1 original + 3 repeats = 4 tool turns, then the forced final answer.
        self.assertEqual(len(llm.calls), 5)
        self.assertNotIn("tools", llm.calls[-1])
        self.assertEqual(chat.content, '{"summary": "stuck", "comments": []}')
        self.assertEqual(metrics.tool_calls, 4)

    def test_the_model_is_told_it_is_repeating(self) -> None:
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=3)
        llm = self._repeating_llm()
        _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        tool_messages = [
            m for m in llm.calls[-1]["messages"] if m.get("role") == "tool"
        ]
        # First result is clean; every repeat carries the correction.
        self.assertNotIn("[serge]", tool_messages[0]["content"])
        self.assertIn("already made this exact grep call", tool_messages[1]["content"])
        self.assertIn("You are in a loop", tool_messages[-1]["content"])

    def test_emits_the_loop_reason_not_budget_exhausted(self) -> None:
        """Operators read these events to tell a real budget exhaustion from a
        stuck loop — the generic message must not mask the specific one."""
        events: list[tuple[str, str]] = []
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=3)
        _run_agentic_loop(
            self._repeating_llm(),  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
            emit=lambda kind, text: events.append((kind, text)),
        )
        logs = [text for kind, text in events if kind == "log"]
        self.assertTrue(any("Stuck in a tool-call loop" in line for line in logs))
        self.assertFalse(any("Agent budget exhausted" in line for line in logs))

    def test_disabled_guard_lets_the_loop_run_to_its_other_caps(self) -> None:
        cfg = _CfgStub(tool_max_iterations=4, tool_repeat_limit=0)
        llm = self._repeating_llm()
        _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        # Blind-turn cap (4) governs instead of the repeat guard.
        self.assertEqual(len(llm.calls), 5)

    def test_distinct_calls_are_not_cut_off(self) -> None:
        """A genuine investigation making different calls must be untouched."""
        turns = [
            ChatResult(
                content="",
                usage={"prompt_tokens": 1_000, "completion_tokens": 10},
                tool_calls=[
                    ToolCall(
                        id=f"t{i}", name="grep", arguments='{"pattern": "P%d"}' % i
                    )
                ],
            )
            for i in range(8)
        ]
        final = ChatResult(
            content='{"summary": "ok", "comments": []}',
            usage={"prompt_tokens": 1_000, "completion_tokens": 10},
        )
        llm = _FakeLLM(turns + [final])
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=3)
        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        self.assertEqual(metrics.tool_calls, 8)
        self.assertEqual(chat.content, '{"summary": "ok", "comments": []}')


class ValidationGateTests(unittest.TestCase):
    """The optional `validate` callback turns the final answer into a
    verification gate: failures are fed back into the same conversation."""

    def _result(self, content: str) -> ChatResult:
        return ChatResult(
            content=content,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    def test_feedback_reenters_same_conversation_then_accepts(self) -> None:
        cfg = _CfgStub()
        llm = _FakeLLM(
            [self._result('{"patch": "v1"}'), self._result('{"patch": "v2"}')]
        )
        seen: list[str] = []

        def validate(chat: ChatResult) -> str | None:
            seen.append(chat.content)
            return (
                "normalizer failed, fix it"
                if chat.content == '{"patch": "v1"}'
                else None
            )

        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
            validate=validate,
            max_validation_retries=2,
        )
        # Both answers were validated; the corrected one was accepted.
        self.assertEqual(seen, ['{"patch": "v1"}', '{"patch": "v2"}'])
        self.assertEqual(chat.content, '{"patch": "v2"}')
        self.assertEqual(len(llm.calls), 2)
        # The feedback was injected as a user turn in the same conversation.
        second_turn = llm.calls[1]["messages"]
        self.assertTrue(
            any(
                m.get("role") == "user"
                and m.get("content") == "normalizer failed, fix it"
                for m in second_turn
            )
        )
        # And the rejected answer is in the history as an assistant turn.
        self.assertTrue(
            any(
                m.get("role") == "assistant" and m.get("content") == '{"patch": "v1"}'
                for m in second_turn
            )
        )

    def test_retries_exhausted_returns_last_answer(self) -> None:
        cfg = _CfgStub()
        llm = _FakeLLM([self._result('{"patch": "bad"}')])
        validations = {"n": 0}

        def validate(chat: ChatResult) -> str | None:
            validations["n"] += 1
            return "still broken"

        chat, _ = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
            validate=validate,
            max_validation_retries=1,
        )
        # Initial validation + 1 retry, then the last answer is returned even
        # though it never validated.
        self.assertEqual(validations["n"], 2)
        self.assertEqual(chat.content, '{"patch": "bad"}')

    def test_no_validator_is_unchanged(self) -> None:
        cfg = _CfgStub()
        llm = _FakeLLM([self._result('{"patch": "v1"}')])
        chat, _ = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
        )
        self.assertEqual(chat.content, '{"patch": "v1"}')
        self.assertEqual(len(llm.calls), 1)

    def test_force_final_path_still_validates(self) -> None:
        """Regression: exhausting the tool budget must NOT bypass the
        verification gate. The forced final answer has to go through
        ``validate`` (with tool-less corrections) exactly like an in-budget
        final answer, or un-normalized patches reach an opened PR."""
        cfg = _CfgStub(llm_max_input_tokens=1_500_000)
        # Turn 1: a tool call that pushes cumulative input tokens over the cap,
        # forcing the loop into its tool-less final-answer tail. Turns 2/3 are
        # the forced final answer + its tool-less correction.
        results = [
            ChatResult(
                content="",
                usage={"prompt_tokens": 1_200_000, "completion_tokens": 50},
                tool_calls=[ToolCall(id="t0", name="noop", arguments="{}")],
            ),
            ChatResult(
                content='{"patch": "v1"}',
                usage={"prompt_tokens": 400_000, "completion_tokens": 30},
            ),
            ChatResult(
                content='{"patch": "v2"}',
                usage={"prompt_tokens": 400_000, "completion_tokens": 30},
            ),
        ]
        llm = _FakeLLM(results)
        seen: list[str] = []

        def validate(chat: ChatResult) -> str | None:
            seen.append(chat.content)
            return (
                "normalizer failed, fix it"
                if chat.content == '{"patch": "v1"}'
                else None
            )

        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
            prior_prompt_tokens=400_000,
            validate=validate,
            max_validation_retries=2,
        )
        # The forced final answer WAS validated, and its rejection drove a
        # tool-less correction that was then accepted.
        self.assertEqual(seen, ['{"patch": "v1"}', '{"patch": "v2"}'])
        self.assertEqual(chat.content, '{"patch": "v2"}')
        # Turn 1 (tool) + forced final v1 + tool-less correction v2.
        self.assertEqual(len(llm.calls), 3)
        # Both forced-final calls run without tools.
        self.assertNotIn("tools", llm.calls[1])
        self.assertNotIn("tools", llm.calls[2])
        # The normalizer feedback re-entered the same conversation.
        self.assertTrue(
            any(
                m.get("role") == "user"
                and m.get("content") == "normalizer failed, fix it"
                for m in llm.calls[2]["messages"]
            )
        )


class TruncationRecoveryTests(unittest.TestCase):
    """A final answer truncated at the provider's output-token limit
    (finish_reason='length') is re-asked as JSON-only with tools off and
    minimal reasoning, instead of failing the whole task."""

    def test_truncated_final_answer_is_retried_json_only(self) -> None:
        cfg = _CfgStub()
        results = [
            # Turn 1: a final answer cut off at the output limit (reasoning ate
            # the budget), leaving the JSON incomplete.
            ChatResult(
                content='{"patch": "half of a diff th',
                usage={"prompt_tokens": 10, "completion_tokens": 16384},
                finish_reason="length",
            ),
            # Turn 2 (recovery): the complete JSON.
            ChatResult(
                content='{"patch": "v2"}',
                usage={"prompt_tokens": 20, "completion_tokens": 30},
                finish_reason="stop",
            ),
        ]
        llm = _FakeLLM(results)
        chat, _ = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        # The complete answer from the recovery turn is returned.
        self.assertEqual(chat.content, '{"patch": "v2"}')
        self.assertEqual(len(llm.calls), 2)
        # Turn 1 had tools; the recovery turn disabled them and forced low
        # reasoning so the whole output budget goes to the JSON.
        self.assertIsNotNone(llm.calls[0]["tools"])
        self.assertIsNone(llm.calls[1]["tools"])
        self.assertEqual(llm.calls[1]["extra"], {"reasoning_effort": "low"})
        # The recovery instruction re-entered the same conversation.
        self.assertTrue(
            any(
                m.get("role") == "user"
                and "output-token limit" in str(m.get("content"))
                for m in llm.calls[1]["messages"]
            )
        )

    def test_truncation_recovery_is_bounded(self) -> None:
        cfg = _CfgStub()
        # Always truncated: recovery must give up after _MAX_TRUNCATION_RETRIES
        # and return the last answer rather than loop forever.
        always_trunc = ChatResult(
            content='{"patch": "nope',
            usage={"prompt_tokens": 10, "completion_tokens": 16384},
            finish_reason="length",
        )
        llm = _FakeLLM([always_trunc])
        chat, _ = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "go"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
        )
        # Initial answer + _MAX_TRUNCATION_RETRIES recovery attempts.
        self.assertEqual(len(llm.calls), 1 + _MAX_TRUNCATION_RETRIES)
        self.assertEqual(chat.finish_reason, "length")


class LeakedTextToolCallLoopTests(unittest.TestCase):
    """End-to-end regression for serge#79: Kimi wrote a tool call into
    `message.content` as special tokens with an empty `tool_calls` field and
    finish_reason="stop". The loop read that as the final answer and the markup
    was published as the review body. Drives the *real* client (mocked HTTP) so
    the whole chain — parse, recover, execute, continue — is covered."""

    LEAKED = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.read_file:6"
        '<|tool_call_argument_begin|>{"path": "hello.py"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    @staticmethod
    def _response(body: dict) -> Mock:
        return Mock(
            status_code=200, raise_for_status=Mock(), json=Mock(return_value=body)
        )

    def _turn(self, content: str) -> Mock:
        return self._response(
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }
        )

    def test_leaked_tool_call_keeps_the_loop_going(self) -> None:
        with tempfile.TemporaryDirectory() as repo:
            with open(os.path.join(repo, "hello.py"), "w") as fh:
                fh.write("print('the file the model asked for')\n")

            final = '{"summary": "reviewed hello.py", "comments": []}'
            with patch(
                "reviewbot.llm_client.requests.post",
                side_effect=[self._turn(self.LEAKED), self._turn(final)],
            ) as mock_post:
                llm = ChatCompletionClient(
                    "https://example.com/v1", "token", "moonshotai/Kimi-K2.6"
                )
                chat, metrics = _run_agentic_loop(
                    llm,
                    [{"role": "user", "content": "review this"}],
                    cfg=_CfgStub(),  # type: ignore[arg-type]
                    tool_env=ToolEnv(repo_root=repo),
                )

            # The turn was treated as a tool turn, not as the final answer.
            self.assertEqual(metrics.tool_calls, 1)
            self.assertEqual(chat.content, final)
            self.assertEqual(mock_post.call_count, 2)

            # The second request carried the executed tool's output back, so the
            # model actually got the file it asked for.
            follow_up = json.loads(mock_post.call_args_list[1].kwargs["data"])
            tool_msgs = [m for m in follow_up["messages"] if m["role"] == "tool"]
            self.assertEqual(len(tool_msgs), 1)
            self.assertEqual(tool_msgs[0]["name"], "read_file")
            self.assertIn("the file the model asked for", tool_msgs[0]["content"])

            # The assistant turn that requested it round-trips as a structured
            # tool_calls message, with the markup stripped from its content.
            assistant = [
                m
                for m in follow_up["messages"]
                if m["role"] == "assistant" and m.get("tool_calls")
            ]
            self.assertEqual(len(assistant), 1)
            self.assertIsNone(assistant[0]["content"])

    def test_final_review_is_parsed_not_the_leaked_tool_arguments(self) -> None:
        # The second half of the serge#79 chain: even if the markup does reach
        # the parser, the leaked arguments must not pass as a review.
        with self.assertRaises(ValueError):
            _extract_json(self.LEAKED, _REVIEW_JSON_KEYS)


if __name__ == "__main__":
    unittest.main()


class StopReasonTests(unittest.TestCase):
    """Which guard ended the session, recorded per job.

    In the one prod window still readable when this was written, all seven task
    sessions that ran LLM turns were terminated by a guard and none by the model
    deciding it was done — a fact that was nowhere in the data, only in the log
    lines of jobs that have since been pruned."""

    def _tool_turn(self, prompt_tokens: int = 40_000) -> ChatResult:
        return ChatResult(
            content="",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 20},
            tool_calls=[ToolCall(id="t", name="grep", arguments='{"pattern": "Foo"}')],
        )

    def _answer(self) -> ChatResult:
        return ChatResult(
            content='{"summary": "ok", "comments": []}',
            usage={"prompt_tokens": 1_000, "completion_tokens": 10},
        )

    def test_a_model_that_finishes_is_recorded_as_answered(self) -> None:
        cfg = _CfgStub()
        _, metrics = _run_agentic_loop(
            _FakeLLM([self._answer()]),  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
        )
        self.assertEqual(metrics.stop_reason, STOP_ANSWERED)

    def test_input_token_cap(self) -> None:
        cfg = _CfgStub(llm_max_input_tokens=100_000)
        _, metrics = _run_agentic_loop(
            _FakeLLM([self._tool_turn(120_000), self._answer()]),  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        self.assertEqual(metrics.stop_reason, STOP_INPUT_TOKEN_CAP)

    def test_repeat_guard(self) -> None:
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=2)
        _, metrics = _run_agentic_loop(
            _FakeLLM([self._tool_turn()] * 3 + [self._answer()]),  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        self.assertEqual(metrics.stop_reason, STOP_REPEAT_GUARD)

    def test_blind_turn_cap(self) -> None:
        cfg = _CfgStub(tool_max_iterations=2, tool_repeat_limit=0)
        turns = [
            ChatResult(
                content="",
                usage={"prompt_tokens": 1_000, "completion_tokens": 10},
                tool_calls=[
                    ToolCall(
                        id=f"t{i}", name="grep", arguments='{"pattern": "P%d"}' % i
                    )
                ],
            )
            for i in range(4)
        ]
        _, metrics = _run_agentic_loop(
            _FakeLLM(turns + [self._answer()]),  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        self.assertEqual(metrics.stop_reason, STOP_BLIND_TURN_CAP)

    def test_the_browse_record_rides_out_with_the_metrics(self) -> None:
        """Whatever exit the loop takes, the counters must be complete."""
        cfg = _CfgStub(tool_max_iterations=0, tool_repeat_limit=0)
        reads = [
            ChatResult(
                content="",
                usage={"prompt_tokens": 1_000, "completion_tokens": 10},
                tool_calls=[
                    ToolCall(
                        id=f"t{i}",
                        name="read_file",
                        arguments='{"path": "a.py", "start_line": %d}' % (i * 50),
                    )
                ],
            )
            for i in range(3)
        ]
        _, metrics = _run_agentic_loop(
            _FakeLLM(reads + [self._answer()]),  # type: ignore[arg-type]
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=ToolEnv(repo_root="/tmp"),
        )
        # Three reads of one file at different offsets: no exact repeat, but two
        # of the three calls re-opened a file already in the context.
        self.assertEqual(metrics.repeats, 0)
        self.assertEqual(metrics.distinct_paths, 1)
        self.assertEqual(metrics.path_revisits, 2)
        self.assertEqual(metrics.stop_reason, STOP_ANSWERED)


class SessionRecordTests(unittest.TestCase):
    def test_record_is_flat_and_json_safe(self) -> None:
        cfg = _CfgStub()
        _, metrics = _run_agentic_loop(
            _FakeLLM(  # type: ignore[arg-type]
                [
                    ChatResult(
                        content='{"summary": "ok", "comments": []}',
                        usage={"prompt_tokens": 7, "completion_tokens": 3},
                    )
                ]
            ),
            [{"role": "user", "content": "x"}],
            cfg=cfg,  # type: ignore[arg-type]
            tool_env=None,
        )
        record = session_record(metrics)
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(record["turns"], 1)
        self.assertEqual(record["prompt_tokens"], 7)
        self.assertEqual(record["rounds"], 1)
        self.assertEqual(record["stop_reason"], STOP_ANSWERED)

    def test_merging_sums_the_bill_and_counts_the_rounds(self) -> None:
        a = {
            "turns": 10,
            "tool_calls": 5,
            "prompt_tokens": 900,
            "rounds": 1,
            "stop_reason": STOP_ANSWERED,
            "distinct_paths": 3,
            "seconds": 1.5,
        }
        b = {
            "turns": 7,
            "tool_calls": 9,
            "prompt_tokens": 1_200,
            "rounds": 1,
            "stop_reason": STOP_INPUT_TOKEN_CAP,
            "distinct_paths": 8,
            "seconds": 2.0,
        }
        merged = merge_session_records(a, b)
        self.assertEqual(merged["turns"], 17)
        self.assertEqual(merged["prompt_tokens"], 2_100)
        self.assertEqual(merged["rounds"], 2)
        self.assertEqual(merged["seconds"], 3.5)
        # One cut-off round is what starves the patch that gets published, so a
        # guard anywhere in the job is what the job reports.
        self.assertEqual(merged["stop_reason"], STOP_INPUT_TOKEN_CAP)
        # Paths cannot be summed without double-counting a file two rounds both
        # opened; the larger round is the honest bound.
        self.assertEqual(merged["distinct_paths"], 8)

    def test_merging_an_accumulation_keeps_its_round_count(self) -> None:
        already = {"turns": 4, "rounds": 3, "stop_reason": STOP_ANSWERED}
        self.assertEqual(merge_session_records({}, already)["rounds"], 3)
        self.assertEqual(
            merge_session_records({"turns": 1, "rounds": 2}, already)["rounds"], 5
        )

    def test_a_zero_round_record_is_not_read_as_one(self) -> None:
        """`rounds: 0` is a real value — a job that never reached the loop — so
        the arithmetic must not fall back to 1 and invent a loop."""
        never_ran = no_llm_session_record()
        self.assertEqual(merge_session_records({}, never_ran)["rounds"], 0)
        one_loop = {"turns": 5, "rounds": 1, "stop_reason": STOP_ANSWERED}
        self.assertEqual(merge_session_records(never_ran, one_loop)["rounds"], 1)
        self.assertEqual(merge_session_records(one_loop, never_ran)["rounds"], 1)

    def test_a_record_without_a_rounds_key_counts_as_one_loop(self) -> None:
        """Missing is not zero: a session record written before `rounds` existed
        still describes exactly one loop."""
        legacy = {"turns": 5, "stop_reason": STOP_ANSWERED}
        self.assertEqual(merge_session_records({}, legacy)["rounds"], 1)
        self.assertEqual(merge_session_records(legacy, dict(legacy))["rounds"], 2)

    def test_merging_nothing_changes_nothing(self) -> None:
        a = {"turns": 3, "rounds": 1}
        self.assertEqual(merge_session_records(a, None), a)
        self.assertEqual(merge_session_records(a, {}), a)
        self.assertEqual(merge_session_records(None, None), {})

    def test_a_job_that_never_ran_the_loop_still_reports(self) -> None:
        """Reproduce-first classifying a group ENVIRONMENT is 0 LLM turns, and
        that has to be countable — otherwise it is indistinguishable from serge
        never having been asked."""
        record = no_llm_session_record()
        self.assertEqual(record["rounds"], 0)
        self.assertEqual(record["turns"], 0)
        self.assertEqual(record["stop_reason"], STOP_NO_LLM_TURNS)
