import json
import unittest

from reviewbot.llm_client import ChatResult, ToolCall
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
    _extract_json,
    _merge_chunk_event,
    _merge_chunk_summaries,
    _run_agentic_loop,
    _summarize_rejected_comments,
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
            tool_calls=[ToolCall(id="t0", name="read_file", arguments='{"path":"a.py"}')],
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
    ) -> None:
        self.llm_max_tokens = llm_max_tokens
        self.tool_max_iterations = tool_max_iterations
        self.llm_max_input_tokens = llm_max_input_tokens
        self.llm_reasoning_effort = llm_reasoning_effort


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


if __name__ == "__main__":
    unittest.main()
