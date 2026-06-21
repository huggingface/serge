import json
import unittest
from unittest.mock import Mock, patch

import requests

from reviewbot.llm_client import ChatCompletionClient, LLMResponseError


def _interrupted_iter_lines(prefix_lines: list[str], exc: Exception):
    def gen(*_args, **_kwargs):
        for line in prefix_lines:
            yield line
        raise exc

    return gen


class ChatCompletionClientTests(unittest.TestCase):
    def test_complete_uses_explicit_model_without_discovery(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get") as mock_get,
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        mock_get.assert_not_called()
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "fixed-model")

    def test_complete_discovers_first_model_once(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get") as mock_get,
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(
                    return_value={"data": [{"id": "auto-model"}, {"id": "other"}]}
                ),
            )
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token")
            first = client.complete([{"role": "user", "content": "one"}])
            second = client.complete([{"role": "user", "content": "two"}])

        self.assertEqual(first.content, "ok")
        self.assertEqual(second.content, "ok")
        mock_get.assert_called_once_with(
            "https://example.com/v1/models",
            headers={
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        first_payload = json.loads(mock_post.call_args_list[0].kwargs["data"])
        second_payload = json.loads(mock_post.call_args_list[1].kwargs["data"])
        self.assertEqual(first_payload["model"], "auto-model")
        self.assertEqual(second_payload["model"], "auto-model")

    def test_complete_adds_v1_when_base_is_unversioned(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get") as mock_get,
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": [{"id": "auto-model"}]}),
            )
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com", "token")
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        mock_get.assert_called_once_with(
            "https://example.com/v1/models",
            headers={
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self.assertEqual(
            mock_post.call_args.kwargs["url"]
            if "url" in mock_post.call_args.kwargs
            else mock_post.call_args.args[0],
            "https://example.com/v1/chat/completions",
        )

    def test_complete_sends_bill_to_header_when_configured(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get") as mock_get,
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", bill_to="my-org"
            )
            client.complete([{"role": "user", "content": "hi"}])

        mock_get.assert_not_called()
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["X-HF-Bill-To"], "my-org"
        )

    def test_complete_omits_bill_to_header_when_not_configured(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get"),
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            client.complete([{"role": "user", "content": "hi"}])

        self.assertNotIn("X-HF-Bill-To", mock_post.call_args.kwargs["headers"])

    def test_complete_retries_on_5xx_then_succeeds(self) -> None:
        flaky = Mock(
            status_code=504,
            json=Mock(return_value={}),
            raise_for_status=Mock(side_effect=AssertionError("should not be called")),
        )
        ok = Mock(
            status_code=200,
            json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
            raise_for_status=Mock(),
        )
        with (
            patch("reviewbot.llm_client.time.sleep"),
            patch(
                "reviewbot.llm_client.requests.post", side_effect=[flaky, ok]
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        self.assertEqual(mock_post.call_count, 2)

    def test_complete_retries_on_429_then_succeeds(self) -> None:
        rate_limited = Mock(
            status_code=429,
            ok=False,
            reason="Too Many Requests",
            text="rate limit",
            headers={"Retry-After": "1"},
        )
        ok = Mock(
            status_code=200,
            json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
            raise_for_status=Mock(),
        )
        with (
            patch("reviewbot.llm_client.time.sleep") as mock_sleep,
            patch(
                "reviewbot.llm_client.requests.post", side_effect=[rate_limited, ok]
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        self.assertEqual(mock_post.call_count, 2)
        # Honored the Retry-After: 1 header, capped to the per-wait ceiling.
        mock_sleep.assert_called_once_with(1.0)

    def test_complete_raises_llmresponseerror_with_status_and_body_after_4xx(
        self,
    ) -> None:
        bad = Mock(
            status_code=400,
            ok=False,
            reason="Bad Request",
            text='{"error":{"message":"response_format.type bad"}}',
            headers={},
        )
        with patch("reviewbot.llm_client.requests.post", return_value=bad):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            with self.assertRaises(LLMResponseError) as ctx:
                client.complete([{"role": "user", "content": "hi"}])

        exc = ctx.exception
        self.assertEqual(exc.status_code, 400)
        self.assertEqual(exc.reason, "Bad Request")
        self.assertIn("response_format.type bad", exc.body_preview)
        # Also a real requests.HTTPError so existing handlers still catch it.
        self.assertIsInstance(exc, requests.HTTPError)

    def test_complete_consumes_sse_stream_when_streaming_enabled(self) -> None:
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
            "data: [DONE]",
        ]
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                raise_for_status=Mock(),
                iter_lines=Mock(return_value=iter(sse_lines)),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.completion_tokens, 2)
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertTrue(mock_post.call_args.kwargs["stream"])

    def test_complete_retries_on_stream_interruption_then_succeeds(self) -> None:
        sse_full = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]
        flaky = Mock(
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=_interrupted_iter_lines(
                [sse_full[0]],
                requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            ),
        )
        ok = Mock(
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=Mock(return_value=iter(sse_full)),
        )
        with (
            patch("reviewbot.llm_client.time.sleep"),
            patch(
                "reviewbot.llm_client.requests.post", side_effect=[flaky, ok]
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "hello")
        self.assertEqual(mock_post.call_count, 2)

    def test_complete_raises_after_exhausting_stream_retries(self) -> None:
        flaky = lambda: Mock(  # noqa: E731
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=_interrupted_iter_lines(
                [],
                requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            ),
        )
        with (
            patch("reviewbot.llm_client.time.sleep"),
            patch(
                "reviewbot.llm_client.requests.post",
                side_effect=[flaky(), flaky(), flaky()],
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(mock_post.call_count, 3)

    def test_stream_assembles_tool_calls_across_chunks(self) -> None:
        # Two tool_calls, each split across multiple chunks. The arguments
        # for index 0 (`read_file`) and index 1 (`grep`) arrive piecewise.
        sse_lines = [
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\\"pat"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"function":{"arguments":"h\\":\\"src/main.py\\"}"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":1,"id":"call_b","function":{"name":"grep","arguments":"{\\"pat"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":1,"function":{"arguments":"tern\\":\\"hello\\"}"}}'
            "]}}]}",
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            "data: [DONE]",
        ]
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                raise_for_status=Mock(),
                iter_lines=Mock(return_value=iter(sse_lines)),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].id, "call_a")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(
            json.loads(result.tool_calls[0].arguments), {"path": "src/main.py"}
        )
        self.assertEqual(result.tool_calls[1].id, "call_b")
        self.assertEqual(result.tool_calls[1].name, "grep")
        self.assertEqual(
            json.loads(result.tool_calls[1].arguments), {"pattern": "hello"}
        )

    def test_non_stream_parses_tool_calls(self) -> None:
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_x",
                                            "type": "function",
                                            "function": {
                                                "name": "list_dir",
                                                "arguments": '{"path":"src"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ),
                raise_for_status=Mock(),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=False
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "list_dir"}}],
            )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "list_dir")
        self.assertEqual(json.loads(result.tool_calls[0].arguments), {"path": "src"})

    def test_non_stream_captures_gemini_thought_signature(self) -> None:
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_x",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":"a.py"}',
                                            },
                                            "extra_content": {
                                                "google": {
                                                    "thought_signature": "sig-abc"
                                                }
                                            },
                                        },
                                        {
                                            "id": "call_y",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":"b.py"}',
                                            },
                                        },
                                    ],
                                },
                            }
                        ]
                    }
                ),
                raise_for_status=Mock(),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "gemini-3-flash", stream=False
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        # Only the first parallel call carries the signature; the second is None.
        self.assertEqual(result.tool_calls[0].thought_signature, "sig-abc")
        self.assertIsNone(result.tool_calls[1].thought_signature)

    def test_non_stream_tool_calls_without_signature_are_none(self) -> None:
        # OpenAI / HF Router shape: no extra_content => no signature.
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_x",
                                            "type": "function",
                                            "function": {
                                                "name": "list_dir",
                                                "arguments": '{"path":"src"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ),
                raise_for_status=Mock(),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=False
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "list_dir"}}],
            )

        self.assertIsNone(result.tool_calls[0].thought_signature)

    def test_stream_captures_gemini_thought_signature(self) -> None:
        sse_lines = [
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call_a","function":{"name":"read_file",'
            '"arguments":"{}"},"extra_content":{"google":'
            '{"thought_signature":"sig-stream"}}}'
            "]}}]}",
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            "data: [DONE]",
        ]
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                raise_for_status=Mock(),
                iter_lines=Mock(return_value=iter(sse_lines)),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "gemini-3-flash", stream=True
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].thought_signature, "sig-stream")

    def test_estimate_input_tokens_counts_thought_signature(self) -> None:
        # The replayed assistant turn carries Gemini's signature; the live
        # "in" counter should account for those chars, not silently drop them.
        base = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            }
        ]
        with_sig = [
            {
                **base[0],
                "tool_calls": [
                    {
                        **base[0]["tool_calls"][0],
                        "extra_content": {"google": {"thought_signature": "x" * 400}},
                    }
                ],
            }
        ]
        self.assertEqual(
            ChatCompletionClient._estimate_input_tokens(with_sig, None)
            - ChatCompletionClient._estimate_input_tokens(base, None),
            100,  # 400 signature chars / 4 chars-per-token
        )

    def test_complete_falls_back_when_endpoint_rejects_tools(self) -> None:
        # First call (with tools) returns 400; second call (no tools) succeeds.
        rejected = Mock(
            status_code=400,
            ok=False,
            reason="Bad Request",
            text='{"error":"tools not supported by this model"}',
            raise_for_status=Mock(side_effect=AssertionError("should not be reached")),
        )
        ok = Mock(
            status_code=200,
            ok=True,
            json=Mock(
                return_value={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                }
            ),
            raise_for_status=Mock(),
        )
        with (
            patch("reviewbot.llm_client.time.sleep"),
            patch(
                "reviewbot.llm_client.requests.post", side_effect=[rejected, ok]
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        self.assertEqual(result.content, "ok")
        # Two POSTs: first with tools, second without.
        self.assertEqual(mock_post.call_count, 2)
        first_payload = json.loads(mock_post.call_args_list[0].kwargs["data"])
        second_payload = json.loads(mock_post.call_args_list[1].kwargs["data"])
        self.assertIn("tools", first_payload)
        self.assertNotIn("tools", second_payload)

    def test_complete_does_not_fall_back_on_400_without_tools(self) -> None:
        # If we never sent tools, a 400 should bubble up — not silently retry.
        rejected = Mock(
            status_code=400,
            ok=False,
            reason="Bad Request",
            text='{"error":"bad messages"}',
            raise_for_status=Mock(side_effect=requests.HTTPError("400")),
        )
        with (
            patch("reviewbot.llm_client.time.sleep"),
            patch(
                "reviewbot.llm_client.requests.post", return_value=rejected
            ) as mock_post,
        ):
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            with self.assertRaises(requests.HTTPError):
                client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(mock_post.call_count, 1)

    def test_complete_raises_when_discovery_returns_no_models(self) -> None:
        with (
            patch("reviewbot.llm_client.requests.get") as mock_get,
            patch("reviewbot.llm_client.requests.post") as mock_post,
        ):
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": []}),
            )

            client = ChatCompletionClient("https://example.com/v1", "token")
            with self.assertRaisesRegex(RuntimeError, "No models found"):
                client.complete([{"role": "user", "content": "hi"}])

        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
