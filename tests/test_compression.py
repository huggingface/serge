import json
import os
import unittest
from unittest.mock import Mock, patch

from reviewbot.compression import CompressionConfig, MessageCompressor
from reviewbot.llm_client import ChatCompletionClient


class _FakeCompressConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _fake_result(messages, *, before=100, after=40):
    return Mock(
        messages=messages,
        tokens_before=before,
        tokens_after=after,
        tokens_saved=before - after,
        compression_ratio=(before - after) / before,
    )


class CompressionConfigEnvTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = CompressionConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.compress_user_messages)
        self.assertEqual(cfg.protect_recent, 4)
        self.assertIsNone(cfg.target_ratio)

    def test_reads_overrides(self) -> None:
        env = {
            "HEADROOM_COMPRESS": "true",
            "HEADROOM_COMPRESS_USER_MESSAGES": "1",
            "HEADROOM_PROTECT_RECENT": "2",
            "HEADROOM_TARGET_RATIO": "0.5",
            "HEADROOM_MIN_TOKENS": "100",
            "HEADROOM_KOMPRESS_MODEL": "disabled",
            "HEADROOM_MODEL_LIMIT": "128000",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = CompressionConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.compress_user_messages)
        self.assertEqual(cfg.protect_recent, 2)
        self.assertEqual(cfg.target_ratio, 0.5)
        self.assertEqual(cfg.min_tokens_to_compress, 100)
        self.assertEqual(cfg.kompress_model, "disabled")
        self.assertEqual(cfg.model_limit, 128000)

    def test_bad_ratio_falls_back_to_default(self) -> None:
        with patch.dict(os.environ, {"HEADROOM_TARGET_RATIO": "nope"}, clear=True):
            cfg = CompressionConfig.from_env()
        self.assertIsNone(cfg.target_ratio)


class MessageCompressorTests(unittest.TestCase):
    def test_disabled_is_passthrough_and_never_loads(self) -> None:
        c = MessageCompressor(CompressionConfig(enabled=False))
        msgs = [{"role": "user", "content": "hi"}]
        with patch.object(c, "_load") as load:
            self.assertIs(c.compress(msgs, model="gpt-4o"), msgs)
            load.assert_not_called()

    def test_empty_messages_passthrough(self) -> None:
        c = MessageCompressor(CompressionConfig(enabled=True))
        self.assertEqual(c.compress([], model="gpt-4o"), [])

    def test_unavailable_package_passes_through_and_warns_once(self) -> None:
        c = MessageCompressor(CompressionConfig(enabled=True))
        msgs = [{"role": "tool", "content": "x" * 500}]
        # Force the import to fail.
        with patch.dict("sys.modules", {"headroom": None}):
            with self.assertLogs("reviewbot.compression", level="WARNING") as logs:
                self.assertIs(c.compress(msgs, model="gpt-4o"), msgs)
                # Second call must not emit another warning.
                self.assertIs(c.compress(msgs, model="gpt-4o"), msgs)
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(c._unavailable)

    def test_compresses_when_available(self) -> None:
        c = MessageCompressor(
            CompressionConfig(enabled=True, protect_recent=2, target_ratio=0.5)
        )
        msgs = [{"role": "tool", "content": "noisy " * 200}]
        compressed = [{"role": "tool", "content": "noisy"}]
        compress_fn = Mock(return_value=_fake_result(compressed))
        c._headroom = (compress_fn, _FakeCompressConfig)

        out = c.compress(msgs, model="claude-sonnet-4-5")

        self.assertIs(out, compressed)
        # model + config forwarded
        _, kwargs = compress_fn.call_args
        self.assertEqual(kwargs["model"], "claude-sonnet-4-5")
        cfg = kwargs["config"]
        self.assertEqual(cfg.kwargs["protect_recent"], 2)
        self.assertEqual(cfg.kwargs["target_ratio"], 0.5)

    def test_default_model_when_none(self) -> None:
        c = MessageCompressor(CompressionConfig(enabled=True))
        msgs = [{"role": "tool", "content": "x"}]
        compress_fn = Mock(return_value=_fake_result(msgs))
        c._headroom = (compress_fn, _FakeCompressConfig)

        c.compress(msgs, model=None)

        self.assertEqual(compress_fn.call_args.kwargs["model"], "gpt-4o")

    def test_compression_failure_returns_original(self) -> None:
        c = MessageCompressor(CompressionConfig(enabled=True))
        msgs = [{"role": "tool", "content": "x"}]
        compress_fn = Mock(side_effect=RuntimeError("boom"))
        c._headroom = (compress_fn, _FakeCompressConfig)

        with self.assertLogs("reviewbot.compression", level="WARNING"):
            out = c.compress(msgs, model="gpt-4o")

        self.assertIs(out, msgs)


class ClientIntegrationTests(unittest.TestCase):
    def test_client_compresses_messages_before_post(self) -> None:
        compressor = MessageCompressor(CompressionConfig(enabled=True))
        compressed = [{"role": "user", "content": "short"}]
        compress_fn = Mock(return_value=_fake_result(compressed))
        compressor._headroom = (compress_fn, _FakeCompressConfig)

        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                ok=True,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
            )
            client = ChatCompletionClient(
                "https://example.com/v1",
                "token",
                "fixed-model",
                compressor=compressor,
            )
            client.complete([{"role": "user", "content": "looong " * 100}])

        # Resolved model is forwarded to the compressor.
        self.assertEqual(compress_fn.call_args.kwargs["model"], "fixed-model")
        # The compressed messages are what actually get POSTed.
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertEqual(payload["messages"], compressed)

    def test_client_without_compressor_sends_original(self) -> None:
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                ok=True,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model"
            )
            original = [{"role": "user", "content": "hi"}]
            client.complete(original)

        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertEqual(payload["messages"], original)


if __name__ == "__main__":
    unittest.main()
