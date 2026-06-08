import hashlib
import hmac
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional web extra
    TestClient = None


def _signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _inline_payload() -> dict:
    return {
        "action": "created",
        "installation": {"id": 1234},
        "comment": {
            "id": 3322017554,
            "body": "@askserge can you check this?",
            "author_association": "MEMBER",
            "user": {"login": "reviewer"},
            "path": "src/foo.py",
            "line": 68,
            "side": "RIGHT",
            "diff_hunk": "@@ -65,4 +65,4 @@",
        },
        "pull_request": {"number": 42, "state": "open"},
        "repository": {"full_name": "acme/widgets"},
    }


class WebappWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _import_webapp(self):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": " llm-token\n",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def test_webhook_rejects_bad_signature(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        body = json.dumps(_inline_payload()).encode()

        response = client.post(
            "/webhook",
            content=body,
            headers={
                "X-GitHub-Event": "pull_request_review_comment",
                "X-Hub-Signature-256": "sha256=bad",
            },
        )

        self.assertEqual(response.status_code, 401)

    def test_webhook_accepts_inline_comment_and_dispatches_worker(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        body = json.dumps(_inline_payload()).encode()

        with (
            patch.object(
                webapp._WEBHOOK_REVIEW_POOL,
                "submit",
                side_effect=lambda fn, *args: fn(*args),
            ) as submit,
            patch.object(webapp, "installation_token", return_value="github-token"),
            patch.object(webapp, "GitHubClient") as github_client,
            patch.object(webapp, "run_followup") as run_followup,
        ):
            response = client.post(
                "/webhook",
                content=body,
                headers={
                    "X-GitHub-Event": "pull_request_review_comment",
                    "X-Hub-Signature-256": _signature("webhook-secret", body),
                },
            )

        self.assertEqual(response.status_code, 202)
        submit.assert_called_once()
        run_followup.assert_called_once()
        cfg = run_followup.call_args.args[0]
        gh = run_followup.call_args.args[1]
        req = run_followup.call_args.args[2]
        self.assertEqual(cfg.llm_api_key, "llm-token")
        self.assertIs(gh, github_client.return_value)
        self.assertEqual(req.owner, "acme")
        self.assertEqual(req.repo, "widgets")
        self.assertEqual(req.number, 42)
        self.assertIsNotNone(req.inline)

    def test_webhook_worker_uses_db_provider_config_for_repo(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")
        webapp = self._import_webapp()
        # A provider_config matching the repo should drive the worker's
        # LLM credentials — repo-only match, no logged-in user needed.
        webapp._store.insert_provider_config(
            id="cfg-1",
            provider="anthropic",
            api_key="db-anthropic-key",
            api_base=None,
            default_model="claude-opus-4-6",
            repo_pattern="acme/widgets",
            allowed_users=[],
            allowed_orgs=["acme-org"],
            created_by="admin",
        )
        client = TestClient(webapp.app)
        body = json.dumps(_inline_payload()).encode()

        with (
            patch.object(
                webapp._WEBHOOK_REVIEW_POOL,
                "submit",
                side_effect=lambda fn, *args: fn(*args),
            ),
            patch.object(webapp, "installation_token", return_value="github-token"),
            patch.object(webapp, "GitHubClient"),
            patch.object(webapp, "run_followup") as run_followup,
        ):
            response = client.post(
                "/webhook",
                content=body,
                headers={
                    "X-GitHub-Event": "pull_request_review_comment",
                    "X-Hub-Signature-256": _signature("webhook-secret", body),
                },
            )

        self.assertEqual(response.status_code, 202)
        run_followup.assert_called_once()
        cfg = run_followup.call_args.args[0]
        # DB row wins over the global LLM_API_KEY / base / model.
        self.assertEqual(cfg.llm_api_key, "db-anthropic-key")
        self.assertEqual(cfg.llm_api_base, "https://api.anthropic.com")
        self.assertEqual(cfg.llm_model, "claude-opus-4-6")


class HfModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _import_webapp(self):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def test_parser_keeps_only_live_tool_capable_models_sorted(self) -> None:
        webapp = self._import_webapp()
        payload = {
            "data": [
                {
                    "id": "zeta/Tooler",
                    "providers": [{"status": "live", "supports_tools": True}],
                },
                {
                    "id": "alpha/Tooler",
                    "providers": [
                        {"status": "error", "supports_tools": True},
                        {"status": "live", "supports_tools": True},
                    ],
                },
                {
                    # No tool support anywhere — dropped.
                    "id": "beta/NoTools",
                    "providers": [{"status": "live", "supports_tools": False}],
                },
                {
                    # Tool support but not live — dropped.
                    "id": "gamma/Staging",
                    "providers": [{"status": "staging", "supports_tools": True}],
                },
                {"id": "", "providers": [{"status": "live", "supports_tools": True}]},
            ]
        }
        self.assertEqual(
            webapp._tool_capable_hf_models(payload),
            ["alpha/Tooler", "zeta/Tooler"],
        )

    def test_parser_tolerates_garbage(self) -> None:
        webapp = self._import_webapp()
        self.assertEqual(webapp._tool_capable_hf_models({}), [])
        self.assertEqual(webapp._tool_capable_hf_models({"data": "nope"}), [])
        self.assertEqual(webapp._tool_capable_hf_models("nope"), [])

    def test_endpoint_returns_cached_models(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi is not installed")
        webapp = self._import_webapp()
        # Seed the in-process cache so the endpoint serves it without
        # touching the network.
        webapp._hf_models_cache["models"] = ["Qwen/Qwen3", "meta-llama/Llama-4"]
        webapp._hf_models_cache["fetched_at"] = webapp.time.monotonic()
        client = TestClient(webapp.app)
        response = client.get("/llm-options/hf-models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["models"], ["Qwen/Qwen3", "meta-llama/Llama-4"]
        )

    def test_empty_fetch_result_is_cached(self) -> None:
        import asyncio

        webapp = self._import_webapp()
        # Start from "never fetched" so the first call hits the network.
        webapp._hf_models_cache["models"] = []
        webapp._hf_models_cache["fetched_at"] = 0.0

        get_calls: list[str] = []

        class _Resp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                # A valid response with no tool-capable models.
                return {"data": []}

        class _Client:
            def __init__(self, *a, **k) -> None:
                pass

            async def __aenter__(self) -> "_Client":
                return self

            async def __aexit__(self, *a) -> bool:
                return False

            async def get(self, url: str) -> "_Resp":
                get_calls.append(url)
                return _Resp()

        with patch.object(webapp.httpx, "AsyncClient", _Client):
            first = asyncio.run(webapp._fetch_hf_router_models())
            second = asyncio.run(webapp._fetch_hf_router_models())

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        # The empty-but-valid result is cached, so the second call is served
        # from cache rather than re-hitting the router.
        self.assertEqual(len(get_calls), 1)


if __name__ == "__main__":
    unittest.main()
