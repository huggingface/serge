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


if __name__ == "__main__":
    unittest.main()
