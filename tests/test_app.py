import importlib
import os
import sys
import unittest
from unittest.mock import patch


class AppWebhookWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("reviewbot.app", None)

    def _import_app(self):
        env = {
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "token",
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.app")

    def test_direct_pr_review_forces_comment_event(self) -> None:
        app_module = self._import_app()
        req = app_module.ReviewRequest(
            owner="acme",
            repo="widgets",
            number=42,
            trigger_comment_id=123,
            trigger_comment_body="@askserge please review",
            commenter="reviewer",
        )

        with (
            patch.object(app_module, "installation_token", return_value="github-token"),
            patch.object(app_module, "GitHubClient"),
            patch.object(app_module, "run_review") as run_review,
        ):
            app_module._review_worker(1234, req)

        self.assertTrue(run_review.call_args.kwargs["force_comment_event"])


if __name__ == "__main__":
    unittest.main()
