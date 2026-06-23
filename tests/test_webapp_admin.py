"""Tests for the admin view bypass: web-UI reviews are private to their
submitter, except users listed in WEB_ADMIN_USERS, who may follow any
user's review via a shared link. Webhook jobs stay viewable by anyone."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


class WebappAdminViewTests(unittest.TestCase):
    def _build(self, admin_users: str = ""):
        """Import a fresh webapp with DEV_NO_AUTH on (current user == "dev")
        and the given WEB_ADMIN_USERS list, returning a TestClient."""
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "WEB_ADMIN_USERS": admin_users,
        }
        with patch.dict(os.environ, env, clear=True):
            webapp = importlib.import_module("reviewbot.webapp")
        return webapp, TestClient(webapp.app)

    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _insert_web_job(self, webapp, job_id="job-web", user="octocat"):
        webapp._store.insert_job(
            id=job_id,
            user=user,
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=1.0,
            status="done",
            source="web",
            kind="review",
            task_spec_json=None,
        )

    def test_non_admin_cannot_view_other_users_web_review(self):
        webapp, client = self._build(admin_users="")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["detail"], "not_your_job")

    def test_admin_can_view_other_users_web_review(self):
        webapp, client = self._build(admin_users="dev")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], "job-web")

    def test_admin_match_is_case_insensitive(self):
        webapp, client = self._build(admin_users="DEV")
        self._insert_web_job(webapp)
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)

    def test_owner_can_still_view_own_review(self):
        webapp, client = self._build(admin_users="")
        self._insert_web_job(webapp, user="dev")
        r = client.get("/reviews/acme/widgets/7/job-web/info")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
