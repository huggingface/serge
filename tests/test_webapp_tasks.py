"""Tests for the POST /tasks route: the feature flag, OIDC auth, the
repository-claim authorization, and the write opt-in gate. The OIDC
verification and the task worker are stubbed so no network/LLM is touched —
we assert the route's accept/reject behavior and that an accepted task is
queued."""

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


class _Claims:
    def __init__(self, repository="acme/widgets", actor="octocat"):
        self.repository = repository
        self.actor = actor
        self.workflow_ref = "acme/widgets/.github/workflows/fix.yml@refs/heads/main"
        self.raw = {}


class WebappTasksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _import_webapp(self, *, task_api_enabled=True):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "TASK_API_ENABLED": "1" if task_api_enabled else "0",
            "TASK_OIDC_AUDIENCE": "serge",
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def _seed_write_config(self, webapp, *, task_write_enabled=True):
        webapp._store.insert_provider_config(
            id="c1",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model="some-model",
            repo_pattern="acme/widgets",
            allowed_users=["octocat"],
            allowed_orgs=[],
            created_by="admin",
            task_write_enabled=task_write_enabled,
        )

    def test_disabled_returns_404(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(task_api_enabled=False)
        client = TestClient(webapp.app)
        r = client.post("/tasks", json={"instruction": "x"})
        self.assertEqual(r.status_code, 404)

    def test_missing_bearer_returns_401(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        r = client.post("/tasks", json={"instruction": "x"})
        self.assertEqual(r.status_code, 401)

    def test_bad_oidc_returns_401(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        with patch.object(webapp, "verify_token", side_effect=webapp.OIDCError("bad")):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "x"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 401)

    def test_no_write_config_returns_403(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        # config exists but write not enabled
        self._seed_write_config(webapp, task_write_enabled=False)
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "fix tests", "context": "log"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 403)

    def test_repo_claim_mismatch_returns_403(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_write_config(webapp)
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "x", "repo": "evil/other"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 403)

    def test_accepted_task_is_queued(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_write_config(webapp)
        submitted = {}

        def fake_submit(fn, job, worker_cfg, req):
            submitted["job"] = job
            submitted["req"] = req

        with (
            patch.object(webapp, "verify_token", return_value=_Claims()),
            patch.object(webapp._TASK_POOL, "submit", side_effect=fake_submit),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "fix the failing tests", "context": "trace"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 202)
        body = r.json()
        self.assertEqual(body["repo"], "acme/widgets")
        self.assertEqual(body["mode"], "new_pr")
        self.assertTrue(body["url"].startswith("/tasks/acme/widgets/"))
        self.assertEqual(submitted["job"].kind, "task")
        self.assertEqual(submitted["req"].instruction, "fix the failing tests")
        # Persisted with task kind.
        row = webapp._store.load(body["id"])
        self.assertEqual(row["kind"], "task")


if __name__ == "__main__":
    unittest.main()
