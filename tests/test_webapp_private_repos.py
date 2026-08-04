"""Tests for the private-repo gate: public repos stay open to every
signed-in user, while a private repo's review/task is only submittable and
readable by a collaborator (checked App-side, fail-closed)."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests

from reviewbot.github_auth import AppNotInstalledError

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


class _FakeGH:
    """Stands in for the installation-scoped GitHubClient, recording which
    endpoints the gate actually touched."""

    def __init__(self, private: bool, collaborator=None, error: bool = False):
        self.private = private
        self.collaborator = collaborator
        self.error = error
        self.calls: list[str] = []

    def get_repo(self, owner, repo):
        self.calls.append("get_repo")
        if self.error:
            raise requests.ConnectionError("github unreachable")
        return {"private": self.private, "full_name": f"{owner}/{repo}"}

    def is_collaborator(self, owner, repo, username):
        self.calls.append("is_collaborator")
        return self.collaborator


class PrivateRepoGateTests(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.webapp = self._import_webapp()
        self._seed_config(self.webapp)

    def _import_webapp(self, dev_no_auth: str = "0"):
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "DEV_NO_AUTH": dev_no_auth,
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "GITHUB_OAUTH_CLIENT_ID": "oauth-client",
            "GITHUB_OAUTH_CLIENT_SECRET": "oauth-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_SESSION_SECRET": "session-secret",
            "WEB_ALLOWED_USERS": "dev",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def _seed_config(self, webapp) -> None:
        webapp._store.insert_provider_config(
            id="c1",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model="some-model",
            repo_pattern="acme/widgets",
            allowed_users=["dev"],
            allowed_orgs=[],
            created_by="admin",
        )

    def _client(self, user: str = "dev"):
        client = TestClient(self.webapp.app)
        client.cookies.set(
            self.webapp._SESSION_COOKIE,
            self.webapp._serializer.dumps({"user": user, "orgs": []}),
        )
        return client

    def _submit(self, client):
        return client.post(
            "/reviews",
            json={"pr": "acme/widgets#7", "llm_provider": "hf"},
            headers={"Origin": "http://testserver"},
        )

    def _insert_job(self, *, job_id="job-1", user="someone-else", source, kind):
        self.webapp._store.insert_job(
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
            source=source,
            kind=kind,
            task_spec_json=None,
        )

    # --- the check itself ---------------------------------------------
    def test_public_repo_skips_the_collaborator_check(self):
        gh = _FakeGH(private=False)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            self.assertTrue(self.webapp._user_may_access_repo("dev", "acme", "widgets"))
        self.assertEqual(gh.calls, ["get_repo"])

    def test_private_repo_allows_a_collaborator(self):
        gh = _FakeGH(private=True, collaborator=True)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            self.assertTrue(self.webapp._user_may_access_repo("dev", "acme", "widgets"))
        self.assertEqual(gh.calls, ["get_repo", "is_collaborator"])

    def test_private_repo_denies_a_non_collaborator(self):
        gh = _FakeGH(private=True, collaborator=False)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            self.assertFalse(
                self.webapp._user_may_access_repo("dev", "acme", "widgets")
            )

    def test_inconclusive_collaborator_answer_denies(self):
        # is_collaborator returns None when the App lacks the permission.
        gh = _FakeGH(private=True, collaborator=None)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            self.assertFalse(
                self.webapp._user_may_access_repo("dev", "acme", "widgets")
            )

    def test_github_error_denies_and_is_not_cached(self):
        gh = _FakeGH(private=True, error=True)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            self.assertFalse(
                self.webapp._user_may_access_repo("dev", "acme", "widgets")
            )
        ok = _FakeGH(private=True, collaborator=True)
        with patch.object(self.webapp, "_installation_client", return_value=ok):
            self.assertTrue(self.webapp._user_may_access_repo("dev", "acme", "widgets"))

    def test_verdict_is_cached_per_user(self):
        gh = _FakeGH(private=True, collaborator=True)
        with patch.object(self.webapp, "_installation_client", return_value=gh) as mk:
            self.webapp._user_may_access_repo("dev", "acme", "widgets")
            self.webapp._user_may_access_repo("dev", "acme", "widgets")
            self.assertEqual(mk.call_count, 1)
            # A different user is a different verdict, so it re-checks.
            self.webapp._user_may_access_repo("other", "acme", "widgets")
            self.assertEqual(mk.call_count, 2)

    def test_dev_no_auth_skips_the_gate(self):
        webapp = self._import_webapp(dev_no_auth="1")
        with patch.object(webapp, "_installation_client") as mk:
            self.assertTrue(webapp._user_may_access_repo("dev", "acme", "widgets"))
        mk.assert_not_called()

    # --- submit path ---------------------------------------------------
    def test_submit_rejects_private_repo_without_access(self):
        gh = _FakeGH(private=True, collaborator=False)
        with (
            patch.object(self.webapp, "_installation_client", return_value=gh),
            patch.object(self.webapp, "_run_review_worker") as worker,
        ):
            r = self._submit(self._client())
        self.assertEqual(r.status_code, 403)
        self.assertIn("no_repo_access", r.json()["detail"])
        worker.assert_not_called()

    def test_submit_accepts_private_repo_for_a_collaborator(self):
        gh = _FakeGH(private=True, collaborator=True)
        with (
            patch.object(self.webapp, "_installation_client", return_value=gh),
            patch.object(self.webapp, "_run_review_worker") as worker,
        ):
            r = self._submit(self._client())
        self.assertEqual(r.status_code, 200, r.text)
        worker.assert_called_once()

    def test_submit_surfaces_missing_installation(self):
        with (
            patch.object(
                self.webapp,
                "_installation_client",
                side_effect=AppNotInstalledError("acme", "widgets"),
            ),
            patch.object(self.webapp, "_run_review_worker") as worker,
        ):
            r = self._submit(self._client())
        self.assertEqual(r.status_code, 400)
        self.assertIn("not installed", r.json()["detail"])
        worker.assert_not_called()

    # --- view path -----------------------------------------------------
    def test_webhook_review_on_private_repo_is_not_world_readable(self):
        self._insert_job(job_id="job-hook", source="webhook", kind="review")
        gh = _FakeGH(private=True, collaborator=False)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            r = self._client().get("/reviews/acme/widgets/7/job-hook/info")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["detail"], "no_repo_access")

    def test_webhook_review_on_public_repo_stays_world_readable(self):
        self._insert_job(job_id="job-hook", source="webhook", kind="review")
        gh = _FakeGH(private=False)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            r = self._client().get("/reviews/acme/widgets/7/job-hook/info")
        self.assertEqual(r.status_code, 200, r.text)

    def test_task_on_private_repo_is_not_world_readable(self):
        self._insert_job(job_id="job-task", source="task", kind="task")
        gh = _FakeGH(private=True, collaborator=False)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            r = self._client().get("/tasks/acme/widgets/job-task/info")
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["detail"], "no_repo_access")

    def test_task_on_private_repo_is_readable_by_a_collaborator(self):
        self._insert_job(job_id="job-task", source="task", kind="task")
        gh = _FakeGH(private=True, collaborator=True)
        with patch.object(self.webapp, "_installation_client", return_value=gh):
            r = self._client().get("/tasks/acme/widgets/job-task/info")
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
