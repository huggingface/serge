"""Tests for the build stamp: the /version endpoint, the ``serge`` field
injected into every JSON body by middleware, the X-Serge-* response headers,
and the kind-aware journal/reviews links (a ``task`` job links to /tasks/,
a review links to /reviews/)."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import reviewbot

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


class WebappVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)
        # git_sha() is lru_cached at package level; pin it to a known SHA for
        # the whole test (the middleware reads it at request time, outside the
        # import-env patch below) and clear the cache before and after.
        reviewbot.git_sha.cache_clear()
        os.environ["SERGE_GIT_SHA"] = "deadbeefcafe9999"
        self.addCleanup(os.environ.pop, "SERGE_GIT_SHA", None)
        self.addCleanup(reviewbot.git_sha.cache_clear)
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "SERGE_GIT_SHA": "deadbeefcafe9999",
        }
        with patch.dict(os.environ, env, clear=True):
            self.webapp = importlib.import_module("reviewbot.webapp")
        self.client = TestClient(self.webapp.app)

    def test_version_endpoint(self):
        r = self.client.get("/version")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["version"], self.webapp.__version__)
        # env SHA is trimmed to 12 chars.
        self.assertEqual(body["commit"], "deadbeefcafe")

    def test_headers_on_every_response(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.headers["x-serge-version"], self.webapp.__version__)
        self.assertEqual(r.headers["x-serge-commit"], "deadbeefcafe")

    def test_serge_stamp_injected_into_json_body(self):
        body = self.client.get("/healthz").json()
        self.assertEqual(
            body["serge"],
            {"version": self.webapp.__version__, "commit": "deadbeefcafe"},
        )

    def test_journal_links_task_to_tasks_route(self):
        self.webapp._store.insert_job(
            id="job-task",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=0,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=1.0,
            status="done",
            source="task",
            kind="task",
            task_spec_json="{}",
        )
        self.webapp._store.insert_job(
            id="job-review",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=2.0,
            status="done",
            source="webhook",
            kind="review",
            task_spec_json=None,
        )
        entries = {
            e["id"]: e for e in self.client.get("/journal/data").json()["entries"]
        }
        self.assertEqual(entries["job-task"]["url"], "/tasks/acme/widgets/job-task")
        self.assertEqual(
            entries["job-review"]["url"], "/reviews/acme/widgets/7/job-review"
        )

    def test_journal_page_renders_rows_without_javascript(self):
        self.webapp._store.insert_job(
            id="job-review",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=2.0,
            status="published",
            source="webhook",
            kind="review",
            task_spec_json=None,
        )
        html = self.client.get("/journal").text
        self.assertIn('id="journal-count">(1)</span>', html)
        self.assertIn("acme/widgets#7", html)
        self.assertIn("/reviews/acme/widgets/7/job-review", html)
        self.assertIn("octocat", html)
        self.assertIn("webhook", html)
        self.assertIn("published", html)


if __name__ == "__main__":
    unittest.main()
