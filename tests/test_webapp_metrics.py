"""Tests for ``GET /metrics``.

The endpoint has to be scrapeable without a session (Prometheus holds no
cookie), must not carry review content, and must answer even when the store is
empty or a row is unreadable — a scrape that 500s loses the history quietly.
"""

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
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


SESSION = {
    "turns": 46,
    "tool_calls": 48,
    "prompt_tokens": 2_111_885,
    "completion_tokens": 9_000,
    "seconds": 1234.5,
    "stop_reason": "input_token_cap",
    "repeats": 2,
    "distinct_paths": 12,
    "path_revisits": 14,
    "validation_retries": 0,
    "truncation_retries": 0,
    "rounds": 1,
}


class WebappMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)
        env = {
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "WEB_JOB_RETENTION": "25",
            # Real auth, not DEV_NO_AUTH: the point of the first test is that a
            # scraper with no session still gets the body.
            "GITHUB_OAUTH_CLIENT_ID": "oauth-client",
            "GITHUB_OAUTH_CLIENT_SECRET": "oauth-secret",
            "WEB_SESSION_SECRET": "session-secret",
            "WEB_ALLOWED_ORG": "huggingface",
        }
        with patch.dict(os.environ, env, clear=True):
            self.webapp = importlib.import_module("reviewbot.webapp")
        self.client = TestClient(self.webapp.app)

    def _finished_job(self, job_id: str, *, session=SESSION, **result) -> None:
        store = self.webapp._store
        store.insert_job(
            id=job_id,
            user="alice",
            target_owner="huggingface",
            target_repo="transformers",
            target_number=48322,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base="https://router.huggingface.co",
            llm_model="kimi-k2",
            created_at=1.0,
            status="running",
            kind="task",
        )
        if result:
            store.save_task_result(job_id, json.dumps(result))
        store.save_terminal(
            job_id,
            status="published",
            error=None,
            raw_llm_output=None,
            draft=None,
            history=[],
            session=session,
        )

    def test_scrapeable_without_a_session_cookie(self) -> None:
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        self.assertIn("version=0.0.4", r.headers["content-type"])

    def test_a_json_stamping_middleware_does_not_rewrite_the_body(self) -> None:
        """The build stamp is merged into JSON bodies; a text exposition must
        come back byte-for-byte or Prometheus rejects it."""
        r = self.client.get("/metrics")
        self.assertNotIn('{"serge"', r.text)
        for line in r.text.splitlines():
            self.assertTrue(line.startswith("#") or " " in line, line)

    def test_empty_store_is_a_valid_scrape(self) -> None:
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("serge_job_retention 25", r.text)
        self.assertNotIn("serge_job_turns{", r.text)

    def test_a_finished_job_is_exported(self) -> None:
        self._finished_job("job1", pr_number=48322, verify_verdict="fixed")
        body = self.client.get("/metrics").text
        self.assertIn('job_id="job1"', body)
        self.assertIn('stop_reason="input_token_cap"', body)
        self.assertIn('pr="48322"', body)
        self.assertIn('serge_job_input_tokens{job_id="job1"} 2111885', body)
        self.assertIn('serge_job_path_revisits{job_id="job1"} 14', body)

    def test_a_running_job_is_not_exported(self) -> None:
        self.webapp._store.insert_job(
            id="live",
            user="alice",
            target_owner="huggingface",
            target_repo="transformers",
            target_number=1,
            trigger_comment="x",
            llm_provider="hf",
            llm_api_base="b",
            llm_model="m",
            created_at=1.0,
            status="running",
        )
        self.assertNotIn('job_id="live"', self.client.get("/metrics").text)

    def test_no_review_content_reaches_the_exposition(self) -> None:
        self._finished_job("job1")
        body = self.client.get("/metrics").text
        for leaked in ("trigger_comment", "@askserge", "alice"):
            self.assertNotIn(leaked, body)

    def test_a_broken_store_answers_instead_of_failing_the_scrape(self) -> None:
        with patch.object(
            self.webapp._store, "list_sessions", side_effect=RuntimeError("db gone")
        ):
            r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("serge_job_retention 25", r.text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
