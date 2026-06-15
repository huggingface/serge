"""Tests for the per-review "max input tokens" override exposed on the new
review form: the payload parser, that an accepted submission threads the
override onto the queued job, and that /llm-options advertises the
deployment default so the UI can show it."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None
    HTTPException = None


class WebappReviewsTests(unittest.TestCase):
    def setUp(self) -> None:
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)
        self.webapp = self._import_webapp()
        self._seed_config(self.webapp)

    def _import_webapp(self):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "LLM_MAX_INPUT_TOKENS": "2000000",
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

    def _submit(self, client, **overrides):
        payload = {
            "pr": "acme/widgets#7",
            "comment": "@askserge please review",
            "llm_provider": "hf",
            "llm_model": "some-model",
        }
        payload.update(overrides)
        return client.post(
            "/reviews", json=payload, headers={"Origin": "http://testserver"}
        )

    # --- parser -------------------------------------------------------
    def test_parse_blank_and_missing_yield_none(self):
        parse = self.webapp._parse_max_input_tokens
        self.assertIsNone(parse({}))
        self.assertIsNone(parse({"llm_max_input_tokens": ""}))
        self.assertIsNone(parse({"llm_max_input_tokens": "   "}))
        self.assertIsNone(parse({"llm_max_input_tokens": None}))

    def test_parse_accepts_zero_and_positive(self):
        parse = self.webapp._parse_max_input_tokens
        self.assertEqual(parse({"llm_max_input_tokens": "0"}), 0)
        self.assertEqual(parse({"llm_max_input_tokens": 500000}), 500000)
        self.assertEqual(parse({"llm_max_input_tokens": "500000"}), 500000)

    def test_parse_rejects_negative_and_garbage(self):
        parse = self.webapp._parse_max_input_tokens
        for bad in ("-1", -5, "lots", "1.5"):
            with self.assertRaises(HTTPException) as ctx:
                parse({"llm_max_input_tokens": bad})
            self.assertEqual(ctx.exception.status_code, 400)

    # --- submission threading ----------------------------------------
    def test_submit_threads_override_onto_job(self):
        client = TestClient(self.webapp.app)
        with patch.object(self.webapp, "_run_review_worker") as worker:
            r = self._submit(client, llm_max_input_tokens="750000")
        self.assertEqual(r.status_code, 200, r.text)
        job = worker.call_args.args[0]
        self.assertEqual(job.llm_max_input_tokens, 750000)

    def test_submit_without_override_leaves_job_default(self):
        client = TestClient(self.webapp.app)
        with patch.object(self.webapp, "_run_review_worker") as worker:
            r = self._submit(client)
        self.assertEqual(r.status_code, 200, r.text)
        job = worker.call_args.args[0]
        self.assertIsNone(job.llm_max_input_tokens)

    def test_submit_rejects_bad_override(self):
        client = TestClient(self.webapp.app)
        with patch.object(self.webapp, "_run_review_worker"):
            r = self._submit(client, llm_max_input_tokens="-3")
        self.assertEqual(r.status_code, 400)

    # --- options ------------------------------------------------------
    def test_llm_options_advertises_default(self):
        client = TestClient(self.webapp.app)
        r = client.get("/llm-options")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["default_max_input_tokens"], 2000000)


if __name__ == "__main__":
    unittest.main()
