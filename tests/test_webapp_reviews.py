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

from reviewbot.reviewer import (
    DraftComment,
    ReviewDraft,
    ReviewEdits,
    effective_draft,
)

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

    # --- effective_draft (single source of truth for what gets posted) ---
    def _sample_draft(self, **overrides):
        kwargs = dict(
            owner="acme",
            repo="widgets",
            number=7,
            head_sha="abc",
            summary="generated summary",
            event="COMMENT",
            comments=[
                DraftComment(
                    id="c1", path="a.py", side="RIGHT", line=10, body="generated body"
                ),
                DraftComment(
                    id="c2", path="b.py", side="RIGHT", line=20, body="discard me"
                ),
            ],
        )
        kwargs.update(overrides)
        return ReviewDraft(**kwargs)

    def test_effective_draft_materializes_edits(self):
        draft = self._sample_draft()
        edits = ReviewEdits(
            summary="edited summary",
            event="REQUEST_CHANGES",
            comment_overrides={"c1": "edited body"},
            discarded_comment_ids={"c2"},
        )

        published = effective_draft(draft, edits, allow_approve=True)

        self.assertEqual(published.summary, "edited summary")
        self.assertEqual(published.event, "REQUEST_CHANGES")
        self.assertEqual(len(published.comments), 1)
        self.assertEqual(published.comments[0].id, "c1")
        self.assertEqual(published.comments[0].body, "edited body")

    def test_effective_draft_downgrades_approve_when_disallowed(self):
        # The generated draft asks to APPROVE and the user leaves it
        # untouched. With allow_approve off, GitHub receives COMMENT, so the
        # effective (= published) draft must reflect COMMENT too.
        draft = self._sample_draft(event="APPROVE")

        published = effective_draft(draft, None, allow_approve=False)

        self.assertEqual(published.event, "COMMENT")

    def test_effective_draft_keeps_approve_when_allowed(self):
        draft = self._sample_draft(event="APPROVE")

        published = effective_draft(draft, None, allow_approve=True)

        self.assertEqual(published.event, "APPROVE")

    # --- _review_changes (audit diff between generated and published) -----
    def test_review_changes_event_and_summary(self):
        generated = self._sample_draft(event="APPROVE")
        published = self._sample_draft(event="COMMENT", summary="edited")
        changes = self.webapp._review_changes(generated, published)
        self.assertTrue(changes["event"])
        self.assertTrue(changes["summary"])
        self.assertEqual(changes["edited_comment_ids"], [])
        self.assertEqual(changes["discarded_comment_ids"], [])

    def test_review_changes_edited_comment(self):
        generated = self._sample_draft()
        published = effective_draft(
            generated,
            ReviewEdits(comment_overrides={"c1": "edited body"}),
            allow_approve=True,
        )
        changes = self.webapp._review_changes(generated, published)
        self.assertEqual(changes["edited_comment_ids"], ["c1"])
        self.assertEqual(changes["discarded_comment_ids"], [])

    def test_review_changes_discarded_comment(self):
        generated = self._sample_draft()
        published = effective_draft(
            generated,
            ReviewEdits(discarded_comment_ids={"c2"}),
            allow_approve=True,
        )
        changes = self.webapp._review_changes(generated, published)
        self.assertEqual(changes["discarded_comment_ids"], ["c2"])
        self.assertEqual(changes["edited_comment_ids"], [])

    def test_review_changes_no_changes(self):
        generated = self._sample_draft()
        published = effective_draft(generated, None, allow_approve=True)
        changes = self.webapp._review_changes(generated, published)
        self.assertFalse(changes["summary"])
        self.assertFalse(changes["event"])
        self.assertEqual(changes["edited_comment_ids"], [])
        self.assertEqual(changes["discarded_comment_ids"], [])

    # --- audit endpoint payload shape ------------------------------------
    def test_review_draft_endpoint_returns_audit(self):
        generated = self._sample_draft(event="APPROVE")
        published = effective_draft(generated, None, allow_approve=False)
        job = self.webapp.Job(
            id="job-audit-1",
            user="dev",
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="@askserge please review",
            llm_provider="hf",
            llm_api_base="",
            llm_model="some-model",
            created_at=0.0,
            source="webhook",
            status="published",
            draft=generated,
            published_draft=published,
            review_edits={
                "summary": None,
                "event": None,
                "comment_overrides": {},
                "discarded_comment_ids": [],
            },
        )
        with self.webapp._jobs_lock:
            self.webapp._jobs[job.id] = job

        client = TestClient(self.webapp.app)
        r = client.get("/reviews/acme/widgets/7/job-audit-1/draft")
        self.assertEqual(r.status_code, 200, r.text)
        audit = r.json()["audit"]
        self.assertEqual(
            set(audit),
            {
                "trigger_comment",
                "generated_draft",
                "published_draft",
                "review_edits",
                "changes",
                "trace",
            },
        )
        self.assertEqual(audit["trigger_comment"], "@askserge please review")
        self.assertEqual(audit["generated_draft"]["event"], "APPROVE")
        self.assertEqual(audit["published_draft"]["event"], "COMMENT")
        # The APPROVE -> COMMENT downgrade must surface as an event change.
        self.assertTrue(audit["changes"]["event"])


if __name__ == "__main__":
    unittest.main()
