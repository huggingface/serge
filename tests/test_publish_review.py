"""Tests for the prepare/publish split. We don't go through the LLM —
we hand-craft a ReviewDraft and exercise publish_review's body-format
and edit-application rules."""

import unittest
from unittest.mock import MagicMock, patch

from reviewbot.config import Config
from reviewbot.reviewer import (
    DraftComment,
    ReviewDraft,
    ReviewEdits,
    ReviewRequest,
    publish_review,
    run_review,
)


def _make_cfg(**overrides) -> Config:
    """Build a Config that's minimal but valid for publish_review."""
    base = dict(
        github_app_id=None,
        github_private_key=None,
        github_webhook_secret=None,
        llm_api_base="https://example.com/v1",
        llm_api_key="x",
        llm_model=None,
        llm_bill_to=None,
        llm_max_tokens=4096,
        llm_stream=False,
        mention_trigger="@askserge",
        review_event="COMMENT",
        max_diff_chars=200000,
        review_rules_path=".ai/review-rules.md",
        helper_tools_path=".ai/review-tools.json",
        default_review_rules="",
        allow_approve=False,
        persona_header="🤗 **Serge** says:",
        context_script_path=".ai/context-script",
        context_script_timeout=30,
        repo_checkout_path="",
        tool_max_iterations=8,
        llm_max_input_tokens=2_000_000,
    )
    base.update(overrides)
    return Config(**base)


def _make_draft(**overrides) -> ReviewDraft:
    base = dict(
        owner="acme",
        repo="widgets",
        number=42,
        head_sha="deadbeef",
        summary="LGTM overall.",
        event="COMMENT",
        comments=[
            DraftComment(
                id="c0", path="a.py", side="RIGHT", line=10, body="nit: rename"
            ),
            DraftComment(
                id="c1", path="b.py", side="RIGHT", line=20, body="this is wrong"
            ),
        ],
        rejected_count=0,
        metrics_line="2 LLM turns · 0 tool calls · 3.4s",
    )
    base.update(overrides)
    return ReviewDraft(**base)


class PublishReviewTests(unittest.TestCase):
    def test_publishes_all_comments_unedited(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        gh.create_review.assert_called_once()
        kwargs = gh.create_review.call_args.kwargs
        self.assertEqual(kwargs["commit_id"], "deadbeef")
        self.assertEqual(kwargs["event"], "COMMENT")
        self.assertEqual(len(kwargs["comments"]), 2)
        self.assertIn("🤗 **Serge** says:", kwargs["body"])
        self.assertIn("LGTM overall.", kwargs["body"])
        self.assertIn("2 LLM turns", kwargs["body"])

    def test_summary_override_replaces_summary(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft, edits=ReviewEdits(summary="Edited summary."))
        body = gh.create_review.call_args.kwargs["body"]
        self.assertIn("Edited summary.", body)
        self.assertNotIn("LGTM overall.", body)

    def test_comment_override_replaces_body(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        edits = ReviewEdits(comment_overrides={"c0": "rephrased comment"})
        publish_review(cfg, gh, draft, edits=edits)
        comments = gh.create_review.call_args.kwargs["comments"]
        bodies = [c["body"] for c in comments]
        self.assertIn("rephrased comment", bodies)
        self.assertIn("this is wrong", bodies)

    def test_discarded_comment_is_dropped(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        edits = ReviewEdits(discarded_comment_ids={"c1"})
        publish_review(cfg, gh, draft, edits=edits)
        comments = gh.create_review.call_args.kwargs["comments"]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]["path"], "a.py")

    def test_empty_override_drops_the_comment(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        edits = ReviewEdits(comment_overrides={"c0": "   "})
        publish_review(cfg, gh, draft, edits=edits)
        comments = gh.create_review.call_args.kwargs["comments"]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]["path"], "b.py")

    def test_event_override_changes_event(self) -> None:
        cfg = _make_cfg(allow_approve=True)
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft, edits=ReviewEdits(event="REQUEST_CHANGES"))
        self.assertEqual(gh.create_review.call_args.kwargs["event"], "REQUEST_CHANGES")

    def test_approve_downgrades_when_disallowed(self) -> None:
        cfg = _make_cfg(allow_approve=False)
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft, edits=ReviewEdits(event="APPROVE"))
        self.assertEqual(gh.create_review.call_args.kwargs["event"], "COMMENT")

    def test_approve_passes_through_when_allowed(self) -> None:
        cfg = _make_cfg(allow_approve=True)
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft, edits=ReviewEdits(event="APPROVE"))
        self.assertEqual(gh.create_review.call_args.kwargs["event"], "APPROVE")

    def test_invalid_event_falls_back_to_draft_event(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(event="REQUEST_CHANGES")
        gh = MagicMock()
        publish_review(cfg, gh, draft, edits=ReviewEdits(event="GARBAGE"))
        self.assertEqual(gh.create_review.call_args.kwargs["event"], "REQUEST_CHANGES")

    def test_rejected_count_note_in_body(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(rejected_count=3)
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertIn("3 suggested inline comment(s) were dropped", body)

    def test_model_in_footer(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(model="acme/cool-model")
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertIn("model: `acme/cool-model`", body)
        # Model and metrics share one footer line.
        self.assertIn("model: `acme/cool-model` · 2 LLM turns", body)

    def test_no_model_footer_when_unset(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(model=None)
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertNotIn("model:", body)
        # Metrics line still renders on its own.
        self.assertIn("2 LLM turns", body)

    def test_staging_note_in_body(self) -> None:
        cfg = _make_cfg(is_staging=True)
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertIn("posted from a staging deployment", body)

    def test_no_staging_note_by_default(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertNotIn("staging", body)

    def test_no_persona_header_when_disabled(self) -> None:
        cfg = _make_cfg(persona_header="")
        draft = _make_draft()
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertNotIn("Serge", body.split("LGTM overall")[0])

    def test_empty_summary_renders_fallback(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(summary="")
        gh = MagicMock()
        publish_review(cfg, gh, draft)
        body = gh.create_review.call_args.kwargs["body"]
        self.assertIn("(no overall summary provided)", body)

    def test_run_review_can_force_comment_event_for_direct_publish(self) -> None:
        cfg = _make_cfg()
        draft = _make_draft(event="REQUEST_CHANGES")
        gh = MagicMock()
        req = ReviewRequest(
            owner="acme",
            repo="widgets",
            number=42,
            trigger_comment_id=123,
            trigger_comment_body="@askserge please review",
            commenter="reviewer",
        )

        with patch("reviewbot.reviewer.prepare_review", return_value=draft):
            run_review(cfg, gh, req, force_comment_event=True)

        self.assertEqual(gh.create_review.call_args.kwargs["event"], "COMMENT")


if __name__ == "__main__":
    unittest.main()
