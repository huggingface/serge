import unittest

from reviewbot.triggers import build_review_request


class TriggerTests(unittest.TestCase):
    def test_build_review_request_accepts_serge_trigger(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "@askserge please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        self.assertIsNotNone(req)
        assert req is not None
        self.assertEqual(req.owner, "acme")
        self.assertEqual(req.repo, "project")
        self.assertEqual(req.number, 7)
        self.assertEqual(req.trigger_comment_id, 123)

    def test_review_comment_event_captures_inline_context(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "id": 4242,
                "body": "@askserge could you help me understand this line?",
                "author_association": "COLLABORATOR",
                "user": {"login": "alice"},
                "path": "src/foo.py",
                "line": 42,
                "side": "RIGHT",
                "diff_hunk": "@@ -40,3 +40,3 @@\n-old\n+new line under inspection",
                "in_reply_to_id": None,
            },
            "pull_request": {"number": 9, "state": "open"},
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("pull_request_review_comment", payload, "@askserge")

        self.assertIsNotNone(req)
        assert req is not None
        self.assertEqual(req.number, 9)
        self.assertIsNotNone(req.inline)
        assert req.inline is not None
        self.assertEqual(req.inline.comment_id, 4242)
        self.assertEqual(req.inline.path, "src/foo.py")
        self.assertEqual(req.inline.line, 42)
        self.assertEqual(req.inline.side, "RIGHT")
        self.assertIn("new line under inspection", req.inline.diff_hunk)

    def test_review_comment_event_falls_back_to_original_line(self) -> None:
        # When a thread becomes "outdated" GitHub nulls line/side and
        # only keeps original_line/original_side; we should still anchor
        # the follow-up to the line the commenter saw.
        payload = {
            "action": "created",
            "comment": {
                "id": 1,
                "body": "@askserge what is this for?",
                "author_association": "MEMBER",
                "user": {"login": "bob"},
                "path": "x.py",
                "line": None,
                "side": None,
                "original_line": 17,
                "original_side": "LEFT",
                "diff_hunk": "@@ ...",
            },
            "pull_request": {"number": 1, "state": "open"},
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("pull_request_review_comment", payload, "@askserge")

        assert req is not None and req.inline is not None
        self.assertEqual(req.inline.line, 17)
        self.assertEqual(req.inline.side, "LEFT")

    def test_review_comment_event_rejects_closed_pr(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "id": 1,
                "body": "@askserge",
                "author_association": "MEMBER",
                "user": {"login": "bob"},
                "path": "x.py",
                "line": 5,
                "side": "RIGHT",
                "diff_hunk": "",
            },
            "pull_request": {"number": 1, "state": "closed"},
            "repository": {"full_name": "acme/project"},
        }

        self.assertIsNone(
            build_review_request("pull_request_review_comment", payload, "@askserge")
        )

    def test_issue_comment_event_has_no_inline_context(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "@askserge please review",
                "author_association": "MEMBER",
                "id": 1,
                "user": {"login": "alice"},
            },
            "issue": {
                "pull_request": {"url": "..."},
                "state": "open",
                "number": 3,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        assert req is not None
        self.assertIsNone(req.inline)

    def test_build_review_request_rejects_bot_authored_comment(self) -> None:
        # In App mode the webhook receives the App's own comments; a stray
        # trigger phrase in bot output must not start a review loop.
        payload = {
            "action": "created",
            "comment": {
                "body": "@askserge please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "serge[bot]", "type": "Bot"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        self.assertIsNone(req)

    def test_build_review_request_accepts_trigger_after_leading_whitespace(
        self,
    ) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "  @askserge please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        self.assertIsNotNone(req)

    def test_build_review_request_rejects_mid_comment_trigger(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": (
                    "Right, a maintainer need to trigger a new review with "
                    "@askserge and it would review again"
                ),
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        self.assertIsNone(req)

    def test_build_review_request_rejects_non_matching_trigger(self) -> None:
        payload = {
            "action": "created",
            "comment": {
                "body": "@claude please review",
                "author_association": "MEMBER",
                "id": 123,
                "user": {"login": "reviewer"},
            },
            "issue": {
                "pull_request": {
                    "url": "https://api.github.com/repos/acme/project/pulls/7"
                },
                "state": "open",
                "number": 7,
            },
            "repository": {"full_name": "acme/project"},
        }

        req = build_review_request("issue_comment", payload, "@askserge")

        self.assertIsNone(req)


if __name__ == "__main__":
    unittest.main()
