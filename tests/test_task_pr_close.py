"""Tests for closing serge's own task PRs (reviewbot/task_pr_close.py)."""

import unittest

from reviewbot.task_pr_close import (
    CLOSE_REASONS,
    MAX_COMMENT_CHARS,
    ClosePRError,
    close_task_pr,
)


def _pr(**overrides):
    pr = {
        "number": 7,
        "state": "open",
        "body": "Fixes the test.\n<!-- serge-task:integration-failure-triage:sha256:abc -->",
        "user": {"login": "sergereview[bot]", "type": "Bot"},
        "head": {
            "ref": "serge/fix/itf-abc123",
            "repo": {"full_name": "acme/widgets"},
        },
    }
    pr.update(overrides)
    return pr


class FakeClient:
    def __init__(self, pr):
        self.pr = pr
        self.comments = []
        self.closed = []

    def get_pr(self, owner, repo, number):
        return self.pr

    def post_issue_comment(self, owner, repo, number, body):
        self.comments.append((number, body))
        return {}

    def close_pull_request(self, owner, repo, number, *, body=None):
        self.closed.append((number, body))
        return {}


PAYLOAD = {"reason": "stale", "comment": "The target test passes on main."}


class CloseTaskPRTests(unittest.TestCase):
    def test_closes_with_comment_and_marker(self):
        client = FakeClient(_pr())
        result = close_task_pr(client, "acme", "widgets", 7, PAYLOAD)
        self.assertEqual(result, {"closed": True, "number": 7, "reason": "stale"})
        self.assertEqual(len(client.comments), 1)
        self.assertTrue(
            client.comments[0][1].startswith("The target test passes on main.")
        )
        number, body = client.closed[0]
        self.assertEqual(number, 7)
        self.assertTrue(body.endswith(CLOSE_REASONS["stale"]))
        # The original body, including the triage's fingerprint marker, survives.
        self.assertIn("serge-task:integration-failure-triage", body)

    def test_marker_not_duplicated(self):
        client = FakeClient(_pr(body="x\n\n" + CLOSE_REASONS["superseded"]))
        close_task_pr(
            client, "acme", "widgets", 7, {"reason": "superseded", "comment": "dup"}
        )
        self.assertEqual(client.closed[0][1].count(CLOSE_REASONS["superseded"]), 1)

    def test_already_closed_is_a_no_op(self):
        client = FakeClient(_pr(state="closed"))
        result = close_task_pr(client, "acme", "widgets", 7, PAYLOAD)
        self.assertEqual(result["closed"], False)
        self.assertEqual(client.comments, [])
        self.assertEqual(client.closed, [])

    def test_refuses_non_serge_branch(self):
        pr = _pr(head={"ref": "fix-bug", "repo": {"full_name": "acme/widgets"}})
        client = FakeClient(pr)
        with self.assertRaises(ClosePRError) as cm:
            close_task_pr(client, "acme", "widgets", 7, PAYLOAD)
        self.assertEqual(cm.exception.status, 403)
        self.assertEqual(client.closed, [])

    def test_refuses_fork_with_serge_named_branch(self):
        pr = _pr(head={"ref": "serge/fix/x", "repo": {"full_name": "evil/widgets"}})
        with self.assertRaises(ClosePRError) as cm:
            close_task_pr(FakeClient(pr), "acme", "widgets", 7, PAYLOAD)
        self.assertEqual(cm.exception.status, 403)

    def test_refuses_human_author(self):
        pr = _pr(user={"login": "octocat", "type": "User"})
        with self.assertRaises(ClosePRError) as cm:
            close_task_pr(FakeClient(pr), "acme", "widgets", 7, PAYLOAD)
        self.assertEqual(cm.exception.status, 403)

    def test_validates_reason_and_comment(self):
        for payload in (
            None,
            {"reason": "because", "comment": "x"},
            {"reason": "stale", "comment": "  "},
            {"reason": "stale", "comment": "x" * (MAX_COMMENT_CHARS + 1)},
        ):
            with self.assertRaises(ClosePRError) as cm:
                close_task_pr(FakeClient(_pr()), "acme", "widgets", 7, payload)
            self.assertEqual(cm.exception.status, 400)


if __name__ == "__main__":
    unittest.main()
