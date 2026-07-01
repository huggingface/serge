import unittest
from unittest.mock import Mock, patch

from reviewbot.slack_tool import (
    SLACK_POST_MESSAGE_URL,
    post_task_finished_notification,
    post_task_pr_created_notification,
)


class SlackToolTests(unittest.TestCase):
    def test_missing_config_is_noop(self):
        self.assertFalse(
            post_task_pr_created_notification(
                token=None,
                channel="#ci",
                repo_full_name="o/r",
                pr_number=1,
                pr_url="https://github.com/o/r/pull/1",
                title="Fix tests",
                branch="serge/fix-1",
                changed_files=["a.py"],
            )
        )

    def test_posts_to_dynamic_channel(self):
        response = Mock()
        response.json.return_value = {"ok": True}
        response.raise_for_status.return_value = None

        with patch("reviewbot.slack_tool.requests.post", return_value=response) as post:
            ok = post_task_pr_created_notification(
                token="tok",
                channel="#dynamic-ci",
                repo_full_name="o/r",
                pr_number=12,
                pr_url="https://github.com/o/r/pull/12",
                title="Fix failing tests",
                branch="serge/fix-12",
                changed_files=["a.py", "b.py"],
            )

        self.assertTrue(ok)
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], SLACK_POST_MESSAGE_URL)
        self.assertEqual(post.call_args.kwargs["json"]["channel"], "#dynamic-ci")
        self.assertIn("Fix failing tests", post.call_args.kwargs["json"]["text"])

    def test_posts_task_finished_notification(self):
        response = Mock()
        response.json.return_value = {"ok": True}
        response.raise_for_status.return_value = None

        with patch("reviewbot.slack_tool.requests.post", return_value=response) as post:
            ok = post_task_finished_notification(
                token="tok",
                channel="#finished-ci",
                repo_full_name="o/r",
                status="done",
                message="Opened PR #12.",
                pr_number=12,
                pr_url="https://github.com/o/r/pull/12",
                job_id="abcdef1234567890",
            )

        self.assertTrue(ok)
        self.assertEqual(post.call_args.kwargs["json"]["channel"], "#finished-ci")
        self.assertIn("Serge task finished", post.call_args.kwargs["json"]["text"])


if __name__ == "__main__":
    unittest.main()
