import json
import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from reviewbot import action_runner
from reviewbot.llm_client import LLMResponseError


def _write_event(payload: dict) -> str:
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)
    return path


def _inline_comment_payload(*, fork: bool = False) -> dict:
    payload = {
        "action": "created",
        "comment": {
            "id": 3322017554,
            "body": "@askserge can you check this?",
            "author_association": "MEMBER",
            "user": {"login": "reviewer"},
            "path": "src/foo.py",
            "line": 68,
            "side": "RIGHT",
            "diff_hunk": "@@ -65,4 +65,4 @@",
        },
        "pull_request": {"number": 13827, "state": "open"},
        "repository": {"full_name": "huggingface/diffusers"},
    }
    if fork:
        payload["pull_request"]["head"] = {
            "repo": {"full_name": "akshan-main/diffusers"}
        }
        payload["pull_request"]["base"] = {
            "repo": {"full_name": "huggingface/diffusers"}
        }
    return payload


def _issue_comment_payload() -> dict:
    return {
        "action": "created",
        "comment": {
            "id": 123,
            "body": "@askserge please review",
            "author_association": "MEMBER",
            "user": {"login": "reviewer"},
        },
        "issue": {
            "number": 13827,
            "state": "open",
            "pull_request": {
                "url": "https://api.github.com/repos/huggingface/diffusers/pulls/13827"
            },
        },
        "repository": {"full_name": "huggingface/diffusers"},
    }


class ActionRunnerTests(unittest.TestCase):
    def test_empty_llm_api_key_fails_before_review(self) -> None:
        event_path = _write_event(_inline_comment_payload())
        self.addCleanup(os.remove, event_path)
        env = {
            "GITHUB_EVENT_NAME": "pull_request_review_comment",
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_TOKEN": "github-token",
            "LLM_API_KEY": "",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("reviewbot.action_runner.run_followup") as run_followup,
            self.assertLogs("ai-reviewer.action", level="ERROR") as logs,
        ):
            code = action_runner.main()

        self.assertEqual(code, 1)
        run_followup.assert_not_called()
        self.assertIn("LLM_API_KEY missing", "\n".join(logs.output))

    def test_forked_pr_with_missing_llm_api_key_uses_action_message(self) -> None:
        event_path = _write_event(_inline_comment_payload(fork=True))
        self.addCleanup(os.remove, event_path)
        fd, summary_path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.remove, summary_path)
        env = {
            "GITHUB_EVENT_NAME": "pull_request_review_comment",
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_STEP_SUMMARY": summary_path,
            "GITHUB_TOKEN": "github-token",
            "LLM_API_KEY": "",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("reviewbot.action_runner.run_followup") as run_followup,
            patch("reviewbot.action_runner.GitHubClient") as github_client,
            self.assertLogs("ai-reviewer.action", level="WARNING") as logs,
        ):
            github_client.return_value.reply_to_review_comment.side_effect = (
                requests.HTTPError("403 Resource not accessible by integration")
            )
            code = action_runner.main()

        self.assertEqual(code, 1)
        run_followup.assert_not_called()
        github_client.return_value.reply_to_review_comment.assert_called_once()
        body = github_client.return_value.reply_to_review_comment.call_args.args[-1]
        self.assertIn(action_runner.FORKED_PR_ACTION_MESSAGE, body)
        output = "\n".join(logs.output)
        self.assertIn(action_runner.FORKED_PR_ACTION_MESSAGE, output)
        with open(summary_path) as f:
            self.assertIn(action_runner.FORKED_PR_ACTION_MESSAGE, f.read())

    def test_llm_api_key_is_stripped_before_review(self) -> None:
        event_path = _write_event(_inline_comment_payload())
        self.addCleanup(os.remove, event_path)
        env = {
            "GITHUB_EVENT_NAME": "pull_request_review_comment",
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_TOKEN": "github-token",
            "LLM_API_KEY": "  token-with-newline\n",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("reviewbot.action_runner.run_followup") as run_followup,
            patch("reviewbot.action_runner.GitHubClient"),
        ):
            code = action_runner.main()

        self.assertEqual(code, 0)
        cfg = run_followup.call_args.args[0]
        self.assertEqual(cfg.llm_api_key, "token-with-newline")

    def test_direct_pr_review_forces_comment_event(self) -> None:
        event_path = _write_event(_issue_comment_payload())
        self.addCleanup(os.remove, event_path)
        env = {
            "GITHUB_EVENT_NAME": "issue_comment",
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_TOKEN": "github-token",
            "LLM_API_KEY": "token",
        }

        with (
            patch.dict(os.environ, env, clear=True),
            patch("reviewbot.action_runner.run_review") as run_review,
            patch("reviewbot.action_runner.GitHubClient"),
        ):
            code = action_runner.main()

        self.assertEqual(code, 0)
        self.assertTrue(run_review.call_args.kwargs["force_comment_event"])

    def test_llm_response_error_is_logged_without_traceback(self) -> None:
        event_path = _write_event(_inline_comment_payload())
        self.addCleanup(os.remove, event_path)
        env = {
            "GITHUB_EVENT_NAME": "pull_request_review_comment",
            "GITHUB_EVENT_PATH": event_path,
            "GITHUB_TOKEN": "github-token",
            "LLM_API_KEY": "bad-token",
        }
        llm_error = LLMResponseError(
            401,
            "Unauthorized",
            "https://api.anthropic.com/v1/chat/completions",
            '{"error":{"message":"Invalid bearer token"}}',
        )
        post_error = requests.HTTPError(
            '403 replying to review comment: {"message":"Resource not accessible by integration"}'
        )

        with (
            patch.dict(os.environ, env, clear=True),
            patch("reviewbot.action_runner.run_followup", side_effect=llm_error),
            patch("reviewbot.action_runner.GitHubClient") as github_client,
            self.assertLogs("ai-reviewer.action", level="WARNING") as logs,
        ):
            github_client.return_value.reply_to_review_comment.side_effect = post_error
            code = action_runner.main()

        self.assertEqual(code, 1)
        output = "\n".join(logs.output)
        self.assertIn("LLM endpoint returned 401 Unauthorized", output)
        self.assertIn("failed to post failure comment to PR: 403", output)
        self.assertNotIn("Traceback", output)
        github_client.return_value.reply_to_review_comment.assert_called_once()


if __name__ == "__main__":
    unittest.main()
