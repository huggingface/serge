"""Tests for the in-pod review runner (reviewbot/review_runner.py): request
reconstruction and the run() outcome/callback behavior, with prepare_review and
the checkout stubbed so no network/LLM/git is touched."""

import os
import unittest
from unittest import mock

from reviewbot import review_runner
from reviewbot.reviewer import ReviewDraft, ReviewRequest
from reviewbot.store import decode_draft
from reviewbot.task_runner import RunnerSpec


def _spec(**overrides):
    request = {
        "owner": "acme",
        "repo": "widgets",
        "number": 7,
        "trigger_comment_id": 0,
        "trigger_comment_body": "@serge review",
        "commenter": "octocat",
    }
    request.update(overrides.pop("request", {}))
    data = {
        "job_id": "job-rev-1",
        "request": request,
        "github_token": "gh-token",
        "request_type": "review",
        "llm": {},
        "config": {},
        "callback": {"url": "", "token": ""},
    }
    data.update(overrides)
    return RunnerSpec.from_file(_write_spec(data))


def _write_spec(data):
    import json
    import tempfile

    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(data, fh)
    fh.close()
    return fh.name


class _RecordingEmitter:
    def __init__(self, url, token, job_id):
        self.events = []
        self.terminal = None

    def emit(self, kind, text):
        self.events.append((kind, text))

    def finish(self, status, *, result=None, error=None, raw_llm_output=None):
        self.terminal = {
            "status": status,
            "result": result,
            "error": error,
            "raw_llm_output": raw_llm_output,
        }


class _FakeCache:
    def __init__(self, *a, **k):
        pass

    def acquire(self, *a, **k):
        return None  # review runs without browse tools; avoids git

    def release(self, checkout):
        pass


class ReviewRunnerTests(unittest.TestCase):
    def setUp(self):
        # build_runner_config reads Config.from_env; give it the minimum.
        self._env = mock.patch.dict(
            os.environ,
            {"LLM_API_KEY": "k", "WEB_CLONE_CACHE_DIR": "/tmp/rr-clones"},
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _run(self, prepare_return=None, prepare_side_effect=None):
        emitter = _RecordingEmitter(None, None, "job-rev-1")
        with (
            mock.patch.object(review_runner, "CloneCache", _FakeCache),
            mock.patch.object(review_runner, "GitHubClient", lambda *a, **k: object()),
            mock.patch.object(
                review_runner, "CallbackEmitter", lambda *a, **k: emitter
            ),
            mock.patch.object(
                review_runner,
                "prepare_review",
                return_value=prepare_return,
                side_effect=prepare_side_effect,
            ),
        ):
            code = review_runner.run(_spec())
        return code, emitter

    def test_build_review_request_ignores_extras(self):
        spec = _spec(request={"extra_field": "ignored"})
        req = review_runner.build_review_request(spec)
        self.assertIsInstance(req, ReviewRequest)
        self.assertEqual(req.owner, "acme")
        self.assertEqual(req.number, 7)
        self.assertIsNone(req.inline)

    def test_run_reports_draft(self):
        draft = ReviewDraft(
            owner="acme",
            repo="widgets",
            number=7,
            head_sha="deadbeef",
            summary="looks good",
            event="COMMENT",
            comments=[],
            prompt_tokens=100,
            completion_tokens=42,
        )
        code, emitter = self._run(prepare_return=draft)
        self.assertEqual(code, 0)
        self.assertEqual(emitter.terminal["status"], "done")
        result = emitter.terminal["result"]
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertEqual(result["completion_tokens"], 42)
        # The encoded draft round-trips back to a ReviewDraft.
        rebuilt = decode_draft(result["draft"])
        self.assertEqual(rebuilt.summary, "looks good")
        self.assertEqual(rebuilt.head_sha, "deadbeef")

    def test_run_no_reviewable_diff(self):
        code, emitter = self._run(prepare_return=None)
        self.assertEqual(code, 0)
        self.assertEqual(emitter.terminal["status"], "done")
        self.assertIsNone(emitter.terminal["result"])

    def test_run_reports_error_on_crash(self):
        code, emitter = self._run(prepare_side_effect=RuntimeError("boom"))
        self.assertEqual(code, 1)
        self.assertEqual(emitter.terminal["status"], "error")
        # The runner pod is reaped as soon as it self-reports, so the crash
        # cause must travel in the reported error itself — not "(see pod log)".
        self.assertIn("review crashed", emitter.terminal["error"])
        self.assertIn("boom", emitter.terminal["error"])
        self.assertNotIn("see pod log", emitter.terminal["error"])

    def test_run_surfaces_llm_error(self):
        from reviewbot.llm_client import LLMResponseError

        exc = LLMResponseError(
            429, "Too Many Requests", "https://router/v1/chat", "rate limit exceeded"
        )
        code, emitter = self._run(prepare_side_effect=exc)
        self.assertEqual(code, 1)
        self.assertEqual(emitter.terminal["status"], "error")
        # The LLM endpoint's own status + body excerpt lands on the job, so a
        # 429/400/auth failure is legible without the (reaped) pod log.
        self.assertIn("429", emitter.terminal["error"])
        self.assertIn("rate limit exceeded", emitter.terminal["error"])
        self.assertNotIn("see pod log", emitter.terminal["error"])


if __name__ == "__main__":
    unittest.main()
