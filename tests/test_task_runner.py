"""End-to-end test for the in-pod task runner (reviewbot.task_runner).

Exercises the FULL runner path locally with only the true externals faked:

- a **real local git repo** as the clone source (via the spec's repo_remote_url),
  checked out by the real CloneCache;
- the LLM stubbed at ``_run_agentic_loop`` (returns a canned patch JSON), so the
  real ``prepare_task`` / ``publish_task`` run;
- the GitHub Git Data API faked (records the created PR);
- a **real HTTP callback sink** so the callback transport is genuinely exercised.

This proves the runner produces a PR and streams events + a terminal outcome back,
without touching the network, an LLM, or Kubernetes.
"""

import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from reviewbot import task_runner
from reviewbot.reviewer import _AggregateMetrics

_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}


def _git(cwd, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, env=_GIT_ENV
    ).stdout.decode()


class _FakeChat:
    def __init__(self, content):
        self.content = content
        self.finish_reason = "stop"
        self.reasoning_chars = 0


class _FakeGH:
    """Minimal Git Data API fake — records the PR it 'opens'."""

    def __init__(self):
        self.created_pr = None
        self.marked_ready = None

    def get_ref_sha(self, owner, repo, ref):
        return f"parent-of-{ref}"

    def get_commit_tree_sha(self, owner, repo, commit_sha):
        return f"tree-of-{commit_sha}"

    def create_blob(self, owner, repo, content):
        return "blob1"

    def create_tree(self, owner, repo, base_tree, entries):
        return "newtree"

    def create_commit(self, owner, repo, *, message, tree_sha, parents):
        return "newcommit"

    def create_ref(self, owner, repo, ref, sha):
        return {"ref": ref}

    def create_pull_request(self, owner, repo, *, title, head, base, body, draft=False):
        self.created_pr = {"title": title, "head": head, "base": base}
        return {
            "number": 99,
            "html_url": "https://github.com/o/r/pull/99",
            "node_id": "PR_node_99",
        }

    def mark_pull_request_ready(self, node_id):
        self.marked_ready = node_id


class _CallbackSink:
    """A real HTTP server that captures the runner's callback POSTs."""

    def __init__(self):
        self.events = []
        self.terminal = None

        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if "terminal" in body:
                    sink.terminal = body["terminal"]
                else:
                    sink.events.append(body)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/events"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


class TaskRunnerE2ETests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", self.tmp], check=False))
        # A real local "origin" repo on `main` with one tracked file.
        self.origin = os.path.join(self.tmp, "origin")
        os.makedirs(self.origin)
        _git(self.origin, "init", "-q")
        _git(self.origin, "checkout", "-q", "-B", "main")
        with open(os.path.join(self.origin, "hello.txt"), "w") as fh:
            fh.write("hi\n")
        _git(self.origin, "add", "hello.txt")
        _git(self.origin, "commit", "-q", "-m", "init")
        # Produce a valid unified diff (hi -> hello) the LLM will "return".
        with open(os.path.join(self.origin, "hello.txt"), "w") as fh:
            fh.write("hello\n")
        self.patch = _git(self.origin, "diff")
        _git(self.origin, "checkout", "-q", "--", "hello.txt")

    def _spec(self, callback_url):
        return task_runner.RunnerSpec(
            job_id="job-e2e-0001",
            request={
                "owner": "o",
                "repo": "r",
                "base_ref": "main",
                "instruction": "make hi say hello",
                "context": "",
                "mode": "new_pr",
                "branch_prefix": "serge/fix",
            },
            github_token="ghs-fake-token",
            llm={"api_base": "https://example.com/v1", "api_key": "k", "model": "m"},
            callback={"url": callback_url, "token": "cb-token"},
            repo_remote_url=self.origin,
        )

    def test_full_run_opens_pr_and_reports_back(self):
        answer = json.dumps(
            {"title": "Say hello", "body": "Fixes the greeting.", "patch": self.patch}
        )
        fake_gh = _FakeGH()

        env = {
            "DEV_NO_AUTH": "1",
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmp, "clones"),
            # No TASK_NORMALIZE_COMMAND -> the in-loop normalize gate is skipped,
            # so this test needs no sandbox/toolchain.
        }

        with _CallbackSink() as sink:
            spec = self._spec(sink.url)
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "reviewbot.tasks._run_agentic_loop",
                    return_value=(_FakeChat(answer), _AggregateMetrics(turns=1)),
                ),
                patch.object(task_runner, "GitHubClient", return_value=fake_gh),
            ):
                rc = task_runner.run(spec)

        self.assertEqual(rc, 0)
        # The runner published a PR via the (faked) Git Data API.
        self.assertIsNotNone(fake_gh.created_pr)
        self.assertEqual(fake_gh.created_pr["base"], "main")
        self.assertEqual(fake_gh.marked_ready, "PR_node_99")
        # It streamed events and a terminal outcome to the real callback.
        kinds = [e["kind"] for e in sink.events]
        self.assertIn("clone", [e["text"] for e in sink.events if e["kind"] == "step"])
        self.assertIn("done", kinds)
        self.assertIsNotNone(sink.terminal)
        self.assertEqual(sink.terminal["status"], "published")
        self.assertEqual(sink.terminal["result"]["pr_number"], 99)

    def test_no_patch_reports_no_fix(self):
        answer = json.dumps({"title": "No fix", "body": "No safe fix.", "patch": ""})
        fake_gh = _FakeGH()
        env = {"DEV_NO_AUTH": "1", "WEB_CLONE_CACHE_DIR": os.path.join(self.tmp, "cl2")}

        with _CallbackSink() as sink:
            spec = self._spec(sink.url)
            with (
                patch.dict(os.environ, env, clear=False),
                patch(
                    "reviewbot.tasks._run_agentic_loop",
                    return_value=(_FakeChat(answer), _AggregateMetrics(turns=1)),
                ),
                patch.object(task_runner, "GitHubClient", return_value=fake_gh),
            ):
                rc = task_runner.run(spec)

        self.assertEqual(rc, 0)
        self.assertIsNone(fake_gh.created_pr)
        self.assertEqual(sink.terminal["status"], "no_fix")


if __name__ == "__main__":
    unittest.main()
