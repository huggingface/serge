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


class RunnerCrashReportingTests(unittest.TestCase):
    """A crash *before* the agent loop (spec/config/clone) must still POST a
    terminal ``error`` callback so serge shows *why* instead of the opaque
    "task runner exited without reporting (exit code 1)"."""

    def _spec(self, callback_url):
        return task_runner.RunnerSpec(
            job_id="job-crash-0001",
            request={
                "owner": "o",
                "repo": "r",
                "base_ref": "main",
                "instruction": "x",
                "context": "",
            },
            github_token="t",
            llm={"api_base": "https://e/v1", "api_key": "k", "model": "m"},
            callback={"url": callback_url, "token": "cb"},
        )

    def test_startup_crash_reports_terminal_error(self):
        def _boom(_spec):
            raise RuntimeError("checkout blew up on the big repo")

        with _CallbackSink() as sink:
            spec = self._spec(sink.url)
            with patch.object(task_runner, "build_runner_config", _boom):
                rc = task_runner.run(spec)

        self.assertEqual(rc, 1)
        self.assertIsNotNone(sink.terminal)
        self.assertEqual(sink.terminal["status"], "error")
        # The cause + the crashing frame are surfaced, not swallowed.
        self.assertIn("RuntimeError", sink.terminal["error"])
        self.assertIn("checkout blew up on the big repo", sink.terminal["error"])
        self.assertIn("(at ", sink.terminal["error"])

    def test_main_backstop_reports_when_run_escapes(self):
        # If run() itself raises (e.g. couldn't even build the emitter), main's
        # backstop still POSTs a terminal error from the spec's callback info.
        boom = RuntimeError("run() escaped")
        with _CallbackSink() as sink:
            spec = self._spec(sink.url)
            tmp = tempfile.mkdtemp()
            self.addCleanup(lambda: subprocess.run(["rm", "-rf", tmp], check=False))
            spec_path = os.path.join(tmp, "task.json")
            with open(spec_path, "w") as fh:
                json.dump(
                    {
                        "job_id": spec.job_id,
                        "request": spec.request,
                        "github_token": spec.github_token,
                        "callback": spec.callback,
                    },
                    fh,
                )
            with patch.object(task_runner, "run", side_effect=boom):
                rc = task_runner.main(["--spec", spec_path])

        self.assertEqual(rc, 1)
        self.assertIsNotNone(sink.terminal)
        self.assertEqual(sink.terminal["status"], "error")
        self.assertIn("run() escaped", sink.terminal["error"])


class BuildRunnerConfigTests(unittest.TestCase):
    """The spec's resolved-config subset (per-task caps + operator normalize/
    review settings) must reach the in-pod Config; the LLM dict wins for provider
    settings; and the in-pod sandboxes are forced off (the pod is the sandbox)."""

    def test_applies_config_and_llm_overrides_and_forces_sandboxes_off(self):
        spec = task_runner.RunnerSpec(
            job_id="j",
            request={
                "owner": "o",
                "repo": "r",
                "base_ref": "main",
                "instruction": "x",
                "context": "",
            },
            github_token="t",
            llm={
                "api_base": "https://llm.example/v1",
                "api_key": "secret-key",
                "model": "m",
                "stream": False,
            },
            config={
                "task_normalize_command": ["make", "fix-repo"],
                "task_normalize_max_retries": 4,
                "review_rules_path": ".ai/AGENTS.md",
                "tool_max_iterations": 7,
                "tool_max_iterations_strict": True,
                "llm_max_tokens": 12345,
                "llm_max_input_tokens": 250000,
            },
        )
        with patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=True):
            cfg = task_runner.build_runner_config(spec)

        # Operator/repo + per-task caps came across.
        self.assertEqual(cfg.task_normalize_command, ["make", "fix-repo"])
        self.assertEqual(cfg.task_normalize_max_retries, 4)
        self.assertEqual(cfg.review_rules_path, ".ai/AGENTS.md")
        self.assertEqual(cfg.tool_max_iterations, 7)
        self.assertTrue(cfg.tool_max_iterations_strict)
        self.assertEqual(cfg.llm_max_tokens, 12345)
        self.assertEqual(cfg.llm_max_input_tokens, 250000)
        # LLM provider settings win from the llm dict.
        self.assertEqual(cfg.llm_api_key, "secret-key")
        self.assertEqual(cfg.llm_api_base, "https://llm.example/v1")
        self.assertFalse(cfg.llm_stream)
        # The pod is the isolation boundary — no nested sandboxes.
        self.assertEqual(cfg.task_sandbox_backend, "off")
        self.assertEqual(cfg.helper_sandbox, "off")


class RunnerConfigPassthroughTest(unittest.TestCase):
    """The runner pod rebuilds Config from a near-empty env and applies only the
    RUNNER_CONFIG_FIELDS subset from the spec. A verify_* field missing from that
    list silently disables that behavior in the pod that actually runs tasks —
    exactly the bug where verify_reproduce_first never reached the runner."""

    def test_all_verify_fields_are_threaded_to_the_runner(self):
        import dataclasses

        from reviewbot import launcher
        from reviewbot.config import Config

        verify_fields = {
            f.name for f in dataclasses.fields(Config) if f.name.startswith("verify_")
        }
        missing = verify_fields - set(launcher.RUNNER_CONFIG_FIELDS)
        self.assertEqual(
            missing,
            set(),
            f"verify_* Config fields not threaded to the runner pod: {missing}",
        )

    def test_runner_config_threads_reproduce_first(self):
        import types

        from reviewbot import launcher

        fake = types.SimpleNamespace(
            **{
                field: (field in ("verify_on_gpu", "verify_reproduce_first"))
                for field in launcher.RUNNER_CONFIG_FIELDS
            }
        )
        out = launcher.runner_config(fake)
        self.assertTrue(out["verify_reproduce_first"])
        self.assertTrue(out["verify_on_gpu"])


if __name__ == "__main__":
    unittest.main()
