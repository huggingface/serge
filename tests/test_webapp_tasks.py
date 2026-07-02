"""Tests for the POST /tasks route: the feature flag, OIDC auth, the
repository-claim authorization, and the write opt-in gate. The OIDC
verification and the task worker are stubbed so no network/LLM is touched —
we assert the route's accept/reject behavior and that an accepted task is
queued."""

import importlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None


class _Claims:
    def __init__(self, repository="acme/widgets", actor="octocat"):
        self.repository = repository
        self.actor = actor
        self.workflow_ref = "acme/widgets/.github/workflows/fix.yml@refs/heads/main"
        self.raw = {}


class WebappTasksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _import_webapp(self, *, task_api_enabled=True, **extra_env):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "LLM_MAX_TOKENS": "4096",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "TASK_API_ENABLED": "1" if task_api_enabled else "0",
            "TASK_OIDC_AUDIENCE": "serge",
            "TASK_LLM_MAX_TOKENS": "16384",
            "TASK_LLM_MAX_INPUT_TOKENS": "250000",
            "TASK_TOOL_MAX_ITERATIONS": "8",
        }
        env.update({k: v for k, v in extra_env.items() if v is not None})
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def _seed_write_config(self, webapp, *, task_write_enabled=True):
        webapp._store.insert_provider_config(
            id="c1",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model="some-model",
            repo_pattern="acme/widgets",
            allowed_users=["octocat"],
            allowed_orgs=[],
            created_by="admin",
            task_write_enabled=task_write_enabled,
        )

    def test_disabled_returns_404(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(task_api_enabled=False)
        client = TestClient(webapp.app)
        r = client.post("/tasks", json={"instruction": "x"})
        self.assertEqual(r.status_code, 404)

    def test_missing_bearer_returns_401(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        r = client.post("/tasks", json={"instruction": "x"})
        self.assertEqual(r.status_code, 401)

    def test_bad_oidc_returns_401(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        with patch.object(webapp, "verify_token", side_effect=webapp.OIDCError("bad")):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "x"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 401)

    def test_no_write_config_returns_403(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        # config exists but write not enabled
        self._seed_write_config(webapp, task_write_enabled=False)
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "fix tests", "context": "log"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 403)

    def test_repo_claim_mismatch_returns_403(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_write_config(webapp)
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "x", "repo": "evil/other"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 403)

    def test_accepted_task_is_queued(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_write_config(webapp)
        submitted = {}

        def fake_submit(fn, job, worker_cfg, req):
            submitted["job"] = job
            submitted["worker_cfg"] = worker_cfg
            submitted["req"] = req

        with (
            patch.object(webapp, "verify_token", return_value=_Claims()),
            patch.object(webapp._TASK_POOL, "submit", side_effect=fake_submit),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={
                    "instruction": "fix the failing tests",
                    "context": "trace",
                    "notifications": {
                        "slack_channel": "#dynamic-ci",
                        "task_finished": True,
                    },
                },
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 202)
        body = r.json()
        self.assertEqual(body["repo"], "acme/widgets")
        self.assertEqual(body["mode"], "new_pr")
        self.assertTrue(body["url"].startswith("/tasks/acme/widgets/"))
        self.assertEqual(submitted["job"].kind, "task")
        self.assertEqual(submitted["req"].instruction, "fix the failing tests")
        self.assertEqual(submitted["req"].slack_channel, "#dynamic-ci")
        self.assertTrue(submitted["req"].slack_notify_task_finished)
        self.assertEqual(webapp.cfg.llm_max_tokens, 4096)
        self.assertEqual(submitted["worker_cfg"].llm_max_tokens, 16384)
        self.assertEqual(submitted["worker_cfg"].llm_max_input_tokens, 250000)
        self.assertEqual(submitted["worker_cfg"].tool_max_iterations, 8)
        self.assertTrue(submitted["worker_cfg"].tool_max_iterations_strict)
        # Persisted with task kind.
        row = webapp._store.load(body["id"])
        self.assertEqual(row["kind"], "task")

    def test_inprocess_execution_dispatches_to_worker(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()  # TASK_EXECUTION defaults to inprocess
        self._seed_write_config(webapp)
        submitted = {}

        with (
            patch.object(webapp, "verify_token", return_value=_Claims()),
            patch.object(
                webapp._TASK_POOL,
                "submit",
                side_effect=lambda fn, *a: submitted.setdefault("fn", fn),
            ),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "fix", "context": "trace"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 202)
        self.assertIs(submitted["fn"], webapp._run_task_worker)

    def test_docker_execution_dispatches_to_launcher(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(
            TASK_EXECUTION="docker",
            TASK_RUNNER_IMAGE="serge/runner:latest",
            TASK_CALLBACK_BASE_URL="http://serge:8000",
        )
        self.assertEqual(webapp.cfg.task_execution, "docker")
        self._seed_write_config(webapp)
        submitted = {}

        with (
            patch.object(webapp, "verify_token", return_value=_Claims()),
            patch.object(
                webapp._TASK_POOL,
                "submit",
                side_effect=lambda fn, *a: submitted.setdefault("fn", fn),
            ),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                "/tasks",
                json={"instruction": "fix", "context": "trace"},
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 202)
        self.assertIs(submitted["fn"], webapp._launch_task_pod)

    def test_task_finished_notification_uses_request_channel(self):
        webapp = self._import_webapp()
        worker_cfg = webapp.dataclasses.replace(
            webapp.cfg,
            slack_bot_token="tok",
            slack_report_channel="#default-ci",
        )
        req = webapp.TaskRequest(
            owner="acme",
            repo="widgets",
            base_ref="main",
            instruction="fix",
            context="trace",
            slack_channel="#dynamic-ci",
            slack_notify_task_finished=True,
        )
        job = webapp.Job(
            id="abcdef1234567890",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=0,
            trigger_comment="fix",
            llm_provider="hf",
            llm_api_base="https://example.com/v1",
            llm_model="model",
            created_at=0,
            status="done",
            source="task",
            kind="task",
            task_result={
                "message": "Opened PR #99.",
                "pr_number": 99,
                "url": "https://github.com/acme/widgets/pull/99",
            },
        )

        with patch.object(webapp, "post_task_finished_notification") as notify:
            webapp._notify_task_finished(worker_cfg, req, job)

        notify.assert_called_once()
        self.assertEqual(notify.call_args.kwargs["token"], "tok")
        self.assertEqual(notify.call_args.kwargs["channel"], "#dynamic-ci")
        self.assertEqual(notify.call_args.kwargs["pr_number"], 99)

    def _run_task_worker_with(self, webapp, task_result):
        """Drive _run_task_worker with the heavy deps stubbed so the only thing
        under test is how a TaskResult maps to the terminal job.status."""
        worker_cfg = webapp.dataclasses.replace(
            webapp.cfg, github_app_id="123", github_private_key="key"
        )
        req = webapp.TaskRequest(
            owner="acme",
            repo="widgets",
            base_ref="main",
            instruction="fix",
            context="trace",
        )
        job = webapp.Job(
            id="abcdef1234567890",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=0,
            trigger_comment="fix",
            llm_provider="hf",
            llm_api_base="https://example.com/v1",
            llm_model="model",
            created_at=0,
            status="running",
            source="task",
            kind="task",
        )
        # prepare_task/publish_task are both stubbed, so the plan value is never
        # inspected — any sentinel works.
        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="tok"),
            patch.object(webapp, "GitHubClient"),
            patch.object(
                webapp._clone_cache,
                "acquire_ref",
                return_value=types.SimpleNamespace(path="/tmp/co"),
            ),
            patch.object(webapp._clone_cache, "release"),
            patch.object(webapp, "_persist_terminal"),
            patch.object(webapp, "prepare_task", return_value=object()),
            patch.object(webapp, "publish_task", return_value=task_result),
        ):
            webapp._run_task_worker(job, worker_cfg, req)
        return job

    def test_published_status_when_pr_opened(self):
        webapp = self._import_webapp()
        result = webapp.TaskResult(mode="new_pr", pr_number=99, no_change=False)
        job = self._run_task_worker_with(webapp, result)
        self.assertEqual(job.status, "published")

    def test_no_fix_status_when_no_patch(self):
        webapp = self._import_webapp()
        result = webapp.TaskResult(mode="new_pr", no_change=True)
        job = self._run_task_worker_with(webapp, result)
        self.assertEqual(job.status, "no_fix")


class TaskLauncherTests(unittest.TestCase):
    """The docker launcher (_launch_task_pod) and the callback-ingest endpoint
    (POST /internal/tasks/{id}/events) added in Phase 2."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _import_webapp(self, **extra_env):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "TASK_API_ENABLED": "1",
            "TASK_EXECUTION": "docker",
            "TASK_RUNNER_IMAGE": "serge/runner:latest",
            "TASK_CALLBACK_BASE_URL": "http://serge:8000",
        }
        env.update(extra_env)
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def _make_job(self, webapp, *, status="running", callback_token="cbtok"):
        job = webapp.Job(
            id="job123abc456",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=0,
            trigger_comment="fix",
            llm_provider="hf",
            llm_api_base="https://example.com/v1",
            llm_model="model",
            created_at=0,
            status=status,
            source="task",
            kind="task",
            callback_token=callback_token,
        )
        job.loop = None  # _push_event tolerates a None loop (no live SSE)
        with webapp._jobs_lock:
            webapp._jobs[job.id] = job
        return job

    # --- _launch_task_pod ------------------------------------------------
    def _worker_cfg_and_req(self, webapp):
        worker_cfg = webapp.dataclasses.replace(
            webapp.cfg,
            llm_api_base="https://llm.example/v1",
            llm_api_key="secret-llm-key",
            llm_model="some-model",
            llm_bill_to="acme-org",
        )
        req = webapp.TaskRequest(
            owner="acme",
            repo="widgets",
            base_ref="main",
            instruction="fix",
            context="trace",
        )
        return worker_cfg, req

    def test_launch_builds_spec_and_reconciles_on_no_callback(self):
        webapp = self._import_webapp()
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)
        captured = {}

        def fake_launch(spec, opts, *, wait, timeout):
            captured["spec"] = spec
            captured["opts"] = opts
            captured["wait"] = wait
            return 0, "container-id"

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(webapp, "launch_docker", side_effect=fake_launch),
            patch.object(webapp, "_notify_task_finished"),
            patch.object(webapp, "_persist_terminal"),
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        spec = captured["spec"]
        self.assertTrue(captured["wait"])
        self.assertEqual(spec["job_id"], job.id)
        self.assertEqual(spec["github_token"], "gh-token")
        self.assertEqual(spec["llm"]["api_key"], "secret-llm-key")
        self.assertEqual(spec["llm"]["bill_to"], "acme-org")
        self.assertEqual(
            spec["callback"]["url"],
            f"http://serge:8000/internal/tasks/{job.id}/events",
        )
        self.assertTrue(spec["callback"]["token"])
        self.assertEqual(captured["opts"].image, "serge/runner:latest")
        # No terminal callback arrived → reconcile to error; token is cleared.
        self.assertEqual(job.status, "error")
        self.assertIsNone(job.callback_token)

    def test_launch_leaves_terminal_status_from_callback(self):
        webapp = self._import_webapp()
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)

        def fake_launch(spec, opts, *, wait, timeout):
            # Simulate the runner's terminal callback landing before exit.
            job.status = "published"
            return 0, "cid"

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(webapp, "launch_docker", side_effect=fake_launch),
            patch.object(webapp, "_notify_task_finished"),
            patch.object(webapp, "_persist_terminal"),
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        self.assertEqual(job.status, "published")

    def test_launch_kubernetes_dispatches_and_reconciles(self):
        webapp = self._import_webapp(
            TASK_EXECUTION="kubernetes",
            TASK_K8S_NAMESPACE="serge",
            TASK_K8S_SERVICE_ACCOUNT="serge-task",
            TASK_K8S_NODE_SELECTOR="pool=tasks",
            TASK_RUNNER_PROXY="http://egress:3128",
            TASK_RUNNER_NO_PROXY=".svc.cluster.local",
        )
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)
        captured = {}

        def fake_launch(spec, opts, *, timeout, poll_interval=2.0):
            captured["spec"] = spec
            captured["opts"] = opts
            return 0, "pod log"

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(webapp, "launch_kubernetes", side_effect=fake_launch),
            patch.object(webapp, "_notify_task_finished"),
            patch.object(webapp, "_persist_terminal"),
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        opts = captured["opts"]
        self.assertEqual(opts.image, "serge/runner:latest")
        self.assertEqual(opts.namespace, "serge")
        self.assertEqual(opts.service_account, "serge-task")
        self.assertEqual(opts.node_selector, {"pool": "tasks"})
        self.assertEqual(opts.proxy, "http://egress:3128")
        self.assertEqual(opts.no_proxy, ".svc.cluster.local")
        self.assertEqual(captured["spec"]["github_token"], "gh-token")
        # No terminal callback arrived → reconcile to error.
        self.assertEqual(job.status, "error")
        self.assertIsNone(job.callback_token)

    def test_launch_kubernetes_surfaces_sandbox_error(self):
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes")
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(
                webapp,
                "launch_kubernetes",
                side_effect=webapp.K8sSandboxError("no cluster"),
            ),
            patch.object(webapp, "_notify_task_finished"),
            patch.object(webapp, "_persist_terminal"),
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "no cluster")

    # --- POST /internal/tasks/{id}/events --------------------------------
    def test_ingest_rejects_missing_token(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._make_job(webapp)
        client = TestClient(webapp.app)
        r = client.post("/internal/tasks/job123abc456/events", json={"kind": "log"})
        self.assertEqual(r.status_code, 401)

    def test_ingest_rejects_bad_token(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._make_job(webapp, callback_token="right")
        client = TestClient(webapp.app)
        r = client.post(
            "/internal/tasks/job123abc456/events",
            json={"kind": "log", "text": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        self.assertEqual(r.status_code, 401)

    def test_ingest_rejects_unknown_job(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        r = client.post(
            "/internal/tasks/does-not-exist/events",
            json={"kind": "log"},
            headers={"Authorization": "Bearer whatever"},
        )
        self.assertEqual(r.status_code, 401)

    def test_ingest_event_appends_to_history(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        job = self._make_job(webapp, callback_token="tok")
        client = TestClient(webapp.app)
        r = client.post(
            "/internal/tasks/job123abc456/events",
            json={"kind": "log", "text": "checking out…"},
            headers={"Authorization": "Bearer tok"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(
            any(e["text"] == "checking out…" for e in job.history),
        )
        self.assertEqual(job.status, "running")  # non-terminal

    def test_ingest_terminal_records_outcome(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        job = self._make_job(webapp, callback_token="tok")
        client = TestClient(webapp.app)
        r = client.post(
            "/internal/tasks/job123abc456/events",
            json={
                "terminal": {
                    "status": "published",
                    "result": {"mode": "new_pr", "pr_number": 42},
                    "error": None,
                }
            },
            headers={"Authorization": "Bearer tok"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(job.status, "published")
        self.assertEqual(job.task_result["pr_number"], 42)

    def test_ingest_terminal_records_error(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        job = self._make_job(webapp, callback_token="tok")
        client = TestClient(webapp.app)
        r = client.post(
            "/internal/tasks/job123abc456/events",
            json={"terminal": {"status": "error", "error": "boom"}},
            headers={"Authorization": "Bearer tok"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "boom")


if __name__ == "__main__":
    unittest.main()
