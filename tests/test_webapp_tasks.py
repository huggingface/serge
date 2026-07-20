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

    def test_debug_max_tasks_caps_a_burst(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_write_config(webapp)
        webapp._task_debug_state.update(count=0, last=0.0)
        submitted = []

        def fake_submit(fn, job, worker_cfg, req):
            submitted.append(job.id)

        with (
            patch.dict(os.environ, {"TASK_DEBUG_MAX_TASKS": "2"}),
            patch.object(webapp, "verify_token", return_value=_Claims()),
            patch.object(webapp._TASK_POOL, "submit", side_effect=fake_submit),
        ):
            client = TestClient(webapp.app)
            results = [
                client.post(
                    "/tasks",
                    json={"instruction": f"fix {i}", "context": ""},
                    headers={"Authorization": "Bearer tok"},
                )
                for i in range(3)
            ]

        # All three are accepted (202); only the first two actually queue a
        # worker — the third is a no-op skip.
        self.assertEqual([r.status_code for r in results], [202, 202, 202])
        self.assertEqual(len(submitted), 2)
        self.assertNotIn("skipped", results[0].json())
        self.assertNotIn("skipped", results[1].json())
        self.assertTrue(results[2].json().get("skipped"))

    def _seed_task_job(self, webapp, *, status="no_fix", repo="acme/widgets"):
        owner, name = repo.split("/", 1)
        webapp._store.insert_job(
            id="task-status-1",
            user="octocat",
            target_owner=owner,
            target_repo=name,
            target_number=0,
            trigger_comment="fix it",
            llm_provider="hf",
            llm_api_base=None,
            llm_model="m",
            created_at=1.0,
            status=status,
            source="task",
            kind="task",
            task_spec_json="{}",
        )

    def test_status_endpoint_oidc_returns_no_fix(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_task_job(webapp, status="no_fix")
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.get(
                "/tasks/acme/widgets/task-status-1/status",
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["id"], "task-status-1")
        self.assertEqual(body["status"], "no_fix")
        self.assertEqual(body["target"], "acme/widgets")

    def test_status_endpoint_rejects_foreign_repo_claim(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        self._seed_task_job(webapp, status="no_fix")
        # A token minted for a different repo must not read acme/widgets tasks.
        with patch.object(
            webapp, "verify_token", return_value=_Claims(repository="evil/repo")
        ):
            client = TestClient(webapp.app)
            r = client.get(
                "/tasks/acme/widgets/task-status-1/status",
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 403)

    def test_status_endpoint_404_when_api_disabled(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(task_api_enabled=False)
        with patch.object(webapp, "verify_token", return_value=_Claims()):
            client = TestClient(webapp.app)
            r = client.get(
                "/tasks/acme/widgets/task-status-1/status",
                headers={"Authorization": "Bearer tok"},
            )
        self.assertEqual(r.status_code, 404)

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
        # The per-candidate prepare+publish (incl. any GPU verify retry loop) is
        # stubbed to return the given result, so we only assert status mapping.
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
            patch.object(
                webapp, "prepare_and_publish_candidate", return_value=task_result
            ),
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
            llm_max_tokens=16384,
            tool_max_iterations=8,
            tool_max_iterations_strict=True,
            task_normalize_command=["make", "fix-repo"],
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
        # The resolved worker-Config subset (per-task caps + normalize settings)
        # is transmitted so the runner doesn't fall back to env defaults.
        self.assertEqual(spec["config"]["llm_max_tokens"], 16384)
        self.assertEqual(spec["config"]["tool_max_iterations"], 8)
        self.assertTrue(spec["config"]["tool_max_iterations_strict"])
        self.assertEqual(spec["config"]["task_normalize_command"], ["make", "fix-repo"])
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

    def test_launch_kubernetes_is_nonblocking(self):
        # The kubernetes backend creates the Job and returns immediately: it does
        # NOT block on the pod, NOT finalize, and leaves the callback token live
        # so the runner can report. The watcher/callback reconcile later.
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

        def fake_create(spec, opts, *, timeout):
            captured["spec"] = spec
            captured["opts"] = opts
            return "serge-task-xyz", "serge"

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(webapp, "create_kubernetes", side_effect=fake_create),
            patch.object(webapp, "_notify_task_finished") as notify,
            patch.object(webapp, "_persist_terminal") as persist,
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        opts = captured["opts"]
        self.assertEqual(opts.namespace, "serge")
        self.assertEqual(opts.service_account, "serge-task")
        self.assertEqual(opts.node_selector, {"pool": "tasks"})
        self.assertEqual(opts.proxy, "http://egress:3128")
        self.assertEqual(captured["spec"]["github_token"], "gh-token")
        # Non-blocking: still running, token live, no finalize yet.
        self.assertEqual(job.status, "running")
        self.assertIsNotNone(job.callback_token)
        notify.assert_not_called()
        persist.assert_not_called()
        # A pod task is tracked for the watcher, with the created Job's name.
        task = webapp._pod_tasks[job.id]
        self.assertEqual(task.backend, "kubernetes")
        self.assertEqual(task.job_name, "serge-task-xyz")
        self.assertEqual(task.namespace, "serge")

    def test_launch_kubernetes_surfaces_create_error(self):
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes")
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)

        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="gh-token"),
            patch.object(
                webapp,
                "create_kubernetes",
                side_effect=webapp.K8sSandboxError("no cluster"),
            ),
            patch.object(webapp, "_notify_task_finished") as notify,
            patch.object(webapp, "_persist_terminal") as persist,
        ):
            webapp._launch_task_pod(job, worker_cfg, req)

        # Launch failure finalizes immediately (notify + persist) and untracks.
        self.assertEqual(job.status, "error")
        self.assertEqual(job.error, "no cluster")
        notify.assert_called_once()
        persist.assert_called_once()
        self.assertNotIn(job.id, webapp._pod_tasks)

    def test_watcher_reconciles_crashed_pod(self):
        # A k8s Job that reached a terminal state while serge's job is still
        # "running" and no callback landed within the grace window → the watcher
        # marks it errored, finalizes once, and reaps the Job.
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes")
        job = self._make_job(webapp)
        worker_cfg, req = self._worker_cfg_and_req(webapp)
        task = webapp._PodTask(
            job=job,
            worker_cfg=worker_cfg,
            req=req,
            backend="kubernetes",
            job_name="serge-task-xyz",
            namespace="serge",
            deadline=webapp.time.monotonic() + 3600,
            terminal_since=webapp.time.monotonic() - 100,  # past the grace window
        )
        webapp._pod_tasks[job.id] = task

        with (
            patch.object(webapp, "poll_task_job", return_value="failed"),
            patch.object(webapp, "collect_task_result", return_value=(1, "boom")),
            patch.object(webapp, "cleanup_task_job") as cleanup,
            patch.object(webapp, "_notify_task_finished") as notify,
            patch.object(webapp, "_persist_terminal") as persist,
        ):
            webapp._reconcile_pod_task(task)

        self.assertEqual(job.status, "error")
        self.assertIsNone(job.callback_token)
        # The exit code AND the pod's log tail are surfaced on the job, so the
        # crash cause is visible without a live repro against the reaped pod.
        self.assertIn("exit code 1", job.error)
        self.assertIn("boom", job.error)
        notify.assert_called_once()
        persist.assert_called_once()
        cleanup.assert_called_once_with("serge-task-xyz", "serge")
        self.assertNotIn(job.id, webapp._pod_tasks)

    def test_watcher_reaps_finalized_pod(self):
        # Happy path: the callback already finalized. The watcher just reaps the
        # Job and untracks — no second notify/persist.
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes")
        job = self._make_job(webapp)
        job.status = "published"
        worker_cfg, req = self._worker_cfg_and_req(webapp)
        task = webapp._PodTask(
            job=job,
            worker_cfg=worker_cfg,
            req=req,
            backend="kubernetes",
            job_name="serge-task-xyz",
            namespace="serge",
            finalized=True,
        )
        webapp._pod_tasks[job.id] = task

        with (
            patch.object(webapp, "poll_task_job") as poll,
            patch.object(webapp, "cleanup_task_job") as cleanup,
        ):
            webapp._reconcile_pod_task(task)

        poll.assert_not_called()  # finalized → skip straight to reap
        cleanup.assert_called_once_with("serge-task-xyz", "serge")
        self.assertNotIn(job.id, webapp._pod_tasks)

    # --- /admin/pods -----------------------------------------------------
    def _register_task(self, webapp, *, backend, job_name, status="running"):
        job = self._make_job(webapp)
        job.status = status
        worker_cfg, req = self._worker_cfg_and_req(webapp)
        task = webapp._PodTask(
            job=job,
            worker_cfg=worker_cfg,
            req=req,
            backend=backend,
            job_name=job_name,
            namespace="serge",
        )
        webapp._pod_tasks[job.id] = task
        return job, task

    def test_admin_pods_requires_admin(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        # dev user is not in WEB_ADMIN_USERS → 403.
        webapp = self._import_webapp()
        client = TestClient(webapp.app)
        self.assertEqual(client.get("/admin/pods/data").status_code, 403)

    def test_admin_pods_lists_kubernetes_pods(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(
            TASK_EXECUTION="kubernetes",
            TASK_K8S_NAMESPACE="serge",
            WEB_ADMIN_USERS="dev",
        )
        job, _task = self._register_task(
            webapp, backend="kubernetes", job_name="serge-task-tracked"
        )
        # One pod matches the tracked task; one is an orphan (serge restarted).
        pods = [
            {
                "pod": "serge-task-tracked-abc",
                "job_name": "serge-task-tracked",
                "phase": "Running",
                "node": "node-1",
                "start_epoch": 1000.0,
            },
            {
                "pod": "serge-task-orphan-xyz",
                "job_name": "serge-task-orphan",
                "phase": "Running",
                "node": "node-2",
                "start_epoch": 900.0,
            },
        ]
        with patch.object(webapp, "list_task_pods", return_value=("serge", pods)):
            client = TestClient(webapp.app)
            r = client.get("/admin/pods/data")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["backend"], "kubernetes")
        self.assertIsNone(data["error"])
        rows = {row["job_name"]: row for row in data["pods"]}
        # Tracked pod is joined with serge's job → repo + user filled in.
        self.assertEqual(rows["serge-task-tracked"]["repo"], "acme/widgets")
        self.assertEqual(rows["serge-task-tracked"]["job_id"], job.id)
        self.assertEqual(rows["serge-task-tracked"]["phase"], "Running")
        # Orphan pod still shows, with k8s data but no serge context.
        self.assertEqual(rows["serge-task-orphan"]["node"], "node-2")
        self.assertEqual(rows["serge-task-orphan"]["repo"], "")

    def test_admin_pods_shows_just_launched_task_without_pod(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes", WEB_ADMIN_USERS="dev")
        self._register_task(webapp, backend="kubernetes", job_name="serge-task-new")
        # No pod in the cluster yet (just created).
        with patch.object(webapp, "list_task_pods", return_value=("serge", [])):
            client = TestClient(webapp.app)
            data = client.get("/admin/pods/data").json()
        names = [row["job_name"] for row in data["pods"]]
        self.assertIn("serge-task-new", names)

    def test_admin_pods_docker_backend_lists_tracked(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(TASK_EXECUTION="docker", WEB_ADMIN_USERS="dev")
        self._register_task(webapp, backend="docker", job_name="serge-task-dkr")
        client = TestClient(webapp.app)
        data = client.get("/admin/pods/data").json()
        self.assertEqual(data["backend"], "docker")
        self.assertEqual([r["job_name"] for r in data["pods"]], ["serge-task-dkr"])

    def test_admin_pods_kill(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(TASK_EXECUTION="kubernetes", WEB_ADMIN_USERS="dev")
        with patch.object(webapp, "cleanup_task_job") as cleanup:
            client = TestClient(webapp.app)
            r = client.post(
                "/admin/pods/kill",
                json={"job_name": "serge-task-x", "namespace": "serge"},
                headers={"origin": "http://testserver"},
            )
        self.assertEqual(r.status_code, 200)
        cleanup.assert_called_once_with("serge-task-x", "serge")

    # --- review pods (REVIEW_EXECUTION) ----------------------------------
    def _review_job(self, webapp):
        job = webapp.Job(
            id="rev123abc456",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=7,
            trigger_comment="@serge review",
            llm_provider="hf",
            llm_api_base="https://example.com/v1",
            llm_model="model",
            created_at=0,
            status="running",
            source="web",
            kind="review",
            callback_token="cbtok",
        )
        job.loop = None
        with webapp._jobs_lock:
            webapp._jobs[job.id] = job
        return job

    def test_launch_review_pod_kubernetes_is_nonblocking(self):
        webapp = self._import_webapp(
            REVIEW_EXECUTION="kubernetes", TASK_RUNNER_IMAGE="serge/runner:latest"
        )
        job = self._review_job(webapp)
        req = webapp.ReviewRequest(
            owner="acme",
            repo="widgets",
            number=7,
            trigger_comment_id=0,
            trigger_comment_body="@serge review",
            commenter="octocat",
        )
        captured = {}

        def fake_create(spec, opts, *, timeout):
            captured["spec"] = spec
            return "serge-task-rev", "serge"

        with (
            patch.object(webapp, "create_kubernetes", side_effect=fake_create),
            patch.object(webapp, "_notify_task_finished") as notify,
            patch.object(webapp, "_persist_terminal") as persist,
        ):
            webapp._launch_review_pod(job, webapp.cfg, req, "gh-token")

        self.assertEqual(captured["spec"]["request_type"], "review")
        self.assertEqual(captured["spec"]["github_token"], "gh-token")
        # Non-blocking: still running, token live, not finalized.
        self.assertEqual(job.status, "running")
        self.assertIsNotNone(job.callback_token)
        notify.assert_not_called()
        persist.assert_not_called()
        task = webapp._pod_tasks[job.id]
        self.assertEqual(task.backend, "kubernetes")
        self.assertEqual(task.job_name, "serge-task-rev")

    def test_finalize_review_does_not_slack_notify(self):
        # Reviews carry a ReviewRequest (no slack fields); finalize must persist
        # but never call the task slack notifier.
        webapp = self._import_webapp(REVIEW_EXECUTION="kubernetes")
        job = self._review_job(webapp)
        req = webapp.ReviewRequest(
            owner="acme",
            repo="widgets",
            number=7,
            trigger_comment_id=0,
            trigger_comment_body="x",
            commenter="octocat",
        )
        webapp._pod_tasks[job.id] = webapp._PodTask(
            job=job, worker_cfg=webapp.cfg, req=req, backend="kubernetes"
        )
        with (
            patch.object(webapp, "_notify_task_finished") as notify,
            patch.object(webapp, "_persist_terminal") as persist,
        ):
            self.assertTrue(webapp._finalize_task(job.id))
        notify.assert_not_called()
        persist.assert_called_once()

    def test_ingest_review_reconstructs_draft(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(REVIEW_EXECUTION="kubernetes")
        job = self._review_job(webapp)
        # Encode a draft the way the review pod would.
        draft = webapp.decode_draft(
            '{"owner":"acme","repo":"widgets","number":7,"head_sha":"abc",'
            '"summary":"looks fine","event":"COMMENT","comments":[]}'
        )
        from reviewbot.store import encode_draft

        with (
            patch.object(webapp, "_persist_terminal"),
            patch.object(webapp, "_notify_task_finished"),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                f"/internal/tasks/{job.id}/events",
                json={
                    "terminal": {
                        "status": "done",
                        "result": {
                            "draft": encode_draft(draft),
                            "prompt_tokens": 100,
                            "completion_tokens": 25,
                        },
                    }
                },
                headers={"Authorization": "Bearer cbtok"},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(job.status, "done")
        self.assertIsNotNone(job.draft)
        self.assertEqual(job.draft.summary, "looks fine")
        self.assertEqual(job.draft.prompt_tokens, 100)
        self.assertEqual(job.draft.completion_tokens, 25)

    def _webhook_review_job(self, webapp):
        job = self._review_job(webapp)
        job.source = "webhook"
        req = webapp.ReviewRequest(
            owner="acme",
            repo="widgets",
            number=7,
            trigger_comment_id=0,
            trigger_comment_body="@serge review",
            commenter="octocat",
        )
        webapp._pod_tasks[job.id] = webapp._PodTask(
            job=job, worker_cfg=webapp.cfg, req=req, backend="kubernetes"
        )
        return job

    def test_webhook_review_auto_publishes_on_callback(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(REVIEW_EXECUTION="kubernetes")
        job = self._webhook_review_job(webapp)
        from reviewbot.store import encode_draft

        draft = webapp.decode_draft(
            '{"owner":"acme","repo":"widgets","number":7,"head_sha":"abc",'
            '"summary":"lgtm","event":"COMMENT","comments":[]}'
        )
        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="tok"),
            patch.object(webapp, "GitHubClient", lambda *a, **k: object()),
            patch.object(webapp, "publish_review", return_value=draft) as pub,
            patch.object(webapp, "_persist_terminal"),
            patch.object(webapp, "_notify_task_finished"),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                f"/internal/tasks/{job.id}/events",
                json={
                    "terminal": {
                        "status": "done",
                        "result": {"draft": encode_draft(draft)},
                    }
                },
                headers={"Authorization": "Bearer cbtok"},
            )
        self.assertEqual(r.status_code, 200)
        pub.assert_called_once()
        self.assertEqual(job.status, "published")
        self.assertIsNotNone(job.published_draft)

    def test_webhook_review_failure_posts_comment(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp(REVIEW_EXECUTION="kubernetes")
        job = self._webhook_review_job(webapp)
        with (
            patch.object(webapp, "installation_id_for_repo", return_value=1),
            patch.object(webapp, "installation_token", return_value="tok"),
            patch.object(webapp, "GitHubClient", lambda *a, **k: object()),
            patch.object(webapp, "_post_webhook_failure_comment") as comment,
            patch.object(webapp, "publish_review") as pub,
            patch.object(webapp, "_persist_terminal"),
            patch.object(webapp, "_notify_task_finished"),
        ):
            client = TestClient(webapp.app)
            r = client.post(
                f"/internal/tasks/{job.id}/events",
                json={"terminal": {"status": "error", "error": "model exploded"}},
                headers={"Authorization": "Bearer cbtok"},
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(job.status, "error")
        comment.assert_called_once()
        pub.assert_not_called()

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

    def test_task_info_includes_filtered_trace(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        webapp = self._import_webapp()
        job = self._make_job(webapp, status="error")
        webapp._push_event(job, "step", "normalize")
        webapp._push_event(job, "log", "Validating the patch with `make style`…")
        webapp._push_event(job, "normalize_error", "Normalizer failed:\nactual error")
        webapp._push_event(job, "token", "noisy")
        job.error = "task runner exited without reporting (exit code 1)"
        client = TestClient(webapp.app)

        r = client.get("/tasks/acme/widgets/job123abc456/info")

        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(
            [(e["kind"], e["text"]) for e in data["trace"]],
            [
                ("step", "normalize"),
                ("log", "Validating the patch with `make style`…"),
                ("normalize_error", "Normalizer failed:\nactual error"),
            ],
        )

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
