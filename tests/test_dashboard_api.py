"""Tests for the dashboard API (/dashboard/*): the service token, the acting
user's write check, and the allowlists on cancel and dispatch. GitHub is a fake;
no network is touched. Also pins that /tasks is unaffected by the new config."""

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from reviewbot import dashboard_api

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None

REPO = "huggingface/transformers"
CANCEL = (".github/workflows/pr-ci-caller.yml", ".github/workflows/self-comment-ci.yml")
DISPATCH = ("rerun-failed-cpu.yml", "rerun-failed-gpu.yml")


class FakeGitHub:
    def __init__(self, role="write", run=None, cancel_status=202):
        self.role = role
        self.run = run or {
            "path": ".github/workflows/pr-ci-caller.yml",
            "status": "in_progress",
        }
        self.cancel_status = cancel_status
        self.role_calls = 0
        self.cancelled = []
        self.dispatched = []

    def collaborator_role(self, owner, repo, login):
        self.role_calls += 1
        return self.role

    def get_workflow_run(self, owner, repo, run_id):
        return self.run

    def cancel_workflow_run(self, owner, repo, run_id):
        self.cancelled.append(run_id)
        return self.cancel_status

    def get_repo(self, owner, repo):
        return {"default_branch": "main"}

    def dispatch_workflow(self, owner, repo, workflow_file, *, ref, inputs):
        self.dispatched.append((workflow_file, ref, inputs))


def _target(actor="maintainer"):
    return dashboard_api.parse_target({"repository": REPO, "actor": actor}, (REPO,))


class DashboardApiTests(unittest.TestCase):
    def test_service_token(self):
        with self.assertRaises(dashboard_api.DashboardError) as off:
            dashboard_api.authorize_service("", "Bearer anything")
        self.assertEqual(off.exception.status, 404)
        for header in ("", "Basic x", "Bearer wrong"):
            with self.assertRaises(dashboard_api.DashboardError) as bad:
                dashboard_api.authorize_service("secret", header)
            self.assertEqual(bad.exception.status, 401)
        dashboard_api.authorize_service("secret", "Bearer secret")

    def test_target_is_allowlisted_and_validated(self):
        self.assertEqual(_target().repository, REPO)
        for body, status in (
            ({"repository": "acme/widgets", "actor": "a"}, 403),
            ({"repository": "no-slash", "actor": "a"}, 400),
            ({"repository": "../x", "actor": "a"}, 403),
            ({"repository": REPO, "actor": "bad login"}, 400),
            ({"repository": REPO}, 400),
            ("nope", 400),
        ):
            with self.assertRaises(dashboard_api.DashboardError) as error:
                dashboard_api.parse_target(body, (REPO,))
            self.assertEqual(error.exception.status, status, body)

    def test_permission_is_cached(self):
        gh, cache = FakeGitHub(role="maintain"), dashboard_api.RoleCache()
        self.assertEqual(
            dashboard_api.permission(gh, _target(), cache),
            {"actor": "maintainer", "role": "maintain", "can_write": True},
        )
        dashboard_api.permission(gh, _target(), cache)
        self.assertEqual(gh.role_calls, 1)
        self.assertFalse(
            dashboard_api.permission(FakeGitHub(role="triage"), _target("x"), cache)[
                "can_write"
            ]
        )

    def _cancel(self, gh, run_id=120):
        return dashboard_api.cancel_run(
            gh,
            _target(),
            {"run_id": run_id},
            cancel_workflows=CANCEL,
            cache=dashboard_api.RoleCache(),
        )

    def test_cancel_only_allowlisted_workflows_for_writers(self):
        gh = FakeGitHub()
        self.assertEqual(
            self._cancel(gh),
            {"run_id": 120, "github_status": 202, "run_status": "in_progress"},
        )
        self.assertEqual(gh.cancelled, [120])

        other = FakeGitHub(
            run={"path": ".github/workflows/release.yml", "status": "queued"}
        )
        with self.assertRaises(dashboard_api.DashboardError) as error:
            self._cancel(other)
        self.assertEqual(error.exception.detail, "workflow_not_cancellable")
        self.assertEqual(other.cancelled, [])

        reader = FakeGitHub(role="read")
        with self.assertRaises(dashboard_api.DashboardError) as error:
            self._cancel(reader)
        self.assertEqual(error.exception.status, 403)
        self.assertEqual(reader.cancelled, [])

        for run_id in (0, -1, "abc", True, None):
            with self.assertRaises(dashboard_api.DashboardError):
                self._cancel(FakeGitHub(), run_id)

    def test_cancel_reports_finishing_and_finished_runs(self):
        finishing = FakeGitHub(cancel_status=409)
        self.assertEqual(self._cancel(finishing)["github_status"], 409)
        done = FakeGitHub(run={"path": CANCEL[1], "status": "completed"})
        self.assertEqual(self._cancel(done)["github_status"], None)
        self.assertEqual(done.cancelled, [])
        with self.assertRaises(dashboard_api.DashboardError) as error:
            self._cancel(FakeGitHub(cancel_status=500))
        self.assertEqual(error.exception.status, 502)

    def _dispatch(self, gh, workflow="rerun-failed-cpu.yml", inputs=None):
        return dashboard_api.dispatch_workflow(
            gh,
            _target(),
            {
                "workflow": workflow,
                "inputs": {"pr_number": "42"} if inputs is None else inputs,
            },
            dispatch_workflows=DISPATCH,
            cache=dashboard_api.RoleCache(),
        )

    def test_dispatch_only_allowlisted_files_on_the_default_branch(self):
        gh = FakeGitHub()
        self.assertEqual(
            self._dispatch(gh),
            {"workflow": "rerun-failed-cpu.yml", "ref": "main", "dispatched": True},
        )
        self.assertEqual(
            gh.dispatched, [("rerun-failed-cpu.yml", "main", {"pr_number": "42"})]
        )
        for workflow, status in (
            ("serge-verify-caller.yml", 403),
            ("../x.yml", 400),
            ("rerun-failed-cpu", 400),
        ):
            with self.assertRaises(dashboard_api.DashboardError) as error:
                self._dispatch(FakeGitHub(), workflow)
            self.assertEqual(error.exception.status, status, workflow)
        for inputs in (
            {"n": 1},
            {"bad name": "x"},
            {f"k{i}": "x" for i in range(11)},
            {"a": "x" * 65_001},
            "x",
        ):
            with self.assertRaises(dashboard_api.DashboardError):
                self._dispatch(FakeGitHub(), inputs=inputs)
        reader = FakeGitHub(role="triage")
        with self.assertRaises(dashboard_api.DashboardError):
            self._dispatch(reader)
        self.assertEqual(reader.dispatched, [])


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        if TestClient is None:
            self.skipTest("fastapi not installed")
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        sys.modules.pop("reviewbot.webapp", None)

    def _webapp(self, **extra):
        env = {
            "DEV_NO_AUTH": "1",
            "GITHUB_APP_ID": "123",
            "GITHUB_PRIVATE_KEY": "dummy-private-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "LLM_API_KEY": "llm-token",
            "WEB_STORE_PATH": os.path.join(self.tmpdir, "jobs.db"),
            "WEB_CLONE_CACHE_DIR": os.path.join(self.tmpdir, "clones"),
            "TASK_API_ENABLED": "1",
            "TASK_OIDC_AUDIENCE": "serge",
            **extra,
        }
        with patch.dict(os.environ, env, clear=True):
            return importlib.import_module("reviewbot.webapp")

    def test_routes_are_404_without_a_token(self):
        client = TestClient(self._webapp().app)
        for path in (
            "/dashboard/permission",
            "/dashboard/runs/cancel",
            "/dashboard/workflows/dispatch",
        ):
            r = client.post(path, json={"repository": REPO, "actor": "a"})
            self.assertEqual(r.status_code, 404, path)

    def test_cancel_route_end_to_end(self):
        webapp = self._webapp(DASHBOARD_API_TOKEN="secret")
        gh = FakeGitHub()
        client = TestClient(webapp.app)
        with (
            patch.object(webapp, "installation_token_for_repo", return_value="t"),
            patch.object(webapp, "GitHubClient", return_value=gh),
        ):
            body = {"repository": REPO, "actor": "maintainer", "run_id": "120"}
            self.assertEqual(
                client.post("/dashboard/runs/cancel", json=body).status_code, 401
            )
            r = client.post(
                "/dashboard/runs/cancel",
                json=body,
                headers={"Authorization": "Bearer wrong"},
            )
            self.assertEqual(r.status_code, 401)
            r = client.post(
                "/dashboard/runs/cancel",
                json=body,
                headers={"Authorization": "Bearer secret"},
            )
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["github_status"], 202)
            r = client.post(
                "/dashboard/workflows/dispatch",
                json={
                    "repository": REPO,
                    "actor": "maintainer",
                    "workflow": "rerun-failed-gpu.yml",
                    "inputs": {"pr_number": "42"},
                },
                headers={"Authorization": "Bearer secret"},
            )
            self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(gh.cancelled, [120])
        self.assertEqual(gh.dispatched[0][0], "rerun-failed-gpu.yml")

    def test_tasks_still_requires_oidc_with_the_dashboard_token_set(self):
        # The dashboard token is not a /tasks credential: backward compatible.
        client = TestClient(self._webapp(DASHBOARD_API_TOKEN="secret").app)
        r = client.post(
            "/tasks",
            json={"instruction": "x"},
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(r.status_code, 401)
        self.assertIn("oidc_verification_failed", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
