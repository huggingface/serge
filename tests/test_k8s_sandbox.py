"""Tests for the kubernetes normalize backend (reviewbot/k8s_sandbox.py).

The pure manifest/helper logic is tested directly. The Job orchestration
(create -> poll -> logs -> delete) is exercised with fake ``kubernetes`` modules
injected into ``sys.modules``, since the real client is an optional extra and
the live path only runs in-cluster."""

import sys
import types
import unittest
from unittest import mock

from reviewbot import k8s_sandbox
from reviewbot.k8s_sandbox import (
    K8sSandboxError,
    K8sSettings,
    resolve_namespace,
)


def _settings(**kw):
    base = dict(
        namespace="serge",
        service_account="serge-task",
    )
    base.update(kw)
    return K8sSettings(**base)


class HelperTests(unittest.TestCase):
    def test_resolve_namespace_prefers_explicit(self):
        self.assertEqual(resolve_namespace(_settings(namespace="explicit")), "explicit")

    def test_resolve_namespace_reads_incluster_file(self):
        m = mock.mock_open(read_data="from-file\n")
        with mock.patch("builtins.open", m):
            self.assertEqual(resolve_namespace(_settings(namespace=None)), "from-file")

    def test_resolve_namespace_raises_when_unavailable(self):
        with mock.patch("builtins.open", side_effect=OSError):
            with self.assertRaises(K8sSandboxError):
                resolve_namespace(_settings(namespace=None))

    def test_job_terminal_states(self):
        succeeded = types.SimpleNamespace(succeeded=1, failed=None, conditions=None)
        failed = types.SimpleNamespace(succeeded=None, failed=1, conditions=None)
        running = types.SimpleNamespace(succeeded=None, failed=None, conditions=None)
        self.assertEqual(k8s_sandbox._job_terminal(succeeded), "succeeded")
        self.assertEqual(k8s_sandbox._job_terminal(failed), "failed")
        self.assertIsNone(k8s_sandbox._job_terminal(running))
        self.assertIsNone(k8s_sandbox._job_terminal(None))


class _FakeApiException(Exception):
    def __init__(self, reason="boom"):
        super().__init__(reason)
        self.reason = reason


def _install_fake_kubernetes():
    """Inject minimal fake ``kubernetes`` modules so the lazy imports inside
    run_task_job resolve without the real client installed. Returns a cleanup fn."""
    saved = {
        k: sys.modules.get(k)
        for k in ("kubernetes", "kubernetes.client", "kubernetes.client.rest")
    }

    kmod = types.ModuleType("kubernetes")
    cmod = types.ModuleType("kubernetes.client")
    rmod = types.ModuleType("kubernetes.client.rest")
    rmod.ApiException = _FakeApiException
    cmod.rest = rmod
    cmod.V1DeleteOptions = lambda **kw: ("DeleteOptions", kw)
    kmod.client = cmod
    kmod.config = types.SimpleNamespace(
        load_incluster_config=lambda: None, load_kube_config=lambda: None
    )
    sys.modules["kubernetes"] = kmod
    sys.modules["kubernetes.client"] = cmod
    sys.modules["kubernetes.client.rest"] = rmod

    def cleanup():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return cleanup


def _pod_with(exit_code):
    term = types.SimpleNamespace(exit_code=exit_code)
    cs = types.SimpleNamespace(state=types.SimpleNamespace(terminated=term))
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="pod-1"),
        status=types.SimpleNamespace(container_statuses=[cs]),
    )


class TaskJobManifestTests(unittest.TestCase):
    """The per-task-pod Job + Secret builders (TASK_EXECUTION=kubernetes)."""

    def _job(self, **kw):
        params = dict(
            image="serge/runner:latest",
            job_name="serge-task-abc-1234",
            secret_name="serge-task-abc-1234",
            settings=K8sSettings(namespace="serge"),
            timeout=1800,
        )
        params.update(kw)
        return k8s_sandbox.build_task_job_manifest(**params)

    def test_name_is_dns1123_and_bounded(self):
        name = k8s_sandbox.make_task_job_name("Job_ID/With..Junk")
        self.assertTrue(name.startswith("serge-task-"))
        self.assertLessEqual(len(name), 63)
        self.assertRegex(name, r"^[a-z0-9-]+$")

    def test_labels_and_no_sa_token(self):
        m = self._job()
        pod = m["spec"]["template"]["spec"]
        self.assertEqual(
            m["spec"]["template"]["metadata"]["labels"]["serge.io/task-pod"], "true"
        )
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(m["spec"]["backoffLimit"], 0)
        self.assertEqual(m["spec"]["activeDeadlineSeconds"], 1800)

    def test_secret_mounted_at_spec_path(self):
        m = self._job()
        pod = m["spec"]["template"]["spec"]
        vol = next(v for v in pod["volumes"] if v["name"] == "task-spec")
        self.assertEqual(vol["secret"]["secretName"], "serge-task-abc-1234")
        mount = next(
            vm
            for vm in pod["containers"][0]["volumeMounts"]
            if vm["name"] == "task-spec"
        )
        self.assertEqual(mount["mountPath"], "/etc/serge")
        self.assertTrue(mount["readOnly"])

    def test_proxy_and_no_proxy_env(self):
        m = self._job(proxy="http://egress:3128", no_proxy=".svc.cluster.local")
        env = {
            e["name"]: e["value"]
            for e in m["spec"]["template"]["spec"]["containers"][0]["env"]
        }
        self.assertEqual(env["HTTPS_PROXY"], "http://egress:3128")
        self.assertEqual(env["HTTP_PROXY"], "http://egress:3128")
        self.assertEqual(env["NO_PROXY"], ".svc.cluster.local")
        self.assertEqual(env["SERGE_TASK_SPEC"], "/etc/serge/task.json")
        self.assertEqual(env["WEB_CLONE_CACHE_DIR"], "/tmp/serge-clones")

    def test_service_account_and_node_selector(self):
        m = self._job(
            settings=K8sSettings(
                namespace="serge",
                service_account="serge-task",
                node_selector={"pool": "tasks"},
            )
        )
        pod = m["spec"]["template"]["spec"]
        self.assertEqual(pod["serviceAccountName"], "serge-task")
        self.assertEqual(pod["nodeSelector"], {"pool": "tasks"})

    def test_memory_limit(self):
        m = self._job(memory="4Gi")
        res = m["spec"]["template"]["spec"]["containers"][0]["resources"]
        self.assertEqual(res["limits"]["memory"], "4Gi")

    def test_gpu_resources_and_tolerations(self):
        # issue #20: GPU tasks reserve the extended resource and tolerate the
        # GPU node taint.
        tol = [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
        m = self._job(
            memory="64Gi",
            gpu_resource="nvidia.com/gpu",
            gpu_count=2,
            settings=K8sSettings(
                namespace="serge",
                node_selector={"gpu": "a100"},
                tolerations=tol,
            ),
        )
        pod = m["spec"]["template"]["spec"]
        limits = pod["containers"][0]["resources"]["limits"]
        self.assertEqual(limits["memory"], "64Gi")
        self.assertEqual(limits["nvidia.com/gpu"], "2")
        self.assertEqual(pod["nodeSelector"], {"gpu": "a100"})
        self.assertEqual(pod["tolerations"], tol)

    def test_no_gpu_resource_without_count(self):
        m = self._job()
        self.assertNotIn("resources", m["spec"]["template"]["spec"]["containers"][0])
        self.assertNotIn("tolerations", m["spec"]["template"]["spec"])

    def test_missing_image_raises(self):
        with self.assertRaises(K8sSandboxError):
            self._job(image="")

    def test_secret_manifest_owner_ref_and_data(self):
        import base64
        import json

        spec = {"job_id": "j1", "github_token": "tok"}
        m = k8s_sandbox.build_task_secret_manifest(
            name="serge-task-j1-abcd",
            spec_json=json.dumps(spec),
            job_name="serge-task-j1-abcd",
            job_uid="uid-123",
            namespace="serge",
        )
        self.assertEqual(m["kind"], "Secret")
        owner = m["metadata"]["ownerReferences"][0]
        self.assertEqual(owner["kind"], "Job")
        self.assertEqual(owner["uid"], "uid-123")
        self.assertTrue(owner["controller"])
        decoded = base64.b64decode(m["data"]["task.json"]).decode()
        self.assertEqual(json.loads(decoded), spec)


class RunTaskJobTests(unittest.TestCase):
    def setUp(self):
        self.cleanup = _install_fake_kubernetes()
        self.addCleanup(self.cleanup)

    def _clients(self, *, statuses, pod, log_text="ok"):
        batch = mock.Mock()
        batch.create_namespaced_job.return_value = types.SimpleNamespace(
            metadata=types.SimpleNamespace(uid="job-uid-1")
        )
        batch.read_namespaced_job_status.side_effect = [
            types.SimpleNamespace(status=s) for s in statuses
        ]
        core = mock.Mock()
        core.list_namespaced_pod.return_value = types.SimpleNamespace(items=[pod])
        core.read_namespaced_pod_log.return_value = log_text
        return batch, core

    def test_creates_secret_with_job_owner_and_deletes_both(self):
        running = types.SimpleNamespace(succeeded=None, failed=None, conditions=None)
        done = types.SimpleNamespace(succeeded=1, failed=None, conditions=None)
        batch, core = self._clients(statuses=[running, done], pod=_pod_with(0))
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            rc, tail = k8s_sandbox.run_task_job(
                {"job_id": "j1", "github_token": "tok"},
                image="serge/runner:latest",
                settings=K8sSettings(namespace="serge"),
                timeout=30,
                proxy="http://egress:3128",
                poll_interval=0.0,
            )
        self.assertEqual(rc, 0)
        # Job created before the Secret; Secret carries the Job's uid as owner.
        self.assertTrue(batch.create_namespaced_job.called)
        self.assertTrue(core.create_namespaced_secret.called)
        secret = core.create_namespaced_secret.call_args.args[1]
        self.assertEqual(secret["metadata"]["ownerReferences"][0]["uid"], "job-uid-1")
        # Both are cleaned up.
        self.assertTrue(batch.delete_namespaced_job.called)
        self.assertTrue(core.delete_namespaced_secret.called)

    def test_secret_create_failure_deletes_job(self):
        done = types.SimpleNamespace(succeeded=1, failed=None, conditions=None)
        batch, core = self._clients(statuses=[done], pod=_pod_with(0))
        core.create_namespaced_secret.side_effect = _FakeApiException("forbidden")
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            with self.assertRaises(K8sSandboxError):
                k8s_sandbox.run_task_job(
                    {"job_id": "j1", "github_token": "tok"},
                    image="serge/runner:latest",
                    settings=K8sSettings(namespace="serge"),
                    timeout=30,
                    poll_interval=0.0,
                )
        self.assertTrue(batch.delete_namespaced_job.called)


class NonBlockingLaunchTests(unittest.TestCase):
    """The non-blocking primitives (SERGE_ORCHESTRATOR_PODS_PLAN.md Phase 1):
    create_task_job / poll_task_job / collect_task_result / cleanup_task_job."""

    def setUp(self):
        self.addCleanup(_install_fake_kubernetes())

    def _clients(self):
        batch = mock.Mock()
        batch.create_namespaced_job.return_value = types.SimpleNamespace(
            metadata=types.SimpleNamespace(uid="job-uid-1")
        )
        core = mock.Mock()
        return batch, core

    def test_create_returns_without_waiting(self):
        batch, core = self._clients()
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            name, ns = k8s_sandbox.create_task_job(
                {"job_id": "j1", "github_token": "tok"},
                image="serge/runner:latest",
                settings=K8sSettings(namespace="serge"),
                timeout=30,
            )
        self.assertEqual(ns, "serge")
        self.assertTrue(name.startswith("serge-task-"))
        self.assertTrue(batch.create_namespaced_job.called)
        self.assertTrue(core.create_namespaced_secret.called)
        # Crucially, it never polls or deletes — the watcher owns that.
        self.assertFalse(batch.read_namespaced_job_status.called)
        self.assertFalse(batch.delete_namespaced_job.called)

    def test_poll_running_then_terminal(self):
        batch, core = self._clients()
        running = types.SimpleNamespace(succeeded=None, failed=None, conditions=None)
        done = types.SimpleNamespace(succeeded=1, failed=None, conditions=None)
        batch.read_namespaced_job_status.side_effect = [
            types.SimpleNamespace(status=running),
            types.SimpleNamespace(status=done),
        ]
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            self.assertIsNone(k8s_sandbox.poll_task_job("j", "serge"))
            self.assertEqual(k8s_sandbox.poll_task_job("j", "serge"), "succeeded")

    def test_poll_treats_404_as_failed(self):
        batch, core = self._clients()
        exc = _FakeApiException("gone")
        exc.status = 404
        batch.read_namespaced_job_status.side_effect = exc
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            self.assertEqual(k8s_sandbox.poll_task_job("j", "serge"), "failed")

    def test_cleanup_deletes_job_and_secret(self):
        batch, core = self._clients()
        with mock.patch.object(
            k8s_sandbox, "_load_clients", return_value=(batch, core)
        ):
            k8s_sandbox.cleanup_task_job("serge-task-x", "serge")
        self.assertTrue(batch.delete_namespaced_job.called)
        self.assertTrue(core.delete_namespaced_secret.called)


if __name__ == "__main__":
    unittest.main()
