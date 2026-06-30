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
    build_job_manifest,
    make_job_name,
    resolve_namespace,
)


def _settings(**kw):
    base = dict(
        worktree_pvc="serge-worktrees",
        worktree_volume_root="/data/clones",
        namespace="serge",
        service_account="serge-task",
    )
    base.update(kw)
    return K8sSettings(**base)


class ManifestTests(unittest.TestCase):
    def _build(self, **kw):
        params = dict(
            command=["make", "style"],
            image="quality:1",
            workdir="/data/clones/worktrees/wt1",
            write_root="/data/clones/worktrees/wt1",
            job_name="serge-nrm-wt1-abcd",
            settings=_settings(),
            uid=1000,
            gid=1000,
            timeout=600,
        )
        params.update(kw)
        command = params.pop("command")
        return build_job_manifest(command, **params)

    def test_basic_shape_and_isolation(self):
        m = self._build(memory="4Gi")
        self.assertEqual(m["apiVersion"], "batch/v1")
        self.assertEqual(m["kind"], "Job")
        spec = m["spec"]
        self.assertEqual(spec["backoffLimit"], 0)
        self.assertEqual(spec["activeDeadlineSeconds"], 600)
        pod = spec["template"]["spec"]
        self.assertEqual(pod["restartPolicy"], "Never")
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(pod["serviceAccountName"], "serge-task")
        # deny-all-egress NetworkPolicy selector label
        self.assertEqual(
            spec["template"]["metadata"]["labels"][k8s_sandbox.SANDBOX_LABEL_KEY],
            k8s_sandbox.SANDBOX_LABEL_VALUE,
        )
        c = pod["containers"][0]
        self.assertEqual(c["command"], ["make", "style"])
        self.assertEqual(c["workingDir"], "/data/clones/worktrees/wt1")
        self.assertEqual(c["resources"]["limits"]["memory"], "4Gi")
        sc = c["securityContext"]
        self.assertFalse(sc["allowPrivilegeEscalation"])
        self.assertTrue(sc["readOnlyRootFilesystem"])
        self.assertEqual(sc["capabilities"]["drop"], ["ALL"])
        self.assertTrue(pod["securityContext"]["runAsNonRoot"])
        self.assertEqual(pod["securityContext"]["runAsUser"], 1000)

    def test_worktree_mounted_by_subpath_at_write_root(self):
        m = self._build(
            workdir="/data/clones/worktrees/acme__w__job7",
            write_root="/data/clones/worktrees/acme__w__job7",
        )
        mounts = m["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        wt = next(v for v in mounts if v["name"] == "worktree")
        # only the worktree subtree is exposed, at the same absolute path
        self.assertEqual(wt["mountPath"], "/data/clones/worktrees/acme__w__job7")
        self.assertEqual(wt["subPath"], "worktrees/acme__w__job7")
        vols = {v["name"]: v for v in m["spec"]["template"]["spec"]["volumes"]}
        self.assertEqual(
            vols["worktree"]["persistentVolumeClaim"]["claimName"], "serge-worktrees"
        )
        self.assertIn("emptyDir", vols["tmp"])

    def test_root_uid_omits_run_as_non_root(self):
        # serge running as uid 0 must not assert runAsNonRoot (would contradict
        # the explicit runAsUser and be rejected by the kubelet).
        m = self._build(uid=0, gid=0)
        self.assertNotIn("runAsNonRoot", m["spec"]["template"]["spec"]["securityContext"])

    def test_missing_image_pvc_or_root_raise(self):
        with self.assertRaises(K8sSandboxError):
            self._build(image="")
        with self.assertRaises(K8sSandboxError):
            self._build(settings=_settings(worktree_pvc=None))
        with self.assertRaises(K8sSandboxError):
            self._build(settings=_settings(worktree_volume_root=None))

    def test_worktree_outside_volume_root_rejected(self):
        with self.assertRaises(K8sSandboxError):
            self._build(
                write_root="/somewhere/else/wt1",
                workdir="/somewhere/else/wt1",
            )


class HelperTests(unittest.TestCase):
    def test_job_name_is_dns1123_and_unique(self):
        name = make_job_name("/data/clones/worktrees/Acme__Widget__job1")
        self.assertLessEqual(len(name), 63)
        self.assertTrue(all(c.islower() or c.isdigit() or c == "-" for c in name))
        self.assertFalse(name.startswith("-") or name.endswith("-"))
        self.assertNotEqual(name, make_job_name("/data/clones/worktrees/Acme__Widget__job1"))

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
    run_job resolve without the real client installed. Returns a cleanup fn."""
    saved = {k: sys.modules.get(k) for k in (
        "kubernetes", "kubernetes.client", "kubernetes.client.rest")}

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


class RunJobTests(unittest.TestCase):
    def setUp(self):
        self.cleanup = _install_fake_kubernetes()
        self.addCleanup(self.cleanup)

    def _clients(self, *, statuses, pod, log_text="ok-log"):
        """Build fake BatchV1Api/CoreV1Api. ``statuses`` is the sequence of
        ``.status`` objects read_namespaced_job_status yields per poll."""
        batch = mock.Mock()
        batch.create_namespaced_job.return_value = None
        batch.read_namespaced_job_status.side_effect = [
            types.SimpleNamespace(status=s) for s in statuses
        ]
        core = mock.Mock()
        core.list_namespaced_pod.return_value = types.SimpleNamespace(items=[pod])
        core.read_namespaced_pod_log.return_value = log_text
        return batch, core

    def test_success_returns_exit_code_and_logs_and_deletes(self):
        running = types.SimpleNamespace(succeeded=None, failed=None, conditions=None)
        done = types.SimpleNamespace(succeeded=1, failed=None, conditions=None)
        batch, core = self._clients(
            statuses=[running, done], pod=_pod_with(0), log_text="a\nb\nc"
        )
        with mock.patch.object(k8s_sandbox, "_load_clients", return_value=(batch, core)):
            rc, tail = k8s_sandbox.run_job(
                ["make", "style"],
                image="img:1",
                workdir="/data/clones/worktrees/wt1",
                write_root="/data/clones/worktrees/wt1",
                settings=_settings(),
                uid=1000,
                gid=1000,
                timeout=30,
                poll_interval=0.0,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(tail, "a\nb\nc")
        self.assertTrue(batch.create_namespaced_job.called)
        self.assertTrue(batch.delete_namespaced_job.called)

    def test_failed_job_surfaces_container_exit_code(self):
        done = types.SimpleNamespace(succeeded=None, failed=1, conditions=None)
        batch, core = self._clients(statuses=[done], pod=_pod_with(2))
        with mock.patch.object(k8s_sandbox, "_load_clients", return_value=(batch, core)):
            rc, _ = k8s_sandbox.run_job(
                ["make", "style"],
                image="img:1",
                workdir="/data/clones/worktrees/wt1",
                write_root="/data/clones/worktrees/wt1",
                settings=_settings(),
                uid=1000,
                gid=1000,
                timeout=30,
                poll_interval=0.0,
            )
        self.assertEqual(rc, 2)

    def test_failed_without_exit_code_defaults_nonzero(self):
        done = types.SimpleNamespace(succeeded=None, failed=1, conditions=None)
        # pod has no terminated container status (e.g. deadline exceeded)
        pod = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="pod-1"),
            status=types.SimpleNamespace(container_statuses=None),
        )
        batch, core = self._clients(statuses=[done], pod=pod)
        with mock.patch.object(k8s_sandbox, "_load_clients", return_value=(batch, core)):
            rc, _ = k8s_sandbox.run_job(
                ["make", "style"],
                image="img:1",
                workdir="/data/clones/worktrees/wt1",
                write_root="/data/clones/worktrees/wt1",
                settings=_settings(),
                uid=1000,
                gid=1000,
                timeout=30,
                poll_interval=0.0,
            )
        self.assertEqual(rc, 1)

    def test_timeout_raises_and_still_deletes(self):
        running = types.SimpleNamespace(succeeded=None, failed=None, conditions=None)
        batch = mock.Mock()
        batch.read_namespaced_job_status.return_value = types.SimpleNamespace(
            status=running
        )
        core = mock.Mock()
        with mock.patch.object(k8s_sandbox, "_load_clients", return_value=(batch, core)):
            with self.assertRaises(K8sSandboxError):
                k8s_sandbox.run_job(
                    ["make", "style"],
                    image="img:1",
                    workdir="/data/clones/worktrees/wt1",
                    write_root="/data/clones/worktrees/wt1",
                    settings=_settings(),
                    uid=1000,
                    gid=1000,
                    timeout=0,
                    poll_interval=0.0,
                )
        self.assertTrue(batch.delete_namespaced_job.called)

    def test_create_failure_raises_k8s_error(self):
        batch = mock.Mock()
        batch.create_namespaced_job.side_effect = _FakeApiException("forbidden")
        core = mock.Mock()
        with mock.patch.object(k8s_sandbox, "_load_clients", return_value=(batch, core)):
            with self.assertRaises(K8sSandboxError):
                k8s_sandbox.run_job(
                    ["make", "style"],
                    image="img:1",
                    workdir="/data/clones/worktrees/wt1",
                    write_root="/data/clones/worktrees/wt1",
                    settings=_settings(),
                    uid=1000,
                    gid=1000,
                    timeout=30,
                    poll_interval=0.0,
                )


if __name__ == "__main__":
    unittest.main()
