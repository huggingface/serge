"""Kubernetes normalize backend: run the repo normalizer as a one-shot Job.

This is the proper-isolation backend for the /tasks normalize gate (Backend B
in ``SERGE_NORMALIZE_PLAN.md`` §5), the alternative to a privileged
docker-in-docker sidecar. For each patch-validation attempt serge creates a
locked-down ``batch/v1`` Job that runs the normalizer command against the
task's worktree, waits for it, reads the pod logs, and deletes the Job.

Isolation contract (mirrors the docker backend, stronger where k8s allows):

- **Worktree on a shared RWX PVC.** serge writes the *self-contained* clone
  (``clone_cache.acquire_ref(standalone=True)``) to the PVC; the Job mounts the
  same claim with a ``subPath`` so the pod sees **only** that worktree, at the
  same absolute path serge used. Because the checkout is self-contained, no
  bare-repo mount is needed and in-pod ``git`` works.
- **No network.** Egress is denied by a cluster ``NetworkPolicy`` selecting the
  ``serge.io/sandbox: normalize`` pod label (the ``--network none`` equivalent;
  see ``deploy/helm``). The image must already carry the repo toolchain.
- **Least privilege.** Non-root, ``readOnlyRootFilesystem`` + a writable
  ``/tmp`` ``emptyDir``, all capabilities dropped, no privilege escalation,
  ``RuntimeDefault`` seccomp, and ``automountServiceAccountToken: false`` so the
  normalize container has no API credentials.

The Job is created by serge's own ServiceAccount (which needs RBAC to manage
Jobs/pods/logs — see ``deploy/helm``), never by the sandboxed pod.

The kubernetes client is imported lazily so non-k8s installs never need it;
:class:`K8sSandboxError` is raised for any infrastructure failure, which the
caller (:func:`reviewbot.normalize._run_kubernetes`) turns into a
``NormalizeError`` (best-effort accept, not the model's fault).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

log = logging.getLogger(__name__)

# Pod label the deny-all-egress NetworkPolicy selects on.
SANDBOX_LABEL_KEY = "serge.io/sandbox"
SANDBOX_LABEL_VALUE = "normalize"

# Pod label the per-task-pod allowlist-egress NetworkPolicy selects on
# (SERGE_PERTASK_POD_PLAN.md). Distinct from the normalize deny-all label so the
# two policies never overlap while both backends coexist (Phase 3 → 4).
TASK_POD_LABEL_KEY = "serge.io/task-pod"
TASK_POD_LABEL_VALUE = "true"

# Where the per-job Secret (task.json) is projected inside the runner pod. The
# Secret's single ``task.json`` key becomes ``{_SPEC_MOUNT_DIR}/task.json``,
# which the runner reads via ``SERGE_TASK_SPEC``.
_SPEC_MOUNT_DIR = "/etc/serge"
_SPEC_KEY = "task.json"
_SPEC_MOUNT_PATH = f"{_SPEC_MOUNT_DIR}/{_SPEC_KEY}"

# In-cluster files the kubelet projects for the pod's ServiceAccount.
_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

_DNS1123_MAX = 63
_TTL_AFTER_FINISHED = 300
_DEFAULT_POLL_INTERVAL = 2.0
_LOG_TAIL_LINES = 200


class K8sSandboxError(RuntimeError):
    """The normalize Job could not be run to completion (client unavailable,
    misconfiguration, API error, or timeout). The message is safe to log."""


@dataclass(frozen=True)
class K8sSettings:
    """Deployment-supplied wiring for the kubernetes normalize backend.

    ``worktree_volume_root`` is where the worktree PVC is mounted *in serge*;
    the worktree's path relative to it becomes the Job's volume ``subPath``.
    ``namespace`` defaults to the in-cluster namespace when unset."""

    worktree_pvc: Optional[str] = None
    worktree_volume_root: Optional[str] = None
    namespace: Optional[str] = None
    service_account: Optional[str] = None
    node_selector: Optional[dict] = None


def parse_node_selector(raw: Optional[str]) -> Optional[dict]:
    """Parse a ``"key=value,key2=value2"`` string into a nodeSelector dict.

    Keys may contain ``/`` and ``.`` (e.g. ``scheduling.cast.ai/node-template``);
    only the first ``=`` is used as the separator. Returns ``None`` when empty."""
    if not raw:
        return None
    out: dict = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        key = key.strip()
        if key and sep:
            out[key] = value.strip()
    return out or None


def _sanitize_dns1123(text: str) -> str:
    """Lowercase, keep ``[a-z0-9-]``, collapse/trim to a DNS-1123 fragment."""
    out = []
    prev_dash = False
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    return "".join(out).strip("-")


def make_job_name(write_root: str) -> str:
    """A unique, DNS-1123-safe Job name derived from the worktree dir."""
    base = _sanitize_dns1123(os.path.basename(write_root.rstrip("/"))) or "task"
    suffix = uuid.uuid4().hex[:8]
    prefix = "serge-nrm-"
    keep = _DNS1123_MAX - len(prefix) - len(suffix) - 1
    return f"{prefix}{base[:keep]}-{suffix}"


def resolve_namespace(settings: K8sSettings) -> str:
    """Explicit namespace, else the in-cluster ServiceAccount namespace."""
    if settings.namespace:
        return settings.namespace
    try:
        with open(_NAMESPACE_FILE, encoding="utf-8") as fh:
            ns = fh.read().strip()
        if ns:
            return ns
    except OSError:
        pass
    raise K8sSandboxError(
        "kubernetes namespace not configured (set TASK_K8S_NAMESPACE) and the "
        "in-cluster namespace file is unavailable"
    )


def _worktree_subpath(write_root: str, volume_root: str) -> str:
    """The worktree's path relative to the PVC mount root, for the volume
    ``subPath``. Refuses paths outside the volume root."""
    write_root = os.path.realpath(write_root)
    volume_root = os.path.realpath(volume_root)
    rel = os.path.relpath(write_root, volume_root)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        raise K8sSandboxError(
            f"worktree {write_root!r} is not under the configured worktree "
            f"volume root {volume_root!r}; cannot mount it by subPath"
        )
    return rel


def build_job_manifest(
    command: list[str],
    *,
    image: str,
    workdir: str,
    write_root: str,
    job_name: str,
    settings: K8sSettings,
    uid: int,
    gid: int,
    timeout: int,
    memory: Optional[str] = None,
) -> dict:
    """Build the ``batch/v1`` Job manifest (pure; no API calls).

    The worktree PVC is mounted at ``write_root`` via ``subPath`` so the pod
    sees only its own checkout at the same absolute path serge wrote it to."""
    if not image:
        raise K8sSandboxError(
            "kubernetes normalize backend requires a configured image "
            "(TASK_NORMALIZE_IMAGE)"
        )
    if not settings.worktree_pvc:
        raise K8sSandboxError(
            "kubernetes normalize backend requires a worktree PVC "
            "(TASK_K8S_WORKTREE_PVC)"
        )
    if not settings.worktree_volume_root:
        raise K8sSandboxError(
            "kubernetes normalize backend requires the worktree volume root "
            "(TASK_K8S_WORKTREE_VOLUME_ROOT, or WEB_CLONE_CACHE_DIR)"
        )

    sub_path = _worktree_subpath(write_root, settings.worktree_volume_root)

    pod_security: dict = {
        "runAsUser": uid,
        "runAsGroup": gid,
        "fsGroup": gid,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    # runAsNonRoot must not be asserted when serge itself runs as uid 0, or the
    # kubelet rejects the pod for contradicting the explicit runAsUser.
    if uid != 0:
        pod_security["runAsNonRoot"] = True

    container: dict = {
        "name": "normalize",
        "image": image,
        "command": list(command),
        "workingDir": workdir,
        "env": [
            {"name": "HOME", "value": "/tmp"},
            {"name": "TMPDIR", "value": "/tmp"},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        ],
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "worktree", "mountPath": write_root, "subPath": sub_path},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
    }
    if memory:
        container["resources"] = {"limits": {"memory": memory}}

    pod_spec: dict = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "securityContext": pod_security,
        "containers": [container],
        "volumes": [
            {
                "name": "worktree",
                "persistentVolumeClaim": {"claimName": settings.worktree_pvc},
            },
            {"name": "tmp", "emptyDir": {}},
        ],
    }
    if settings.service_account:
        pod_spec["serviceAccountName"] = settings.service_account
    if settings.node_selector:
        pod_spec["nodeSelector"] = dict(settings.node_selector)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                "app.kubernetes.io/managed-by": "serge",
                "app.kubernetes.io/component": "normalize",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": timeout,
            "ttlSecondsAfterFinished": _TTL_AFTER_FINISHED,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/managed-by": "serge",
                        "app.kubernetes.io/component": "normalize",
                        SANDBOX_LABEL_KEY: SANDBOX_LABEL_VALUE,
                    }
                },
                "spec": pod_spec,
            },
        },
    }


def _load_clients():
    """Lazily import the kubernetes client and load in-cluster (else kube)
    config. Returns ``(BatchV1Api, CoreV1Api)``."""
    try:
        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover - dep-not-installed path
        raise K8sSandboxError(
            "kubernetes backend selected but the 'kubernetes' client is not "
            "installed (pip install 'reviewbot[kubernetes]')"
        ) from exc

    try:
        config.load_incluster_config()
    except Exception:  # noqa: BLE001 - fall back to a local kubeconfig (dev)
        try:
            config.load_kube_config()
        except Exception as exc:  # noqa: BLE001
            raise K8sSandboxError(
                f"could not load kubernetes config (in-cluster or kubeconfig): {exc}"
            ) from exc
    return client.BatchV1Api(), client.CoreV1Api()


def _job_terminal(status) -> Optional[str]:
    """Return ``"succeeded"``/``"failed"`` once the Job reaches a terminal
    state, else ``None``. Reads the typed status conditions."""
    if status is None:
        return None
    if getattr(status, "succeeded", None):
        return "succeeded"
    if getattr(status, "failed", None):
        return "failed"
    for cond in getattr(status, "conditions", None) or []:
        if cond.status == "True" and cond.type in ("Complete", "Failed"):
            return "succeeded" if cond.type == "Complete" else "failed"
    return None


def run_job(
    command: list[str],
    *,
    image: str,
    workdir: str,
    write_root: str,
    settings: K8sSettings,
    uid: int,
    gid: int,
    timeout: int,
    memory: Optional[str] = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> tuple[int, str]:
    """Create the normalize Job, wait for it, return ``(exit_code, log_tail)``.

    Raises :class:`K8sSandboxError` on any infrastructure failure or if the Job
    does not finish within ``timeout``. The Job is always deleted on the way
    out (a leftover would also be reaped by ``ttlSecondsAfterFinished``)."""
    from kubernetes.client.rest import ApiException

    namespace = resolve_namespace(settings)
    job_name = make_job_name(write_root)
    manifest = build_job_manifest(
        command,
        image=image,
        workdir=workdir,
        write_root=write_root,
        job_name=job_name,
        settings=settings,
        uid=uid,
        gid=gid,
        timeout=timeout,
        memory=memory,
    )
    batch, core = _load_clients()

    try:
        batch.create_namespaced_job(namespace, manifest)
    except ApiException as exc:
        raise K8sSandboxError(
            f"could not create normalize Job {job_name}: {exc.reason or exc}"
        ) from exc

    try:
        # Give the Job its full deadline plus a grace margin to be observed
        # terminal; activeDeadlineSeconds caps the pod itself.
        deadline = time.monotonic() + timeout + poll_interval * 2
        outcome: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                status = batch.read_namespaced_job_status(job_name, namespace).status
            except ApiException as exc:
                raise K8sSandboxError(
                    f"could not read normalize Job status: {exc.reason or exc}"
                ) from exc
            outcome = _job_terminal(status)
            if outcome is not None:
                break
            time.sleep(poll_interval)
        if outcome is None:
            raise K8sSandboxError(
                f"normalize Job {job_name} did not finish within {timeout}s"
            )

        exit_code, tail = _collect_pod_result(core, namespace, job_name)
        # A Job can fail without a container exit code (e.g. deadline exceeded);
        # surface a non-zero so the caller treats it as a rejected patch.
        if outcome == "failed" and exit_code == 0:
            exit_code = 1
        return exit_code, tail
    finally:
        _delete_job(batch, job_name, namespace)


def _collect_pod_result(core, namespace: str, job_name: str) -> tuple[int, str]:
    """Read the Job pod's container exit code and a tail of its logs."""
    from kubernetes.client.rest import ApiException

    try:
        pods = core.list_namespaced_pod(
            namespace, label_selector=f"job-name={job_name}"
        ).items
    except ApiException as exc:
        raise K8sSandboxError(
            f"could not list normalize Job pods: {exc.reason or exc}"
        ) from exc
    if not pods:
        raise K8sSandboxError(f"normalize Job {job_name} produced no pod")

    pod = pods[0]
    pod_name = pod.metadata.name

    exit_code = 1
    for cs in getattr(pod.status, "container_statuses", None) or []:
        term = getattr(cs.state, "terminated", None) if cs.state else None
        if term is not None and term.exit_code is not None:
            exit_code = term.exit_code
            break

    try:
        logs = core.read_namespaced_pod_log(
            pod_name, namespace, tail_lines=_LOG_TAIL_LINES
        )
    except ApiException as exc:
        # Logs are best-effort; the exit code is what gates the patch.
        log.warning("could not read normalize pod logs: %s", exc.reason or exc)
        logs = ""
    tail = "\n".join((logs or "").splitlines()[-40:])
    return exit_code, tail


def _delete_job(batch, job_name: str, namespace: str) -> None:
    """Best-effort Job deletion with background propagation (reaps the pod)."""
    from kubernetes.client import V1DeleteOptions
    from kubernetes.client.rest import ApiException

    try:
        batch.delete_namespaced_job(
            job_name,
            namespace,
            body=V1DeleteOptions(propagation_policy="Background"),
        )
    except ApiException as exc:  # pragma: no cover - cleanup is best-effort
        log.warning(
            "could not delete normalize Job %s: %s", job_name, exc.reason or exc
        )


# ---------------------------------------------------------------------------
# Per-task-pod backend (TASK_EXECUTION=kubernetes) — the whole write-capable
# task (checkout + agent loop + in-process normalize) runs in one Job pod, which
# streams results back to serge over the HTTP callback. Unlike the normalize
# backend there is no shared PVC: the pod does its own checkout into an ephemeral
# ``emptyDir`` (SERGE_PERTASK_POD_PLAN.md, "the pod checks out its own copy").
# ---------------------------------------------------------------------------
def make_task_job_name(job_id: str) -> str:
    """A unique, DNS-1123-safe name for the task Job (and its per-job Secret,
    which shares the name — a Job and a Secret can coexist under one name)."""
    base = _sanitize_dns1123(job_id) or "task"
    prefix = "serge-task-"
    suffix = uuid.uuid4().hex[:8]
    keep = _DNS1123_MAX - len(prefix) - len(suffix) - 1
    return f"{prefix}{base[:keep]}-{suffix}"


def build_task_job_manifest(
    *,
    image: str,
    job_name: str,
    secret_name: str,
    settings: K8sSettings,
    timeout: int,
    proxy: Optional[str] = None,
    no_proxy: Optional[str] = None,
    memory: Optional[str] = None,
    clone_dir: str = "/tmp/serge-clones",
    uid: Optional[int] = None,
    gid: Optional[int] = None,
    ttl: int = _TTL_AFTER_FINISHED,
) -> dict:
    """Build the ``batch/v1`` Job manifest for one task runner pod (pure).

    The per-job Secret is projected read-only at ``/etc/serge/task.json``; the
    pod clones into an ``emptyDir`` under ``/tmp`` (no shared PVC). ``proxy`` is
    the allowlisting egress gateway (injected as ``HTTPS_PROXY``/``HTTP_PROXY``);
    ``no_proxy`` keeps in-cluster traffic (serge's callback) off the proxy. The
    pod holds no API token (``automountServiceAccountToken: false``) — it creates
    no sub-Jobs; the network firewall is the isolation boundary."""
    if not image:
        raise K8sSandboxError(
            "kubernetes task backend requires a configured runner image "
            "(TASK_RUNNER_IMAGE)"
        )

    env: list[dict] = [
        {"name": "HOME", "value": "/tmp"},
        {"name": "TMPDIR", "value": "/tmp"},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "SERGE_TASK_SPEC", "value": _SPEC_MOUNT_PATH},
        {"name": "WEB_CLONE_CACHE_DIR", "value": clone_dir},
    ]
    if proxy:
        for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env.append({"name": name, "value": proxy})
    if no_proxy:
        env.append({"name": "NO_PROXY", "value": no_proxy})
        env.append({"name": "no_proxy", "value": no_proxy})

    container: dict = {
        "name": "runner",
        "image": image,
        "env": env,
        "securityContext": {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        },
        "volumeMounts": [
            {"name": "task-spec", "mountPath": _SPEC_MOUNT_DIR, "readOnly": True},
            {"name": "tmp", "mountPath": "/tmp"},
        ],
    }
    if memory:
        container["resources"] = {"limits": {"memory": memory}}

    pod_security: dict = {"seccompProfile": {"type": "RuntimeDefault"}}
    # The runner runs the repo's own build (``make fix-repo``), so we don't force
    # readOnlyRootFilesystem or a fixed user — that would fight the toolchain
    # image. Isolation is the ephemeral pod + the egress firewall, not the FS.
    if uid is not None:
        pod_security["runAsUser"] = uid
        pod_security["runAsGroup"] = gid if gid is not None else uid
        pod_security["fsGroup"] = gid if gid is not None else uid
        if uid != 0:
            pod_security["runAsNonRoot"] = True

    pod_spec: dict = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "securityContext": pod_security,
        "containers": [container],
        "volumes": [
            {"name": "task-spec", "secret": {"secretName": secret_name}},
            {"name": "tmp", "emptyDir": {}},
        ],
    }
    if settings.service_account:
        pod_spec["serviceAccountName"] = settings.service_account
    if settings.node_selector:
        pod_spec["nodeSelector"] = dict(settings.node_selector)

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                "app.kubernetes.io/managed-by": "serge",
                "app.kubernetes.io/component": "task",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "completions": 1,
            "parallelism": 1,
            "activeDeadlineSeconds": timeout,
            "ttlSecondsAfterFinished": ttl,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/managed-by": "serge",
                        "app.kubernetes.io/component": "task",
                        TASK_POD_LABEL_KEY: TASK_POD_LABEL_VALUE,
                    }
                },
                "spec": pod_spec,
            },
        },
    }


def build_task_secret_manifest(
    *,
    name: str,
    spec_json: str,
    job_name: str,
    job_uid: str,
    namespace: str,
) -> dict:
    """Build the per-job Secret carrying ``task.json`` (spec + short-lived
    GitHub token + LLM key + callback token). ``ownerReferences`` points at the
    Job so the Secret is garbage-collected with it (auto-GC on crash), on top of
    the launcher's explicit delete."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "type": "Opaque",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/managed-by": "serge",
                "app.kubernetes.io/component": "task",
            },
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name,
                    "uid": job_uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "data": {
            _SPEC_KEY: base64.b64encode(spec_json.encode("utf-8")).decode("ascii"),
        },
    }


def run_task_job(
    spec: dict[str, Any],
    *,
    image: str,
    settings: K8sSettings,
    timeout: int,
    proxy: Optional[str] = None,
    no_proxy: Optional[str] = None,
    memory: Optional[str] = None,
    clone_dir: str = "/tmp/serge-clones",
    uid: Optional[int] = None,
    gid: Optional[int] = None,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
) -> tuple[int, str]:
    """Launch one task runner Job for ``spec`` and block until it terminates.

    Creates the Job, then a per-job Secret (ownerReferenced to the Job) holding
    ``task.json``, polls the Job to a terminal state, and returns
    ``(exit_code, log_tail)``. The task's real outcome is streamed to serge over
    the HTTP callback; the exit code is only used to reconcile a runner that died
    without reporting. The Job (and, via ownerRef, the Secret) is always deleted
    on the way out. Raises :class:`K8sSandboxError` on infrastructure failure or
    timeout."""
    from kubernetes.client.rest import ApiException

    namespace = resolve_namespace(settings)
    job_name = make_task_job_name(str(spec.get("job_id") or "task"))
    secret_name = job_name
    spec_json = json.dumps(spec)

    manifest = build_task_job_manifest(
        image=image,
        job_name=job_name,
        secret_name=secret_name,
        settings=settings,
        timeout=timeout,
        proxy=proxy,
        no_proxy=no_proxy,
        memory=memory,
        clone_dir=clone_dir,
        uid=uid,
        gid=gid,
    )
    batch, core = _load_clients()

    try:
        created = batch.create_namespaced_job(namespace, manifest)
    except ApiException as exc:
        raise K8sSandboxError(
            f"could not create task Job {job_name}: {exc.reason or exc}"
        ) from exc

    job_uid = getattr(getattr(created, "metadata", None), "uid", None) or ""
    secret_manifest = build_task_secret_manifest(
        name=secret_name,
        spec_json=spec_json,
        job_name=job_name,
        job_uid=job_uid,
        namespace=namespace,
    )
    try:
        core.create_namespaced_secret(namespace, secret_manifest)
    except ApiException as exc:
        _delete_job(batch, job_name, namespace)
        raise K8sSandboxError(
            f"could not create task Secret {secret_name}: {exc.reason or exc}"
        ) from exc

    try:
        deadline = time.monotonic() + timeout + poll_interval * 2
        outcome: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                status = batch.read_namespaced_job_status(job_name, namespace).status
            except ApiException as exc:
                raise K8sSandboxError(
                    f"could not read task Job status: {exc.reason or exc}"
                ) from exc
            outcome = _job_terminal(status)
            if outcome is not None:
                break
            time.sleep(poll_interval)
        if outcome is None:
            raise K8sSandboxError(
                f"task Job {job_name} did not finish within {timeout}s"
            )

        exit_code, tail = _collect_pod_result(core, namespace, job_name)
        if outcome == "failed" and exit_code == 0:
            exit_code = 1
        return exit_code, tail
    finally:
        _delete_job(batch, job_name, namespace)
        _delete_secret(core, secret_name, namespace)


def _delete_secret(core, secret_name: str, namespace: str) -> None:
    """Best-effort Secret deletion (the ownerRef also GCs it with the Job)."""
    from kubernetes.client.rest import ApiException

    try:
        core.delete_namespaced_secret(secret_name, namespace)
    except ApiException as exc:  # pragma: no cover - cleanup is best-effort
        log.warning(
            "could not delete task Secret %s: %s", secret_name, exc.reason or exc
        )
