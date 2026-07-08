"""Kubernetes per-task-pod backend: run a whole /tasks request as one Job.

For ``TASK_EXECUTION=kubernetes`` serge launches one locked-down ``batch/v1``
Job per task; the pod runs the full write-capable loop — checkout, agentic loop,
**in-process** normalize, PR publish — and streams events + the terminal result
back to serge over the HTTP callback (see ``SERGE_PERTASK_POD_PLAN.md``). serge
watches the Job to a terminal state only to reconcile a runner that died without
reporting; it then deletes the Job (and, via ``ownerReferences``, the per-job
Secret carrying ``task.json``).

Isolation contract:

- **Nothing shared but the spec.** The pod does its own checkout into an
  ephemeral ``emptyDir`` (no shared PVC); the only thing serge injects is the
  small per-job Secret at ``/etc/serge/task.json``.
- **Allowlist egress.** The pod's ``NetworkPolicy`` permits egress only to the
  ``serge-egress`` proxy (git + LLM), serge's callback, and kube-dns; the proxy
  CONNECT-allows GitHub + the HF LLM only (see ``deploy/helm``).
- **Least privilege.** No API token (``automountServiceAccountToken: false`` —
  the pod spawns no sub-Jobs), all capabilities dropped, no privilege
  escalation, ``RuntimeDefault`` seccomp.

The Job is created by serge's own ServiceAccount (RBAC to manage Jobs/pods/logs
+ per-job Secrets — see ``deploy/helm``). The kubernetes client is imported
lazily so non-k8s installs never need it; :class:`K8sSandboxError` is raised for
any infrastructure failure.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

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
    """The task Job could not be run to completion (client unavailable,
    misconfiguration, API error, or timeout). The message is safe to log."""


@dataclass(frozen=True)
class K8sSettings:
    """Deployment-supplied placement for the per-task runner Job.

    ``namespace`` defaults to the in-cluster namespace when unset;
    ``service_account`` is the task *pod's* SA (it holds no API token
    regardless); ``node_selector`` pins the pods to a node pool;
    ``tolerations`` (issue #20) lets GPU task pods schedule onto tainted GPU
    nodes (a list of v1.Toleration dicts)."""

    namespace: Optional[str] = None
    service_account: Optional[str] = None
    node_selector: Optional[dict] = None
    tolerations: Optional[list] = None


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


# --- egress allowlist sync ---------------------------------------------------
# The serge-egress proxy CONNECT-allows only the hosts in its tinyproxy filter
# (rendered by helm as git + the default LLM host). But provider configs are
# dynamic — an admin can add an OpenAI/Anthropic/custom-base provider on the
# Settings page at any time, and pod-based reviews route their LLM call through
# this proxy. So serge keeps the filter in sync with the configured LLM hosts.
_EGRESS_FILTER_KEY = "filter"
# Marks the last sync on the egress Deployment's pod template; changing it
# triggers a rollout so tinyproxy re-reads its filter file on start.
_EGRESS_SYNC_ANNOTATION = "serge.io/allowlist-synced-at"


def _host_filter_regex(base: str) -> Optional[str]:
    """ERE that matches exactly one host, for the tinyproxy allowlist. Accepts a
    full URL or a bare host; returns ``None`` if no host can be parsed."""
    raw = (base or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "https://" + raw
    host = urllib.parse.urlparse(raw).hostname
    if not host:
        return None
    return "^" + re.escape(host) + "$"


def sync_egress_allowlist(
    llm_bases: Iterable[str],
    *,
    egress_name: str,
    namespace: Optional[str] = None,
) -> bool:
    """Ensure the ``serge-egress`` proxy allowlists every LLM host in
    ``llm_bases`` (URLs or bare hosts). Additively unions the host regexes into
    the proxy's tinyproxy filter ConfigMap; if anything was missing, patches the
    ConfigMap and rollout-restarts the egress Deployment (both named
    ``egress_name``) so tinyproxy re-reads the filter. Returns ``True`` when it
    changed the allowlist.

    Fail-soft by contract: any missing dependency or API error is logged and
    swallowed (returns ``False``) so serge keeps serving — a stale allowlist
    only affects reviews to not-yet-allowlisted LLM hosts, and never the app."""
    if not egress_name:
        return False
    want = {rx for b in llm_bases if (rx := _host_filter_regex(b))}
    if not want:
        return False
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        log.warning("egress allowlist: kubernetes client unavailable; skipping sync")
        return False

    try:
        try:
            config.load_incluster_config()
        except Exception:  # noqa: BLE001 - local kubeconfig (dev)
            config.load_kube_config()
        ns = resolve_namespace(K8sSettings(namespace=namespace))
        core = client.CoreV1Api()
        apps = client.AppsV1Api()

        cm = core.read_namespaced_config_map(egress_name, ns)
        current = (cm.data or {}).get(_EGRESS_FILTER_KEY, "") or ""
        lines = [ln for ln in current.splitlines() if ln.strip()]
        missing = sorted(want - set(lines))
        if not missing:
            return False

        new_filter = "\n".join(lines + missing) + "\n"
        core.patch_namespaced_config_map(
            egress_name, ns, {"data": {_EGRESS_FILTER_KEY: new_filter}}
        )
        # tinyproxy reads its filter only at startup; roll the Deployment so the
        # new pod picks up the patched ConfigMap.
        apps.patch_namespaced_deployment(
            egress_name,
            ns,
            {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {_EGRESS_SYNC_ANNOTATION: str(time.time())}
                        }
                    }
                }
            },
        )
        log.info(
            "egress allowlist: added %d LLM host(s) and rolled %s: %s",
            len(missing),
            egress_name,
            ", ".join(missing),
        )
        return True
    except ApiException as exc:
        log.warning(
            "egress allowlist sync failed (%s); reviews to newly-added LLM "
            "hosts may be blocked until this succeeds",
            getattr(exc, "reason", exc),
        )
        return False
    except Exception:  # noqa: BLE001 - never let allowlist upkeep break serge
        log.warning("egress allowlist sync failed", exc_info=True)
        return False


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
    gpu_resource: Optional[str] = None,
    gpu_count: Optional[int] = None,
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
    # Resource limits: memory and, for GPU tasks (issue #20), the GPU extended
    # resource (e.g. ``nvidia.com/gpu: "2"``). An extended resource is declared
    # only under ``limits``; the scheduler mirrors it as the request.
    limits: dict = {}
    if memory:
        limits["memory"] = memory
    if gpu_resource and gpu_count:
        limits[gpu_resource] = str(gpu_count)
    if limits:
        container["resources"] = {"limits": limits}

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
    if settings.tolerations:
        pod_spec["tolerations"] = list(settings.tolerations)

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
    gpu_resource: Optional[str] = None,
    gpu_count: Optional[int] = None,
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
    job_name, namespace = create_task_job(
        spec,
        image=image,
        settings=settings,
        timeout=timeout,
        proxy=proxy,
        no_proxy=no_proxy,
        memory=memory,
        gpu_resource=gpu_resource,
        gpu_count=gpu_count,
        clone_dir=clone_dir,
        uid=uid,
        gid=gid,
    )
    _, core = _load_clients()
    try:
        deadline = time.monotonic() + timeout + poll_interval * 2
        outcome: Optional[str] = None
        while time.monotonic() < deadline:
            outcome = poll_task_job(job_name, namespace)
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
        cleanup_task_job(job_name, namespace)


def create_task_job(
    spec: dict[str, Any],
    *,
    image: str,
    settings: K8sSettings,
    timeout: int,
    proxy: Optional[str] = None,
    no_proxy: Optional[str] = None,
    memory: Optional[str] = None,
    gpu_resource: Optional[str] = None,
    gpu_count: Optional[int] = None,
    clone_dir: str = "/tmp/serge-clones",
    uid: Optional[int] = None,
    gid: Optional[int] = None,
) -> tuple[str, str]:
    """Create the task Job + its per-job Secret and return ``(job_name,
    namespace)`` **without waiting** — the non-blocking launch path
    (SERGE_ORCHESTRATOR_PODS_PLAN.md). The caller reconciles completion
    out-of-band via :func:`poll_task_job` + a watcher, so no serge thread is
    parked for the pod's lifetime. Raises :class:`K8sSandboxError` on failure
    (the Job is cleaned up if the Secret create fails)."""
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
        gpu_resource=gpu_resource,
        gpu_count=gpu_count,
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
    return job_name, namespace


def poll_task_job(job_name: str, namespace: str) -> Optional[str]:
    """Return the Job's terminal outcome (``"succeeded"``/``"failed"``) or
    ``None`` if it is still running. A vanished Job (404 — deleted or TTL-GC'd)
    counts as ``"failed"`` (terminal). Raises :class:`K8sSandboxError` on any
    other API error."""
    from kubernetes.client.rest import ApiException

    batch, _ = _load_clients()
    try:
        status = batch.read_namespaced_job_status(job_name, namespace).status
    except ApiException as exc:
        if getattr(exc, "status", None) == 404:
            return "failed"
        raise K8sSandboxError(
            f"could not read task Job status: {exc.reason or exc}"
        ) from exc
    return _job_terminal(status)


def collect_task_result(job_name: str, namespace: str) -> tuple[int, str]:
    """``(exit_code, log_tail)`` for a terminated task Job pod. Best-effort: a
    missing pod / unreadable logs yields ``(1, "")`` rather than raising, so the
    watcher can always finish reconciling."""
    _, core = _load_clients()
    try:
        return _collect_pod_result(core, namespace, job_name)
    except K8sSandboxError:
        return 1, ""


def cleanup_task_job(job_name: str, namespace: str) -> None:
    """Delete the task Job (+ its Secret via ownerRef). ``ttlSecondsAfterFinished``
    on the Job is the backstop if this is ever missed."""
    batch, core = _load_clients()
    _delete_job(batch, job_name, namespace)
    _delete_secret(core, job_name, namespace)


def list_task_pods(namespace: Optional[str] = None) -> tuple[str, list[dict]]:
    """List live task-runner pods in the cluster for the admin view
    (SERGE_ORCHESTRATOR_PODS_PLAN.md Phase 2). Returns ``(namespace, pods)``
    where each pod is a plain dict: ``pod`` (name), ``job_name`` (the owning
    Job, from the auto-applied ``job-name`` label — used to join back to serge's
    tracked task), ``phase``, ``node``, ``start_epoch``. Raises
    :class:`K8sSandboxError` on API failure."""
    from kubernetes.client.rest import ApiException

    ns = resolve_namespace(K8sSettings(namespace=namespace))
    _, core = _load_clients()
    selector = "app.kubernetes.io/managed-by=serge,app.kubernetes.io/component=task"
    try:
        pods = core.list_namespaced_pod(ns, label_selector=selector).items
    except ApiException as exc:
        raise K8sSandboxError(f"could not list task pods: {exc.reason or exc}") from exc
    out: list[dict] = []
    for pod in pods:
        meta = getattr(pod, "metadata", None)
        st = getattr(pod, "status", None)
        spec = getattr(pod, "spec", None)
        labels = getattr(meta, "labels", None) or {}
        start = getattr(st, "start_time", None)
        out.append(
            {
                "pod": getattr(meta, "name", "") or "",
                "job_name": labels.get("job-name", "") or "",
                "phase": getattr(st, "phase", "") or "",
                "node": getattr(spec, "node_name", "") or "",
                "start_epoch": start.timestamp() if start else None,
            }
        )
    return ns, out


def _delete_secret(core, secret_name: str, namespace: str) -> None:
    """Best-effort Secret deletion (the ownerRef also GCs it with the Job)."""
    from kubernetes.client.rest import ApiException

    try:
        core.delete_namespaced_secret(secret_name, namespace)
    except ApiException as exc:  # pragma: no cover - cleanup is best-effort
        log.warning(
            "could not delete task Secret %s: %s", secret_name, exc.reason or exc
        )
