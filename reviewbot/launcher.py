"""Launch a per-task runner (:mod:`reviewbot.task_runner`) out-of-process.

serge stays a thin orchestrator: instead of running the write-capable task in a
thread, it launches a **runner pod/container** that runs the whole loop and
streams back over the HTTP callback (see ``SERGE_PERTASK_POD_PLAN.md``). This
module builds the per-task spec and starts the runner. Two backends:

- ``docker``: ``docker run`` the runner image. Works on any Docker host — no
  Kubernetes needed. When serge itself is containerized this is
  "docker-in-docker" via a mounted ``/var/run/docker.sock`` (docker-out-of-docker,
  really). This is also the local-dev / self-hosted path.
- ``kubernetes``: a one-shot Job (implemented in a later phase; the k8s Job
  helpers in :mod:`reviewbot.k8s_sandbox` are reused there).

The spec carries secrets (a short-lived GitHub token, the LLM key) + the callback
coordinates. For docker it is written to a ``0600`` temp file bind-mounted at
``/etc/serge/task.json``; for kubernetes it becomes a per-job Secret.

**Network firewall.** The runner runs arbitrary repo build code alongside the
secrets, so its egress must be allowlisted (git + the LLM only). Pass ``proxy``
to route egress through an allowlisting forward proxy and attach the container to
an ``internal`` docker network (``network``) that has no route out except the
proxy — the docker analogue of the k8s egress-proxy + NetworkPolicy. For local
e2e against host mocks, use ``network="host"`` and no proxy.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_SPEC_MOUNT_PATH = "/etc/serge/task.json"

# Config fields serge resolves per-task (or per-deployment) that the runner
# cannot recover from its own environment — the runner rebuilds a base Config
# from env, then applies these overrides from the spec. Covers the per-task LLM
# caps + strict tool mode resolved by ``_resolve_task_worker_cfg`` and the
# operator/repo normalize + review settings. LLM provider settings travel
# separately in ``llm`` (and win); secrets (App key, session secret) are never
# transmitted — the runner needs none of them.
RUNNER_CONFIG_FIELDS: tuple[str, ...] = (
    "task_normalize_command",
    "task_normalize_guidance",
    "task_normalize_timeout",
    "task_normalize_max_retries",
    "task_normalize_memory",
    "task_max_followups",
    "review_rules_path",
    "helper_tools_path",
    "default_review_rules",
    "max_diff_chars",
    "persona_header",
    "context_script_path",
    "context_script_timeout",
    "allow_approve",
    "llm_reasoning_effort",
    "is_staging",
    "llm_max_tokens",
    "llm_max_input_tokens",
    "tool_max_iterations",
    "tool_max_iterations_strict",
)


def runner_config(cfg: Any) -> dict[str, Any]:
    """Extract the :data:`RUNNER_CONFIG_FIELDS` subset of a resolved worker
    ``Config`` for transmission in the spec. Duck-typed on ``cfg`` so this
    module stays free of a hard :class:`Config` import."""
    return {field: getattr(cfg, field) for field in RUNNER_CONFIG_FIELDS}


def build_spec(
    *,
    job_id: str,
    request: dict[str, Any],
    github_token: str,
    llm: dict[str, Any],
    callback_url: str,
    callback_token: str,
    config: Optional[dict[str, Any]] = None,
    repo_remote_url: Optional[str] = None,
    request_type: str = "task",
) -> dict[str, Any]:
    """Assemble the ``task.json`` payload the runner reads. ``request`` is a
    serialized :class:`reviewbot.tasks.TaskRequest`; ``llm`` is the per-repo
    resolved provider settings (``api_base``/``api_key``/``model``/``bill_to``/
    ``stream``); ``config`` is the resolved-worker-Config subset (see
    :func:`runner_config`) the runner applies over its env-built base;
    ``github_token`` is the short-lived installation token serge minted for this
    task."""
    spec: dict[str, Any] = {
        "job_id": job_id,
        "request": request,
        "github_token": github_token,
        "request_type": request_type,
        "llm": llm,
        "config": config or {},
        "callback": {"url": callback_url, "token": callback_token},
    }
    if repo_remote_url:
        spec["repo_remote_url"] = repo_remote_url
    return spec


@dataclass
class DockerLaunchOptions:
    """Wiring for the docker backend. ``network``/``proxy`` set the egress
    firewall; the rest are escape hatches for dev/e2e (extra mounts, host
    resolution, keeping the container for inspection)."""

    image: str
    network: Optional[str] = None  # e.g. an `internal` net, or "host" for e2e
    proxy: Optional[str] = None  # HTTPS_PROXY/HTTP_PROXY for allowlisted egress
    add_hosts: dict[str, str] = field(default_factory=dict)  # name -> ip/host-gateway
    volumes: dict[str, str] = field(default_factory=dict)  # host_path -> ctr_path[:ro]
    env: dict[str, str] = field(default_factory=dict)
    memory: Optional[str] = None
    name: Optional[str] = None
    remove: bool = True


def _docker_run_argv(spec_path: str, opts: DockerLaunchOptions) -> list[str]:
    argv = ["docker", "run"]
    if opts.remove:
        argv.append("--rm")
    if opts.name:
        argv += ["--name", opts.name]
    if opts.network:
        argv += ["--network", opts.network]
    for host, ip in opts.add_hosts.items():
        argv += ["--add-host", f"{host}:{ip}"]
    if opts.memory:
        argv += ["--memory", opts.memory]
    # The spec (secrets + callback) is mounted read-only at the well-known path.
    argv += ["-v", f"{spec_path}:{_SPEC_MOUNT_PATH}:ro"]
    for host_path, ctr_path in opts.volumes.items():
        argv += ["-v", f"{host_path}:{ctr_path}"]
    env = dict(opts.env)
    env.setdefault("SERGE_TASK_SPEC", _SPEC_MOUNT_PATH)
    if opts.proxy:
        # Allowlisted egress: everything the runner does (git, LLM) goes via the
        # proxy; the container's own network is otherwise cut off (internal net).
        env.setdefault("HTTPS_PROXY", opts.proxy)
        env.setdefault("HTTP_PROXY", opts.proxy)
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    argv.append(opts.image)
    return argv


def launch_docker(
    spec: dict[str, Any],
    opts: DockerLaunchOptions,
    *,
    wait: bool = False,
    timeout: Optional[int] = None,
) -> tuple[int, str]:
    """Start the runner container for ``spec``.

    With ``wait=False`` the container is detached and this returns immediately
    ``(0, container_id)`` — the runner reports its outcome over the callback.
    The bind-mounted spec temp file is **left in place**: unlinking it before the
    detached container has read it would break the mount. The caller (or a
    reaper keyed on container exit) removes ``serge-task-*.json`` afterwards.
    With ``wait=True`` it blocks up to ``timeout`` seconds, removes the spec on
    the way out, and returns ``(exit_code, combined_output)`` (tests /
    synchronous callers)."""
    fd, spec_path = tempfile.mkstemp(prefix="serge-task-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(spec, fh)
    os.chmod(spec_path, 0o600)

    argv = _docker_run_argv(spec_path, opts)
    if not wait:
        argv.insert(2, "-d")  # `docker run -d ...`
    log.info("launching task runner container (wait=%s): %s", wait, opts.image)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except BaseException:
        _unlink(spec_path)
        raise

    out = (proc.stdout or "") + (proc.stderr or "")
    if not wait:
        if proc.returncode != 0:
            _unlink(spec_path)
            raise RuntimeError(f"docker run failed: {out.strip()}")
        return 0, out.strip()  # container id; spec file left for the reaper
    _unlink(spec_path)
    return proc.returncode, out


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass
class K8sLaunchOptions:
    """Wiring for the kubernetes backend. ``proxy``/``no_proxy`` set the egress
    firewall (the allowlisting ``serge-egress`` gateway + the in-cluster hosts
    that bypass it); ``namespace``/``service_account``/``node_selector`` place
    the Job. The per-job Secret carries the spec — nothing is shared but that."""

    image: str
    namespace: Optional[str] = None
    service_account: Optional[str] = None
    node_selector: Optional[dict[str, str]] = None
    proxy: Optional[str] = None  # HTTPS_PROXY → serge-egress gateway ClusterIP
    no_proxy: Optional[str] = None  # hosts that skip the proxy (serge callback)
    memory: Optional[str] = None
    uid: Optional[int] = None
    gid: Optional[int] = None
    clone_dir: str = "/tmp/serge-clones"


def launch_kubernetes(
    spec: dict[str, Any],
    opts: K8sLaunchOptions,
    *,
    timeout: int,
    poll_interval: float = 2.0,
) -> tuple[int, str]:
    """Launch the runner as a one-shot Job and block until it terminates,
    returning ``(exit_code, log_tail)``. The task's outcome is streamed to serge
    over the HTTP callback; the exit code only reconciles a runner that died
    without reporting. Delegates to :func:`reviewbot.k8s_sandbox.run_task_job`
    (kubernetes client imported lazily there)."""
    from .k8s_sandbox import K8sSettings, run_task_job

    settings = K8sSettings(
        namespace=opts.namespace,
        service_account=opts.service_account,
        node_selector=opts.node_selector,
    )
    return run_task_job(
        spec,
        image=opts.image,
        settings=settings,
        timeout=timeout,
        proxy=opts.proxy,
        no_proxy=opts.no_proxy,
        memory=opts.memory,
        clone_dir=opts.clone_dir,
        uid=opts.uid,
        gid=opts.gid,
        poll_interval=poll_interval,
    )


def create_kubernetes(
    spec: dict[str, Any],
    opts: K8sLaunchOptions,
    *,
    timeout: int,
) -> tuple[str, str]:
    """Non-blocking launch: create the runner Job + Secret and return
    ``(job_name, namespace)`` immediately (SERGE_ORCHESTRATOR_PODS_PLAN.md
    Phase 1). The caller reconciles completion via the callback + a Job watcher,
    so no serge thread is parked for the pod's lifetime. Delegates to
    :func:`reviewbot.k8s_sandbox.create_task_job` (client imported lazily)."""
    from .k8s_sandbox import K8sSettings, create_task_job

    settings = K8sSettings(
        namespace=opts.namespace,
        service_account=opts.service_account,
        node_selector=opts.node_selector,
    )
    return create_task_job(
        spec,
        image=opts.image,
        settings=settings,
        timeout=timeout,
        proxy=opts.proxy,
        no_proxy=opts.no_proxy,
        memory=opts.memory,
        clone_dir=opts.clone_dir,
        uid=opts.uid,
        gid=opts.gid,
    )
