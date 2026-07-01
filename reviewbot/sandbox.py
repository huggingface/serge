"""Filesystem + network isolation for subprocesses that touch the PR tree.

A review runs three kinds of subprocess against a PR checkout: helper-tool
commands, the per-repo ``pip install`` hook, and the context-script. Any
of them may execute or import code a PR author influenced, so on the
persistent review host they must not be able to read host secrets (the
GitHub App key, LLM keys, ``jobs.db``), reach the network, or escape the
worktree.

We wrap each one in ``bubblewrap`` (``bwrap``): an unprivileged,
daemonless user-namespace sandbox, one invocation per subprocess. See
``docs/security-architecture.md`` for the full rationale.

The sandbox is deny-by-default: it binds ``/usr`` (+ usrmerge symlinks),
the Python venv, and a short allowlist of ``/etc`` files read-only, plus
the worktree read-write, and nothing else. ``--unshare-all
--unshare-net`` removes the network entirely (the main review loop, which
talks to the LLM, is a separate unsandboxed process), which also makes the
EC2 instance-metadata endpoint unreachable.

``HELPER_SANDBOX`` selects behaviour when bwrap is unavailable:

- ``require`` (production): raise :class:`SandboxUnavailable` — the caller
  surfaces an error and does **not** run the subprocess.
- ``auto`` (default): sandbox if ``bwrap`` is on PATH, else run unwrapped
  (local dev on macOS, ephemeral CI runners).
- ``off``: never wrap.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

log = logging.getLogger(__name__)

BWRAP = "bwrap"
DOCKER = "docker"

# Modes for the HELPER_SANDBOX setting.
REQUIRE = "require"
AUTO = "auto"
OFF = "off"
_VALID_MODES = frozenset({REQUIRE, AUTO, OFF})

# Backends for the task-command sandbox (TASK_SANDBOX_BACKEND). Distinct
# from HELPER_SANDBOX's require/auto/off (which selects bwrap-or-nothing for
# the read-only review subprocesses). A task command — e.g. the post-LLM
# normalize hook running ``make fix-repo`` — runs arbitrary repo build code
# that needs the *target repo's* dependency environment, which serge's own
# venv does not provide. The ``docker`` backend runs it in a throwaway,
# network-isolated container built from a per-repo image with those deps baked
# in; ``kubernetes`` runs it as a one-shot Job in an isolated namespace (for
# k8s deployments); ``bwrap`` reuses serge's venv (only viable when the command
# needs no extra deps). Docker stays a first-class backend so a non-k8s
# deployment never needs Kubernetes.
BWRAP_BACKEND = "bwrap"
DOCKER_BACKEND = "docker"
KUBERNETES_BACKEND = "kubernetes"
AUTO_BACKEND = "auto"
_VALID_BACKENDS = frozenset(
    {BWRAP_BACKEND, DOCKER_BACKEND, KUBERNETES_BACKEND, AUTO_BACKEND}
)

# Read-only /etc files a sandboxed tool plausibly needs (name resolution,
# user lookup, timezone, TLS roots) — deliberately NOT all of /etc, which
# would expose /etc/reviewbot and any other host secrets.
_ETC_ALLOW = (
    "/etc/passwd",
    "/etc/group",
    "/etc/nsswitch.conf",
    "/etc/localtime",
    "/etc/ssl",
    "/etc/pki",
    "/etc/ca-certificates",
)

# usrmerge symlinks recreated inside the (otherwise empty) sandbox root so
# that ``/bin/sh``, the dynamic linker under ``/lib64``, etc. resolve into
# the read-only ``/usr`` bind.
_USRMERGE_SYMLINKS = (
    ("usr/bin", "/bin"),
    ("usr/sbin", "/sbin"),
    ("usr/lib", "/lib"),
    ("usr/lib64", "/lib64"),
)


class SandboxUnavailable(RuntimeError):
    """Raised when sandboxing is required but ``bwrap`` is not usable."""


class DockerUnavailable(RuntimeError):
    """Raised when the docker command-task backend is required but the
    ``docker`` CLI is not on PATH."""


def normalize_mode(raw: str | None) -> str:
    mode = (raw or AUTO).strip().lower()
    return mode if mode in _VALID_MODES else AUTO


def normalize_backend(raw: str | None) -> str:
    backend = (raw or AUTO_BACKEND).strip().lower()
    return backend if backend in _VALID_BACKENDS else AUTO_BACKEND


def sandbox_available() -> bool:
    return shutil.which(BWRAP) is not None


def docker_available() -> bool:
    return shutil.which(DOCKER) is not None


def _venv_root() -> str | None:
    """Directory tree to bind so the running interpreter and any
    pip-installed helper console scripts resolve inside the sandbox.

    ``sys.prefix`` differs from ``sys.base_prefix`` in a venv; bind the
    venv prefix when we're in one (its ``bin`` holds installed helpers and
    its ``pyvenv.cfg`` points back at the base interpreter under /usr,
    which is already bound)."""
    prefix = getattr(sys, "prefix", None)
    base = getattr(sys, "base_prefix", None)
    if prefix and base and os.path.realpath(prefix) != os.path.realpath(base):
        return os.path.realpath(prefix)
    return None


def build_bwrap_argv(command: list[str], *, workdir: str, write_root: str) -> list[str]:
    """Return the ``bwrap ... -- <command>`` argv for ``command``.

    ``write_root`` (the worktree) is bound read-write and is the only
    writable host path; ``workdir`` is the cwd inside the sandbox (must be
    within ``write_root``). The caller still passes the scrubbed ``env``
    to ``subprocess.run``; we do not ``--clearenv`` so that single
    allowlist stays the source of truth, but we pin HOME/TMPDIR to the
    in-sandbox tmpfs."""
    write_root = os.path.realpath(write_root)
    argv: list[str] = [
        BWRAP,
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for target, link in _USRMERGE_SYMLINKS:
        argv += ["--symlink", target, link]
    for path in _ETC_ALLOW:
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    venv = _venv_root()
    if venv and venv != "/usr" and not venv.startswith("/usr/"):
        argv += ["--ro-bind", venv, venv]
    argv += ["--bind", write_root, write_root]
    argv += ["--chdir", workdir]
    argv += ["--", *command]
    return argv


def build_docker_argv(
    command: list[str],
    *,
    image: str,
    workdir: str,
    write_root: str,
    network: bool = False,
    uid: int | None = None,
    gid: int | None = None,
    memory: str | None = None,
    pids_limit: int = 512,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    """Return the ``docker run ... <image> <command>`` argv for a command task.

    The container is a throwaway sandbox with the same trust properties as
    the bwrap one, just stronger isolation for running arbitrary repo build
    commands:

    - ``--network none`` (default): no network. The repo's dependencies must
      already be baked into ``image`` — serge never pip-installs at command
      time. Pass ``network=True`` only for trusted, deliberately online tasks.
    - ``--read-only`` rootfs + a writable ``/tmp`` tmpfs. The image's
      site-packages live in read-only layers (readable); the only writable
      host path is the worktree.
    - The worktree (``write_root``) is bind-mounted **at the same absolute
      path** read-write, and is the cwd (``workdir`` must be within it). The
      command's file edits land straight in the host worktree, where
      ``collect_changes`` picks them up.
    - ``--cap-drop ALL`` + ``--security-opt no-new-privileges`` + a pids cap.
    - Run as the host ``uid:gid`` so files the command creates in the
      worktree are owned by serge, not root.
    - **No serge secrets are ever passed** — the env is a minimal allowlist,
      and (unlike the helper install hook) the container has no network to
      exfiltrate anything regardless.

    ``image`` is operator-controlled (per-repo, via config), never
    caller-controlled — the OIDC ``repository`` claim authorizes *which* repo,
    not *what image* runs.
    """
    write_root = os.path.realpath(write_root)
    if uid is None:
        uid = os.getuid() if hasattr(os, "getuid") else 0
    if gid is None:
        gid = os.getgid() if hasattr(os, "getgid") else 0

    argv: list[str] = [
        DOCKER,
        "run",
        "--rm",
        "--init",
        "--network",
        "none" if not network else "bridge",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,nodev",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids_limit),
        "--user",
        f"{uid}:{gid}",
        "--volume",
        f"{write_root}:{write_root}:rw",
        "--workdir",
        workdir,
        # HOME/TMPDIR point at the writable tmpfs; don't write .pyc into the
        # read-only site-packages.
        "--env",
        "HOME=/tmp",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if memory:
        argv += ["--memory", memory]
    for key, value in (extra_env or {}).items():
        argv += ["--env", f"{key}={value}"]
    argv += [image, *command]
    return argv


def wrap_command(
    command: list[str],
    *,
    workdir: str,
    write_root: str,
    mode: str,
) -> list[str]:
    """Wrap ``command`` in bwrap per ``mode``.

    Returns the argv to execute: a bwrap-wrapped command when sandboxing
    is active, or ``command`` unchanged when it is not. Raises
    :class:`SandboxUnavailable` when ``mode == "require"`` but bwrap is
    not on PATH."""
    mode = normalize_mode(mode)
    if mode == OFF:
        return command
    if sandbox_available():
        return build_bwrap_argv(command, workdir=workdir, write_root=write_root)
    if mode == REQUIRE:
        raise SandboxUnavailable(
            "HELPER_SANDBOX=require but bubblewrap (bwrap) is not installed; "
            "refusing to run untrusted subprocess unsandboxed"
        )
    log.warning(
        "bubblewrap not found; running subprocess WITHOUT sandbox (HELPER_SANDBOX=auto)"
    )
    return command


def wrap_task_command(
    command: list[str],
    *,
    workdir: str,
    write_root: str,
    backend: str,
    image: str | None,
    mode: str,
    network: bool = False,
    memory: str | None = None,
) -> list[str]:
    """Resolve a local (subprocess) task-command sandbox backend and return
    the argv to run.

    ``backend`` is ``bwrap`` | ``docker`` | ``auto``. ``auto`` picks docker
    when an ``image`` is configured and the docker CLI is present, else falls
    back to bwrap (which uses serge's own venv and the ``mode`` require/auto/off
    semantics). Raises :class:`DockerUnavailable` when docker is explicitly
    required but unusable, or :class:`SandboxUnavailable` when bwrap is required
    but unusable. The ``kubernetes`` backend is not handled here — it does not
    run as a local subprocess; see ``reviewbot/normalize.py``."""
    backend = normalize_backend(backend)
    if backend == AUTO_BACKEND:
        backend = DOCKER_BACKEND if (image and docker_available()) else BWRAP_BACKEND

    if backend == DOCKER_BACKEND:
        if not image:
            raise DockerUnavailable(
                "docker task-command backend requires a configured image "
                "(TASK_NORMALIZE_IMAGE)"
            )
        if not docker_available():
            raise DockerUnavailable(
                "docker command-task backend selected but the 'docker' CLI is "
                "not on PATH"
            )
        return build_docker_argv(
            command,
            image=image,
            workdir=workdir,
            write_root=write_root,
            network=network,
            memory=memory,
        )

    return wrap_command(command, workdir=workdir, write_root=write_root, mode=mode)
