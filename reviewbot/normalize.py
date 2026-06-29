"""Repo-normalizer execution primitive for the in-loop patch verification gate.

The /tasks flow validates each LLM patch by applying it to the worktree and
running the target repo's own normalizer (e.g. ``make style && make fix-repo``)
*inside the agentic loop* — a non-zero exit is fed back to the model so it
corrects the patch (see ``tasks._validate_patch``). This module is the part
that actually runs the normalizer in a sandbox; it does not decide policy.

This is **opt-in, per deployment/repo**: when no normalize command is
configured, the caller skips this module entirely and serge stays
repo-agnostic. The command is **operator/repo config, never request-supplied**,
so there is no command-injection surface to contain.

The normalizer runs arbitrary repo build code, so it executes in a
network-isolated sandbox selected by ``TASK_SANDBOX_BACKEND``:

- ``bwrap`` / ``off``: a local subprocess wrapped in bubblewrap, reusing
  serge's own venv (dev/test only — no target-repo deps).
- ``docker``: a throwaway, network-isolated container built from an image with
  the repo's toolchain baked in (the portable backend; works on any host with
  a Docker daemon, no Kubernetes required).
- ``kubernetes``: a one-shot Job in a locked-down namespace (for k8s
  deployments). Implemented in Phase 1 — see ``reviewbot/k8s_sandbox.py``.

A non-zero exit is **returned** (the caller turns it into model feedback). An
infrastructure failure (sandbox unavailable, timeout, launch failure) is
**raised** as :class:`NormalizeError` — that is not the model's fault, so the
caller accepts the applied patch best-effort rather than blaming the LLM.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

from . import sandbox
from .tools import _helper_subprocess_env

log = logging.getLogger(__name__)


class NormalizeError(RuntimeError):
    """The normalize hook could not run to completion (sandbox unavailable,
    timeout, or launch failure). The message is safe to log/surface. The
    caller treats this as best-effort: log and commit the raw LLM patch."""


def run_normalize(
    command: list[str],
    *,
    workdir: str,
    write_root: str,
    backend: str,
    image: Optional[str],
    mode: str,
    timeout: int,
    memory: Optional[str] = None,
    network: bool = False,
) -> tuple[int, str]:
    """Run ``command`` against the worktree in the selected sandbox backend.

    Returns ``(returncode, output_tail)``. Does not stage or commit — the
    caller stages the worktree afterwards and collects the combined diff.

    ``backend`` is ``bwrap`` | ``docker`` | ``kubernetes`` | ``auto``. The
    ``bwrap``/``docker``/``auto`` backends run as a local subprocess (the same
    isolation as the read-only review subprocesses, just stronger for docker);
    ``kubernetes`` runs as a one-shot Job. Raises :class:`NormalizeError` when
    the sandbox is unavailable, the command times out, or it cannot be
    launched."""
    backend = sandbox.normalize_backend(backend)
    if backend == sandbox.KUBERNETES_BACKEND:
        return _run_kubernetes(
            command,
            workdir=workdir,
            write_root=write_root,
            image=image,
            timeout=timeout,
            memory=memory,
            network=network,
        )
    return _run_subprocess(
        command,
        workdir=workdir,
        write_root=write_root,
        backend=backend,
        image=image,
        mode=mode,
        timeout=timeout,
        memory=memory,
        network=network,
    )


def _run_subprocess(
    command: list[str],
    *,
    workdir: str,
    write_root: str,
    backend: str,
    image: Optional[str],
    mode: str,
    timeout: int,
    memory: Optional[str],
    network: bool,
) -> tuple[int, str]:
    """bwrap / docker / auto backends: wrap the command and run it locally."""
    try:
        argv = sandbox.wrap_task_command(
            command,
            workdir=workdir,
            write_root=write_root,
            backend=backend,
            image=image,
            mode=mode,
            network=network,
            memory=memory,
        )
    except (sandbox.DockerUnavailable, sandbox.SandboxUnavailable) as exc:
        raise NormalizeError(f"normalize sandbox unavailable: {exc}") from exc

    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=_helper_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise NormalizeError(f"normalize command timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        # e.g. the docker CLI vanished between the availability check and run.
        raise NormalizeError(f"could not launch normalize command: {exc}") from exc

    output = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(output.splitlines()[-40:])
    return proc.returncode, tail


def _run_kubernetes(
    command: list[str],
    *,
    workdir: str,
    write_root: str,
    image: Optional[str],
    timeout: int,
    memory: Optional[str],
    network: bool,
) -> tuple[int, str]:
    """kubernetes backend: run the command as a one-shot Job in a locked-down
    namespace. Implemented in Phase 1 (``reviewbot/k8s_sandbox.py``)."""
    raise NormalizeError(
        "kubernetes normalize backend is not implemented yet (Phase 1); "
        "set TASK_SANDBOX_BACKEND=docker for now"
    )
