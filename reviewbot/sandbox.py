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

# Modes for the HELPER_SANDBOX setting.
REQUIRE = "require"
AUTO = "auto"
OFF = "off"
_VALID_MODES = frozenset({REQUIRE, AUTO, OFF})

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


def normalize_mode(raw: str | None) -> str:
    mode = (raw or AUTO).strip().lower()
    return mode if mode in _VALID_MODES else AUTO


def sandbox_available() -> bool:
    return shutil.which(BWRAP) is not None


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
