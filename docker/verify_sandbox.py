#!/usr/bin/env python3
"""Exercise the real bubblewrap helper-sandbox and assert its guarantees.

Unlike tests/test_sandbox.py (which mocks bwrap), this actually executes
sandboxed subprocesses and checks the four properties the sandbox exists to
enforce: a sandboxed command runs, the worktree is writable, host secrets
outside the worktree are NOT readable, and there is no network. Run inside the
docker/Dockerfile image (Linux + bubblewrap), where bwrap can actually run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from reviewbot import sandbox
from reviewbot.sandbox import REQUIRE, build_bwrap_argv, wrap_command

SECRET = "/etc/reviewbot/github-app.pem"
# Minimal env handed to the subprocess; the bwrap profile is the real boundary.
# Includes /usr/local/bin (slim-image python) and the venv bin, both under
# read-only host binds inside the sandbox.
ENV = {"PATH": "/opt/app/.venv/bin:/usr/local/bin:/usr/bin:/bin"}


def run_sandboxed(command, *, workdir, write_root):
    argv = wrap_command(command, workdir=workdir, write_root=write_root, mode=REQUIRE)
    return subprocess.run(argv, env=ENV, capture_output=True, text=True, timeout=30)


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))

    # 0. bwrap must be present (require mode would otherwise raise).
    check("bwrap available on PATH", sandbox.sandbox_available(),
          sandbox.shutil.which(sandbox.BWRAP) or "not found")

    worktree = tempfile.mkdtemp(prefix="worktree-")

    # 1. A sandboxed command runs and sees the bound worktree.
    r = run_sandboxed(["/bin/sh", "-c", "echo ok"], workdir=worktree, write_root=worktree)
    check("sandboxed command runs", r.returncode == 0 and r.stdout.strip() == "ok",
          f"rc={r.returncode} out={r.stdout!r} err={r.stderr.strip()!r}")

    # 2. The worktree is writable from inside the sandbox.
    marker = os.path.join(worktree, "written-from-sandbox")
    r = run_sandboxed(["/bin/sh", "-c", f"echo hi > {marker}"],
                      workdir=worktree, write_root=worktree)
    check("worktree is writable", r.returncode == 0 and os.path.exists(marker),
          f"rc={r.returncode} err={r.stderr.strip()!r} exists={os.path.exists(marker)}")

    # 3. A host secret OUTSIDE the worktree is NOT readable.
    r = run_sandboxed(["/bin/sh", "-c", f"cat {SECRET}"],
                      workdir=worktree, write_root=worktree)
    leaked = "TOP-SECRET" in r.stdout
    check("host secret is NOT readable", r.returncode != 0 and not leaked,
          f"rc={r.returncode} stdout={r.stdout.strip()!r}")

    # 4. Host paths bound read-only cannot be modified. (A write to a path
    # that only exists on the sandbox's ephemeral tmpfs root would succeed
    # harmlessly; the real guarantee is that actual host binds are read-only.)
    r = run_sandboxed(["/bin/sh", "-c", "echo pwned > /opt/app/.venv/pwned"],
                      workdir=worktree, write_root=worktree)
    escaped = os.path.exists("/opt/app/.venv/pwned")
    check("read-only host bind is not writable", r.returncode != 0 and not escaped,
          f"rc={r.returncode} err={r.stderr.strip()!r} escaped={escaped}")

    # 5. No network: a TCP connect must fail (also blocks EC2 metadata).
    net = (
        "import socket,sys\n"
        "try:\n"
        "  socket.create_connection(('1.1.1.1',443),timeout=5); print('CONNECTED')\n"
        "except Exception as e:\n"
        "  print('blocked:',type(e).__name__); sys.exit(7)\n"
    )
    r = run_sandboxed(["python", "-c", net], workdir=worktree, write_root=worktree)
    check("network is unshared", r.returncode == 7 and "CONNECTED" not in r.stdout,
          f"rc={r.returncode} out={r.stdout.strip()!r} err={r.stderr.strip()!r}")

    # Show the exact argv the sandbox builds, for the record.
    print("bwrap argv for a helper command:")
    print("  " + " ".join(build_bwrap_argv(["ruff", "check"],
                                            workdir=worktree, write_root=worktree)))
    print()

    width = max(len(n) for n, _, _ in results)
    all_ok = True
    for name, ok, detail in results:
        all_ok &= ok
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name.ljust(width)}"
        if not ok and detail:
            line += f"   <- {detail}"
        print(line)

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
