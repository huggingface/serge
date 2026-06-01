"""Shared bare-clone + per-job worktree cache.

Without this, every review does a cold ``git init`` + ``git fetch
--depth 50`` of the PR head into its own throwaway tmpdir. N concurrent
reviews on the same repo means N cold fetches at once — an outbound
bandwidth and EBS-burst disaster (see ``SCALE_UP_PLAN.md`` phase 3).

Instead we keep one bare repo per ``(owner, repo)`` and hand each job a
cheap ``git worktree`` checkout off it. The first review on a repo pays
for the fetch; subsequent ones are incremental (the objects are already
local) and the worktree add is near-instant.

Security properties carried over from the old ``_clone_pr_head``:

- The installation token is passed via ``http.extraHeader`` (``-c``), so
  it never lands in process listings (``/proc/<pid>/cmdline``), the
  remote URL, or ``.git/config`` on disk.
- ``GIT_TERMINAL_PROMPT=0`` / ``GIT_ASKPASS=/bin/false`` refuse
  interactive auth prompts that would otherwise hang on failure.
- ``GIT_CONFIG_NOSYSTEM=1`` ignores host-wide git config.
- ``core.symlinks=false`` (set on the bare repo and on checkout) forces
  symlinks in the PR tree to be written as plain files, so helper tools
  rooted at the worktree cannot follow a symlink out of the checkout.
- Any token-shaped substring is scrubbed from git stderr before it is
  logged.
"""

import base64
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional


log = logging.getLogger(__name__)

# git refs / branch names disallow a handful of characters; owner/repo
# from GitHub are already restricted to [A-Za-z0-9._-], but sanitize
# defensively so a surprising value can never escape the cache root or
# inject a refspec.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")

# Config directory whose contents we force to the repo's default branch.
_AI_DIR = ".ai"
# Per-bare-repo ref holding the default-branch tip used for the overlay.
_BASE_REF = "_reviewbot_base"


def _slug(value: str) -> str:
    return _SAFE.sub("-", value)


@dataclass
class Checkout:
    """A live worktree handed to one review worker. Pass it back to
    :meth:`CloneCache.release` when the job finishes."""

    path: str  # worktree directory rooted at the PR head (repo_checkout_path)
    branch: str  # per-job branch in the shared bare repo
    bare: str  # backing bare repo path
    owner: str
    repo: str


class CloneCache:
    """Per-``(owner, repo)`` shared bare repo + per-job worktrees.

    Thread-safe: a per-bare-repo lock serializes fetches and worktree
    mutations into the same backing repo (git locks ``index.lock`` itself,
    but a Python lock turns fail-and-retry into clean serialization)."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self._repos_dir = os.path.join(self.root, "repos")
        self._worktrees_dir = os.path.join(self.root, "worktrees")
        # Maps bare-repo path -> Lock. Guarded by _locks_guard.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ASKPASS": "/bin/false",
        }
        os.makedirs(self._repos_dir, exist_ok=True)
        os.makedirs(self._worktrees_dir, exist_ok=True)

    # -- internals ---------------------------------------------------------

    def _bare_path(self, owner: str, repo: str) -> str:
        return os.path.join(self._repos_dir, f"{_slug(owner)}__{_slug(repo)}.git")

    def _lock_for(self, bare: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(bare)
            if lock is None:
                lock = threading.Lock()
                self._locks[bare] = lock
            return lock

    def _git(
        self,
        repo_dir: str,
        *args: str,
        timeout: int = 120,
        check: bool = True,
        redact: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess:
        cmd = ["git", "-C", repo_dir, *args]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, env=self._env)
        if check and proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            for secret in redact:
                if secret:
                    stderr = stderr.replace(secret, "<redacted>")
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=stderr.encode()
            )
        return proc

    def _ensure_bare(self, bare: str) -> None:
        """Create the bare repo if it doesn't exist yet. Caller holds the
        per-repo lock."""
        if os.path.isdir(os.path.join(bare, "objects")):
            return
        subprocess.run(
            ["git", "init", "--bare", "--quiet", bare],
            check=True,
            capture_output=True,
            timeout=30,
            env=self._env,
        )
        # Applied to every worktree checkout off this repo.
        self._git(bare, "config", "core.symlinks", "false", timeout=30)

    def _overlay_base_ai(
        self,
        bare: str,
        wt: str,
        url: str,
        auth_args: list[str],
        redact: tuple[str, ...],
    ) -> None:
        """Replace the worktree's ``.ai/`` with the repo's default-branch
        copy. The "clone main, grab .ai/, then check out the fork" flow,
        expressed against the shared bare repo.

        Fail-closed: if the default branch can't be fetched we drop the
        PR's ``.ai/`` entirely rather than trust a copy the PR author
        could have tampered with. The caller holds the per-repo lock."""
        ai_path = os.path.join(wt, _AI_DIR)
        try:
            # Fetch the remote's default branch (HEAD) shallowly into a
            # stable ref. "+" forces it so a moved default tip overwrites.
            self._git(
                bare,
                *auth_args,
                "fetch",
                "--depth",
                "1",
                url,
                f"+HEAD:{_BASE_REF}",
                timeout=120,
                redact=redact,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log.warning(
                "could not fetch default branch for .ai overlay (%s); "
                "dropping the PR's .ai/ to stay fail-closed",
                type(exc).__name__,
            )
            shutil.rmtree(ai_path, ignore_errors=True)
            return
        # Whatever the PR shipped under .ai/ is discarded unconditionally.
        shutil.rmtree(ai_path, ignore_errors=True)
        # Only materialize the upstream .ai/ if the default branch has one.
        exists = self._git(
            bare, "cat-file", "-e", f"{_BASE_REF}:{_AI_DIR}", check=False, timeout=30
        )
        if exists.returncode != 0:
            return
        self._git(
            wt,
            "checkout",
            _BASE_REF,
            "--",
            _AI_DIR,
            timeout=60,
            redact=redact,
        )

    # -- public API --------------------------------------------------------

    def acquire(
        self,
        token: Optional[str],
        owner: str,
        repo: str,
        number: int,
        *,
        job_id: str,
        depth: int = 50,
        remote_url: Optional[str] = None,
    ) -> Optional[Checkout]:
        """Fetch ``pull/<number>/head`` into the shared bare repo and add a
        worktree for this job. Returns the checkout, or ``None`` if anything
        went wrong (the caller then runs the review without browse tools).

        ``remote_url`` defaults to the public GitHub HTTPS URL; it exists so
        tests can point at a local repo. When ``token`` is falsy no auth
        header is attached (public repos / tests)."""
        bare = self._bare_path(owner, repo)
        branch = f"pr-{number}-{_slug(job_id)}"
        wt = os.path.join(
            self._worktrees_dir,
            f"{_slug(owner)}__{_slug(repo)}__{number}__{_slug(job_id)}",
        )
        url = remote_url or f"https://github.com/{owner}/{repo}.git"

        auth_args: list[str] = []
        basic = ""
        if token:
            # Basic auth with x-access-token as the username is the
            # documented form for GitHub App installation tokens.
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            auth_args = ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
        redact = (token or "", basic)

        lock = self._lock_for(bare)
        with lock:
            try:
                self._ensure_bare(bare)
                self._git(
                    bare,
                    *auth_args,
                    "-c",
                    "core.symlinks=false",
                    "fetch",
                    "--depth",
                    str(depth),
                    url,
                    f"pull/{number}/head:{branch}",
                    timeout=180,
                    redact=redact,
                )
                # Mark the repo as recently used so GC keeps it alive.
                os.utime(bare, None)
                self._git(
                    bare,
                    "-c",
                    "core.symlinks=false",
                    "worktree",
                    "add",
                    "--quiet",
                    wt,
                    branch,
                    timeout=60,
                    redact=redact,
                )
                # Configuration under .ai/ must come from the repo's own
                # default branch (upstream), never the PR head — a fork PR
                # could otherwise ship a malicious .ai/context-script or
                # helper-tool script that we'd execute. Replace the PR's
                # .ai/ with the default branch's.
                self._overlay_base_ai(bare, wt, url, auth_args, redact)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                stderr = ""
                if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                    stderr = exc.stderr.decode("utf-8", errors="replace")
                log.warning(
                    "clone cache acquire failed for %s/%s#%d (%s): %s",
                    owner,
                    repo,
                    number,
                    type(exc).__name__,
                    stderr or exc,
                )
                # Roll back any partial state so a retry starts clean.
                self._git(
                    bare, "worktree", "remove", "--force", wt, check=False, timeout=30
                )
                self._git(bare, "branch", "-D", branch, check=False, timeout=30)
                shutil.rmtree(wt, ignore_errors=True)
                return None

        return Checkout(path=wt, branch=branch, bare=bare, owner=owner, repo=repo)

    def release(self, checkout: Optional[Checkout]) -> None:
        """Drop a job's worktree and its branch. Best-effort; the bare repo
        and its objects stay for the next review on the same repo."""
        if checkout is None:
            return
        lock = self._lock_for(checkout.bare)
        with lock:
            self._git(
                checkout.bare,
                "worktree",
                "remove",
                "--force",
                checkout.path,
                check=False,
                timeout=60,
            )
            self._git(
                checkout.bare, "branch", "-D", checkout.branch, check=False, timeout=30
            )
        shutil.rmtree(checkout.path, ignore_errors=True)

    def reset_worktrees(self) -> None:
        """Clear orphaned worktrees left by a previous process (their jobs
        were marked crashed on restart). Called once at startup."""
        for name in (
            os.listdir(self._worktrees_dir)
            if os.path.isdir(self._worktrees_dir)
            else []
        ):
            shutil.rmtree(os.path.join(self._worktrees_dir, name), ignore_errors=True)
        if os.path.isdir(self._repos_dir):
            for name in os.listdir(self._repos_dir):
                bare = os.path.join(self._repos_dir, name)
                if os.path.isdir(bare):
                    self._git(bare, "worktree", "prune", check=False, timeout=30)

    def gc(self, max_age_seconds: int, *, now: Optional[float] = None) -> int:
        """Drop bare repos untouched for longer than ``max_age_seconds``.
        ``utime`` is bumped on every :meth:`acquire`, so an actively-used
        repo is never collected. Returns the number of repos removed."""
        if max_age_seconds <= 0 and now is None:
            return 0
        now = time.time() if now is None else now
        removed = 0
        if not os.path.isdir(self._repos_dir):
            return 0
        for name in os.listdir(self._repos_dir):
            bare = os.path.join(self._repos_dir, name)
            if not os.path.isdir(bare):
                continue
            try:
                age = now - os.path.getmtime(bare)
            except OSError:
                continue
            if age < max_age_seconds:
                continue
            lock = self._lock_for(bare)
            with lock:
                self._git(bare, "worktree", "prune", check=False, timeout=30)
                shutil.rmtree(bare, ignore_errors=True)
            removed += 1
        if removed:
            log.info("clone cache GC removed %d stale bare repo(s)", removed)
        return removed
