"""In-process warm git-mirror keeper (SERGE_SHARED_MIRROR_PLAN.md).

serge keeps a warm bare mirror per repo under ``web_mirror_dir`` and refreshes it
on a schedule. Task/review pods (and the in-process/docker backends) borrow those
objects read-only as a fetch seed via :meth:`CloneCache.update_mirror` /
``acquire_ref(mirror_bare=…)``, so their GitHub fetch shrinks to a delta.

Design constraints:

- **No Kubernetes dependency.** This is a plain background thread, not a k8s
  CronJob, so it works identically for the ``inprocess`` / ``docker`` /
  ``kubernetes`` backends. serge must always run without a cluster.
- **Mirror-on-first-request.** A repo enters the warmed set the first time a task
  targets it (:meth:`register`). No static allowlist. The very first request for a
  repo clones from GitHub as before (no mirror yet); subsequent ones are seeded.
- **Fail-soft.** A repo whose App token can't be minted (App not installed) or
  whose fetch fails is logged and skipped, never crashing the loop.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .clone_cache import CloneCache

log = logging.getLogger(__name__)

# Callable[[owner, repo], token|None] — mints a short-lived token for a repo, or
# returns None to skip it (e.g. the App is not installed there).
TokenProvider = Callable[[str, str], Optional[str]]


class MirrorWarmer:
    """Tracks a set of ``(owner, repo)`` and refreshes their bare mirrors on an
    interval. Thread-safe registration; a single background thread does the
    refreshes sequentially (warming is not latency-critical)."""

    def __init__(
        self,
        mirror_dir: str,
        token_provider: TokenProvider,
        *,
        interval_seconds: int = 300,
    ) -> None:
        self._cache = CloneCache(mirror_dir)
        self._token_provider = token_provider
        self._interval = max(30, int(interval_seconds))
        self._repos: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def register(self, owner: str, repo: str) -> bool:
        """Add ``owner/repo`` to the warmed set. Returns True if it was new.
        Cheap and safe to call on every task submission."""
        key = (owner, repo)
        with self._lock:
            if key in self._repos:
                return False
            self._repos.add(key)
        log.info("mirror: now tracking %s/%s", owner, repo)
        return True

    def _tracked(self) -> list[tuple[str, str]]:
        with self._lock:
            return sorted(self._repos)

    def refresh_one(self, owner: str, repo: str) -> bool:
        """Refresh a single repo's mirror. Returns True on success, False if it
        was skipped (no token) or failed. Never raises."""
        try:
            token = self._token_provider(owner, repo)
        except Exception:  # noqa: BLE001 — token minting is best-effort
            log.warning("mirror: could not mint token for %s/%s", owner, repo)
            return False
        if not token:
            log.info(
                "mirror: skipping %s/%s (no token / App not installed)", owner, repo
            )
            return False
        t0 = time.monotonic()
        try:
            self._cache.update_mirror(token, owner, repo)
        except Exception:  # noqa: BLE001 — one repo must not kill the loop
            log.warning("mirror: refresh failed for %s/%s", owner, repo, exc_info=True)
            return False
        log.info("mirror: refreshed %s/%s in %.1fs", owner, repo, time.monotonic() - t0)
        return True

    def refresh_all(self) -> None:
        for owner, repo in self._tracked():
            if self._stop.is_set():
                return
            self.refresh_one(owner, repo)

    def run_forever(self) -> None:
        """Refresh loop for the background thread: warm the tracked set, then
        wait ``interval`` (interruptible) and repeat until :meth:`stop`."""
        log.info("mirror warmer started (interval=%ss)", self._interval)
        while not self._stop.is_set():
            self.refresh_all()
            self._stop.wait(self._interval)

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self.run_forever, name="mirror-warmer", daemon=True
        )
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()
