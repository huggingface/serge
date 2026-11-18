"""Keep the task-runner toolchain current without a serge deploy (serge#126).

The task-runner image freezes the target repo's quality toolchain at build time,
and the normalize gate cannot refresh it (``--no-deps``: a task pod has no PyPI).
Every time transformers moved a pin past the image, every task failed the gate
until someone rebuilt and redeployed by hand — four times between 2026-08-17 and
2026-11-16, the last costing two nightlies.

Two halves live here; the third is ``.github/workflows/toolchain.yml``:

- **The workflow** rebuilds the image for a serge commit whenever the target
  repo's pins change, proves the result passes ``make fix-repo`` on a pristine
  checkout, and only then moves the tag ``sha-<serge>-toolchain`` onto it. Same
  serge code, fresher toolchain; every build also keeps an immutable
  ``sha-<serge>-tc-<pins hash>`` tag.
- :func:`resolve_runner_image` runs at each task launch: if the moving tag
  exists for the configured image it launches that, *pinned by digest* so the
  job records exactly what ran; otherwise the configured image unchanged.
- :class:`RefreshDispatcher` closes the gap between scheduled runs: when a task
  reports the gate broken on the pristine checkout, it asks GitHub to run the
  workflow now, for the serge commit this deployment runs.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

log = logging.getLogger(__name__)

# The two messages tasks.py emits (as ``normalize_error`` events) when the gate
# rejects the UNPATCHED checkout — the only failures a rebuild can fix.
GATE_BROKEN_PREFIXES = (
    "Preflight fails on the pristine checkout",
    "Normalizer fails on the pristine checkout",
)

_MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


@dataclass(frozen=True)
class ImageRef:
    registry: str
    repository: str
    tag: str

    @classmethod
    def parse(cls, image: str) -> Optional["ImageRef"]:
        """``registry/repo:tag`` → parts; None for digest refs or no tag."""
        if "@" in image:
            return None
        name, sep, tag = image.rpartition(":")
        if not sep or "/" in tag:
            return None  # no tag (the colon was a registry port)
        first, _, rest = name.partition("/")
        if rest and ("." in first or ":" in first or first == "localhost"):
            return cls(first, rest, tag)
        repo = name if "/" in name else f"library/{name}"
        return cls("registry-1.docker.io", repo, tag)


def _bearer_challenge(header: str) -> dict[str, str]:
    if not header.lower().startswith("bearer "):
        return {}
    return dict(re.findall(r'(\w+)="([^"]*)"', header[7:]))


def manifest_digest(
    registry: str,
    repository: str,
    tag: str,
    *,
    session: Any = requests,
    timeout: float = 10.0,
) -> Optional[str]:
    """The digest ``registry/repository:tag`` points at, or None if absent.

    Anonymous pull (the serge images are public on GHCR): follows the registry's
    Bearer challenge for a pull token. Raises on transport/registry errors so
    the caller can tell "no such tag" from "could not ask"."""
    url = f"https://{registry}/v2/{repository}/manifests/{tag}"
    headers = {"Accept": _MANIFEST_ACCEPT}
    resp = session.head(url, headers=headers, timeout=timeout)
    if resp.status_code == 401:
        challenge = _bearer_challenge(resp.headers.get("WWW-Authenticate", ""))
        if not challenge.get("realm"):
            resp.raise_for_status()
        params = {k: v for k, v in challenge.items() if k in ("service", "scope")}
        params.setdefault("scope", f"repository:{repository}:pull")
        tok = session.get(challenge["realm"], params=params, timeout=timeout)
        tok.raise_for_status()
        body = tok.json()
        token = body.get("token") or body.get("access_token")
        headers["Authorization"] = f"Bearer {token}"
        resp = session.head(url, headers=headers, timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    digest = resp.headers.get("Docker-Content-Digest")
    if digest:
        return digest
    # Some registries omit the header on HEAD; the digest is the body's hash.
    full = session.get(url, headers=headers, timeout=timeout)
    full.raise_for_status()
    return "sha256:" + hashlib.sha256(full.content).hexdigest()


@dataclass(frozen=True)
class ResolvedImage:
    image: str  # what to launch
    refreshed: bool  # True = the toolchain-refreshed build, pinned by digest
    note: str  # one line for the task log


_cache: dict[tuple[str, str], tuple[float, ResolvedImage]] = {}
_cache_lock = threading.Lock()


def resolve_runner_image(
    image: str,
    suffix: str,
    *,
    lookup: Callable[[str, str, str], Optional[str]] = manifest_digest,
    ttl: float = 300.0,
    now: Callable[[], float] = time.monotonic,
) -> ResolvedImage:
    """The image a task should launch with.

    ``image`` is the configured ``TASK_RUNNER_IMAGE``. With a ``suffix`` (e.g.
    ``-toolchain``) and a tagged ref, looks up ``<tag><suffix>`` and returns it
    as ``repo:<tag><suffix>@<digest>`` — the tag for a human, the digest so the
    pod runs exactly the build that was looked up even if the tag moves during
    the pull. Anything else (no suffix, digest ref, tag absent, registry
    unreachable) returns ``image`` unchanged: the refresh can only ever make
    the toolchain newer, never stop a task from launching. Cached for ``ttl``
    seconds so a nightly burst of tasks asks the registry once."""
    if not suffix:
        return ResolvedImage(image, False, "")
    ref = ImageRef.parse(image)
    if ref is None:
        return ResolvedImage(image, False, "")
    key = (image, suffix)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now():
            return hit[1]
    pointer = f"{ref.tag}{suffix}"
    try:
        digest = lookup(ref.registry, ref.repository, pointer)
    except Exception as exc:  # noqa: BLE001 — any failure falls back
        log.warning(
            "toolchain pointer %s:%s lookup failed: %s", ref.repository, pointer, exc
        )
        result = ResolvedImage(
            image, False, f"Toolchain refresh lookup failed ({exc}); using {image}."
        )
    else:
        if digest:
            name = image.rpartition(":")[0]
            result = ResolvedImage(
                f"{name}:{pointer}@{digest}",
                True,
                f"Using the toolchain-refreshed build of this serge commit "
                f"({name}:{pointer}, {digest[:19]}).",
            )
        else:
            result = ResolvedImage(image, False, "")
    with _cache_lock:
        _cache[key] = (now() + ttl, result)
    return result


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def is_gate_broken(kind: str, text: str) -> bool:
    return kind == "normalize_error" and text.startswith(GATE_BROKEN_PREFIXES)


class RefreshDispatcher:
    """Ask GitHub to run the toolchain refresh workflow, at most once per
    ``cooldown`` seconds.

    A broken gate breaks every task of the night, so the first report is all the
    signal there is; the rest would only queue duplicate builds. The workflow
    itself skips when the image for the current pins already exists, so a
    dispatch that races the hourly run costs one quick job."""

    def __init__(
        self,
        *,
        token: Optional[str],
        repo: str,
        workflow: str,
        serge_ref: str,
        cooldown: float = 1800.0,
        session: Any = requests,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.token = token
        self.repo = repo
        self.workflow = workflow
        self.serge_ref = serge_ref
        self.cooldown = cooldown
        self.session = session
        self.now = now
        self._last: Optional[float] = None
        self._lock = threading.Lock()

    def request(self, reason: str) -> bool:
        """Dispatch unless disabled or cooling down. True = dispatched."""
        if not self.token or not self.serge_ref:
            log.warning(
                "toolchain gate broken (%s); no TOOLCHAIN_REFRESH_TOKEN, "
                "waiting for the scheduled refresh",
                reason,
            )
            return False
        with self._lock:
            t = self.now()
            if self._last is not None and t - self._last < self.cooldown:
                return False
            self._last = t
        url = (
            f"https://api.github.com/repos/{self.repo}/actions/workflows/"
            f"{self.workflow}/dispatches"
        )
        try:
            resp = self.session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                },
                json={"ref": "main", "inputs": {"serge_ref": self.serge_ref}},
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            log.warning("toolchain refresh dispatch failed: %s", exc)
            with self._lock:
                self._last = None  # let the next report retry
            return False
        log.info(
            "toolchain gate broken (%s); dispatched %s for %s",
            reason,
            self.workflow,
            self.serge_ref,
        )
        return True
