"""GitHub-native AI code reviewer for any OpenAI-compatible LLM."""

# This package is the serge reviewer.

import functools as _functools
import os as _os
import subprocess as _subprocess
from typing import Optional

__version__ = "0.1.0"


@_functools.lru_cache(maxsize=1)
def git_sha() -> Optional[str]:
    """Short commit SHA of the running build, or ``None`` if unknown.

    Container images ship without the source ``.git`` directory, so CI
    bakes the commit into the ``SERGE_GIT_SHA`` env var at build time (see
    the Dockerfile ``ARG``). When running from a source checkout that env
    var is absent, so we fall back to asking git directly — that keeps the
    SHA accurate during local development.
    """
    env = (_os.environ.get("SERGE_GIT_SHA") or "").strip()
    if env:
        return env[:12]
    try:
        out = _subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=_os.path.dirname(__file__),
        )
    except Exception:  # noqa: BLE001 — git missing / not a checkout / timeout
        return None
    sha = out.stdout.strip()
    return sha or None


def build_info() -> dict:
    """``{"version", "commit"}`` identifying the running Serge build.

    Embedded in every JSON response and the web UI footer so an operator
    can tell at a glance which commit a deployment is actually serving."""
    return {"version": __version__, "commit": git_sha()}
