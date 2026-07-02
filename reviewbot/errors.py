"""Shared error-rendering helpers for the SSE/task surfaces.

These render provider/GitHub failures into a safe, actionable message for the
web UI and the task-runner callback. They live here (rather than in ``webapp``)
so the in-pod task runner (:mod:`reviewbot.task_runner`) can reuse them without
importing the whole FastAPI app.

Both messages come from the upstream service's own response body — no serge
tokens are echoed — so they are safe to surface to callers.
"""

from __future__ import annotations

import requests

from .llm_client import LLMResponseError


def format_llm_error(exc: LLMResponseError) -> str:
    """Render an LLMResponseError for the SSE client. Surfaces the status
    code + a body excerpt so the UI shows whether it was a 429 (rate
    limit), 400 (bad schema), auth, etc. instead of a generic "review
    crashed". The body comes from the LLM provider's own error response —
    no auth tokens of ours are echoed there."""
    excerpt = exc.body_preview.strip()
    if len(excerpt) > 600:
        excerpt = excerpt[:600] + "…"
    reason_part = f" {exc.reason}" if exc.reason else ""
    if excerpt:
        return f"LLM endpoint returned {exc.status_code}{reason_part}: {excerpt}"
    return f"LLM endpoint returned {exc.status_code}{reason_part}"


def format_github_http_error(exc: requests.HTTPError) -> str:
    """Render a GitHub REST error for the task SSE client. The message comes
    from GitHub's own response (no serge tokens), so it's safe to surface and
    far more actionable than a generic "task crashed". Adds a hint for the
    common write-permission failure on the /tasks flow."""
    msg = str(exc) or "GitHub API request failed"
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    lowered = msg.lower()
    if status_code == 403 or "resource not accessible by integration" in lowered:
        msg += (
            "\n\nHint: the GitHub App installation lacks write access. The "
            "/tasks flow needs Contents: write + Pull requests: write — grant "
            "those in the App settings and re-accept the installation on the "
            "org/repo."
        )
    return msg
