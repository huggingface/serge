"""Slack notification helper for Serge-owned automation.

This is intentionally separate from the LLM/repo tool registry: Slack tokens
belong to the Serge process and must never be exposed to model-callable tools
or repo-controlled helper commands.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_TIMEOUT_SECONDS = 10


def post_task_pr_created_notification(
    *,
    token: str | None,
    channel: str | None,
    repo_full_name: str,
    pr_number: int | None,
    pr_url: str | None,
    title: str,
    branch: str,
    changed_files: list[str],
) -> bool:
    """Post a Slack notification for a newly opened /tasks PR.

    Returns True when Slack accepted the message. Missing token/channel is a
    no-op so deployments can leave Slack disabled.
    """
    if not token or not channel:
        return False

    pr_label = f"#{pr_number}" if pr_number is not None else "PR"
    pr_text = (
        f"<{pr_url}|{repo_full_name}{pr_label}>"
        if pr_url
        else f"{repo_full_name}{pr_label}"
    )
    files = ", ".join(f"`{path}`" for path in changed_files[:5])
    if len(changed_files) > 5:
        files += f", +{len(changed_files) - 5} more"
    if not files:
        files = "No changed files reported"

    payload: dict[str, Any] = {
        "channel": channel,
        "text": f"Serge opened {pr_text}: {title}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Serge opened an automated fix PR",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{pr_text}*\n{title}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Branch:* `{branch}`   *Files:* {files}",
                    }
                ],
            },
        ],
    }
    try:
        response = requests.post(
            SLACK_POST_MESSAGE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except Exception:
        log.exception(
            "failed to post Slack notification for %s %s", repo_full_name, pr_label
        )
        return False

    if not body.get("ok"):
        log.warning(
            "Slack rejected task PR notification for %s %s: %s",
            repo_full_name,
            pr_label,
            body.get("error") or body,
        )
        return False
    return True


def post_task_finished_notification(
    *,
    token: str | None,
    channel: str | None,
    repo_full_name: str,
    status: str,
    message: str,
    pr_number: int | None = None,
    pr_url: str | None = None,
    job_id: str | None = None,
    error: str | None = None,
) -> bool:
    """Post a Slack notification when a /tasks job reaches a terminal state."""
    if not token or not channel:
        return False

    status_labels = {
        "published": "completed",
        "no_fix": "completed (no fix)",
        "done": "finished",
        "error": "failed",
    }
    status_label = status_labels.get(status, status)
    title = f"Serge task {status_label} for {repo_full_name}"
    details = message or error or "No task result message was reported."
    if pr_number is not None:
        pr_label = f"#{pr_number}"
        pr_text = (
            f"<{pr_url}|{repo_full_name}{pr_label}>"
            if pr_url
            else f"{repo_full_name}{pr_label}"
        )
        details = f"{details}\n{pr_text}"
    if job_id:
        details = f"{details}\nTask `{job_id[:12]}`"

    payload: dict[str, Any] = {
        "channel": channel,
        "text": f"{title}: {details}",
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": details},
            },
        ],
    }
    try:
        response = requests.post(
            SLACK_POST_MESSAGE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=payload,
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
    except Exception:
        log.exception(
            "failed to post Slack task-finished notification for %s", repo_full_name
        )
        return False

    if not body.get("ok"):
        log.warning(
            "Slack rejected task-finished notification for %s: %s",
            repo_full_name,
            body.get("error") or body,
        )
        return False
    return True
