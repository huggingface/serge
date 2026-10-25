"""Relay narrowly scoped GitHub OIDC progress to the dashboard's internal API."""

import re

import requests

REPOSITORY = "huggingface/transformers-mic"
WORKFLOW_REF = REPOSITORY + "/.github/workflows/cloud-baselines.yml@refs/heads/main"
AUDIENCE = "serge-mic-baselines"


def verified_progress(claims, body):
    if (
        claims.repository != REPOSITORY
        or claims.workflow_ref != WORKFLOW_REF
        or claims.raw.get("event_name") != "workflow_dispatch"
        or claims.raw.get("ref") != "refs/heads/main"
    ):
        raise ValueError("workflow_not_allowed")
    if not isinstance(body, dict) or set(body) != {
        "job_id",
        "profile",
        "stage",
        "completed",
        "total",
        "detail",
        "sequence",
    }:
        raise ValueError("invalid_progress")
    if not isinstance(body["job_id"], str) or not re.fullmatch(
        r"[a-f0-9]{32}", body["job_id"]
    ):
        raise ValueError("invalid_job_id")
    if body["profile"] not in ("cpu", "cuda", "publish") or body["stage"] not in (
        "planning",
        "downloading-models",
        "recording",
        "validating",
        "uploading",
        "complete",
    ):
        raise ValueError("invalid_stage")
    for key in ("completed", "total", "sequence"):
        if type(body[key]) is not int or not 0 <= body[key] <= 1_000_000:
            raise ValueError("invalid_counter")
    if (
        body["completed"] > body["total"]
        or not isinstance(body["detail"], str)
        or len(body["detail"]) > 200
        or any(ord(char) < 32 for char in body["detail"])
    ):
        raise ValueError("invalid_progress")
    run_id, attempt = (
        str(claims.raw.get("run_id", "")),
        str(claims.raw.get("run_attempt", "")),
    )
    if not run_id.isdigit() or not attempt.isdigit():
        raise ValueError("invalid_run_claims")
    return {**body, "run_id": int(run_id), "run_attempt": int(attempt)}


def relay(url, token, payload):
    result = requests.post(
        url,
        json=payload,
        headers={"Authorization": "Bearer " + token},
        timeout=10,
        allow_redirects=False,
    )
    if result.status_code != 200:
        raise ValueError("dashboard_progress_unavailable")
    return {"status": "ok"}
