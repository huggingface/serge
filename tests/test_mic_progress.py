from types import SimpleNamespace

import pytest

from reviewbot import mic_progress
from reviewbot.config import Config


def claims(**changes):
    raw = {
        "event_name": "workflow_dispatch",
        "ref": "refs/heads/main",
        "run_id": "123",
        "run_attempt": "1",
    }
    return SimpleNamespace(
        repository=changes.pop("repository", mic_progress.REPOSITORY),
        workflow_ref=changes.pop("workflow_ref", mic_progress.WORKFLOW_REF),
        raw={**raw, **changes},
    )


def body(**changes):
    return {
        "job_id": "a" * 32,
        "profile": "cuda",
        "stage": "recording",
        "completed": 2,
        "total": 20,
        "detail": "Demo/default",
        "sequence": 3,
        **changes,
    }


def test_claims_supply_run_identity_instead_of_trusting_the_request():
    verified = mic_progress.verified_progress(claims(), body())
    assert verified["run_id"] == 123 and verified["run_attempt"] == 1
    with pytest.raises(ValueError):
        mic_progress.verified_progress(claims(), body(run_id=999))


@pytest.mark.parametrize(
    "change",
    [
        {"repository": "another/repo"},
        {
            "workflow_ref": "huggingface/transformers-mic/.github/workflows/other.yml@refs/heads/main"
        },
        {"ref": "refs/heads/feature"},
        {"event_name": "pull_request"},
        {"run_id": "bad"},
    ],
)
def test_other_repositories_workflows_and_events_are_rejected(change):
    with pytest.raises(ValueError):
        mic_progress.verified_progress(claims(**change), body())


@pytest.mark.parametrize(
    "change",
    [
        {"stage": "shell"},
        {"completed": 21},
        {"sequence": -1},
        {"total": True},
        {"detail": "hello\nforged log"},
        {"profile": "another-job"},
    ],
)
def test_progress_payload_is_bounded(change):
    with pytest.raises(ValueError):
        mic_progress.verified_progress(claims(), body(**change))


def test_relay_uses_internal_service_auth_without_redirects(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mic_progress.requests,
        "post",
        lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(status_code=200),
    )
    assert mic_progress.relay(
        "http://dashboard/mic-jobs/progress", "service-secret", body()
    ) == {"status": "ok"}
    assert calls[0][1]["allow_redirects"] is False
    assert calls[0][1]["headers"]["Authorization"] == "Bearer service-secret"


def test_mic_workflow_is_allowlisted_without_changing_existing_workflows():
    cfg = Config
    assert "huggingface/transformers-mic" in cfg.dashboard_repositories
    assert "cloud-baselines.yml" in cfg.dashboard_dispatch_workflows
    assert "rerun-failed-gpu.yml" in cfg.dashboard_dispatch_workflows
