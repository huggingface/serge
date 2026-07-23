import types

from reviewbot import tasks
from reviewbot.classify import PRODUCT_ISSUE, TEST_ISSUE, ClassifyResult
from reviewbot.verify import (
    DISPATCH_FAILED,
    NOT_REPRODUCED,
    REPRODUCED,
    VerifyOutcome,
)

WHISPER = "tests/models/whisper/test_modeling_whisper.py"
CTX = (
    f"- `{WHISPER}::WhisperModelIntegrationTests::test_x` [multi-gpu] (other, seen 7/7)"
)


def _cfg(reproduce_first=True):
    return types.SimpleNamespace(
        verify_on_gpu=True,
        verify_reproduce_first=reproduce_first,
        verify_max_rounds=0,
        verify_workflow_file="serge-verify-caller.yml",
        verify_ref="main",
        verify_machine_type="aws-g5-12xlarge-cache",
        verify_transformersci_ref="main",
        verify_poll_timeout=100,
        verify_poll_interval=0,
    )


class _FakeGH:
    def __init__(self, ref_exc=None):
        self._ref_exc = ref_exc

    def get_ref_sha(self, owner, repo, ref):
        if self._ref_exc is not None:
            raise self._ref_exc
        return "basesha1234"


class _FakeCloneCache:
    def reset_worktree(self, checkout):
        pass


def _install(monkeypatch, *, repro_outcome, classify_result=None):
    """Stub the GPU reproduce + classifier + prepare/publish. Returns a dict
    recording what prepare_task saw and whether publish ran."""
    rec = {"reproduce_called": False, "prepare_contexts": [], "published": False}

    def fake_reproduce(gh, **kwargs):
        rec["reproduce_called"] = True
        rec["reproduce_kwargs"] = kwargs
        return repro_outcome

    def fake_classify(cfg, node_ids, tracebacks, context, emit):
        return classify_result or ClassifyResult(PRODUCT_ISSUE, reason="crash")

    def fake_prepare(cfg, req, **kwargs):
        rec["prepare_contexts"].append(req.context)
        return tasks.TaskPlan(title="t", body="b", patch="p")

    def fake_publish(cfg, gh, req, plan, **kwargs):
        rec["published"] = True
        return tasks.TaskResult(mode="new_pr", pr_number=1, message="opened")

    monkeypatch.setattr(tasks, "run_gpu_reproduce", fake_reproduce)
    monkeypatch.setattr(tasks, "_classify_reproduced", fake_classify)
    monkeypatch.setattr(tasks, "prepare_task", fake_prepare)
    monkeypatch.setattr(tasks, "publish_task", fake_publish)
    return rec


def _call(cfg, gh=None):
    req = tasks.TaskRequest(
        owner="huggingface",
        repo="transformers",
        base_ref="main",
        instruction="fix it",
        context=CTX,
    )
    return tasks.prepare_and_publish_candidate(
        cfg,
        gh or _FakeGH(),
        req,
        checkout=object(),
        clone_cache=_FakeCloneCache(),
        existing_diff=None,
        job_id="job1234",
        emit=lambda *_a: None,
    )


def test_disabled_skips_reproduce(monkeypatch):
    rec = _install(monkeypatch, repro_outcome=VerifyOutcome(REPRODUCED))
    result = _call(_cfg(reproduce_first=False))
    assert rec["reproduce_called"] is False
    assert rec["prepare_contexts"] == [CTX]  # original context, unseeded
    assert result.pr_number == 1


def test_not_reproduced_bails_without_llm(monkeypatch):
    rec = _install(
        monkeypatch,
        repro_outcome=VerifyOutcome(
            NOT_REPRODUCED, run_url="u", detail="green at base"
        ),
    )
    result = _call(_cfg())
    assert rec["reproduce_called"] is True
    assert rec["prepare_contexts"] == []  # never investigated
    assert rec["published"] is False
    assert result.no_change is True
    assert result.verify_verdict == NOT_REPRODUCED


def test_reproduced_seeds_prompt_and_investigates(monkeypatch):
    rec = _install(
        monkeypatch,
        repro_outcome=VerifyOutcome(
            REPRODUCED, tracebacks={"n": "RuntimeError: boom"}, run_url="u"
        ),
        classify_result=ClassifyResult(PRODUCT_ISSUE, reason="hard crash"),
    )
    result = _call(_cfg())
    assert rec["reproduce_called"] and rec["published"]
    seeded = rec["prepare_contexts"][0]
    assert "REPRODUCED on GPU" in seeded
    assert "RuntimeError: boom" in seeded
    assert "genuine library/model bug" in seeded  # product_issue routing note
    assert result.pr_number == 1
    # reproduce dispatched with a distinct correlation id + resolved base sha.
    assert rec["reproduce_kwargs"]["correlation_id"] == "job1234-repro"
    assert rec["reproduce_kwargs"]["base_sha"] == "basesha1234"


def test_reproduced_test_issue_routing_note(monkeypatch):
    rec = _install(
        monkeypatch,
        repro_outcome=VerifyOutcome(REPRODUCED, tracebacks={"n": "AssertionError"}),
        classify_result=ClassifyResult(TEST_ISSUE, reason="stale expected values"),
    )
    _call(_cfg())
    assert "TEST/expectations issue" in rec["prepare_contexts"][0]


def test_infra_error_fails_open_to_investigate(monkeypatch):
    # A dispatch/timeout failure must NOT block the fix — investigate unseeded.
    rec = _install(monkeypatch, repro_outcome=VerifyOutcome(DISPATCH_FAILED))
    result = _call(_cfg())
    assert rec["reproduce_called"] is True
    assert rec["prepare_contexts"] == [CTX]  # original, unseeded
    assert result.pr_number == 1


def test_base_ref_unresolvable_fails_open(monkeypatch):
    rec = _install(monkeypatch, repro_outcome=VerifyOutcome(REPRODUCED))
    result = _call(_cfg(), gh=_FakeGH(ref_exc=RuntimeError("404")))
    assert rec["reproduce_called"] is False  # never got to dispatch
    assert rec["prepare_contexts"] == [CTX]
    assert result.pr_number == 1


def test_no_nodeids_skips_reproduce(monkeypatch):
    rec = _install(monkeypatch, repro_outcome=VerifyOutcome(REPRODUCED))
    req = tasks.TaskRequest(
        owner="huggingface",
        repo="transformers",
        base_ref="main",
        instruction="fix it",
        context="- no node ids in this context",
    )
    tasks.prepare_and_publish_candidate(
        _cfg(),
        _FakeGH(),
        req,
        checkout=object(),
        clone_cache=_FakeCloneCache(),
        existing_diff=None,
        job_id="job1234",
        emit=lambda *_a: None,
    )
    assert rec["reproduce_called"] is False
    assert rec["prepare_contexts"] == ["- no node ids in this context"]
