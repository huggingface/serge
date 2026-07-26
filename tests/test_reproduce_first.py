import types

from reviewbot import tasks
from reviewbot.classify import (
    ENVIRONMENT_ISSUE,
    PRODUCT_ISSUE,
    TEST_ISSUE,
    ClassifyResult,
)
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


def _cfg(reproduce_first=True, bail_on_env=True):
    return types.SimpleNamespace(
        verify_on_gpu=True,
        verify_reproduce_first=reproduce_first,
        classify_bail_on_environment=bail_on_env,
        verify_max_rounds=0,
        verify_workflow_file="serge-verify-caller.yml",
        verify_ref="main",
        verify_machine_type="aws-g5-12xlarge-cache",
        verify_transformersci_ref="main",
        verify_poll_timeout=100,
        verify_poll_interval=0,
        classify_max_tokens=4096,
        reproduce_tb_chars=12000,
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
        rec["reproduce_run_url"] = req.reproduce_run_url
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
            REPRODUCED,
            tracebacks={"n": "RuntimeError: boom"},
            run_url="https://github.com/o/r/actions/runs/111-repro",
        ),
        classify_result=ClassifyResult(PRODUCT_ISSUE, reason="hard crash"),
    )
    result = _call(_cfg())
    assert rec["reproduce_called"] and rec["published"]
    seeded = rec["prepare_contexts"][0]
    assert "REPRODUCED on GPU" in seeded
    assert "RuntimeError: boom" in seeded
    assert "genuine library/model bug" in seeded  # product_issue routing note
    # the reproduce run is threaded to the request for the PR "failed before" link
    assert rec["reproduce_run_url"] == "https://github.com/o/r/actions/runs/111-repro"
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


def test_environment_issue_bails_after_one_classifier_call(monkeypatch):
    # The failure is REAL on GPU but not patchable (runner OOM, missing dep, a
    # checkpoint gone from the Hub). Investigating could only end `no_fix`, so we
    # stop here — no LLM cycle, no PR. This is the whole point of the label.
    rec = _install(
        monkeypatch,
        repro_outcome=VerifyOutcome(
            REPRODUCED,
            tracebacks={"n": "torch.OutOfMemoryError: CUDA out of memory"},
            run_url="https://github.com/o/r/actions/runs/222-repro",
        ),
        classify_result=ClassifyResult(
            ENVIRONMENT_ISSUE, reason="CUDA out of memory on the runner"
        ),
    )
    result = _call(_cfg())
    assert rec["reproduce_called"] is True
    assert rec["prepare_contexts"] == []  # never investigated
    assert rec["published"] is False
    assert result.no_change is True
    assert (
        result.verify_verdict == REPRODUCED
    )  # it DID reproduce; it just can't be fixed
    # The recap takes the first line as the reason, so it must lead with it.
    assert result.message.splitlines()[0].startswith("GPU reproduce + classify:")
    assert "ENVIRONMENT issue" in result.message
    assert "CUDA out of memory on the runner" in result.message
    assert "222-repro" in result.message
    # The tracebacks travel with the bail so a human sees what happened.
    assert result.verify_tracebacks == {
        "n": "torch.OutOfMemoryError: CUDA out of memory"
    }


def test_environment_issue_investigates_when_bail_disabled(monkeypatch):
    # CLASSIFY_BAIL_ON_ENVIRONMENT=0 restores investigate-everything, but the
    # agent must still be told what the label means rather than dropped in blind.
    rec = _install(
        monkeypatch,
        repro_outcome=VerifyOutcome(REPRODUCED, tracebacks={"n": "ImportError: x"}),
        classify_result=ClassifyResult(ENVIRONMENT_ISSUE, reason="missing dependency"),
    )
    result = _call(_cfg(bail_on_env=False))
    assert rec["published"] is True
    seeded = rec["prepare_contexts"][0]
    assert "ENVIRONMENT/dependency problem" in seeded
    assert "no source patch fixes it" in seeded
    assert result.pr_number == 1


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


def test_verification_footer_links_both_runs():
    footer = tasks._verification_footer(
        verify_run_url="https://gh/run/verify",
        reproduce_run_url="https://gh/run/repro",
    )
    assert "Verified on GPU" in footer
    assert "Passes with this patch:** https://gh/run/verify" in footer
    assert "Failed before the fix (base commit):** https://gh/run/repro" in footer


def test_verification_footer_empty_when_no_runs():
    assert tasks._verification_footer(None, None) == ""


def test_verification_footer_verify_only():
    # reproduce-first disabled / failed-open: still surface the verify run.
    footer = tasks._verification_footer("https://gh/run/verify", None)
    assert "Passes with this patch:** https://gh/run/verify" in footer
    assert "Failed before" not in footer


def test_verification_footer_notes_flakiness_runs():
    footer = tasks._verification_footer(
        "https://gh/run/verify", "https://gh/run/repro", runs=5
    )
    assert (
        "run 5× on both the pre-patch and patched trees to rule out flakiness" in footer
    )


def test_verification_footer_omits_flakiness_note_for_single_run():
    # runs unknown (None) or a single run: no flakiness sentence.
    assert "flakiness" not in tasks._verification_footer(
        "https://gh/run/verify", None, runs=None
    )
    assert "flakiness" not in tasks._verification_footer(
        "https://gh/run/verify", None, runs=1
    )


def test_decorate_body_places_gpu_section_before_disclaimer():
    cfg = types.SimpleNamespace(is_staging=False)
    plan = tasks.TaskPlan(title="t", body="the actual fix body", patch="p")
    req = types.SimpleNamespace(context="")
    footer = tasks._verification_footer("https://gh/run/verify", "https://gh/run/repro")
    body = tasks._decorate_body(cfg, plan, req, verification_footer=footer)
    assert "Verified on GPU" in body
    # The GPU section vouches for the fix, so it sits right under the fix body
    # and ABOVE the "produced automatically by serge" disclaimer.
    assert body.index("the actual fix body") < body.index("Verified on GPU")
    assert body.index("Verified on GPU") < body.index(
        "This change was produced automatically"
    )


def test_decorate_body_no_gpu_section_without_footer():
    cfg = types.SimpleNamespace(is_staging=False)
    plan = tasks.TaskPlan(title="t", body="b", patch="p")
    req = types.SimpleNamespace(context="")
    assert "Verified on GPU" not in tasks._decorate_body(cfg, plan, req)
