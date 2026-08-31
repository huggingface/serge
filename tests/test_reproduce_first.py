import types
import unittest

from reviewbot import prompts, tasks
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
        reproduce_block_chars=32000,
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


class TracebackClipTests(unittest.TestCase):
    """Regression cover for the two truncation bugs the 2026-08-31 reproduce
    artifacts exposed. Both fixtures mirror a real longrepr layout: the
    load-bearing ``E`` block sits at the HEAD in one shape and at the END in the
    other, so a single-ended cut always deletes it for one of them."""

    # mm_grounding_dino: pytest prepends the assertion summary, so everything
    # the model needs is in the first ~2.4k and the rest is --showlocals noise.
    HEAD_SHAPE = (
        "AssertionError: Tensor-likes are not close!\n"
        "Mismatched elements: 9 / 9 (100.0%)\n"
        "        expected_logits = torch.tensor([[-5.1160, -0.2143]])\n"
        ">       torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits)\n"
        "E       AssertionError: Tensor-likes are not close!\n"
    ) + "pixel_values = tensor([[0.2795, 0.3138]])\n" * 4000

    # cwm / kimi_k25: a huge inline Expectations({...}) literal in the test
    # source pushes the E block to within ~10k of the end.
    TAIL_SHAPE = (
        (
            "self = <tests.models.kimi_k25.test_modeling_kimi_k25.KimiK25IntegrationTest>\n"
            "    expectations = Expectations({('cuda', None): [\n"
        )
        + "        [0.123, 0.456, 0.789],\n" * 4000
        + ("E       RuntimeError: index out of range in self\n")
    )

    def test_head_shape_keeps_the_assertion_diff(self) -> None:
        out = tasks._clip_tb(self.HEAD_SHAPE, 12000)
        self.assertLess(len(self.HEAD_SHAPE), 200_000)
        self.assertIn("Tensor-likes are not close", out)
        self.assertIn("expected_logits = torch.tensor", out)
        self.assertIn("E       AssertionError", out)

    def test_tail_shape_keeps_the_error_line(self) -> None:
        out = tasks._clip_tb(self.TAIL_SHAPE, 12000)
        self.assertIn("E       RuntimeError: index out of range in self", out)

    def test_a_tail_only_cut_would_have_lost_the_head_shape(self) -> None:
        """The bug this replaces: the old cut kept only ``text[-max_chars:]``."""
        self.assertNotIn("expected_logits = torch.tensor", self.HEAD_SHAPE[-12000:])

    def test_short_tracebacks_are_returned_whole(self) -> None:
        self.assertEqual(tasks._clip_tb("short", 12000), "short")
        self.assertEqual(tasks._clip_tb(None, 12000), "")

    def test_clip_respects_the_budget(self) -> None:
        out = tasks._clip_tb("x" * 50_000, 12000)
        # The marker adds a bounded constant; the kept text is within budget.
        self.assertLessEqual(len(out), 12000 + 200)

    def test_budget_is_divided_across_the_tracebacks_present(self) -> None:
        self.assertEqual(tasks._tb_budget(1, 32000), 32000)
        self.assertEqual(tasks._tb_budget(2, 32000), 16000)
        # Never below the floor, however many tests failed.
        self.assertEqual(tasks._tb_budget(20, 32000), tasks.MIN_TB_CHARS)
        self.assertEqual(tasks._tb_budget(0, 32000), 32000)

    def test_reproduce_block_stays_within_the_context_reserve(self) -> None:
        """A 5-traceback block used to be able to reach 5x the per-tb cap; the
        block budget is what has to fit inside CONTEXT_TAIL_RESERVE_CHARS."""
        outcome = VerifyOutcome(
            verdict=REPRODUCED,
            run_url="u",
            detail="d",
            tracebacks={f"t{i}": "y" * 200_000 for i in range(5)},
        )
        block = tasks._format_reproduce_feedback(
            outcome, ClassifyResult(TEST_ISSUE, "r"), max_chars=32000
        )
        self.assertLessEqual(len(block), prompts.CONTEXT_TAIL_RESERVE_CHARS)


class ReproduceRefForFollowUpTests(unittest.TestCase):
    """Which commit reproduce-first runs AT.

    The stale-group burn: the nightly re-dispatched PR #48414's group on two
    consecutive nights (`f31aa5d8`, then `9a898563`), each ran a full session
    — 43 and 60 turns, ~2.5M input tokens between them — and each ended
    `verify_verdict=already_passing`. Reproduce-first asked "is this still
    broken on main?", which stays true until the fix PR merges, while the end
    verify gate used the PR head. The two gates disagreed about the baseline
    and the expensive one was wrong.
    """

    def _gh(self, refs):
        asked = []

        class _GH:
            def get_ref_sha(self, owner, repo, ref):
                asked.append(ref)
                return refs[ref]

        return _GH(), asked

    def _req(self, **kw):
        base = dict(
            owner="huggingface",
            repo="transformers",
            base_ref="main",
            instruction="fix",
            context=CTX,
            mode="new_pr",
        )
        base.update(kw)
        return tasks.TaskRequest(**base)

    def _run(self, req, verdict):
        gh, asked = self._gh(
            {"heads/main": "mainsha", "heads/serge/fix/itf-abc": "prheadsha"}
        )
        seen = {}

        def fake_reproduce(_gh, **kw):
            seen.update(kw)
            return VerifyOutcome(
                verdict=verdict, run_url="u", detail="d", tracebacks={"t": "tb"}
            )

        orig = tasks.run_gpu_reproduce
        tasks.run_gpu_reproduce = fake_reproduce
        try:
            bail, out = tasks._maybe_reproduce_first(
                _cfg(), gh, req, "job1", lambda *_a: None
            )
        finally:
            tasks.run_gpu_reproduce = orig
        return bail, out, asked, seen

    def test_a_follow_up_reproduces_at_the_pr_head(self) -> None:
        req = self._req(
            mode="existing_pr", pr_number=48414, head_branch="serge/fix/itf-abc"
        )
        _bail, _out, asked, seen = self._run(req, REPRODUCED)
        self.assertEqual(asked, ["heads/serge/fix/itf-abc"])
        self.assertEqual(seen["base_sha"], "prheadsha")

    def test_a_new_pr_still_reproduces_at_the_base_branch(self) -> None:
        _bail, _out, asked, seen = self._run(self._req(), REPRODUCED)
        self.assertEqual(asked, ["heads/main"])
        self.assertEqual(seen["base_sha"], "mainsha")

    def test_a_follow_up_whose_pr_already_passes_costs_no_llm(self) -> None:
        req = self._req(
            mode="existing_pr", pr_number=48414, head_branch="serge/fix/itf-abc"
        )
        bail, _out, _asked, _seen = self._run(req, NOT_REPRODUCED)
        self.assertIsNotNone(bail)
        self.assertTrue(bail.no_change)
        self.assertEqual(bail.pr_number, 48414)
        self.assertEqual(bail.verify_verdict, NOT_REPRODUCED)

    def test_an_existing_pr_with_no_head_branch_falls_back_to_base(self) -> None:
        req = self._req(mode="existing_pr", pr_number=1)
        _bail, _out, asked, _seen = self._run(req, REPRODUCED)
        self.assertEqual(asked, ["heads/main"])


class FixableOomOverridesTheEnvironmentBailTests(unittest.TestCase):
    """serge's classifier prompt made `environment_issue` cover *any* OOM, so
    every OOM bailed with 0 LLM turns. transformers-ci's triage has known better
    since 2026-08-14 (26 of 54 persistent OOMs were retention-shaped and had
    been deferred as "needs capacity" for weeks) and dispatches the retention
    and load shapes as source patches.

    They disagreed and the less-informed one won because it runs later: job
    `c6836491` (`phimoe`) is a CUDA OOM on the weight-conversion path — `load` —
    and serge killed it. The verdict tool now measures the shape on the GPU box,
    where the whole traceback exists.
    """

    def _oom_outcome(self, shapes):
        return VerifyOutcome(
            verdict=REPRODUCED,
            run_url="u",
            detail="d",
            tracebacks={"t": "tb"},
            result={"oom_shapes": shapes} if shapes is not None else None,
        )

    def test_load_and_retention_are_fixable(self) -> None:
        self.assertEqual(
            tasks.fixable_oom_shapes(
                self._oom_outcome({"a": "load", "b": "retention"})
            ),
            ["a", "b"],
        )

    def test_capacity_and_unknown_are_not(self) -> None:
        self.assertEqual(
            tasks.fixable_oom_shapes(
                self._oom_outcome({"a": "capacity", "b": "unknown"})
            ),
            [],
        )

    def test_a_non_oom_environment_failure_has_no_shapes(self) -> None:
        # A missing dependency is still an environment bail — nothing to override.
        self.assertEqual(tasks.fixable_oom_shapes(self._oom_outcome({})), [])
        self.assertEqual(tasks.fixable_oom_shapes(self._oom_outcome(None)), [])

    def test_an_old_verdict_artifact_without_the_key_still_bails(self) -> None:
        """The verdict tool ships on merge to transformers-ci `main`, so serge
        can see artifacts from before it. Missing key must mean "no override"."""
        self.assertEqual(
            tasks.fixable_oom_shapes(
                VerifyOutcome(verdict=REPRODUCED, run_url="u", detail="d", result={})
            ),
            [],
        )

    def _run_gate(self, shapes, label=ENVIRONMENT_ISSUE):
        """Drive `_maybe_reproduce_first` with a reproduced OOM of `shapes`."""

        class _GH:
            def get_ref_sha(self, owner, repo, ref):
                return "basesha"

        req = tasks.TaskRequest(
            owner="huggingface",
            repo="transformers",
            base_ref="main",
            instruction="fix",
            context=CTX,
            mode="new_pr",
        )
        orig_rep, orig_cls = tasks.run_gpu_reproduce, tasks._classify_reproduced
        tasks.run_gpu_reproduce = lambda _gh, **kw: self._oom_outcome(shapes)
        tasks._classify_reproduced = lambda *a, **k: ClassifyResult(label, "oom")
        try:
            return tasks._maybe_reproduce_first(
                _cfg(), _GH(), req, "job1", lambda *_a: None
            )
        finally:
            tasks.run_gpu_reproduce, tasks._classify_reproduced = orig_rep, orig_cls

    def test_a_patchable_oom_is_investigated_not_skipped(self) -> None:
        bail, seeded = self._run_gate({"t": "load"})
        self.assertIsNone(bail, "a load-shaped OOM must not bail")
        self.assertIn("REPRODUCED on GPU", seeded.context)

    def test_a_capacity_oom_still_skips(self) -> None:
        bail, _ = self._run_gate({"t": "capacity"})
        self.assertIsNotNone(bail)
        self.assertTrue(bail.no_change)
