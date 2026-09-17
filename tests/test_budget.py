"""The runner pod's wall-clock budget.

The failure these cover is job `d2c24049` (2026-09-16): a finished patch, a
title, a body, a clean apply — then the Job's activeDeadlineSeconds fired 5m19s
into the normalize step and Kubernetes killed the pod. serge recorded
`task runner exited without reporting (exit code 1)`, the transformers triage
issue showed `⚠️ task failed`, and the two `qwen3_omni_moe` tests stayed unfixed.
Nothing about the work was wrong; it ran out of clock in the one place where
running out of clock throws everything away.
"""

import time
import types

import pytest

from reviewbot import budget, tasks
from reviewbot.reviewer import STOP_DEADLINE


# ── the budget itself ───────────────────────────────────────────────────────


def test_an_unarmed_budget_is_unbounded_not_expired():
    """The legacy in-process worker (TASK_EXECUTION=inprocess, still the
    default) runs inside serge's own web process, which has been up for days. A
    budget measured from *that* process start would read as exhausted on every
    job forever, so unarmed must mean unbounded — never "no time left"."""
    assert budget.remaining() is None
    assert budget.exhausted() is False
    assert budget.clamp(3600) == 3600


def test_arm_measures_from_process_start():
    t0 = budget.PROCESS_START
    budget.arm(7200, start=t0)
    assert budget.remaining(now=t0 + 1200, reserve=0) == pytest.approx(6000)


def test_a_falsy_timeout_disarms():
    budget.arm(7200)
    assert budget.armed()
    budget.arm(0)
    assert not budget.armed()


def test_exhausted_trips_at_the_reserve_not_at_zero():
    """The point of the reserve is to stop *before* the deadline, with enough
    left to land the work — tripping at zero would be the same kill, later."""
    t0 = budget.PROCESS_START
    budget.arm(7200, start=t0)
    assert budget.exhausted(now=t0 + 7200 - 400, reserve=300) is False
    assert budget.exhausted(now=t0 + 7200 - 200, reserve=300) is True


def test_clamp_never_returns_negative():
    t0 = budget.PROCESS_START
    budget.arm(7200, start=t0)
    assert budget.clamp(1800, now=t0 + 99_999) == 0


# ── the derived tail reserve ────────────────────────────────────────────────


def test_tail_reserve_defaults_to_what_the_tail_has_to_cover():
    """Derived, not a second free-standing number: raising the normalize
    timeout must not silently leave the loop running past the point where its
    own patch can still be normalized and pushed."""
    assert budget.tail_reserve(0, 1800) == 1800 + budget.WINDDOWN_SECONDS


def test_an_explicit_tail_reserve_wins():
    assert budget.tail_reserve(2400, 1800) == 2400


# ── the loop stops on its own terms ─────────────────────────────────────────


def _loop_cfg(**over):
    base = dict(
        tool_max_iterations=0,
        llm_max_input_tokens=0,
        tool_result_window=0,
        tool_repeat_limit=0,
        tool_path_revisit_limit=0,
        tool_path_trip_after=0,
        task_tail_reserve=0,
        task_normalize_timeout=1800,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_the_loop_reserve_comes_from_config():
    cfg = _loop_cfg()
    assert budget.tail_reserve(cfg.task_tail_reserve, cfg.task_normalize_timeout) == (
        1800 + budget.WINDDOWN_SECONDS
    )


def test_stop_deadline_is_its_own_reason():
    """It must not be folded into an existing cap: every other stop reason says
    something about the *model*, and this one says the job was expensive in
    time. Conflating them would hide the deploy-side problem (too little budget,
    or too many GPU verify rounds) behind a model-shaped label."""
    from reviewbot import reviewer

    others = {
        reviewer.STOP_ANSWERED,
        reviewer.STOP_INPUT_TOKEN_CAP,
        reviewer.STOP_REPEAT_GUARD,
        reviewer.STOP_PATH_REVISIT_GUARD,
        reviewer.STOP_BLIND_TURN_CAP,
        reviewer.STOP_STRICT_TOOL_CAP,
        reviewer.STOP_ABSOLUTE_CEILING,
        reviewer.STOP_CHUNK_BUDGET,
        reviewer.STOP_NO_LLM_TURNS,
        reviewer.STOP_RUNNER_LOST,
    }
    assert STOP_DEADLINE not in others


# ── the normalizer gives way instead of being killed ────────────────────────


def _normalize_cfg(**over):
    base = dict(
        task_normalize_command=["make", "style"],
        task_normalize_timeout=1800,
        task_sandbox_backend="off",
        task_normalize_image=None,
        helper_sandbox="off",
        task_normalize_memory=None,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_normalizer_timeout_is_cut_to_the_remaining_budget(monkeypatch):
    seen = {}

    def fake_run_normalize(command, **kwargs):
        seen.update(kwargs)
        return 0, ""

    monkeypatch.setattr(tasks, "run_normalize", fake_run_normalize)
    # _run_repo_normalizer reads the clock itself, so arm a budget that expires
    # 600s from now: 420 usable once the wind-down is held back.
    budget.arm(600, start=time.monotonic())
    rc, _ = tasks._run_repo_normalizer(
        _normalize_cfg(), types.SimpleNamespace(path="/tmp/x"), lambda *a: None
    )
    assert rc == 0
    assert 0 < seen["timeout"] <= 420


def test_normalizer_is_skipped_rather_than_started_with_no_budget(monkeypatch):
    """This is the d2c24049 shape. Starting a 30-minute normalize with 5 minutes
    left does not produce a normalized patch — it produces a dead pod and no PR.
    Accepting it un-normalized is the same outcome the NormalizeError branch
    already treats as acceptable, and CI still catches what the normalizer
    would have."""
    called = False

    def fake_run_normalize(command, **kwargs):
        nonlocal called
        called = True
        return 0, ""

    monkeypatch.setattr(tasks, "run_normalize", fake_run_normalize)
    budget.arm(100, start=time.monotonic())  # inside the 180s wind-down already
    rc, out = tasks._run_repo_normalizer(
        _normalize_cfg(), types.SimpleNamespace(path="/tmp/x"), lambda *a: None
    )
    assert not called, "must not start a step it cannot finish"
    assert (rc, out) == (None, ""), (
        "best-effort pass, same as an unavailable normalizer"
    )


def test_normalizer_is_unclamped_when_the_budget_is_not_armed(monkeypatch):
    seen = {}

    def fake_run_normalize(command, **kwargs):
        seen.update(kwargs)
        return 0, ""

    monkeypatch.setattr(tasks, "run_normalize", fake_run_normalize)
    tasks._run_repo_normalizer(
        _normalize_cfg(), types.SimpleNamespace(path="/tmp/x"), lambda *a: None
    )
    assert seen["timeout"] == 1800


# ── wiring ──────────────────────────────────────────────────────────────────


def test_the_runner_is_told_its_tail_reserve():
    from reviewbot import launcher

    assert "task_tail_reserve" in launcher.RUNNER_CONFIG_FIELDS
    assert "task_runner_timeout" in launcher.RUNNER_CONFIG_FIELDS


def test_building_the_runner_config_arms_the_budget():
    """The single arming site. Both the task runner and the review runner go
    through ``build_runner_config``; nothing else does, which is what keeps the
    in-process worker unbounded."""
    from reviewbot.task_runner import build_runner_config

    assert not budget.armed()
    spec = types.SimpleNamespace(
        config={"task_runner_timeout": 7200},
        llm={},
    )
    build_runner_config(spec)
    assert budget.armed()
    assert budget.remaining(reserve=0) is not None


# ── a fresh cycle is not started if it cannot finish ────────────────────────


def test_round_reserve_is_the_tail_plus_room_to_actually_investigate():
    """A round that only has the tail left cannot investigate: the wall-clock
    guard ends its loop on iteration 1 and the model answers with no tools. The
    round already in hand is on a branch, so starting that one trades a real
    result for an empty one."""
    cfg = _normalize_cfg(task_tail_reserve=0)
    assert tasks.round_reserve(cfg) == (
        1800 + budget.WINDDOWN_SECONDS + tasks._MIN_ROUND_LOOP_SECONDS
    )


def test_verify_retry_round_is_skipped_when_the_budget_cannot_cover_it(monkeypatch):
    cfg = types.SimpleNamespace(
        verify_on_gpu=True,
        verify_max_rounds=2,
        task_tail_reserve=0,
        task_normalize_timeout=1800,
        reproduce_block_chars=1000,
        verify_reproduce_first=False,
    )
    prepared = []

    monkeypatch.setattr(tasks, "_maybe_reproduce_first", lambda *a, **k: (None, a[2]))
    monkeypatch.setattr(
        tasks,
        "prepare_task",
        lambda *a, **k: (
            prepared.append(1)
            or types.SimpleNamespace(session={}, patch="x", title="t", body="b")
        ),
    )

    def fake_publish(*a, **k):
        return tasks.TaskResult(mode="new_pr", verify_verdict="not_fixed")

    monkeypatch.setattr(tasks, "publish_task", fake_publish)
    monkeypatch.setattr(tasks, "should_retry", lambda v: True)
    monkeypatch.setattr(tasks, "_format_verify_feedback", lambda *a, **k: "fb")
    monkeypatch.setattr(tasks, "_with_verify_feedback", lambda req, fb: req)

    # Plenty of budget: the retry rounds run.
    budget.arm(20_000, start=time.monotonic())
    tasks.prepare_and_publish_candidate(
        cfg,
        object(),
        types.SimpleNamespace(),
        checkout=types.SimpleNamespace(path="/tmp/x"),
        clone_cache=types.SimpleNamespace(reset_worktree=lambda c: None),
        existing_diff=None,
        job_id="j",
        emit=lambda *a: None,
    )
    assert len(prepared) == 3, "one initial round plus verify_max_rounds retries"

    # Budget inside the round reserve: the first result is kept instead.
    prepared.clear()
    budget.arm(60, start=time.monotonic())
    tasks.prepare_and_publish_candidate(
        cfg,
        object(),
        types.SimpleNamespace(),
        checkout=types.SimpleNamespace(path="/tmp/x"),
        clone_cache=types.SimpleNamespace(reset_worktree=lambda c: None),
        existing_diff=None,
        job_id="j",
        emit=lambda *a: None,
    )
    assert len(prepared) == 1, "no retry round started with no budget for it"
