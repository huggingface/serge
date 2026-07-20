import types

from reviewbot import tasks, verify


def _cfg(on_gpu: bool, rounds: int = 2):
    return types.SimpleNamespace(verify_on_gpu=on_gpu, verify_max_rounds=rounds)


def _req(context="original context"):
    return tasks.TaskRequest(
        owner="huggingface",
        repo="transformers",
        base_ref="main",
        instruction="fix it",
        context=context,
    )


class _FakeCloneCache:
    def __init__(self):
        self.resets = 0

    def reset_worktree(self, checkout):
        self.resets += 1


def _install(monkeypatch, results):
    """Make prepare_task a no-op and publish_task pop from `results`, recording
    the context each round saw. Returns the recorder."""
    seen_contexts = []

    def fake_prepare(cfg, req, **kwargs):
        seen_contexts.append(req.context)
        return tasks.TaskPlan(title="t", body="b", patch="p")

    def fake_publish(cfg, gh, req, plan, **kwargs):
        return results.pop(0)

    monkeypatch.setattr(tasks, "prepare_task", fake_prepare)
    monkeypatch.setattr(tasks, "publish_task", fake_publish)
    return seen_contexts


def _call(cfg, seen_cache=None):
    cc = seen_cache or _FakeCloneCache()
    result = tasks.prepare_and_publish_candidate(
        cfg,
        gh=object(),
        candidate_req=_req(),
        checkout=object(),
        clone_cache=cc,
        existing_diff=None,
        job_id="job1234",
        emit=lambda *_a: None,
    )
    return result, cc


def _res(verdict=None, no_change=False, tracebacks=None):
    return tasks.TaskResult(
        mode="new_pr",
        no_change=no_change,
        verify_verdict=verdict,
        verify_tracebacks=tracebacks or {},
    )


def test_should_retry():
    assert verify.should_retry("not_fixed")
    assert verify.should_retry("broke_others")
    assert not verify.should_retry("already_passing")
    assert not verify.should_retry("fixed")
    assert not verify.should_retry("timeout")
    assert not verify.should_retry("")


def test_no_retry_when_disabled(monkeypatch):
    # verify off => single attempt even if a verdict is present.
    seen = _install(monkeypatch, [_res(verdict="not_fixed", no_change=True)])
    result, cc = _call(_cfg(on_gpu=False))
    assert result.verify_verdict == "not_fixed"
    assert len(seen) == 1
    assert cc.resets == 0


def test_retries_then_fixes(monkeypatch):
    seen = _install(
        monkeypatch,
        [
            _res(verdict="not_fixed", no_change=True, tracebacks={"t1": "boom1"}),
            _res(verdict="not_fixed", no_change=True, tracebacks={"t2": "boom2"}),
            _res(verdict="fixed"),
        ],
    )
    result, cc = _call(_cfg(on_gpu=True, rounds=2))
    assert result.verify_verdict == "fixed"
    assert len(seen) == 3  # first + 2 retries
    assert cc.resets == 2  # reset before each retry, not the first
    # Round 2 saw round 1's traceback; round 3 saw round 2's traceback.
    assert "boom1" in seen[1]
    assert "boom2" in seen[2]
    # Feedback is appended to the ORIGINAL context (does not compound).
    assert seen[2].startswith("original context")


def test_retries_exhausted_returns_last(monkeypatch):
    _install(
        monkeypatch,
        [_res(verdict="not_fixed", no_change=True) for _ in range(3)],
    )
    result, _ = _call(_cfg(on_gpu=True, rounds=2))
    assert result.verify_verdict == "not_fixed"
    assert result.no_change


def test_non_retryable_verdict_stops_immediately(monkeypatch):
    seen = _install(
        monkeypatch,
        [_res(verdict="already_passing", no_change=True), _res(verdict="fixed")],
    )
    result, _ = _call(_cfg(on_gpu=True, rounds=2))
    assert result.verify_verdict == "already_passing"
    assert len(seen) == 1  # no retry


def test_fixed_first_try_no_retry(monkeypatch):
    seen = _install(monkeypatch, [_res(verdict="fixed")])
    result, cc = _call(_cfg(on_gpu=True, rounds=2))
    assert result.verify_verdict == "fixed"
    assert len(seen) == 1
    assert cc.resets == 0


def test_format_feedback_includes_verdict_and_tracebacks():
    res = _res(
        verdict="not_fixed", tracebacks={"tests/x.py::T::t": "AssertionError: nope"}
    )
    fb = tasks._format_verify_feedback(res)
    assert "not_fixed" in fb
    assert "tests/x.py::T::t" in fb
    assert "AssertionError: nope" in fb


def test_with_verify_feedback_appends_context():
    req = _req("base ctx")
    out = tasks._with_verify_feedback(req, "FEEDBACK")
    assert out.context == "base ctx\n\nFEEDBACK"
    assert req.context == "base ctx"  # original untouched
