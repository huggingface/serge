import types

from reviewbot import tasks, verify


def _cfg(on_gpu: bool, rounds: int = 2, reproduce_first: bool = False):
    return types.SimpleNamespace(
        verify_on_gpu=on_gpu,
        verify_max_rounds=rounds,
        verify_reproduce_first=reproduce_first,
        classify_max_tokens=4096,
        reproduce_block_chars=32000,
    )


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


# ── The verdict must survive a PASSING gate, not only a failing one ──────────
# Every published job in prod stored `verify_verdict=None` while the ones that
# produced no PR carried it, so the field read as "the gate never ran" exactly
# where it had run and passed. Both `_commit_changes` success paths dropped it.


def _full_cfg():
    """The real Config: `_commit_changes` renders a PR body, which the
    SimpleNamespace `_cfg` above (built for the rounds loop) cannot satisfy."""
    from tests.test_tasks import _make_cfg

    return _make_cfg()


class _VerdictGH:
    """Minimal Git Data API fake for the two `_commit_changes` publish paths."""

    def __init__(self):
        self.created_pr = None
        self.deleted = []
        self.updated_refs = []

    def get_ref_sha(self, owner, repo, ref):
        return "parentsha"

    def get_commit_tree_sha(self, owner, repo, commit_sha):
        return "basetree"

    def create_blob(self, owner, repo, content):
        return "blob1"

    def create_tree(self, owner, repo, base_tree, entries):
        return "newtree"

    def create_commit(self, owner, repo, *, message, tree_sha, parents):
        return "newcommit"

    def create_ref(self, owner, repo, ref, sha):
        return {"ref": ref}

    def update_ref(self, owner, repo, ref, sha, *, force=False):
        self.updated_refs.append((ref, sha, force))
        return {}

    def delete_ref(self, owner, repo, ref):
        self.deleted.append(ref)

    def create_pull_request(self, owner, repo, *, title, head, base, body, draft=False):
        self.created_pr = {"head": head}
        return {"number": 99, "html_url": "u", "node_id": "n"}

    def mark_pull_request_ready(self, node_id):
        pass

    def request_reviewers(self, owner, repo, number, reviewers):
        return list(reviewers)


def _commit(mode, verdict, monkeypatch):
    from reviewbot.clone_cache import FileChange

    monkeypatch.setattr(
        tasks, "post_task_pr_created_notification", lambda **_k: None, raising=False
    )
    gh = _VerdictGH()
    req = tasks.TaskRequest(
        owner="acme",
        repo="widget",
        base_ref="main",
        instruction="fix",
        context="",
        mode=mode,
        pr_number=7 if mode == "existing_pr" else None,
        head_branch="serge/fix-old" if mode == "existing_pr" else None,
    )
    result = tasks._commit_changes(
        _full_cfg(),
        gh,
        req,
        changes=[FileChange(path="a.txt", status="M", content=b"x", mode="100644")],
        plan=tasks.TaskPlan(title="t", body="b", patch="p"),
        job_id="job12345",
        emit_fn=lambda *_a: None,
        verify=lambda parent, candidate: verify.VerifyOutcome(verdict=verdict),
    )
    return result, gh


def test_new_pr_records_the_verdict_that_let_it_publish(monkeypatch):
    result, gh = _commit("new_pr", verify.FIXED, monkeypatch)
    assert gh.created_pr is not None, "a fixed verdict must still open the PR"
    assert result.pr_number == 99
    assert result.verify_verdict == "fixed"
    assert result.to_json()["verify_verdict"] == "fixed"


def test_existing_pr_follow_up_records_the_verdict(monkeypatch):
    result, gh = _commit("existing_pr", verify.FIXED, monkeypatch)
    assert result.pr_number == 7
    assert result.verify_verdict == "fixed"


def test_a_failing_gate_still_records_its_verdict(monkeypatch):
    result, gh = _commit("new_pr", verify.NOT_FIXED, monkeypatch)
    assert gh.created_pr is None
    assert result.no_change
    assert result.verify_verdict == "not_fixed"


def test_no_gate_leaves_the_verdict_unset(monkeypatch):
    """`None` must keep meaning "the gate did not run" — that is the only way
    to tell a skipped gate from a passing one."""
    from reviewbot.clone_cache import FileChange

    monkeypatch.setattr(
        tasks, "post_task_pr_created_notification", lambda **_k: None, raising=False
    )
    gh = _VerdictGH()
    req = tasks.TaskRequest(
        owner="a", repo="b", base_ref="main", instruction="i", context="", mode="new_pr"
    )
    result = tasks._commit_changes(
        _full_cfg(),
        gh,
        req,
        changes=[FileChange(path="a.txt", status="M", content=b"x", mode="100644")],
        plan=tasks.TaskPlan(title="t", body="b", patch="p"),
        job_id="job12345",
        emit_fn=lambda *_a: None,
        verify=None,
    )
    assert result.verify_verdict is None


def test_recording_fixed_does_not_trigger_a_retry_round():
    """The rounds loop keys on `result.verify_verdict`, so populating it on the
    success path could have made every published job retry. It cannot: `fixed`
    is not in `_RETRYABLE`."""
    assert not verify.should_retry(verify.FIXED)
    assert not verify.should_retry(verify.ALREADY_PASSING)
