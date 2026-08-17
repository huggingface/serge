import io
import json
import zipfile

from reviewbot import verify


def _zip_with(result: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("verify-result.json", json.dumps(result))
    return buf.getvalue()


class FakeGH:
    def __init__(self, *, runs=None, artifacts=None, zip_bytes=None, dispatch_exc=None):
        self.dispatched = []
        self._runs = runs or []
        self._artifacts = artifacts or []
        self._zip = zip_bytes
        self._dispatch_exc = dispatch_exc

    def dispatch_workflow(self, owner, repo, wf, *, ref, inputs):
        if self._dispatch_exc is not None:
            raise self._dispatch_exc
        self.dispatched.append((owner, repo, wf, ref, inputs))

    def list_workflow_runs(self, owner, repo, wf, *, event=None, per_page=30):
        return self._runs

    def list_run_artifacts(self, owner, repo, run_id):
        return self._artifacts

    def download_artifact_zip(self, owner, repo, artifact_id):
        return self._zip


WHISPER = "tests/models/whisper/test_modeling_whisper.py"
CLS = "WhisperModelIntegrationTests"
BLOCK = [
    f"- `{WHISPER}::{CLS}::test_small_token_timestamp_generation` [multi-gpu] (output_mismatch, seen 7/7)",
    f"- `{WHISPER}::{CLS}::test_tiny_generation` [multi-gpu] (output_mismatch)",
]


class Clock:
    def __init__(self, seq):
        self.seq = list(seq)
        self.i = 0

    def __call__(self):
        v = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return v


# ---- extract_verify_targets --------------------------------------------------


def test_extract_targets_nodeids_model_multi_gpu():
    nodeids, model, machine = verify.extract_verify_targets(BLOCK, "default-mt")
    assert nodeids == [
        f"{WHISPER}::{CLS}::test_small_token_timestamp_generation",
        f"{WHISPER}::{CLS}::test_tiny_generation",
    ]
    assert model == "whisper"
    assert machine == "aws-g5-12xlarge-cache"


def test_extract_targets_single_gpu():
    block = [f"- `{WHISPER}::{CLS}::test_x` [single-gpu] (other)"]
    _, _, machine = verify.extract_verify_targets(block, "default-mt")
    assert machine == "aws-g5-4xlarge-cache"


def test_extract_targets_no_tag_uses_default():
    block = [f"- `{WHISPER}::{CLS}::test_x` (output_mismatch)"]
    _, _, machine = verify.extract_verify_targets(block, "default-mt")
    assert machine == "default-mt"


def test_extract_targets_ignores_non_nodeid_backticks():
    block = ["- `output_mismatch` some prose", "  - `not a nodeid`"]
    nodeids, model, _ = verify.extract_verify_targets(block, "mt")
    assert nodeids == []
    assert model == ""


# ---- parse_verify_result_zip -------------------------------------------------


def test_parse_zip_roundtrip():
    data = verify.parse_verify_result_zip(_zip_with({"verdict": "fixed"}))
    assert data == {"verdict": "fixed"}


def test_parse_zip_bad_bytes():
    assert verify.parse_verify_result_zip(b"not a zip") is None


# ---- run_gpu_verify ----------------------------------------------------------


def _run(gh, **overrides):
    kwargs = dict(
        owner="huggingface",
        repo="transformers",
        base_sha="base",
        commit_sha="cand",
        block_lines=BLOCK,
        correlation_id="corr-123",
        workflow_file="serge-verify-caller.yml",
        ref="main",
        default_machine_type="aws-g5-12xlarge-cache",
        run_collateral=False,
        transformersci_ref="main",
        poll_timeout=100,
        poll_interval=0,
        sleep=lambda _s: None,
        monotonic=Clock([0, 0]),
    )
    kwargs.update(overrides)
    return verify.run_gpu_verify(gh, **kwargs)


def test_run_verify_fixed():
    gh = FakeGH(
        runs=[
            {
                "id": 5,
                "name": "serge verify whisper [corr-123]",
                "status": "completed",
                "html_url": "u",
            }
        ],
        artifacts=[{"id": 9, "name": "serge-verify-result-aws-g5-12xlarge-cache"}],
        zip_bytes=_zip_with({"verdict": "fixed", "tracebacks": {}}),
    )
    out = _run(gh)
    assert out.is_fixed
    assert out.run_url == "u"
    # dispatched with the parsed node-ids + model
    (_o, _r, wf, ref, inputs) = gh.dispatched[0]
    assert wf == "serge-verify-caller.yml" and ref == "main"
    assert inputs["model"] == "whisper"
    assert inputs["correlation_id"] == "corr-123"
    assert "test_tiny_generation" in inputs["test_nodeids"]


def test_run_verify_not_fixed_passes_through_tracebacks():
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "completed", "html_url": "u"}
        ],
        artifacts=[{"id": 9, "name": "serge-verify-result-x"}],
        zip_bytes=_zip_with({"verdict": "not_fixed", "tracebacks": {"t": "boom"}}),
    )
    out = _run(gh)
    assert out.verdict == "not_fixed"
    assert out.tracebacks == {"t": "boom"}


def test_run_verify_no_targets_skips_dispatch():
    gh = FakeGH()
    out = _run(gh, block_lines=["- no node ids here"])
    assert out.verdict == verify.NO_TARGETS
    assert gh.dispatched == []


def test_run_verify_dispatch_failed():
    gh = FakeGH(dispatch_exc=RuntimeError("403 no actions:write"))
    out = _run(gh)
    assert out.verdict == verify.DISPATCH_FAILED
    assert "403" in out.detail


def test_run_verify_timeout_when_run_never_completes():
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "in_progress", "html_url": "u"}
        ]
    )
    # deadline = 0 + 1; enter once (t=0), then t=100 exits the loop still in_progress
    out = _run(gh, poll_timeout=1, monotonic=Clock([0, 0, 100]))
    assert out.verdict == verify.TIMEOUT
    assert out.run_url == "u"


def test_run_verify_no_result_artifact():
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "completed", "html_url": "u"}
        ],
        artifacts=[],
    )
    # The verdict never appears: fetch loop runs until monotonic passes the fetch
    # deadline (a big jump here), then falls open to no_result.
    out = _run(gh, poll_timeout=1, monotonic=Clock([0, 0, 0, 10_000]))
    assert out.verdict == verify.NO_RESULT


def test_run_verify_retries_artifact_past_old_window():
    # Regression for the ~3-min prod listing lag: the artifact stays invisible for
    # far longer than the old fixed 6-attempt (150s) window, then appears. serge
    # must keep polling against the remaining budget rather than bail no_result.
    class VeryLateArtifactGH(FakeGH):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._calls = 0

        def list_run_artifacts(self, owner, repo, run_id):
            self._calls += 1
            if self._calls < 20:  # empty well past the old 6-attempt window
                return []
            return [{"id": 9, "name": "serge-verify-result-x"}]

    gh = VeryLateArtifactGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "completed", "html_url": "u"}
        ],
        zip_bytes=_zip_with({"verdict": "reproduced", "tracebacks": {"t": "boom"}}),
    )
    out = _run_repro(gh)
    assert out.verdict == verify.REPRODUCED
    assert gh._calls == 20  # kept polling far beyond the old 6-attempt cap


def test_run_verify_retries_late_artifact():
    # The artifact list is empty on the first two polls (GitHub still finalizing)
    # and populated on the third — serge must retry rather than report no_result.
    class LateArtifactGH(FakeGH):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._calls = 0

        def list_run_artifacts(self, owner, repo, run_id):
            self._calls += 1
            if self._calls < 3:
                return []
            return [{"id": 9, "name": "serge-verify-result-x"}]

    gh = LateArtifactGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "completed", "html_url": "u"}
        ],
        zip_bytes=_zip_with({"verdict": "reproduced", "tracebacks": {"t": "boom"}}),
    )
    out = _run_repro(gh)
    assert out.verdict == verify.REPRODUCED
    assert gh._calls == 3  # succeeded on the third fetch, not on the first


def test_run_verify_correlation_id_mismatch_times_out():
    # A concurrent run for a different task must not be picked up.
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "serge verify whisper [other-id]", "status": "completed"}
        ]
    )
    out = _run(gh, poll_timeout=1, monotonic=Clock([0, 0, 100]))
    assert out.verdict == verify.TIMEOUT


# ---- run_gpu_reproduce -------------------------------------------------------


def _run_repro(gh, **overrides):
    kwargs = dict(
        owner="huggingface",
        repo="transformers",
        base_sha="base",
        block_lines=BLOCK,
        correlation_id="corr-123",
        workflow_file="serge-verify-caller.yml",
        ref="main",
        default_machine_type="aws-g5-12xlarge-cache",
        transformersci_ref="main",
        poll_timeout=100,
        poll_interval=0,
        sleep=lambda _s: None,
        monotonic=Clock([0, 0]),
    )
    kwargs.update(overrides)
    return verify.run_gpu_reproduce(gh, **kwargs)


def test_run_reproduce_reproduced_dispatches_mode_and_no_commit():
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "x [corr-123]", "status": "completed", "html_url": "u"}
        ],
        artifacts=[{"id": 9, "name": "serge-verify-result-x"}],
        zip_bytes=_zip_with(
            {"verdict": "reproduced", "tracebacks": {"t": "RuntimeError"}}
        ),
    )
    out = _run_repro(gh)
    assert out.verdict == verify.REPRODUCED
    assert out.tracebacks == {"t": "RuntimeError"}
    # dispatched in reproduce mode, with no candidate commit.
    (_o, _r, _wf, _ref, inputs) = gh.dispatched[0]
    assert inputs["mode"] == "reproduce"
    assert "commit_sha" not in inputs
    assert inputs["run_collateral"] == "false"
    assert "test_tiny_generation" in inputs["test_nodeids"]


def test_run_reproduce_not_reproduced_passthrough():
    gh = FakeGH(
        runs=[{"id": 5, "name": "x [corr-123]", "status": "completed"}],
        artifacts=[{"id": 9, "name": "serge-verify-result-x"}],
        zip_bytes=_zip_with({"verdict": "not_reproduced", "tracebacks": {}}),
    )
    assert _run_repro(gh).verdict == verify.NOT_REPRODUCED


def test_run_reproduce_no_targets_skips_dispatch():
    gh = FakeGH()
    out = _run_repro(gh, block_lines=["- no node ids here"])
    assert out.verdict == verify.NO_TARGETS
    assert gh.dispatched == []


def test_run_reproduce_dispatch_failed():
    gh = FakeGH(dispatch_exc=RuntimeError("no actions:write"))
    assert _run_repro(gh).verdict == verify.DISPATCH_FAILED


# ---- groups whose tests are not under tests/models/<model>/ ------------------

# `tests/generation/test_utils.py` has no model folder, so extract_verify_targets
# returns model="". Sending that empty string is a hard 422 from GitHub
# ("Required input 'model' not provided"), which broke both dispatches for this
# group on the 2026-08-16 nightly.
NO_MODEL_BLOCK = [
    "- `tests/generation/test_utils.py::GenerationIntegrationTests"
    "::test_green_red_watermark_generation` [multi-gpu] (other, seen 7/7)",
]


def test_extract_targets_has_no_model_outside_tests_models():
    nodeids, model, _machine = verify.extract_verify_targets(
        NO_MODEL_BLOCK, "default-mt"
    )
    assert nodeids == [
        "tests/generation/test_utils.py::GenerationIntegrationTests"
        "::test_green_red_watermark_generation"
    ]
    assert model == ""


def test_verify_sends_a_placeholder_model_never_an_empty_string():
    # The clock jumps past the deadline so the poll gives up immediately; the
    # dispatch is already recorded and its inputs are all we assert on here.
    gh, clock = FakeGH(), Clock([0, 10**6])
    _run(gh, block_lines=NO_MODEL_BLOCK, monotonic=clock)
    (_o, _r, _wf, _ref, inputs) = gh.dispatched[0]
    assert inputs["model"] == verify._NO_MODEL
    assert inputs["model"]  # the point: `model` is required: true on the caller
    # No model folder means there is no collateral suite to run either.
    assert inputs["run_collateral"] == "false"


def test_reproduce_sends_a_placeholder_model_never_an_empty_string():
    # The clock jumps past the deadline so the poll gives up immediately; the
    # dispatch is already recorded and its inputs are all we assert on here.
    gh, clock = FakeGH(), Clock([0, 10**6])
    _run_repro(gh, block_lines=NO_MODEL_BLOCK, monotonic=clock)
    (_o, _r, _wf, _ref, inputs) = gh.dispatched[0]
    assert inputs["model"] == verify._NO_MODEL


def test_a_real_model_is_still_passed_through():
    # The clock jumps past the deadline so the poll gives up immediately; the
    # dispatch is already recorded and its inputs are all we assert on here.
    gh, clock = FakeGH(), Clock([0, 10**6])
    _run(gh, block_lines=BLOCK, run_collateral=True, monotonic=clock)
    (_o, _r, _wf, _ref, inputs) = gh.dispatched[0]
    assert inputs["model"] == "whisper"
    assert inputs["run_collateral"] == "true"
