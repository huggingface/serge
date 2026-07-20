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
    out = _run(gh)
    assert out.verdict == verify.NO_RESULT


def test_run_verify_correlation_id_mismatch_times_out():
    # A concurrent run for a different task must not be picked up.
    gh = FakeGH(
        runs=[
            {"id": 5, "name": "serge verify whisper [other-id]", "status": "completed"}
        ]
    )
    out = _run(gh, poll_timeout=1, monotonic=Clock([0, 0, 100]))
    assert out.verdict == verify.TIMEOUT
