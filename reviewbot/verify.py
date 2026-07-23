"""GPU verification of a serge candidate patch before a PR is opened.

serge produces a candidate patch for a CI failure group and commits it to a
scratch branch, but it does not run the tests — they are ``@slow``/``[multi-gpu]``
and never run in normal PR CI, so a no-op patch (e.g. transformers#47150) looks
plausible and merges clean. This module closes that loop: it triggers the
``serge-verify-caller.yml`` ``workflow_dispatch`` in the target repo (which runs
the targeted node-ids on GPU on the pre-patch baseline then serge's candidate),
polls the run, downloads the verdict artifact, and reports whether the patch
actually turned the tests red -> green.

The verdict is computed workflow-side by transformers-ci's ``serge-verify-verdict``
console script; this module only orchestrates dispatch + poll + parse. It is
gated behind ``cfg.verify_on_gpu`` (default off) — when disabled serge behaves
exactly as before.

See docs/plans/serge-gpu-verify-loop.md in transformers-ci-playbooks.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .github_client import GitHubClient

# The first backtick-quoted token on a failure bullet is the pytest node-id, e.g.
#   - `tests/models/whisper/..py::WhisperModelIntegrationTests::test_x` [multi-gpu] (output_mismatch, seen 7/7)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_MODEL_RE = re.compile(r"tests/models/([^/]+)/")

_MULTI_GPU = "aws-g5-12xlarge-cache"
_SINGLE_GPU = "aws-g5-4xlarge-cache"

# Verify-mode verdicts produced workflow-side (serge-verify-verdict). Only
# `fixed` opens a PR.
FIXED = "fixed"
NOT_FIXED = "not_fixed"
BROKE_OTHERS = "broke_others"
ALREADY_PASSING = "already_passing"
ERROR = "error"
# Reproduce-mode verdicts (baseline-only run, before serge investigates). Only
# `reproduced` lets serge proceed to investigate the group.
REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
# Orchestration-level verdicts produced here when the workflow can't be run/read.
NO_TARGETS = "no_targets"
DISPATCH_FAILED = "dispatch_failed"
TIMEOUT = "timeout"
NO_RESULT = "no_result"

# Verdicts where re-prompting the LLM with the fresh failures is worth another
# round. `already_passing` (nothing to fix), infra errors and `error` are NOT
# retried — another LLM turn can't change them.
_RETRYABLE = frozenset({NOT_FIXED, BROKE_OTHERS})


def should_retry(verdict: str) -> bool:
    return verdict in _RETRYABLE


@dataclass
class VerifyOutcome:
    verdict: str
    run_url: Optional[str] = None
    result: Optional[dict] = None
    tracebacks: dict[str, str] = field(default_factory=dict)
    detail: str = ""

    @property
    def is_fixed(self) -> bool:
        return self.verdict == FIXED


def _looks_like_nodeid(token: str) -> bool:
    return "::" in token and token.split("::", 1)[0].endswith(".py")


def extract_verify_targets(
    block_lines: list[str], default_machine_type: str
) -> tuple[list[str], str, str]:
    """From one failure group's bullet lines, return
    ``(node_ids, model, machine_type)``.

    node_ids are the backtick-quoted pytest ids; model is the ``tests/models/
    <model>`` folder (``""`` if none); machine_type is ``aws-g5-12xlarge-cache``
    if any bullet is tagged ``[multi-gpu]`` (the superset), else
    ``aws-g5-4xlarge-cache`` for ``[single-gpu]``, else ``default_machine_type``."""
    node_ids: list[str] = []
    seen: set[str] = set()
    machine: Optional[str] = None
    for line in block_lines:
        m = _BACKTICK_RE.search(line)
        if m:
            nid = m.group(1).strip()
            if _looks_like_nodeid(nid) and nid not in seen:
                seen.add(nid)
                node_ids.append(nid)
        low = line.lower()
        if "[multi-gpu]" in low:
            machine = "multi"
        elif "[single-gpu]" in low and machine is None:
            machine = "single"

    model = ""
    for nid in node_ids:
        mm = _MODEL_RE.search(nid)
        if mm:
            model = mm.group(1)
            break

    if machine == "multi":
        machine_type = _MULTI_GPU
    elif machine == "single":
        machine_type = _SINGLE_GPU
    else:
        machine_type = default_machine_type
    return node_ids, model, machine_type


def parse_verify_result_zip(zip_bytes: bytes) -> Optional[dict]:
    """Extract and parse ``verify-result.json`` from an artifact zip."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return None
    for name in zf.namelist():
        if name.endswith("verify-result.json"):
            try:
                return json.loads(zf.read(name))
            except (ValueError, KeyError):
                return None
    return None


def _find_run(runs: list[dict], correlation_id: str) -> Optional[dict]:
    """The caller sets ``run-name`` to echo our correlation id, so we match on
    the run's name/display_title rather than head_sha (a workflow_dispatch run's
    head_sha is the ref, not serge's candidate commit)."""
    for run in runs:
        haystack = f"{run.get('name', '')} {run.get('display_title', '')}"
        if correlation_id in haystack:
            return run
    return None


def run_gpu_verify(
    gh: GitHubClient,
    *,
    owner: str,
    repo: str,
    base_sha: str,
    commit_sha: str,
    block_lines: list[str],
    correlation_id: str,
    workflow_file: str,
    ref: str,
    default_machine_type: str,
    run_collateral: bool,
    transformersci_ref: str,
    poll_timeout: float,
    poll_interval: float,
    emit: Optional[Callable[[str, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> VerifyOutcome:
    """Dispatch the verify workflow (mode=verify) for one failure group, wait for
    it, and return the red→green verdict. Never raises for the expected failure
    modes — it returns a ``VerifyOutcome`` whose ``verdict`` the caller gates on."""
    node_ids, model, machine_type = extract_verify_targets(
        block_lines, default_machine_type
    )
    if not node_ids:
        if emit is not None:
            emit("log", "GPU verify: no node-ids found in the failure group; skipping.")
        return VerifyOutcome(NO_TARGETS, detail="no node-ids parsed from failure group")

    inputs: dict[str, Any] = {
        "mode": "verify",
        "base_sha": base_sha,
        "commit_sha": commit_sha,
        "test_nodeids": " ".join(node_ids),
        "model": model,
        "machine_type": machine_type,
        "run_collateral": "true" if (run_collateral and model) else "false",
        "transformersci_ref": transformersci_ref,
        "correlation_id": correlation_id,
    }
    return _dispatch_poll_fetch(
        gh,
        owner=owner,
        repo=repo,
        workflow_file=workflow_file,
        ref=ref,
        inputs=inputs,
        correlation_id=correlation_id,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        log_label=(
            f"GPU verify: dispatching {workflow_file} on {machine_type} for "
            f"{len(node_ids)} test(s) [{model or 'no-model'}]"
        ),
        emit=emit,
        sleep=sleep,
        monotonic=monotonic,
    )


def run_gpu_reproduce(
    gh: GitHubClient,
    *,
    owner: str,
    repo: str,
    base_sha: str,
    block_lines: list[str],
    correlation_id: str,
    workflow_file: str,
    ref: str,
    default_machine_type: str,
    transformersci_ref: str,
    poll_timeout: float,
    poll_interval: float,
    emit: Optional[Callable[[str, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> VerifyOutcome:
    """Dispatch the verify workflow in ``mode=reproduce`` — a baseline-only run,
    BEFORE serge investigates — to confirm the group's failure is real at
    ``base_sha``. Returns ``reproduced`` (every targeted test red, with the fresh
    tracebacks) so serge can proceed and seed the LLM, or ``not_reproduced`` /
    ``error`` so it bails. No candidate commit is involved. Same never-raise
    contract as :func:`run_gpu_verify`."""
    node_ids, model, machine_type = extract_verify_targets(
        block_lines, default_machine_type
    )
    if not node_ids:
        if emit is not None:
            emit(
                "log",
                "GPU reproduce: no node-ids found in the failure group; skipping.",
            )
        return VerifyOutcome(NO_TARGETS, detail="no node-ids parsed from failure group")

    inputs: dict[str, Any] = {
        "mode": "reproduce",
        "base_sha": base_sha,
        "test_nodeids": " ".join(node_ids),
        "model": model,
        "machine_type": machine_type,
        "run_collateral": "false",
        "transformersci_ref": transformersci_ref,
        "correlation_id": correlation_id,
    }
    return _dispatch_poll_fetch(
        gh,
        owner=owner,
        repo=repo,
        workflow_file=workflow_file,
        ref=ref,
        inputs=inputs,
        correlation_id=correlation_id,
        poll_timeout=poll_timeout,
        poll_interval=poll_interval,
        log_label=(
            f"GPU reproduce: dispatching {workflow_file} on {machine_type} for "
            f"{len(node_ids)} test(s) [{model or 'no-model'}]"
        ),
        emit=emit,
        sleep=sleep,
        monotonic=monotonic,
    )


def _dispatch_poll_fetch(
    gh: GitHubClient,
    *,
    owner: str,
    repo: str,
    workflow_file: str,
    ref: str,
    inputs: dict[str, Any],
    correlation_id: str,
    poll_timeout: float,
    poll_interval: float,
    log_label: str,
    emit: Optional[Callable[[str, str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> VerifyOutcome:
    """Shared dispatch → poll-for-completion → download-verdict plumbing for both
    verify and reproduce modes. The workflow-side verdict (whatever string the
    ``serge-verify-verdict`` tool emitted for the given mode) is returned verbatim
    in ``VerifyOutcome.verdict``; orchestration failures map to the ``*_FAILED`` /
    ``TIMEOUT`` / ``NO_RESULT`` verdicts. Never raises for expected failures."""

    def _emit(kind: str, text: str) -> None:
        if emit is not None:
            emit(kind, text)

    _emit("log", log_label)
    try:
        gh.dispatch_workflow(owner, repo, workflow_file, ref=ref, inputs=inputs)
    except Exception as exc:  # noqa: BLE001 — surface as a verdict, never crash publish
        return VerifyOutcome(DISPATCH_FAILED, detail=str(exc)[:500])

    deadline = monotonic() + poll_timeout
    run: Optional[dict] = None
    while monotonic() < deadline:
        sleep(poll_interval)
        try:
            runs = gh.list_workflow_runs(
                owner, repo, workflow_file, event="workflow_dispatch"
            )
        except Exception as exc:  # noqa: BLE001
            _emit("log", f"GPU verify: run listing failed ({exc}); retrying")
            continue
        found = _find_run(runs, correlation_id)
        if found is not None:
            run = found
            if found.get("status") == "completed":
                break

    if run is None:
        return VerifyOutcome(
            TIMEOUT, detail="verify run never appeared / never completed"
        )
    run_url = run.get("html_url")
    if run.get("status") != "completed":
        return VerifyOutcome(
            TIMEOUT, run_url=run_url, detail="verify run did not complete in time"
        )

    result = _fetch_verdict(gh, owner, repo, int(run["id"]))
    if result is None:
        return VerifyOutcome(
            NO_RESULT, run_url=run_url, detail="no verify-result artifact"
        )
    return VerifyOutcome(
        verdict=result.get("verdict", NO_RESULT),
        run_url=run_url,
        result=result,
        tracebacks=result.get("tracebacks") or {},
    )


def _fetch_verdict(
    gh: GitHubClient, owner: str, repo: str, run_id: int
) -> Optional[dict]:
    try:
        artifacts = gh.list_run_artifacts(owner, repo, run_id)
    except Exception:  # noqa: BLE001
        return None
    for art in artifacts:
        if str(art.get("name", "")).startswith("serge-verify-result"):
            try:
                zip_bytes = gh.download_artifact_zip(owner, repo, int(art["id"]))
            except Exception:  # noqa: BLE001
                return None
            return parse_verify_result_zip(zip_bytes)
    return None
