"""In-pod entrypoint for the per-task-pod execution model.

serge launches one runner pod per ``/tasks`` request (see
``SERGE_PERTASK_POD_PLAN.md``). The pod runs the **full** write-capable task —
checkout, agentic loop, in-process normalize, and PR publish — and streams every
event plus the terminal result back to serge over an authenticated HTTP
callback. serge itself stays a thin orchestrator; it no longer runs the LLM loop
in-process.

This mirrors the body of ``webapp._run_task_worker`` but with two edges swapped:

- **Transport.** ``emit`` POSTs each event to the callback instead of pushing
  onto an in-process SSE queue; the terminal outcome is a final POST.
- **Credentials.** The GitHub installation token is *minted by serge* and handed
  to the pod in the spec — the pod never holds the long-lived App private key.
- **Normalize.** Runs in-process (``TASK_SANDBOX_BACKEND=off``); there is no
  nested Job. The pod's own network firewall (allowlist egress) is the isolation
  boundary.

The task payload is a JSON file (default ``/etc/serge/task.json``, mounted from a
per-job Secret). Static deployment config still comes from the environment via
:meth:`Config.from_env`, exactly as for the serge app; the runner overrides only
the per-task bits (resolved LLM settings, checkout path, sandbox backend).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from . import sandbox
from .clone_cache import Checkout, CloneCache
from .config import Config
from .errors import format_github_http_error, format_llm_error
from .github_client import GitHubClient
from .llm_client import LLMResponseError
from .reviewer import _UnparseableLLMOutput
from .tasks import (
    TaskError,
    TaskRequest,
    TaskResult,
    format_pr_files_diff,
    prepare_task,
    publish_task,
    resolve_existing_pr,
    task_candidate_requests,
)

log = logging.getLogger(__name__)

_DEFAULT_SPEC_PATH = "/etc/serge/task.json"
_DEFAULT_CLONE_DIR = "/tmp/serge-clones"
_CALLBACK_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Spec (task.json)
# ---------------------------------------------------------------------------
@dataclass
class RunnerSpec:
    """The per-task payload serge writes into the pod's mounted Secret.

    ``request`` is the serialized :class:`TaskRequest`; ``github_token`` is a
    short-lived installation token minted by serge; ``llm`` carries the
    per-repo-resolved provider settings; ``callback`` is where to stream events
    and the terminal result."""

    job_id: str
    request: dict[str, Any]
    github_token: str
    llm: dict[str, Any] = field(default_factory=dict)
    # Resolved-worker-Config subset serge sends (see launcher.runner_config): the
    # per-task LLM caps + strict tool mode and the operator/repo normalize/review
    # settings, applied over the env-built base Config.
    config: dict[str, Any] = field(default_factory=dict)
    callback: dict[str, Any] = field(default_factory=dict)
    # Optional clone URL override (GH Enterprise / a mirror / local e2e); when
    # unset the clone uses the public ``https://github.com/owner/repo.git``.
    repo_remote_url: Optional[str] = None

    @classmethod
    def from_file(cls, path: str) -> "RunnerSpec":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        missing = [k for k in ("job_id", "request", "github_token") if not data.get(k)]
        if missing:
            raise ValueError(f"task spec {path} is missing required keys: {missing}")
        return cls(
            job_id=str(data["job_id"]),
            request=dict(data["request"]),
            github_token=str(data["github_token"]),
            llm=dict(data.get("llm") or {}),
            config=dict(data.get("config") or {}),
            callback=dict(data.get("callback") or {}),
            repo_remote_url=(data.get("repo_remote_url") or None),
        )


# ---------------------------------------------------------------------------
# Callback transport
# ---------------------------------------------------------------------------
class CallbackEmitter:
    """POSTs streaming events and the terminal outcome back to serge.

    Best-effort like the in-process ``chunk_callback``: a failed POST is logged
    and swallowed so a flaky callback never crashes the task. Every event also
    goes to stdout so ``kubectl logs`` shows the transcript."""

    def __init__(self, url: Optional[str], token: Optional[str], job_id: str):
        self._url = (url or "").rstrip("/") or None
        self._token = token
        self._job_id = job_id
        self._seq = 0
        self._session = requests.Session()

    def _post(self, payload: dict[str, Any]) -> None:
        if not self._url:
            return
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            self._session.post(
                self._url, json=payload, headers=headers, timeout=_CALLBACK_TIMEOUT
            )
        except requests.RequestException as exc:
            log.warning("callback POST failed (seq=%s): %s", payload.get("seq"), exc)

    def emit(self, kind: str, text: str) -> None:
        """Stream one event (mirrors ``webapp._push_event``'s shape)."""
        self._seq += 1
        log.info("[%s] %s", kind, (text or "")[:200])
        self._post(
            {
                "job_id": self._job_id,
                "seq": self._seq,
                "kind": kind,
                "text": text,
                "ts": time.time(),
            }
        )

    def finish(
        self,
        status: str,
        *,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        raw_llm_output: Optional[str] = None,
    ) -> None:
        """Send the terminal outcome. serge records ``status`` on the job and
        stores ``result`` / ``error`` / ``raw_llm_output``."""
        self._seq += 1
        log.info("terminal status=%s error=%s", status, (error or "")[:200])
        self._post(
            {
                "job_id": self._job_id,
                "seq": self._seq,
                "terminal": {
                    "status": status,
                    "result": result,
                    "error": error,
                    "raw_llm_output": raw_llm_output,
                },
                "ts": time.time(),
            }
        )


# ---------------------------------------------------------------------------
# Config / request assembly
# ---------------------------------------------------------------------------
def build_runner_config(spec: RunnerSpec) -> Config:
    """Deployment config from the environment, with the spec overrides applied:
    the resolved worker-Config subset (``spec.config`` — per-task LLM caps +
    strict tool mode + operator/repo normalize/review settings), the resolved
    LLM provider settings (``spec.llm``, which win over ``config``), a local
    clone dir, and the in-pod sandboxes forced ``off`` — the pod firewall is the
    isolation boundary, so neither the normalize backend nor the helper-tool
    subprocesses need a nested sandbox.

    Built with ``require_app=False`` (the pod holds no GitHub App private key —
    serge mints the token and passes it in the spec) and ``require_web=False``
    (the pod serves no HTTP, so the web-auth env is irrelevant). ``LLM_API_KEY``
    is defaulted so ``from_env`` doesn't demand it; the real key comes from the
    spec override below."""
    os.environ.setdefault("LLM_API_KEY", "")
    cfg = Config.from_env(require_app=False, require_web=False)
    if spec.config:
        valid = {f.name for f in dataclasses.fields(Config)}
        overrides = {k: v for k, v in spec.config.items() if k in valid}
        cfg = dataclasses.replace(cfg, **overrides)
    llm = spec.llm
    clone_dir = (cfg.web_clone_cache_dir or "").strip() or _DEFAULT_CLONE_DIR
    return dataclasses.replace(
        cfg,
        llm_api_base=(llm.get("api_base") or cfg.llm_api_base),
        llm_api_key=(llm.get("api_key") or cfg.llm_api_key),
        llm_model=(llm.get("model") if "model" in llm else cfg.llm_model),
        llm_bill_to=(llm.get("bill_to") if "bill_to" in llm else cfg.llm_bill_to),
        llm_stream=bool(llm.get("stream", cfg.llm_stream)),
        web_clone_cache_dir=clone_dir,
        # Normalize runs in-process; there is no docker daemon / nested Job here.
        task_sandbox_backend="off",
        # The pod itself is the sandbox (ephemeral, allowlisted egress), so the
        # in-pod repo subprocesses (normalize, helper tools, context script) run
        # unwrapped — bubblewrap/docker aren't available and aren't needed.
        helper_sandbox=sandbox.OFF,
    )


def build_task_request(spec: RunnerSpec) -> TaskRequest:
    """Reconstruct the :class:`TaskRequest` from the serialized spec, keeping
    only the fields the dataclass declares (ignores any extras)."""
    fields = {f.name for f in dataclasses.fields(TaskRequest)}
    kwargs = {k: v for k, v in spec.request.items() if k in fields}
    return TaskRequest(**kwargs)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(spec: RunnerSpec) -> int:
    """Execute the task end-to-end. Returns a process exit code: 0 when the
    outcome was reported (published *or* no_fix — both are clean task
    completions), 1 when the task errored. The outcome detail is always sent to
    serge via the callback regardless of exit code."""
    cfg = build_runner_config(spec)
    req = build_task_request(spec)
    emitter = CallbackEmitter(
        spec.callback.get("url"), spec.callback.get("token"), spec.job_id
    )
    clone_cache = CloneCache(cfg.web_clone_cache_dir)
    gh = GitHubClient(spec.github_token)
    checkout: Optional[Checkout] = None

    def emit(kind: str, text: str) -> None:
        emitter.emit(kind, text)

    try:
        existing_diff: Optional[str] = None
        if req.mode == "existing_pr":
            emit("step", "resolve")
            head_branch = resolve_existing_pr(gh, req, cfg)
            emit(
                "log",
                f"Targeting serge branch {head_branch} (PR #{req.pr_number}), "
                f"base {req.base_ref}",
            )
            ref_to_checkout = head_branch
            try:
                pr_files = gh.get_pr_files(req.owner, req.repo, req.pr_number or 0)
                existing_diff = format_pr_files_diff(pr_files)
            except Exception:  # noqa: BLE001
                log.debug("could not fetch prior-attempt diff", exc_info=True)
        else:
            ref_to_checkout = req.base_ref

        emit("step", "clone")
        emit("log", f"Checking out {req.owner}/{req.repo}@{ref_to_checkout}…")
        t0 = time.monotonic()
        # standalone=True: a self-contained clone so in-process git (the repo
        # consistency checkers shell out to git) works, and the checked-out
        # branch is named `main` so the checkers scan all files. The object copy
        # that was slow on EFS is cheap on the pod's local scratch.
        checkout = clone_cache.acquire_ref(
            spec.github_token,
            req.owner,
            req.repo,
            ref_to_checkout,
            job_id=spec.job_id,
            depth=cfg.web_clone_depth,
            remote_url=spec.repo_remote_url,
            standalone=True,
        )
        if checkout is None:
            raise TaskError(
                f"could not check out {req.owner}/{req.repo}@{ref_to_checkout}",
                status_code=502,
            )
        emit("log", f"Checkout ready in {time.monotonic() - t0:.1f}s")
        worker_cfg = dataclasses.replace(cfg, repo_checkout_path=checkout.path)

        candidate_reqs = task_candidate_requests(req)
        last_no_change: Optional[TaskResult] = None
        result: Optional[TaskResult] = None
        for index, candidate_req in enumerate(candidate_reqs, start=1):
            if len(candidate_reqs) > 1:
                emit(
                    "log",
                    f"Starting candidate {index}/{len(candidate_reqs)} in a fresh LLM cycle",
                )
            plan = prepare_task(
                worker_cfg,
                candidate_req,
                checkout=checkout,
                clone_cache=clone_cache,
                existing_diff=existing_diff,
                chunk_callback=emit,
            )
            try:
                attempt_result = publish_task(
                    worker_cfg,
                    gh,
                    candidate_req,
                    plan,
                    checkout=checkout,
                    clone_cache=clone_cache,
                    job_id=spec.job_id,
                    emit=emit,
                )
            except TaskError as exc:
                if exc.status_code == 422 and index < len(candidate_reqs):
                    emit(
                        "log",
                        f"Candidate {index}/{len(candidate_reqs)} did not produce "
                        f"an applicable patch: {exc}. Moving to the next group.",
                    )
                    continue
                raise
            if attempt_result.no_change and index < len(candidate_reqs):
                last_no_change = attempt_result
                emit(
                    "log",
                    f"Candidate {index}/{len(candidate_reqs)} produced no fix. "
                    "Moving to the next group.",
                )
                continue
            result = attempt_result
            break
        if result is None:
            result = last_no_change or TaskResult(
                mode=req.mode,
                no_change=True,
                message="No candidate produced a safe fix.",
            )

        status = "no_fix" if result.no_change else "published"
        emit("log", result.message)
        emit("step", "done")
        emit("done", "")
        emitter.finish(status, result=result.to_json())
        return 0
    except TaskError as exc:
        log.warning("task %s rejected: %s", spec.job_id, exc)
        return _fail(emitter, str(exc))
    except _UnparseableLLMOutput as exc:
        return _fail(emitter, exc.user_message(), raw_llm_output=exc.content)
    except LLMResponseError as exc:
        log.warning(
            "LLM endpoint returned %d for task %s", exc.status_code, spec.job_id
        )
        return _fail(emitter, format_llm_error(exc))
    except requests.HTTPError as exc:
        log.warning("task %s GitHub API error: %s", spec.job_id, exc)
        return _fail(emitter, format_github_http_error(exc))
    except Exception as exc:  # noqa: BLE001
        log.exception("task runner crashed for job %s", spec.job_id)
        return _fail(emitter, f"{type(exc).__name__}: task crashed (see pod log)")
    finally:
        clone_cache.release(checkout)


def _fail(
    emitter: CallbackEmitter, message: str, *, raw_llm_output: Optional[str] = None
) -> int:
    emitter.emit("step", "error")
    emitter.emit("error", message)
    emitter.emit("done", "")
    emitter.finish("error", error=message, raw_llm_output=raw_llm_output)
    return 1


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="reviewbot-task-runner")
    parser.add_argument(
        "--spec",
        default=os.environ.get("SERGE_TASK_SPEC") or _DEFAULT_SPEC_PATH,
        help="Path to the task spec JSON (default: $SERGE_TASK_SPEC or "
        f"{_DEFAULT_SPEC_PATH}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    try:
        spec = RunnerSpec.from_file(args.spec)
    except (OSError, ValueError) as exc:
        log.error("could not load task spec: %s", exc)
        return 2

    return run(spec)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
