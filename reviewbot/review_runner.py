"""In-pod entrypoint for the per-review-pod execution model.

serge launches one runner pod per UI PR review when ``REVIEW_EXECUTION`` is
``docker``/``kubernetes`` (SERGE_ORCHESTRATOR_PODS_PLAN.md Phase 3). The pod runs
the **read-only** review — checkout (standalone, with the trusted ``.ai/``
overlay), then :func:`reviewer.prepare_review` — and streams every event plus the
terminal draft back to serge over the same authenticated HTTP callback the task
pod uses. It never publishes: serge posts the review to GitHub on the human's
approval (publish is a GitHub API call, not LLM work).

This mirrors ``webapp._execute_review`` with two edges swapped, exactly like
``task_runner`` mirrors ``_run_task_worker``: events go to the callback instead of
an in-process SSE queue, and the GitHub token is minted by serge and handed in via
the spec. The dispatch happens in ``task_runner.main`` on ``spec.request_type``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Optional

from .clone_cache import Checkout, CloneCache
from .github_client import GitHubClient
from .reviewer import ReviewRequest, _UnparseableLLMOutput, prepare_review
from .store import encode_draft
from .task_runner import CallbackEmitter, RunnerSpec, build_runner_config

log = logging.getLogger(__name__)


def build_review_request(spec: RunnerSpec) -> ReviewRequest:
    """Reconstruct the :class:`ReviewRequest` from the serialized spec, keeping
    only the fields the dataclass declares. ``inline`` (webhook follow-up
    context) is left at its default — UI reviews never carry it."""
    fields = {f.name for f in dataclasses.fields(ReviewRequest)}
    kwargs = {
        k: v for k, v in spec.request.items() if k in fields and k != "inline"
    }
    return ReviewRequest(**kwargs)


def run(spec: RunnerSpec) -> int:
    """Execute one PR review end-to-end in the pod. Returns a process exit code
    (0 = the outcome was reported, 1 = errored). The outcome — the draft, or an
    error — is always sent to serge over the callback regardless of exit code;
    serge reconstructs the draft and holds it for the human to publish."""
    cfg = build_runner_config(spec)
    req = build_review_request(spec)
    emitter = CallbackEmitter(
        spec.callback.get("url"), spec.callback.get("token"), spec.job_id
    )
    clone_cache = CloneCache(cfg.web_clone_cache_dir)
    gh = GitHubClient(spec.github_token)
    checkout: Optional[Checkout] = None

    def emit(kind: str, text: str) -> None:
        emitter.emit(kind, text)

    try:
        emit("step", "clone")
        emit("log", f"Checking out {req.owner}/{req.repo}#{req.number}…")
        t0 = time.monotonic()
        # standalone=True: a self-contained clone (the pod binds only the
        # checkout dir) with the trusted default-branch .ai/ overlay. A failed
        # checkout just means the review runs without browse tools.
        checkout = clone_cache.acquire(
            spec.github_token,
            req.owner,
            req.repo,
            req.number,
            job_id=spec.job_id,
            depth=cfg.web_clone_depth,
            remote_url=spec.repo_remote_url,
            standalone=True,
        )
        if checkout is not None:
            emit("log", f"Checkout ready in {time.monotonic() - t0:.1f}s")
        else:
            emit("log", "Checkout unavailable; reviewing without browse tools")
        worker_cfg = dataclasses.replace(
            cfg, repo_checkout_path=(checkout.path if checkout else "")
        )

        draft = prepare_review(worker_cfg, gh, req, chunk_callback=emit)
        if draft is None:
            # No reviewable diff — prepare_review already posted a notice to the
            # PR. A clean, draft-less completion.
            emitter.finish("done", result=None)
            return 0

        # Ship the draft (serge rebuilds it via store.decode_draft) plus the
        # token counts, which the draft payload doesn't carry.
        result = {
            "draft": encode_draft(draft),
            "prompt_tokens": draft.prompt_tokens,
            "completion_tokens": draft.completion_tokens,
            "truncated_chunks": draft.truncated_chunks,
        }
        emitter.finish("done", result=result)
        return 0
    except _UnparseableLLMOutput as exc:
        log.warning("review %s: unparseable LLM output", spec.job_id)
        emitter.finish(
            "error",
            error="the model returned an unparseable review",
            raw_llm_output=getattr(exc, "content", None),
        )
        return 1
    except Exception as exc:  # noqa: BLE001 — always report a terminal outcome
        log.exception("review runner crashed for job %s", spec.job_id)
        emitter.finish(
            "error", error=f"{type(exc).__name__}: review crashed (see pod log)"
        )
        return 1
    finally:
        if checkout is not None:
            clone_cache.release(checkout)
