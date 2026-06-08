"""Interactive web mode: a small FastAPI app that lets a logged-in user
trigger a Serge review on a PR, watch it stream live, then tweak the
summary + per-comment text (or discard individual inline comments)
before publishing. The published review still goes out under the
GitHub App identity — OAuth is only used for access control.
"""

import asyncio
import dataclasses
import hashlib
import html as _html
import hmac
import json as _json
import logging
import os
import re
import secrets
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeSerializer

from .clone_cache import CloneCache, Checkout
from .config import Config
from .github_auth import (
    AppNotInstalledError,
    installation_id_for_repo,
    installation_token,
    user_is_org_member,
)
from .github_client import GitHubClient
from .llm_client import LLMResponseError
from .reviewer import (
    DraftComment,
    ReviewDraft,
    ReviewEdits,
    ReviewRequest,
    _UnparseableLLMOutput,
    prepare_review,
    publish_review,
    run_followup,
)
from .store import JobStore, decode_draft
from .triggers import build_review_request

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ai-reviewer.web")

cfg = Config.from_env(require_app=False, require_web=True)
log.info(
    "Config: llm_stream=%s, llm_max_tokens=%d, tool_max_iterations=%s, "
    "llm_max_input_tokens=%s, max_diff_chars=%d, mention_trigger=%r",
    cfg.llm_stream,
    cfg.llm_max_tokens,
    cfg.tool_max_iterations if cfg.tool_max_iterations > 0 else "unlimited",
    cfg.llm_max_input_tokens if cfg.llm_max_input_tokens > 0 else "unlimited",
    cfg.max_diff_chars,
    cfg.mention_trigger,
)

_SESSION_COOKIE = "serge_session"
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# GitHub limits owner / repo names to ASCII alphanumerics plus a few
# punctuation chars; we enforce the same so URL-pattern attacks (`..`,
# encoded slashes, empty strings) can't leak through into API calls.
_GH_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_TRIGGER_COMMENT_CHARS = 4000
_LLM_PROVIDER_HF = "hf"
_LLM_PROVIDER_OPENAI = "openai"
_LLM_PROVIDER_ANTHROPIC = "anthropic"
_LLM_PROVIDER_CUSTOM = "custom"
_LLM_PROVIDER_BASES = {
    _LLM_PROVIDER_HF: "https://router.huggingface.co/v1",
    _LLM_PROVIDER_OPENAI: "https://api.openai.com/v1",
    _LLM_PROVIDER_ANTHROPIC: "https://api.anthropic.com",
}
_LLM_PROVIDER_DEFAULT_MODELS = {
    _LLM_PROVIDER_ANTHROPIC: "claude-opus-4-6",
}


# ---------------------------------------------------------------------------
# Session handling: signed cookies via itsdangerous (no DB).
# ---------------------------------------------------------------------------
def _resolve_session_secret() -> str:
    secret = (cfg.web_session_secret or "").strip()
    if secret:
        return secret
    if not cfg.web_dev_no_auth:
        # Config.from_env(require_web=True) already enforces this when
        # DEV_NO_AUTH is off; the assert is a belt-and-braces guard so we
        # never silently fall back to a known string in production.
        raise RuntimeError("WEB_SESSION_SECRET is required when DEV_NO_AUTH is off")
    # Dev-only path: mint a fresh random secret per process so sessions
    # don't survive restarts (which is fine in dev), and never share a
    # well-known string between deployments.
    ephemeral = secrets.token_urlsafe(32)
    log.warning(
        "DEV_NO_AUTH=1 and no WEB_SESSION_SECRET set; using an ephemeral "
        "random session secret. Existing sessions will not survive restart."
    )
    return ephemeral


_serializer = URLSafeSerializer(_resolve_session_secret(), salt="serge.session")


def _load_session(request: Request) -> dict[str, Any]:
    raw = request.cookies.get(_SESSION_COOKIE)
    if not raw:
        return {}
    try:
        data = _serializer.loads(raw)
    except BadSignature:
        return {}
    return data if isinstance(data, dict) else {}


def _save_session(response: Response, data: dict[str, Any]) -> None:
    response.set_cookie(
        _SESSION_COOKIE,
        _serializer.dumps(data),
        httponly=True,
        samesite="lax",
        # Cookie must travel only over HTTPS in production. Set
        # WEB_INSECURE_COOKIES=1 to relax this for VPN-private HTTP
        # deployments where TLS isn't terminated locally.
        secure=not cfg.web_insecure_cookies,
        max_age=60 * 60 * 24 * 7,
    )


def _clear_session(response: Response) -> None:
    response.delete_cookie(_SESSION_COOKIE)


def _current_user(request: Request) -> Optional[str]:
    if cfg.web_dev_no_auth:
        return "dev"
    sess = _load_session(request)
    user = sess.get("user")
    return user if isinstance(user, str) and user else None


def _current_user_orgs(request: Request) -> list[str]:
    """Orgs the current user belongs to, cached in the signed session
    cookie at login time. Used to match provider_configs that grant
    access to a whole org. Returns [] in dev-no-auth mode."""
    if cfg.web_dev_no_auth:
        return []
    sess = _load_session(request)
    orgs = sess.get("orgs")
    if isinstance(orgs, list):
        return [o for o in orgs if isinstance(o, str)]
    return []


def _effective_user_orgs_for_repo(
    request: Request,
    response: Optional[Response],
    user: str,
    owner: str,
    repo: str,
) -> list[str]:
    """Return the user's effective orgs for matching configs on this
    repo: session-cached orgs plus any App-verified memberships among
    the ``allowed_orgs`` of candidate configs. This is the workaround
    for SAML-protected orgs (which never appear in the user's
    ``/user/orgs`` response, so the login-time cache is empty) and for
    legacy sessions minted before orgs were persisted.

    When new orgs are discovered, the merged list is written back to
    the session cookie so subsequent requests skip the App round-trip.
    """
    base = _current_user_orgs(request)
    if cfg.web_dev_no_auth or not (cfg.github_app_id and cfg.github_private_key):
        return base
    candidates = _store.allowed_orgs_for_repo(owner, repo)
    if not candidates:
        return base
    base_lc = {o.lower() for o in base}
    extra: list[str] = []
    for org in candidates:
        if org.lower() in base_lc:
            continue
        try:
            if user_is_org_member(cfg.github_app_id, cfg.github_private_key, org, user):
                extra.append(org)
        except Exception:  # noqa: BLE001
            log.warning(
                "App-based org membership check failed for %s in %s",
                user,
                org,
                exc_info=True,
            )
    if not extra:
        return base
    merged = list(dict.fromkeys([*base, *extra]))
    log.info("expanded session orgs for %s: %s -> %s", user, base, merged)
    if response is not None:
        sess = _load_session(request)
        sess["user"] = user
        sess["orgs"] = merged
        _save_session(response, sess)
    return merged


def _user_is_allowed(login: str, orgs: list[str]) -> bool:
    if cfg.web_dev_no_auth:
        return True
    if login.lower() in cfg.web_allowed_users:
        return True
    if any(o.lower() in cfg.web_allowed_orgs for o in orgs):
        return True
    return False


# ---------------------------------------------------------------------------
# In-memory job registry. Each Job owns an asyncio.Queue the SSE endpoint
# consumes; the worker thread pushes events via call_soon_threadsafe.
# ---------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    user: str
    target_owner: str
    target_repo: str
    target_number: int
    trigger_comment: str
    llm_provider: str
    llm_api_base: str
    llm_model: Optional[str]
    created_at: float
    # In-memory only — never persisted, never returned through any API.
    # Picked from the matched provider_config at submit time so the
    # worker doesn't need to hit the store again. "" for reconstructed
    # finished jobs that won't be re-executed.
    llm_api_key: str = ""
    # "web" for jobs the logged-in user submitted through the UI; "webhook"
    # for reviews kicked off by a GitHub comment. Webhook jobs have no
    # owning UI user, so any authenticated viewer may follow them.
    source: str = "web"
    status: str = "running"  # running | done | error | discarded | published
    draft: Optional[ReviewDraft] = None
    error: Optional[str] = None
    raw_llm_output: Optional[str] = None  # only set on parse-failure errors
    queue: "asyncio.Queue[dict[str, Any]]" = field(default_factory=asyncio.Queue)
    loop: Optional[asyncio.AbstractEventLoop] = None
    # Replay buffer so a client reconnecting (or arriving late) gets the
    # full console history instead of just events emitted after they
    # opened the EventSource.
    history: list[dict[str, Any]] = field(default_factory=list)
    history_lock: threading.Lock = field(default_factory=threading.Lock)
    # Running tally of "noisy" (token/reasoning) entries currently in
    # history. Lets _push_event do bounded-FIFO eviction in O(1) average
    # instead of scanning the full history on every streaming chunk.
    noisy_history_count: int = 0


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

# Persistent backing store. The in-memory `_jobs` dict is still used as
# a hot cache for live SSE streams (asyncio.Queue, event loop reference)
# — only running jobs strictly need to live here, but we keep finished
# jobs around too until process restart since the bound is tiny.
_store = JobStore(cfg.web_store_path)
_crashed = _store.mark_running_as_crashed()
if _crashed:
    log.warning(
        "Marked %d job(s) left in 'running' state as crashed (server restart)",
        _crashed,
    )

# Shared bare-clone + per-job worktree cache. One fetch per repo, cheap
# worktrees per review (see clone_cache.py / SCALE_UP_PLAN.md phase 3).
_clone_cache = CloneCache(cfg.web_clone_cache_dir)
# Jobs left mid-flight by a previous process are marked crashed above;
# their worktrees are orphaned, so clear them on startup.
_clone_cache.reset_worktrees()

# Same immediate-review webhook behavior as the legacy Flask app, now
# hosted by reviewbot-web at /webhook. Keep this separate from the staged
# review worker pool so a burst of GitHub comments cannot starve UI jobs.
_WEBHOOK_MAX_WORKERS = int(os.environ.get("WEBHOOK_MAX_WORKERS", "2"))
_WEBHOOK_REVIEW_POOL = ThreadPoolExecutor(
    max_workers=_WEBHOOK_MAX_WORKERS,
    thread_name_prefix="webhook-review-worker",
)


def _verify_webhook_signature(body: bytes, header: str) -> bool:
    secret = cfg.github_webhook_secret
    if not secret or not header or not header.startswith("sha256="):
        return False
    mac = hmac.new(secret.encode(), body, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, header)


def _resolve_webhook_worker_cfg(
    req: ReviewRequest,
) -> Optional[tuple[Config, str, str, Optional[str]]]:
    """Resolve the LLM credentials a webhook review should run with.

    Matched on repo only — a webhook has no logged-in user to gate
    ``allowed_users`` / ``allowed_orgs`` on, so the App being installed on
    the repo is the authorization. Falls back to the global env config
    when no ``provider_config`` matches the repo. Returns
    ``(worker_cfg, provider, api_base, model)`` or ``None`` when no usable
    API key is available (misconfiguration — the caller skips the review).
    """
    matched = _store.find_provider_config_for_repo(owner=req.owner, repo=req.repo)
    if matched is not None:
        provider = matched["provider"]
        llm_api_key = (matched.get("api_key") or "").strip()
        llm_api_base = _api_base_for_provider(
            provider, custom_base=matched.get("api_base")
        )
        llm_model = (
            (matched.get("default_model") or "").strip()
            or _LLM_PROVIDER_DEFAULT_MODELS.get(provider, "")
            or cfg.llm_model
        )
        worker_cfg = dataclasses.replace(
            cfg,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
            llm_bill_to=_llm_bill_to_for_provider(provider),
        )
    else:
        provider = _infer_llm_provider(cfg.llm_api_base)
        llm_api_key = cfg.llm_api_key.strip()
        llm_api_base = cfg.llm_api_base
        llm_model = cfg.llm_model
        worker_cfg = dataclasses.replace(cfg, llm_api_key=llm_api_key)
    if not llm_api_key:
        return None
    return worker_cfg, provider, llm_api_base, llm_model or None


def _run_webhook_review_worker(
    job: Job, worker_cfg: Config, installation_id: int, req: ReviewRequest
) -> None:
    """Run a webhook-triggered review on the dedicated webhook pool.

    Mirrors the UI worker (streaming events into the job so the review
    page can follow live), but auto-publishes the result to GitHub —
    there is no human in the loop to edit + publish a draft."""
    try:
        assert cfg.github_app_id and cfg.github_private_key
        token = installation_token(
            cfg.github_app_id, cfg.github_private_key, installation_id
        )
        gh = GitHubClient(token)

        def emit(kind: str, text: str) -> None:
            _push_event(job, kind, text)

        if req.inline is not None:
            # Inline follow-up: a focused reply on the comment thread.
            # There is no draft, so the page just streams the console.
            run_followup(worker_cfg, gh, req, chunk_callback=emit)
            job.status = "done"
            emit("step", "done")
            emit("done", "")
        else:
            _execute_review(job, worker_cfg, gh, token, req, auto_publish=True)
    except _UnparseableLLMOutput as exc:
        # run_followup never raises this (it posts plain markdown); only
        # reachable on the follow-up path if the agentic loop misbehaves.
        job.status = "error"
        job.raw_llm_output = exc.content
        job.error = exc.user_message()
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except AppNotInstalledError as exc:
        log.warning("App not installed for %s/%s (job %s)", exc.owner, exc.repo, job.id)
        job.status = "error"
        job.error = str(exc)
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except LLMResponseError as exc:
        log.warning(
            "LLM endpoint returned %d for webhook job %s: %s",
            exc.status_code,
            job.id,
            exc.body_preview[:400],
        )
        job.status = "error"
        job.error = _format_llm_error(exc)
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except Exception as exc:  # noqa: BLE001
        log.exception("webhook review worker crashed for job %s", job.id)
        job.status = "error"
        job.error = f"{type(exc).__name__}: review crashed (see server log)"
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    finally:
        # Follow-ups stop here; full reviews already persisted inside
        # _execute_review's own finally. Persisting twice is harmless (the
        # second call just re-snapshots the terminal state).
        _persist_terminal(job)


def _infer_llm_provider(api_base: str) -> str:
    normalized = api_base.rstrip("/")
    for provider, base in _LLM_PROVIDER_BASES.items():
        if normalized == base or normalized == base.removesuffix("/v1"):
            return provider
    return _LLM_PROVIDER_CUSTOM


def _normalize_llm_base_url(raw: str) -> str:
    base = raw.strip().rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="bad_llm_base_url")
    return base


def _parse_provider(payload: dict[str, Any]) -> str:
    default_provider = _infer_llm_provider(cfg.llm_api_base)
    provider = (payload.get("llm_provider") or default_provider).strip().lower()
    if provider not in (
        _LLM_PROVIDER_HF,
        _LLM_PROVIDER_OPENAI,
        _LLM_PROVIDER_ANTHROPIC,
        _LLM_PROVIDER_CUSTOM,
    ):
        raise HTTPException(status_code=400, detail="bad_llm_provider")
    return provider


# A repo pattern is either an exact "owner/repo" or "owner/*". Both
# pieces follow GitHub's name rules (alphanumerics plus . _ -). The
# wildcard is a literal "*", not a glob, so the matcher stays trivial.
_REPO_PATTERN_RE = re.compile(r"^[A-Za-z0-9._-]+/([A-Za-z0-9._-]+|\*)$")
_VALID_PROVIDERS = (
    _LLM_PROVIDER_HF,
    _LLM_PROVIDER_OPENAI,
    _LLM_PROVIDER_ANTHROPIC,
    _LLM_PROVIDER_CUSTOM,
)


def _parse_provider_config_payload(
    payload: dict[str, Any], *, require_api_key: bool
) -> dict[str, Any]:
    provider = (payload.get("provider") or "").strip().lower()
    if provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail="bad_provider")
    api_key = payload.get("api_key")
    if require_api_key:
        if not isinstance(api_key, str) or not api_key.strip():
            raise HTTPException(status_code=400, detail="api_key_required")
        api_key = api_key.strip()
    elif api_key is not None:
        if not isinstance(api_key, str):
            raise HTTPException(status_code=400, detail="api_key_must_be_string")
        api_key = api_key.strip() or None
    repo_pattern = (payload.get("repo_pattern") or "").strip()
    if not _REPO_PATTERN_RE.match(repo_pattern):
        raise HTTPException(status_code=400, detail="bad_repo_pattern")
    api_base_raw = (payload.get("api_base") or "").strip()
    api_base: Optional[str] = None
    if provider == _LLM_PROVIDER_CUSTOM:
        if not api_base_raw:
            raise HTTPException(status_code=400, detail="api_base_required_for_custom")
        api_base = _normalize_llm_base_url(api_base_raw)
    default_model = (payload.get("default_model") or "").strip() or None
    allowed_users = _parse_login_list(payload.get("allowed_users"), "allowed_users")
    allowed_orgs = _parse_login_list(payload.get("allowed_orgs"), "allowed_orgs")
    if not allowed_users and not allowed_orgs:
        raise HTTPException(
            status_code=400,
            detail="allowed_users_or_orgs_required",
        )
    return {
        "provider": provider,
        "api_key": api_key,
        "repo_pattern": repo_pattern,
        "api_base": api_base,
        "default_model": default_model,
        "allowed_users": allowed_users,
        "allowed_orgs": allowed_orgs,
    }


def _parse_login_list(raw: Any, field_name: str) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",")]
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if not isinstance(item, str):
                raise HTTPException(
                    status_code=400, detail=f"{field_name}_must_be_string_list"
                )
            items.append(item.strip())
    else:
        raise HTTPException(
            status_code=400, detail=f"{field_name}_must_be_string_or_list"
        )
    cleaned: list[str] = []
    for item in items:
        if not item:
            continue
        if not _GH_NAME_RE.match(item):
            raise HTTPException(
                status_code=400,
                detail=f"{field_name}_contains_invalid_login",
            )
        cleaned.append(item.lower())
    # De-dup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _provider_config_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Scrub api_key before returning to the UI — replaced with a
    short non-reversible hint so admins can tell which row a key
    belongs to without exposing the secret."""
    raw_key = row.get("api_key") or ""
    if raw_key:
        # Length and last-4 chars give just enough fingerprint to spot a
        # stale row without leaking the secret. Short keys (<8 chars)
        # show no tail.
        tail = raw_key[-4:] if len(raw_key) >= 8 else ""
        key_hint = f"set (len={len(raw_key)}, ends={tail})" if tail else "set"
    else:
        key_hint = ""
    return {
        "id": row["id"],
        "provider": row["provider"],
        "api_base": row.get("api_base") or "",
        "default_model": row.get("default_model") or "",
        "repo_pattern": row["repo_pattern"],
        "allowed_users": row.get("allowed_users") or [],
        "allowed_orgs": row.get("allowed_orgs") or [],
        "created_by": row.get("created_by") or "",
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "api_key_status": key_hint,
    }


def _api_base_for_provider(provider: str, custom_base: Optional[str]) -> str:
    """Resolve the base URL for a provider. Built-ins are looked up
    from the static table; custom uses the per-config api_base. Falls
    back to cfg.llm_api_base only when nothing else is available, so
    older deployments don't break mid-rollout."""
    if provider == _LLM_PROVIDER_CUSTOM:
        raw_base = (custom_base or "").strip()
        if not raw_base:
            default_provider = _infer_llm_provider(cfg.llm_api_base)
            raw_base = (
                cfg.llm_api_base if default_provider == _LLM_PROVIDER_CUSTOM else ""
            )
        if not raw_base:
            raise HTTPException(status_code=400, detail="llm_base_url_required")
        return _normalize_llm_base_url(raw_base)
    return _LLM_PROVIDER_BASES[provider]


def _llm_bill_to_for_provider(provider: str) -> Optional[str]:
    return cfg.llm_bill_to if provider == _LLM_PROVIDER_HF else None


def _prune_store() -> None:
    """Keep only the most recent ``web_job_retention`` jobs globally.
    Called on each new submission so we don't need a background sweeper."""
    pruned = _store.prune(cfg.web_job_retention)
    if pruned:
        log.info("Pruned %d old job(s) (retention=%d)", pruned, cfg.web_job_retention)


# Per-kind cap on the replay buffer. "token" and "reasoning" are emitted
# once per LLM streaming chunk and can easily reach 10^5 entries on a
# huge PR (e.g. transformers#44794), which then drowns the SSE replay and
# freezes the page on reload. Structural events ("log", "step", "tool",
# "error", "metrics", "done") are inherently bounded by the agentic loop
# turn count, so they stay unbounded. The cap is FIFO — newer chunks
# evict older ones, since the tail is more relevant on reload.
_NOISY_KINDS = frozenset({"token", "reasoning"})
_NOISY_HISTORY_CAP = 2000


def _push_event(job: Job, kind: str, text: str) -> None:
    """Thread-safe push from the worker thread into the job's queue.
    Also appends to the replay buffer so late SSE subscribers get the
    full transcript."""
    event = {"kind": kind, "text": text, "ts": time.time()}
    with job.history_lock:
        job.history.append(event)
        if kind in _NOISY_KINDS:
            job.noisy_history_count += 1
            if job.noisy_history_count > _NOISY_HISTORY_CAP:
                for i, e in enumerate(job.history):
                    if e["kind"] in _NOISY_KINDS:
                        del job.history[i]
                        job.noisy_history_count -= 1
                        break
    if job.loop is not None:
        job.loop.call_soon_threadsafe(job.queue.put_nowait, event)


def _persist_terminal(job: Job) -> None:
    """Snapshot a finished job into the store so it survives a restart."""
    with job.history_lock:
        history_copy = list(job.history)
    try:
        _store.save_terminal(
            job.id,
            status=job.status,
            error=job.error,
            raw_llm_output=job.raw_llm_output,
            draft=job.draft,
            history=history_copy,
        )
    except Exception:  # noqa: BLE001
        log.exception("failed to persist terminal state for job %s", job.id)


def _format_llm_error(exc: LLMResponseError) -> str:
    """Render an LLMResponseError for the SSE client. Surfaces the status
    code + a body excerpt so the UI shows whether it was a 429 (rate
    limit), 400 (bad schema), auth, etc. instead of a generic "review
    crashed". The body comes from the LLM provider's own error response —
    no auth tokens of ours are echoed there."""
    excerpt = exc.body_preview.strip()
    if len(excerpt) > 600:
        excerpt = excerpt[:600] + "…"
    reason_part = f" {exc.reason}" if exc.reason else ""
    if excerpt:
        return f"LLM endpoint returned {exc.status_code}{reason_part}: {excerpt}"
    return f"LLM endpoint returned {exc.status_code}{reason_part}"


def _execute_review(
    job: Job,
    worker_cfg: Config,
    gh: GitHubClient,
    token: str,
    req: ReviewRequest,
    *,
    auto_publish: bool,
) -> None:
    """Shared review pipeline for both UI and webhook jobs.

    Shallow-clones the PR head so the LLM gets browse tools, runs
    prepare_review streaming events back to the SSE consumer, then either
    stops at the draft (``auto_publish=False`` — UI flow, a human edits +
    publishes) or posts the review immediately (``auto_publish=True`` —
    webhook flow, no human in the loop). Owns its own checkout cleanup +
    terminal persistence in a finally block."""
    checkout: Optional[Checkout] = None
    try:
        # Check out the PR head so the LLM has browse tools (matches Action
        # mode, which gets a checkout via actions/checkout). Backed by the
        # shared clone cache: a worktree off a per-repo bare clone, not a
        # cold clone. If it fails we still run the review — just without tools.
        if not _bool_env_safe("WEB_DISABLE_CHECKOUT", False):
            _push_event(job, "step", "clone")
            _push_event(job, "log", "Preparing PR checkout…")
            t0 = time.monotonic()
            checkout = _clone_cache.acquire(
                token,
                job.target_owner,
                job.target_repo,
                job.target_number,
                job_id=job.id,
                depth=cfg.web_clone_depth,
            )
            if checkout:
                _push_event(
                    job,
                    "log",
                    f"Checkout ready in {time.monotonic() - t0:.1f}s ({checkout.path})",
                )
                worker_cfg = dataclasses.replace(
                    worker_cfg, repo_checkout_path=checkout.path
                )
            else:
                _push_event(
                    job,
                    "log",
                    "Checkout failed; continuing without browse tools",
                )

        draft = prepare_review(
            worker_cfg,
            gh,
            req,
            chunk_callback=lambda kind, text: _push_event(job, kind, text),
        )
        if draft is None:
            job.status = "done"
            job.error = "no reviewable diff (notice was posted to the PR)"
            _push_event(job, "step", "error")
            _push_event(job, "error", job.error)
            _push_event(job, "done", "")
            return
        job.draft = draft
        if auto_publish:
            _push_event(job, "log", "Publishing review to GitHub…")
            publish_review(worker_cfg, gh, draft)
            job.status = "published"
            _push_event(
                job,
                "log",
                f"Published review: {len(draft.comments)} inline comment(s), "
                f"event={draft.event}",
            )
        else:
            job.status = "done"
            _push_event(
                job,
                "log",
                f"Draft ready: {len(draft.comments)} inline comment(s), "
                f"event={draft.event}",
            )
        _push_event(job, "step", "done")
        _push_event(job, "done", "")
    except _UnparseableLLMOutput as exc:
        job.status = "error"
        job.raw_llm_output = exc.content
        job.error = exc.user_message()
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except AppNotInstalledError as exc:
        # Expected failure mode — the App isn't installed on the target
        # repo. Surface the actionable message verbatim instead of the
        # generic "see server log".
        log.warning("App not installed for %s/%s (job %s)", exc.owner, exc.repo, job.id)
        job.status = "error"
        job.error = str(exc)
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except LLMResponseError as exc:
        log.warning(
            "LLM endpoint returned %d for job %s: %s",
            exc.status_code,
            job.id,
            exc.body_preview[:400],
        )
        job.status = "error"
        job.error = _format_llm_error(exc)
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except Exception as exc:  # noqa: BLE001
        log.exception("review worker crashed for job %s", job.id)
        job.status = "error"
        # Exception messages occasionally echo upstream response bodies
        # that may contain auth tokens (e.g. httpx HTTPError). Don't ship
        # the raw repr to the SSE client — the full traceback is in the
        # server log via log.exception above.
        job.error = f"{type(exc).__name__}: review crashed (see server log)"
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    finally:
        # Drop this job's worktree + branch; the bare repo and its objects
        # stay warm for the next review on the same repo.
        _clone_cache.release(checkout)
        # Snapshot the final state into SQLite. Every terminal branch
        # above sets job.status to a non-'running' value, so this also
        # clears the 'running' marker we'd otherwise reap on next restart.
        _persist_terminal(job)


def _run_review_worker(job: Job) -> None:
    """UI entry point. Runs in a background thread: pulls an installation
    token for the target repo, builds the review request from the job, and
    delegates to _execute_review (which streams + stops at the draft for a
    human to edit + publish)."""
    assert cfg.github_app_id and cfg.github_private_key
    try:
        installation_id = installation_id_for_repo(
            cfg.github_app_id,
            cfg.github_private_key,
            job.target_owner,
            job.target_repo,
        )
        token = installation_token(
            cfg.github_app_id, cfg.github_private_key, installation_id
        )
        gh = GitHubClient(token)
    except Exception as exc:  # noqa: BLE001
        # Token / installation lookup failed before we could even start.
        # Mark the job errored + persist so it doesn't hang in 'running'.
        if isinstance(exc, AppNotInstalledError):
            log.warning(
                "App not installed for %s/%s (job %s)", exc.owner, exc.repo, job.id
            )
            job.error = str(exc)
        else:
            log.exception("review worker setup failed for job %s", job.id)
            job.error = f"{type(exc).__name__}: review crashed (see server log)"
        job.status = "error"
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
        _persist_terminal(job)
        return
    req = ReviewRequest(
        owner=job.target_owner,
        repo=job.target_repo,
        number=job.target_number,
        trigger_comment_id=0,
        trigger_comment_body=job.trigger_comment,
        commenter=job.user,
    )
    worker_cfg = dataclasses.replace(
        cfg,
        llm_api_base=job.llm_api_base,
        llm_api_key=job.llm_api_key,
        llm_model=job.llm_model,
        llm_bill_to=_llm_bill_to_for_provider(job.llm_provider),
    )
    _execute_review(job, worker_cfg, gh, token, req, auto_publish=False)


def _bool_env_safe(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# FastAPI app + routes.
# ---------------------------------------------------------------------------
app = FastAPI(title="Serge web reviewer")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.on_event("startup")
async def _start_clone_cache_gc() -> None:
    """Hourly GC of the clone cache: drop bare repos untouched for longer
    than the TTL. Runs in a thread so it never blocks the event loop."""
    ttl = cfg.web_clone_cache_ttl_seconds

    async def _loop() -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                await asyncio.to_thread(_clone_cache.gc, ttl)
            except Exception:  # noqa: BLE001
                log.exception("clone cache GC failed")

    asyncio.create_task(_loop())


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Force browsers to revalidate /static/* on every load. Saves
    users from staring at a stale review.js after we push fixes."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.post("/webhook")
async def github_app_webhook(request: Request) -> Response:
    body = await request.body()
    if not cfg.github_webhook_secret:
        log.error("rejected GitHub webhook: GITHUB_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=503, detail="webhook_not_configured")
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_webhook_signature(body, sig):
        log.warning("rejected GitHub webhook with bad signature")
        raise HTTPException(status_code=401, detail="bad_signature")

    event = request.headers.get("X-GitHub-Event", "")
    try:
        payload = _json.loads(body.decode("utf-8") or "{}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="bad_json") from exc

    if event == "ping":
        return JSONResponse({"pong": True})

    req = build_review_request(event, payload, cfg.mention_trigger)
    if req is None:
        return Response(status_code=204)

    installation = payload.get("installation") or {}
    installation_id = installation.get("id")
    if not isinstance(installation_id, int):
        return Response(status_code=204)

    resolved = _resolve_webhook_worker_cfg(req)
    if resolved is None:
        log.error(
            "webhook review for %s/%s#%d skipped: no LLM API key "
            "(no matching provider_config and LLM_API_KEY is unset)",
            req.owner,
            req.repo,
            req.number,
        )
        return JSONResponse({"status": "skipped_no_key"}, status_code=202)
    worker_cfg, provider, llm_api_base, llm_model = resolved

    # Register a persisted job so the review shows up in the journal and
    # gets a live review page (same as UI-submitted reviews). The
    # triggering commenter is recorded as the "user" for the journal; the
    # "webhook" source lets any authenticated viewer follow it.
    job = Job(
        id=uuid.uuid4().hex,
        user=req.commenter,
        target_owner=req.owner,
        target_repo=req.repo,
        target_number=req.number,
        trigger_comment=req.trigger_comment_body,
        llm_provider=provider,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        created_at=time.time(),
        llm_api_key=worker_cfg.llm_api_key,
        source="webhook",
    )
    job.loop = asyncio.get_running_loop()
    with _jobs_lock:
        _jobs[job.id] = job
    _store.insert_job(
        id=job.id,
        user=job.user,
        target_owner=job.target_owner,
        target_repo=job.target_repo,
        target_number=job.target_number,
        trigger_comment=job.trigger_comment,
        llm_provider=job.llm_provider,
        llm_api_base=job.llm_api_base,
        llm_model=job.llm_model,
        created_at=job.created_at,
        status=job.status,
        source=job.source,
    )
    _prune_store()
    _WEBHOOK_REVIEW_POOL.submit(
        _run_webhook_review_worker, job, worker_cfg, installation_id, req
    )
    log.info(
        "queued webhook job %s for %s/%s#%d (triggered by %s) using %s model=%s",
        job.id,
        req.owner,
        req.repo,
        req.number,
        req.commenter,
        provider,
        llm_model or "<auto>",
    )
    return JSONResponse(
        {
            "status": "accepted",
            "id": job.id,
            "url": (
                f"/reviews/{req.owner}/{req.repo}/{req.number}/{job.id}"
            ),
        },
        status_code=202,
    )


def _require_user(request: Request) -> str:
    user = _current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="not_authenticated"
        )
    return user


def _require_same_origin(request: Request) -> None:
    """CSRF guard for state-changing endpoints. SameSite=Lax already
    blocks most cross-site form posts, but Origin/Referer is the
    backstop — we refuse the request unless one of them points back at
    the host we're serving on."""
    expected_host = (request.url.netloc or "").lower()
    if not expected_host:
        return
    origin = (request.headers.get("origin") or "").strip()
    referer = (request.headers.get("referer") or "").strip()
    for header in (origin, referer):
        if not header:
            continue
        try:
            host = urllib.parse.urlparse(header).netloc.lower()
        except ValueError:
            continue
        if host == expected_host:
            return
    raise HTTPException(status_code=403, detail="bad_origin")


def _serve_static(name: str) -> HTMLResponse:
    path = os.path.join(_STATIC_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    if name.endswith(".html"):
        html = html.replace("<!-- POWERED_BY -->", _powered_by_html())
        html = html.replace("<!-- APP_INSTALL_LINK -->", _app_install_html())
    return HTMLResponse(html)


def _powered_by_html() -> str:
    """Render the fixed Inference Providers badge next to the Serge brand."""
    url = "https://huggingface.co/docs/inference-providers/en/index"
    return (
        '<span class="powered-by">powered by '
        f'<a href="{_html.escape(url, quote=True)}" '
        f'target="_blank" rel="noopener noreferrer">'
        "HF Inference Providers</a></span>"
    )


def _app_install_html() -> str:
    """Render the "install the GitHub App" link for the help page,
    using WEB_GITHUB_APP_URL when set. Falls back to a hint pointing
    operators at the env var so deployments without the variable still
    get a useful page."""
    url = (cfg.web_github_app_url or "").strip()
    if not url:
        return (
            '<span class="hint">Ask your deployment admin for the App '
            "install URL, or set <code>WEB_GITHUB_APP_URL</code> in the "
            "server environment so this page can link to it.</span>"
        )
    escaped = _html.escape(url, quote=True)
    return (
        f'<a href="{escaped}" target="_blank" rel="noopener noreferrer">'
        '<button class="primary" type="button">Install the GitHub App</button>'
        "</a>"
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def index(request: Request) -> Response:
    if not _current_user(request):
        return RedirectResponse("/login", status_code=302)
    return _serve_static("index.html")


@app.get("/journal")
def journal_page(request: Request) -> Response:
    if not _current_user(request):
        return RedirectResponse("/login", status_code=302)
    return _serve_static("journal.html")


@app.get("/help")
def help_page(request: Request) -> Response:
    if not _current_user(request):
        return RedirectResponse("/login", status_code=302)
    return _serve_static("help.html")


@app.get("/login")
def login_page(request: Request) -> Response:
    if _current_user(request):
        return RedirectResponse("/", status_code=302)
    if cfg.web_dev_no_auth:
        # In dev mode there is no OAuth roundtrip; the index page is
        # immediately accessible.
        return RedirectResponse("/", status_code=302)
    return _serve_static("login.html")


@app.get("/auth/login")
def auth_login(request: Request) -> Response:
    if cfg.web_dev_no_auth:
        return RedirectResponse("/", status_code=302)
    state = secrets.token_urlsafe(24)
    sess = _load_session(request)
    sess["oauth_state"] = state
    redirect_uri = cfg.github_oauth_callback_url or str(
        request.url_for("auth_callback")
    )
    params = {
        "client_id": cfg.github_oauth_client_id or "",
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": "read:org",
        "allow_signup": "false",
    }
    qs = urllib.parse.urlencode(params)
    response = RedirectResponse(
        f"https://github.com/login/oauth/authorize?{qs}", status_code=302
    )
    _save_session(response, sess)
    return response


@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request) -> Response:
    if cfg.web_dev_no_auth:
        return RedirectResponse("/", status_code=302)
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    sess = _load_session(request)
    expected_state = sess.pop("oauth_state", None)
    if not code or not state or state != expected_state:
        raise HTTPException(status_code=400, detail="invalid_oauth_state")

    async with httpx.AsyncClient(timeout=30) as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": cfg.github_oauth_client_id,
                "client_secret": cfg.github_oauth_client_secret,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token = token_resp.json().get("access_token")
        if not token:
            raise HTTPException(status_code=400, detail="oauth_token_exchange_failed")

        user_resp = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        user_resp.raise_for_status()
        login = user_resp.json().get("login")
        if not isinstance(login, str) or not login:
            raise HTTPException(status_code=400, detail="oauth_no_login")

        # Always fetch orgs at login: even when web_allowed_orgs is
        # empty, the provider_configs table uses orgs to gate which API
        # key a user may consume, so we need the list cached in the
        # session for later matching.
        orgs: list[str] = []
        orgs_resp = await client.get(
            "https://api.github.com/user/orgs",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        if orgs_resp.is_success:
            orgs = [
                o.get("login", "")
                for o in orgs_resp.json()
                if isinstance(o, dict) and o.get("login")
            ]

    if not _user_is_allowed(login, orgs):
        # /user/orgs honors SAML SSO — if the OAuth token wasn't
        # SSO-authorized for the org, the org won't appear here even when
        # the user is a real member. Fall back to the GitHub App's own
        # view of org membership (public_members first, then the App
        # installation route) which isn't subject to user-side SSO.
        verified_via_app: list[str] = []
        if cfg.web_allowed_orgs and cfg.github_app_id and cfg.github_private_key:
            for org in cfg.web_allowed_orgs:
                if user_is_org_member(
                    cfg.github_app_id, cfg.github_private_key, org, login
                ):
                    verified_via_app.append(org)
                    break
        if not verified_via_app:
            log.warning("denied login attempt by %s (orgs=%s)", login, orgs)
            raise HTTPException(status_code=403, detail="user_not_allowed")
        log.info(
            "user %s authorized via App membership lookup (orgs=%s)",
            login,
            verified_via_app,
        )

    sess["user"] = login
    sess["orgs"] = orgs
    response = RedirectResponse("/", status_code=302)
    _save_session(response, sess)
    log.info("user %s logged in (orgs=%s)", login, orgs)
    return response


@app.post("/auth/logout")
def auth_logout(request: Request) -> Response:
    _require_same_origin(request)
    response = RedirectResponse("/login", status_code=302)
    _clear_session(response)
    return response


@app.get("/reviews")
def list_reviews(request: Request) -> JSONResponse:
    """All persisted jobs the current user has submitted, newest first.
    Reads from the SQLite store so reviews survive process restarts —
    capped globally by ``web_job_retention``. For an in-flight job the
    DB row still says 'running'; we cross-reference the live registry so
    a process restart immediately reflects as 'error' even before the
    startup sweeper has run (it should already have, but belt-and-braces)."""
    user = _require_user(request)
    rows = _store.list_for_user(user, limit=cfg.web_job_retention)
    return JSONResponse(
        {
            "jobs": [
                {
                    "id": r["id"],
                    "owner": r["target_owner"],
                    "repo": r["target_repo"],
                    "number": r["target_number"],
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "url": (
                        f"/reviews/{r['target_owner']}/{r['target_repo']}/"
                        f"{r['target_number']}/{r['id']}"
                    ),
                }
                for r in rows
            ]
        }
    )


@app.get("/journal/data")
def journal_data(request: Request) -> JSONResponse:
    """Cross-user activity log — every persisted review job with its
    token usage, provider/model, and status. Any authenticated user can
    read this; the allowlist already gates who has an account."""
    _require_user(request)
    rows = _store.list_all_calls(limit=cfg.web_job_retention)
    return JSONResponse(
        {
            "entries": [
                {
                    "id": r["id"],
                    "user": r["user"],
                    "owner": r["target_owner"],
                    "repo": r["target_repo"],
                    "number": r["target_number"],
                    "status": r["status"],
                    "source": r["source"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "provider": r["llm_provider"],
                    "model": r["llm_model"],
                    "prompt_tokens": r["prompt_tokens"],
                    "completion_tokens": r["completion_tokens"],
                    "url": (
                        f"/reviews/{r['target_owner']}/{r['target_repo']}/"
                        f"{r['target_number']}/{r['id']}"
                    ),
                }
                for r in rows
            ]
        }
    )


@app.get("/llm-options")
def llm_options(request: Request) -> JSONResponse:
    _require_user(request)
    provider = _infer_llm_provider(cfg.llm_api_base)
    custom_base = cfg.llm_api_base if provider == _LLM_PROVIDER_CUSTOM else ""

    def _provider_entry(pid: str, label: str, base_url: str) -> dict:
        entry: dict[str, Any] = {"id": pid, "label": label, "base_url": base_url}
        # The server-configured cfg.llm_model is the default for whichever
        # provider matches cfg.llm_api_base; other providers fall back to
        # the static per-provider defaults so switching in the UI lands on
        # something sensible instead of an empty box.
        default_model = (
            cfg.llm_model
            if pid == provider and cfg.llm_model
            else _LLM_PROVIDER_DEFAULT_MODELS.get(pid, "")
        )
        if default_model:
            entry["default_model"] = default_model
        return entry

    return JSONResponse(
        {
            "default_provider": provider,
            "default_model": cfg.llm_model or "",
            "custom_base_url": custom_base,
            "providers": [
                _provider_entry(
                    _LLM_PROVIDER_HF, "HF", _LLM_PROVIDER_BASES[_LLM_PROVIDER_HF]
                ),
                _provider_entry(
                    _LLM_PROVIDER_OPENAI,
                    "OpenAI",
                    _LLM_PROVIDER_BASES[_LLM_PROVIDER_OPENAI],
                ),
                _provider_entry(
                    _LLM_PROVIDER_ANTHROPIC,
                    "Anthropic",
                    _LLM_PROVIDER_BASES[_LLM_PROVIDER_ANTHROPIC],
                ),
                _provider_entry(_LLM_PROVIDER_CUSTOM, "Custom", custom_base),
            ],
        }
    )


@app.get("/reviews/lookup-provider")
def lookup_provider(request: Request, owner: str, repo: str) -> JSONResponse:
    """Pre-flight match for the submit form: given a (owner, repo)
    pulled from the PR field as the user types, return the provider +
    default model from the best-matching ``provider_config`` so the UI
    can auto-fill its dropdown. Never returns the api_key — only the
    fields the form is allowed to display. Returns ``match: null`` when
    no config matches; the form then shows a hint that submission will
    be refused until an admin adds one."""
    user = _require_user(request)
    if not _GH_NAME_RE.match(owner) or not _GH_NAME_RE.match(repo):
        raise HTTPException(status_code=400, detail="bad_repo")
    # Build the response first so the org-augmenting helper can attach
    # a refreshed session cookie to it when new memberships are
    # discovered via the App.
    payload: dict[str, Any] = {"match": None}
    placeholder = JSONResponse(payload)
    user_orgs = _effective_user_orgs_for_repo(
        request,
        placeholder,
        user,
        owner,
        repo,
    )
    matched = _store.find_provider_config(
        user=user,
        user_orgs=user_orgs,
        owner=owner,
        repo=repo,
    )
    if matched is not None:
        payload = {
            "match": {
                "provider": matched["provider"],
                "default_model": matched.get("default_model") or "",
                "api_base": matched.get("api_base") or "",
                "repo_pattern": matched["repo_pattern"],
            }
        }
    final = JSONResponse(payload)
    # Forward the session cookie that the helper attached to the
    # placeholder so the augmented orgs persist across requests.
    cookie = placeholder.headers.get("set-cookie")
    if cookie:
        final.raw_headers.append((b"set-cookie", cookie.encode("latin-1")))
    return final


@app.get("/admin")
def admin_page(request: Request) -> Response:
    if not _current_user(request):
        return RedirectResponse("/login", status_code=302)
    return _serve_static("admin.html")


@app.get("/admin/providers")
def admin_list_providers(request: Request) -> JSONResponse:
    _require_user(request)
    rows = _store.list_provider_configs()
    return JSONResponse(
        {
            "providers": _VALID_PROVIDERS,
            "default_models": _LLM_PROVIDER_DEFAULT_MODELS,
            "configs": [_provider_config_summary(r) for r in rows],
        }
    )


@app.post("/admin/providers")
async def admin_create_provider(request: Request) -> JSONResponse:
    _require_same_origin(request)
    user = _require_user(request)
    payload = await request.json()
    fields = _parse_provider_config_payload(payload, require_api_key=True)
    config_id = uuid.uuid4().hex
    _store.insert_provider_config(
        id=config_id,
        provider=fields["provider"],
        api_key=fields["api_key"],
        api_base=fields["api_base"],
        default_model=fields["default_model"],
        repo_pattern=fields["repo_pattern"],
        allowed_users=fields["allowed_users"],
        allowed_orgs=fields["allowed_orgs"],
        created_by=user,
    )
    log.info(
        "user %s added provider config %s (%s for %s)",
        user,
        config_id,
        fields["provider"],
        fields["repo_pattern"],
    )
    row = _store.get_provider_config(config_id)
    assert row is not None
    return JSONResponse(_provider_config_summary(row), status_code=201)


@app.patch("/admin/providers/{config_id}")
async def admin_update_provider(request: Request, config_id: str) -> JSONResponse:
    _require_same_origin(request)
    user = _require_user(request)
    existing = _store.get_provider_config(config_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="config_not_found")
    payload = await request.json()
    # api_key is optional on update — when omitted, keep the stored
    # secret. When present (non-empty), replace it.
    fields = _parse_provider_config_payload(payload, require_api_key=False)
    updated = _store.update_provider_config(
        config_id,
        provider=fields["provider"],
        api_base=fields["api_base"],
        default_model=fields["default_model"],
        repo_pattern=fields["repo_pattern"],
        allowed_users=fields["allowed_users"],
        allowed_orgs=fields["allowed_orgs"],
        new_api_key=fields["api_key"],
    )
    if not updated:
        raise HTTPException(status_code=404, detail="config_not_found")
    log.info(
        "user %s updated provider config %s (%s for %s; key_replaced=%s)",
        user,
        config_id,
        fields["provider"],
        fields["repo_pattern"],
        fields["api_key"] is not None,
    )
    row = _store.get_provider_config(config_id)
    assert row is not None
    return JSONResponse(_provider_config_summary(row))


@app.delete("/admin/providers/{config_id}")
def admin_delete_provider(request: Request, config_id: str) -> JSONResponse:
    _require_same_origin(request)
    user = _require_user(request)
    deleted = _store.delete_provider_config(config_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="config_not_found")
    log.info("user %s deleted provider config %s", user, config_id)
    return JSONResponse({"status": "deleted"})


@app.post("/reviews")
async def submit_review(request: Request) -> JSONResponse:
    _require_same_origin(request)
    user = _require_user(request)
    payload = await request.json()
    pr_ref = (payload.get("pr") or "").strip()
    trigger_comment = (payload.get("comment") or "").strip()
    if not pr_ref:
        raise HTTPException(status_code=400, detail="pr_required")
    if not trigger_comment:
        trigger_comment = f"{cfg.mention_trigger} please review"

    if len(trigger_comment) > _MAX_TRIGGER_COMMENT_CHARS:
        raise HTTPException(status_code=413, detail="comment_too_long")
    owner, repo, number = _parse_pr_ref(pr_ref)
    llm_provider = _parse_provider(payload)
    # Stash any newly-discovered orgs on the eventual response so a
    # SAML-protected user doesn't pay the App-membership round-trip on
    # every subsequent submission.
    session_response = JSONResponse({})
    user_orgs = _effective_user_orgs_for_repo(
        request,
        session_response,
        user,
        owner,
        repo,
    )
    matched = _store.find_provider_config(
        user=user,
        user_orgs=user_orgs,
        owner=owner,
        repo=repo,
        provider=llm_provider,
    )
    if matched is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No provider config grants you access to '{owner}/{repo}' "
                f"with provider '{llm_provider}'. Add one at /admin."
            ),
        )
    # Per-config api_base wins for custom; built-ins use the static map.
    llm_api_base = _api_base_for_provider(
        llm_provider,
        custom_base=matched.get("api_base") or payload.get("llm_base_url"),
    )
    # User-supplied model > config default_model > static per-provider default.
    requested_model = (payload.get("llm_model") or "").strip()
    llm_model = (
        requested_model
        or (matched.get("default_model") or "").strip()
        or _LLM_PROVIDER_DEFAULT_MODELS.get(llm_provider, "")
    ) or None

    job = Job(
        id=uuid.uuid4().hex,
        user=user,
        target_owner=owner,
        target_repo=repo,
        target_number=number,
        trigger_comment=trigger_comment,
        llm_provider=llm_provider,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
        created_at=time.time(),
        llm_api_key=matched["api_key"],
    )
    job.loop = asyncio.get_running_loop()
    with _jobs_lock:
        _jobs[job.id] = job
    _store.insert_job(
        id=job.id,
        user=job.user,
        target_owner=job.target_owner,
        target_repo=job.target_repo,
        target_number=job.target_number,
        trigger_comment=job.trigger_comment,
        llm_provider=job.llm_provider,
        llm_api_base=job.llm_api_base,
        llm_model=job.llm_model,
        created_at=job.created_at,
        status=job.status,
    )
    _prune_store()
    threading.Thread(
        target=_run_review_worker, args=(job,), name=f"job-{job.id}", daemon=True
    ).start()
    log.info(
        "queued job %s for %s/%s#%d by %s using %s model=%s base=%s",
        job.id,
        owner,
        repo,
        number,
        user,
        llm_provider,
        llm_model or "<auto>",
        llm_api_base,
    )
    final = JSONResponse(
        {
            "id": job.id,
            "owner": owner,
            "repo": repo,
            "number": number,
            "url": f"/reviews/{owner}/{repo}/{number}/{job.id}",
        }
    )
    cookie = session_response.headers.get("set-cookie")
    if cookie:
        final.raw_headers.append((b"set-cookie", cookie.encode("latin-1")))
    return final


def _parse_pr_ref(ref: str) -> tuple[str, str, int]:
    """Accept "owner/repo#123", "owner/repo/pull/123", or a full GitHub
    URL. Return (owner, repo, number) or raise HTTPException 400."""
    s = ref.strip()
    if s.startswith("http"):
        # https://github.com/owner/repo/pull/123 (optionally with /files etc.)
        try:
            parts = s.split("github.com/", 1)[1].split("/")
            owner, repo = parts[0], parts[1]
            if parts[2] not in ("pull", "pulls"):
                raise ValueError("not a pull URL")
            number = int(parts[3])
            return _validate_pr_ref(owner, repo, number)
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"bad_pr_url: {exc}") from exc
    if "#" in s:
        repo_part, num_part = s.split("#", 1)
        if "/" not in repo_part:
            raise HTTPException(status_code=400, detail="bad_pr_ref")
        owner, repo = repo_part.split("/", 1)
        try:
            return _validate_pr_ref(owner, repo, int(num_part))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="bad_pr_number") from exc
    if "/" in s and s.count("/") >= 3 and "pull" in s.split("/"):
        parts = s.split("/")
        try:
            i = parts.index("pull")
            return _validate_pr_ref(parts[i - 2], parts[i - 1], int(parts[i + 1]))
        except (ValueError, IndexError) as exc:
            raise HTTPException(status_code=400, detail="bad_pr_ref") from exc
    raise HTTPException(status_code=400, detail="bad_pr_ref")


def _validate_pr_ref(owner: str, repo: str, number: int) -> tuple[str, str, int]:
    if not _GH_NAME_RE.match(owner) or not _GH_NAME_RE.match(repo):
        raise HTTPException(status_code=400, detail="bad_pr_ref")
    if number < 1 or number > 10_000_000:
        raise HTTPException(status_code=400, detail="bad_pr_number")
    return owner, repo, number


def _get_owned_job(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> Job:
    """Resolve a job by its full {owner}/{repo}/{number}/{id} URL. Ensures
    the path identifiers match the job's actual target so users can't
    poke at someone else's job by guessing IDs, and so stale links
    fail-fast instead of silently serving the wrong PR's data.

    Prefers the live in-memory entry (which carries the SSE queue + replay
    history for running streams). Falls back to the SQLite store so
    finished jobs survive a process restart."""
    user = _require_user(request)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        job = _load_job_from_store(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    # Webhook jobs are kicked off by a GitHub comment, not a logged-in
    # user, so there is no real owner — any authenticated viewer may
    # follow them (the journal already exposes them cross-user). UI jobs
    # stay private to the submitter.
    if job.source != "webhook" and job.user != user:
        raise HTTPException(status_code=403, detail="not_your_job")
    if (
        job.target_owner != owner
        or job.target_repo != repo
        or job.target_number != number
    ):
        raise HTTPException(status_code=404, detail="job_target_mismatch")
    return job


def _load_job_from_store(job_id: str) -> Optional[Job]:
    """Reconstruct a finished Job from the SQLite row. The live-only
    fields (queue, loop) stay at their defaults — the SSE generator
    handles the "no live tail" case by replaying history and stopping."""
    row = _store.load(job_id)
    if row is None:
        return None
    job = Job(
        id=row["id"],
        user=row["user"],
        target_owner=row["target_owner"],
        target_repo=row["target_repo"],
        target_number=row["target_number"],
        trigger_comment=row["trigger_comment"],
        llm_provider=row.get("llm_provider") or _infer_llm_provider(cfg.llm_api_base),
        llm_api_base=row.get("llm_api_base") or cfg.llm_api_base,
        llm_model=row.get("llm_model") or cfg.llm_model,
        created_at=row["created_at"],
        status=row["status"],
        source=row.get("source") or "web",
        error=row["error"],
        raw_llm_output=row["raw_llm_output"],
        draft=decode_draft(row["draft_json"]) if row.get("draft_json") else None,
    )
    job.history = list(row.get("history") or [])
    return job


@app.get("/reviews/{owner}/{repo}/{number}/{job_id}")
def review_page(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> Response:
    # Ownership is enforced by the JSON endpoints; the HTML page is
    # static and safe to serve to any logged-in user.
    _require_user(request)
    return _serve_static("review.html")


@app.get("/reviews/{owner}/{repo}/{number}/{job_id}/info")
def review_info(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> JSONResponse:
    job = _get_owned_job(request, owner, repo, number, job_id)
    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "target": f"{job.target_owner}/{job.target_repo}#{job.target_number}",
            "trigger_comment": job.trigger_comment,
            "llm_provider": job.llm_provider,
            "llm_base_url": job.llm_api_base,
            "llm_model": job.llm_model or "",
            "error": job.error,
        }
    )


@app.get("/reviews/{owner}/{repo}/{number}/{job_id}/draft")
def review_draft(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> JSONResponse:
    job = _get_owned_job(request, owner, repo, number, job_id)
    if job.draft is None:
        return JSONResponse({"status": job.status, "error": job.error, "draft": None})
    return JSONResponse(
        {
            "status": job.status,
            "error": job.error,
            "draft": _draft_to_dict(job.draft),
        }
    )


def _draft_to_dict(draft: ReviewDraft) -> dict[str, Any]:
    return {
        "owner": draft.owner,
        "repo": draft.repo,
        "number": draft.number,
        "head_sha": draft.head_sha,
        "summary": draft.summary,
        "event": draft.event,
        "rejected_count": draft.rejected_count,
        "metrics_line": draft.metrics_line,
        "comments": [dataclasses.asdict(c) for c in draft.comments],
    }


@app.get("/reviews/{owner}/{repo}/{number}/{job_id}/stream")
async def review_stream(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> StreamingResponse:
    job = _get_owned_job(request, owner, repo, number, job_id)

    async def generator():
        # Replay history first so reloads / late subscribers see the full
        # transcript. For finished jobs we strip token/reasoning chunks:
        # the worker may have emitted 10^5 of them on a huge PR (see
        # _NOISY_HISTORY_CAP) and replaying them on every reload freezes
        # the page. The draft is what matters once the job is done; the
        # remaining structural events still show clone/fetch/llm/tool
        # progress so the console isn't blank.
        finished = job.status in ("done", "error", "discarded", "published")
        with job.history_lock:
            if finished:
                replay = [e for e in job.history if e["kind"] not in _NOISY_KINDS]
            else:
                replay = list(job.history)
        for event in replay:
            yield _sse_format(event)
        # If the job already finished while we were replaying, stop here.
        if finished:
            # Make sure the final "done" event is included.
            if not any(e.get("kind") == "done" for e in replay):
                yield _sse_format({"kind": "done", "text": ""})
            return
        # Otherwise, stream live events. Use a short timeout so a client
        # disconnect propagates quickly.
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(job.queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Heartbeat keeps proxies (nginx, cloudflare) from closing
                # the connection on idle long streams.
                yield ": keepalive\n\n"
                continue
            yield _sse_format(event)
            if event.get("kind") == "done":
                return

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        generator(), media_type="text/event-stream", headers=headers
    )


def _sse_format(event: dict[str, Any]) -> str:
    kind = event.get("kind", "log")
    text = event.get("text", "")
    # Use SSE event= for non-default kinds; "log" stays the default
    # message event so the browser handler doesn't need to special-case.
    if kind == "log":
        return f"data: {_json_inline(text)}\n\n"
    return f"event: {kind}\ndata: {_json_inline(text)}\n\n"


def _json_inline(s: str) -> str:
    # SSE data lines can't contain bare newlines; encode as JSON so the
    # client gets a well-defined string back.
    return _json.dumps(s, ensure_ascii=False)


@app.post("/reviews/{owner}/{repo}/{number}/{job_id}/publish")
async def publish(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> JSONResponse:
    _require_same_origin(request)
    job = _get_owned_job(request, owner, repo, number, job_id)
    if job.draft is None:
        raise HTTPException(status_code=409, detail="draft_not_ready")
    payload = await request.json()
    edits = _edits_from_payload(payload, job.draft)
    assert cfg.github_app_id and cfg.github_private_key
    installation_id = installation_id_for_repo(
        cfg.github_app_id,
        cfg.github_private_key,
        job.draft.owner,
        job.draft.repo,
    )
    token = installation_token(
        cfg.github_app_id, cfg.github_private_key, installation_id
    )
    gh = GitHubClient(token)
    publish_review(cfg, gh, job.draft, edits=edits)
    job.status = "published"
    _store.update_status(job.id, "published")
    return JSONResponse({"status": "published"})


def _edits_from_payload(payload: dict[str, Any], draft: ReviewDraft) -> ReviewEdits:
    summary = payload.get("summary")
    if summary is not None and not isinstance(summary, str):
        raise HTTPException(status_code=400, detail="summary_must_be_string")
    event = payload.get("event")
    if event is not None and event not in ("COMMENT", "REQUEST_CHANGES", "APPROVE"):
        raise HTTPException(status_code=400, detail="bad_event")
    # APPROVE can be picked by the model from attacker-controlled diff/body,
    # so unless the operator has opted in via ALLOW_APPROVE we refuse to
    # publish it — even if a distracted reviewer clicked through.
    if event == "APPROVE" and not cfg.allow_approve:
        raise HTTPException(status_code=403, detail="approve_disabled")
    overrides_raw = payload.get("comment_overrides") or {}
    if not isinstance(overrides_raw, dict):
        raise HTTPException(status_code=400, detail="comment_overrides_must_be_object")
    discarded_raw = payload.get("discarded_comment_ids") or []
    if not isinstance(discarded_raw, list):
        raise HTTPException(status_code=400, detail="discarded_must_be_array")

    known_ids = {c.id for c in draft.comments}
    overrides = {
        k: v
        for k, v in overrides_raw.items()
        if isinstance(k, str) and k in known_ids and isinstance(v, str)
    }
    discarded = {k for k in discarded_raw if isinstance(k, str) and k in known_ids}
    return ReviewEdits(
        summary=summary,
        event=event,
        comment_overrides=overrides,
        discarded_comment_ids=discarded,
    )


@app.post("/reviews/{owner}/{repo}/{number}/{job_id}/discard")
def discard(
    request: Request, owner: str, repo: str, number: int, job_id: str
) -> JSONResponse:
    """Discard a draft entirely — removes it from both the in-memory
    registry and the SQLite store. Refusing a draft shouldn't clutter
    the user's history with dead rows; if they want a record, they
    should publish instead."""
    _require_same_origin(request)
    job = _get_owned_job(request, owner, repo, number, job_id)
    with _jobs_lock:
        _jobs.pop(job.id, None)
    _store.delete(job.id)
    return JSONResponse({"status": "discarded"})


# Suppress an unused-import warning for DraftComment (re-exported via
# dataclasses.asdict in _draft_to_dict, but pyright loses track).
_ = DraftComment


def main() -> int:
    """Console entry point: runs uvicorn on 0.0.0.0:PORT."""
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(
        "reviewbot.webapp:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # single worker — in-memory jobs registry
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
