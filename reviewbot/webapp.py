"""Interactive web mode: a small FastAPI app that lets a logged-in user
trigger a Serge review on a PR, watch it stream live, then tweak the
summary + per-comment text (or discard individual inline comments)
before publishing. The published review still goes out under the
GitHub App identity — OAuth is only used for access control.
"""
import asyncio
import base64
import dataclasses
import html as _html
import json as _json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import uuid
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

from .config import Config
from .github_auth import (
    AppNotInstalledError,
    installation_id_for_repo,
    installation_token,
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
)
from .store import JobStore, decode_draft

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
        raise RuntimeError(
            "WEB_SESSION_SECRET is required when DEV_NO_AUTH is off"
        )
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
        # Cookie must travel only over HTTPS in production. In dev mode
        # (DEV_NO_AUTH) we relax this so localhost http:// flows work.
        secure=not cfg.web_dev_no_auth,
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


def _llm_selection_from_payload(payload: dict[str, Any]) -> tuple[str, str, Optional[str]]:
    default_provider = _infer_llm_provider(cfg.llm_api_base)
    provider = (payload.get("llm_provider") or default_provider).strip().lower()
    if provider not in (
        _LLM_PROVIDER_HF,
        _LLM_PROVIDER_OPENAI,
        _LLM_PROVIDER_ANTHROPIC,
        _LLM_PROVIDER_CUSTOM,
    ):
        raise HTTPException(status_code=400, detail="bad_llm_provider")

    if provider == _LLM_PROVIDER_CUSTOM:
        raw_base = (payload.get("llm_base_url") or "").strip()
        if not raw_base:
            raw_base = cfg.llm_api_base if default_provider == _LLM_PROVIDER_CUSTOM else ""
        if not raw_base:
            raise HTTPException(status_code=400, detail="llm_base_url_required")
        api_base = _normalize_llm_base_url(raw_base)
    else:
        api_base = _LLM_PROVIDER_BASES[provider]

    model = (
        payload.get("llm_model")
        or _LLM_PROVIDER_DEFAULT_MODELS.get(provider)
        or cfg.llm_model
        or ""
    ).strip() or None
    return provider, api_base, model


def _configured_llm_api_key_for_provider(provider: str) -> Optional[str]:
    default_provider = _infer_llm_provider(cfg.llm_api_base)
    if provider == _LLM_PROVIDER_HF:
        return (
            os.environ.get("HF_API_KEY")
            or os.environ.get("HUGGINGFACE_API_KEY")
            or os.environ.get("HF_TOKEN")
            or cfg.llm_api_key
        )
    if provider == _LLM_PROVIDER_OPENAI:
        return os.environ.get("OPENAI_API_KEY") or (
            cfg.llm_api_key if default_provider == _LLM_PROVIDER_OPENAI else None
        )
    if provider == _LLM_PROVIDER_ANTHROPIC:
        return os.environ.get("ANTHROPIC_API_KEY") or (
            cfg.llm_api_key if default_provider == _LLM_PROVIDER_ANTHROPIC else None
        )
    return os.environ.get("CUSTOM_LLM_API_KEY") or (
        cfg.llm_api_key if default_provider == _LLM_PROVIDER_CUSTOM else None
    )


def _llm_key_hint(provider: str) -> str:
    if provider == _LLM_PROVIDER_HF:
        return "Set HF_API_KEY, HUGGINGFACE_API_KEY, HF_TOKEN, or LLM_API_KEY."
    if provider == _LLM_PROVIDER_OPENAI:
        return "Set OPENAI_API_KEY."
    if provider == _LLM_PROVIDER_ANTHROPIC:
        return "Set ANTHROPIC_API_KEY."
    return "Set CUSTOM_LLM_API_KEY, or make the custom endpoint the default LLM_API_BASE with LLM_API_KEY."


def _require_llm_api_key_for_provider(provider: str) -> None:
    if _configured_llm_api_key_for_provider(provider):
        return
    raise HTTPException(
        status_code=400,
        detail=f"No API key is configured for provider '{provider}'. {_llm_key_hint(provider)}",
    )


def _llm_api_key_for_provider(provider: str) -> str:
    key = _configured_llm_api_key_for_provider(provider)
    if key:
        return key
    raise RuntimeError(f"No API key configured for provider {provider!r}")


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


def _clone_pr_head(
    token: str, owner: str, repo: str, number: int, *, depth: int = 50
) -> Optional[str]:
    """Shallow-clone the PR head ref (``pull/<n>/head``) into a temp dir
    so the LLM gets browse tools rooted at the PR's working tree. Works
    for forked PRs too — GitHub exposes the merge ref on the base repo.

    The installation token is passed via ``http.extraHeader`` rather than
    embedded in the remote URL — that way it never lands in process
    listings (``/proc/<pid>/cmdline``) or git's own error messages.

    Returns the local path, or None if anything went wrong. The caller
    is responsible for deleting the directory."""
    tmpdir = tempfile.mkdtemp(prefix=f"serge-{owner}-{repo}-{number}-")
    url = f"https://github.com/{owner}/{repo}.git"
    # Basic-auth with x-access-token as the username and the token as the
    # password is the documented form for GitHub App installation tokens.
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    auth_header = f"Authorization: Basic {basic}"

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        # Refuse interactive credential prompts (would otherwise hang on
        # auth failure) and ignore host-wide git config so a quirky
        # /etc/gitconfig can't influence behavior.
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        # The token-bearing header is only attached via -c; nothing on
        # disk references it.
        "GIT_ASKPASS": "/bin/false",
    }

    def _run(*args: str, timeout: int = 120, with_auth: bool = False) -> None:
        cmd = ["git", "-C", tmpdir]
        if with_auth:
            cmd += ["-c", f"http.extraHeader={auth_header}"]
        cmd += list(args)
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )

    try:
        subprocess.run(
            ["git", "init", "--quiet", tmpdir],
            check=True,
            capture_output=True,
            timeout=30,
            env=env,
        )
        _run("remote", "add", "origin", url)
        # core.symlinks=false forces symlinks in the PR tree to be written
        # as plain files, so helper tools rooted at the checkout cannot
        # follow them out of the repo.
        _run(
            "-c", "core.symlinks=false",
            "fetch",
            "--depth",
            str(depth),
            "origin",
            f"pull/{number}/head:pr",
            timeout=180,
            with_auth=True,
        )
        _run("-c", "core.symlinks=false", "checkout", "--quiet", "pr")
        return tmpdir
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # Defense in depth: scrub anything that looks like the token in
        # case git or its transport ever echoed it back. The Authorization
        # header itself is never on argv, so this is belt-and-braces.
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            stderr = stderr.replace(token, "<redacted-token>")
            stderr = stderr.replace(basic, "<redacted-basic>")
        log.warning(
            "git clone failed for %s/%s#%d (%s): %s",
            owner,
            repo,
            number,
            type(exc).__name__,
            stderr or exc,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None


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


def _run_review_worker(job: Job) -> None:
    """Runs in a background thread. Pulls an installation token for the
    target repo, shallow-clones the PR head so browse tools work, then
    calls prepare_review with a chunk_callback that streams events back
    to the SSE consumer."""
    clone_path: Optional[str] = None
    try:
        assert cfg.github_app_id and cfg.github_private_key
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
        req = ReviewRequest(
            owner=job.target_owner,
            repo=job.target_repo,
            number=job.target_number,
            trigger_comment_id=0,
            trigger_comment_body=job.trigger_comment,
            commenter=job.user,
        )

        # Shallow-clone so the LLM has browse tools (matches Action mode,
        # which gets a checkout via actions/checkout). If the clone fails
        # we still run the review — just without tools.
        worker_cfg = dataclasses.replace(
            cfg,
            llm_api_base=job.llm_api_base,
            llm_api_key=_llm_api_key_for_provider(job.llm_provider),
            llm_model=job.llm_model,
            llm_bill_to=_llm_bill_to_for_provider(job.llm_provider),
        )
        if not _bool_env_safe("WEB_DISABLE_CHECKOUT", False):
            _push_event(job, "step", "clone")
            _push_event(job, "log", "Cloning PR head…")
            t0 = time.monotonic()
            clone_path = _clone_pr_head(
                token, job.target_owner, job.target_repo, job.target_number
            )
            if clone_path:
                _push_event(
                    job,
                    "log",
                    f"Clone ready in {time.monotonic() - t0:.1f}s ({clone_path})",
                )
                worker_cfg = dataclasses.replace(worker_cfg, repo_checkout_path=clone_path)
            else:
                _push_event(
                    job,
                    "log",
                    "Clone failed; continuing without browse tools",
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
        job.status = "done"
        _push_event(
            job,
            "log",
            f"Draft ready: {len(draft.comments)} inline comment(s), event={draft.event}",
        )
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
        log.warning(
            "App not installed for %s/%s (job %s)",
            exc.owner,
            exc.repo,
            job.id,
        )
        job.status = "error"
        job.error = str(exc)
        _push_event(job, "step", "error")
        _push_event(job, "error", job.error)
        _push_event(job, "done", "")
    except LLMResponseError as exc:
        # Upstream chat-completions endpoint returned a non-OK status
        # (and either wasn't retryable, or retries didn't recover).
        # Surface the status code + a body excerpt directly to the SSE
        # client — without it the UI just shows "review crashed (see
        # server log)" and the user has to SSH into the box to figure
        # out whether it was a 429 (rate limit), a 400 (bad schema),
        # auth, etc. The body comes from the LLM provider's own error
        # response — no auth tokens of ours are echoed there.
        log.warning(
            "LLM endpoint returned %d for job %s: %s",
            exc.status_code,
            job.id,
            exc.body_preview[:400],
        )
        job.status = "error"
        excerpt = exc.body_preview.strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        reason_part = f" {exc.reason}" if exc.reason else ""
        job.error = (
            f"LLM endpoint returned {exc.status_code}{reason_part}: {excerpt}"
            if excerpt
            else f"LLM endpoint returned {exc.status_code}{reason_part}"
        )
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
        if clone_path:
            shutil.rmtree(clone_path, ignore_errors=True)
        # Snapshot the final state into SQLite. Every terminal branch
        # above sets job.status to a non-'running' value, so this also
        # clears the 'running' marker we'd otherwise reap on next restart.
        _persist_terminal(job)


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


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
def index(request: Request) -> Response:
    if not _current_user(request):
        return RedirectResponse("/login", status_code=302)
    return _serve_static("index.html")


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
    redirect_uri = cfg.github_oauth_callback_url or str(request.url_for("auth_callback"))
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

        orgs: list[str] = []
        if cfg.web_allowed_orgs:
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
                    if isinstance(o, dict)
                ]

    if not _user_is_allowed(login, orgs):
        log.warning("denied login attempt by %s (orgs=%s)", login, orgs)
        raise HTTPException(status_code=403, detail="user_not_allowed")

    sess["user"] = login
    response = RedirectResponse("/", status_code=302)
    _save_session(response, sess)
    log.info("user %s logged in", login)
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


@app.get("/llm-options")
def llm_options(request: Request) -> JSONResponse:
    _require_user(request)
    provider = _infer_llm_provider(cfg.llm_api_base)
    custom_base = cfg.llm_api_base if provider == _LLM_PROVIDER_CUSTOM else ""
    return JSONResponse(
        {
            "default_provider": provider,
            "default_model": cfg.llm_model or "",
            "custom_base_url": custom_base,
            "providers": [
                {
                    "id": _LLM_PROVIDER_HF,
                    "label": "HF",
                    "base_url": _LLM_PROVIDER_BASES[_LLM_PROVIDER_HF],
                },
                {
                    "id": _LLM_PROVIDER_OPENAI,
                    "label": "OpenAI",
                    "base_url": _LLM_PROVIDER_BASES[_LLM_PROVIDER_OPENAI],
                },
                {
                    "id": _LLM_PROVIDER_ANTHROPIC,
                    "label": "Anthropic",
                    "base_url": _LLM_PROVIDER_BASES[_LLM_PROVIDER_ANTHROPIC],
                    "default_model": _LLM_PROVIDER_DEFAULT_MODELS[
                        _LLM_PROVIDER_ANTHROPIC
                    ],
                },
                {
                    "id": _LLM_PROVIDER_CUSTOM,
                    "label": "Custom",
                    "base_url": custom_base,
                },
            ],
        }
    )


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
    llm_provider, llm_api_base, llm_model = _llm_selection_from_payload(payload)
    _require_llm_api_key_for_provider(llm_provider)

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
    return JSONResponse(
        {
            "id": job.id,
            "owner": owner,
            "repo": repo,
            "number": number,
            "url": f"/reviews/{owner}/{repo}/{number}/{job.id}",
        }
    )


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
    if job.user != user:
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
        return JSONResponse(
            {"status": job.status, "error": job.error, "draft": None}
        )
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
