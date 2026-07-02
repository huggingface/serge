import logging
import os
import shlex
import stat
import tempfile
from dataclasses import dataclass
from typing import Optional

from . import sandbox
from .sandbox import normalize_mode as normalize_sandbox_mode


log = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """Like int(os.environ[name]) with a default, but also treats an empty
    string as "use default" so unset GitHub Action secrets (which forward as
    "") don't blow up int parsing."""
    raw = (os.environ.get(name) or "").strip()
    return int(raw) if raw else default


def _load_private_key() -> Optional[str]:
    inline = os.environ.get("GITHUB_PRIVATE_KEY")
    if inline:
        return inline.replace("\\n", "\n")
    path = os.environ.get("GITHUB_PRIVATE_KEY_PATH")
    if not path:
        return None
    try:
        mode = os.stat(path).st_mode
        if stat.S_IMODE(mode) & 0o077:
            log.warning(
                "GITHUB_PRIVATE_KEY_PATH %s is group/world-readable "
                "(mode=%o); tighten permissions with `chmod 600 %s`",
                path,
                stat.S_IMODE(mode),
                path,
            )
    except OSError:
        pass
    with open(path, "r") as f:
        return f.read()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Config:
    # Only used in webhook mode (GitHub App). In Action mode the runner
    # provides GITHUB_TOKEN directly so these may be absent.
    github_app_id: Optional[str]
    github_private_key: Optional[str]
    github_webhook_secret: Optional[str]

    llm_api_base: str
    llm_api_key: str
    llm_model: Optional[str]
    llm_bill_to: Optional[str]
    llm_max_tokens: int
    llm_stream: bool

    mention_trigger: str
    review_event: str
    max_diff_chars: int
    review_rules_path: str
    helper_tools_path: str
    default_review_rules: str
    allow_approve: bool
    persona_header: str
    context_script_path: str
    context_script_timeout: int
    # Path to the checked-out PR head; when set, the LLM gets read-only
    # browse tools (read_file/list_dir/grep) rooted here. Empty disables
    # tool use entirely.
    repo_checkout_path: str
    tool_max_iterations: int
    # When true, tool_max_iterations caps total tool calls instead of only
    # blind tool turns. Used by /tasks to preserve final-answer budget.
    tool_max_iterations_strict: bool = False
    # Hard cap on cumulative *input* tokens consumed by LLM calls during a
    # single review (across all chunks and tool turns). When exceeded we
    # stop the agentic loop, ask the model for a final review with tools
    # off, and skip any remaining diff chunks. Set to 0 to disable.
    llm_max_input_tokens: int = 2_000_000

    # When true, published reviews carry a note that they came from a
    # non-production (staging) deployment. Set via the STAGING env var.
    is_staging: bool = False

    # Isolation policy for subprocesses that touch the PR tree (helper
    # tools, the .ai/context-script). One of "require" | "auto" | "off";
    # see reviewbot/sandbox.py and docs/security-architecture.md.
    # Production sets "require"; defaults to "auto" so unit tests and
    # local dev (no bubblewrap) run unsandboxed.
    helper_sandbox: str = sandbox.AUTO

    # Web-mode (reviewbot-web) settings. All optional in webhook/Action
    # modes; required only when require_web=True.
    github_oauth_client_id: Optional[str] = None
    github_oauth_client_secret: Optional[str] = None
    github_oauth_callback_url: Optional[str] = None
    web_session_secret: Optional[str] = None
    # Comma-separated lists. Either may be empty when DEV_NO_AUTH is on.
    web_allowed_users: tuple[str, ...] = ()
    web_allowed_orgs: tuple[str, ...] = ()
    # Admins (comma-separated GitHub logins) may view any user's review,
    # not just their own. Used to follow a shared review link that was
    # submitted by someone else via the web UI.
    web_admin_users: tuple[str, ...] = ()
    # SQLite file used by the web app to persist job metadata, drafts,
    # and structural event history. Default is relative to CWD so dev
    # works out of the box; deploy sets an absolute path.
    web_store_path: str = "jobs.db"
    # Global cap on persisted jobs. Older finished jobs are pruned;
    # running jobs are never pruned.
    web_job_retention: int = 25
    web_dev_no_auth: bool = False
    # Shared bare-clone + per-job worktree cache (see clone_cache.py).
    # Default lives under the system temp dir so dev works out of the box;
    # deploy points this at a dedicated EBS volume (e.g.
    # /var/lib/reviewbot/clones). TTL is how long an untouched bare repo
    # survives GC; depth is the shallow-fetch depth of the PR head.
    web_clone_cache_dir: str = ""
    web_clone_cache_ttl_seconds: int = 7 * 24 * 3600
    web_clone_depth: int = 50
    # Drop the Secure flag from the session cookie so plain-HTTP works
    # (typical for VPN-private deployments without TLS termination).
    # Independent of DEV_NO_AUTH: you can have mandatory auth without
    # HTTPS as long as the network path is trusted. Default off.
    web_insecure_cookies: bool = False
    # Optional ``reasoning_effort`` passed through on /v1/chat/completions.
    # Supported by some endpoints (OpenAI o-series, HF Router for the
    # Kimi-K2 thinking variants, etc.). Common values: "low", "medium",
    # "high". Leave empty to omit the parameter entirely.
    llm_reasoning_effort: Optional[str] = None
    # Public URL to install/configure the GitHub App that backs this
    # deployment. Surfaced on the /help page as the "Install the app"
    # link. Defaults to the Hugging Face Serge App; override per-deploy
    # via WEB_GITHUB_APP_URL.
    web_github_app_url: Optional[str] = "https://github.com/apps/sergereview"

    # --- Tasks flow (POST /tasks) -------------------------------------
    # serge's write-capable endpoint: a GitHub Actions job posts an
    # instruction + context (e.g. a failing-test report) and serge opens a
    # PR with a fix. Off by default — it requires the App to hold
    # Contents:write + Pull Requests:write and is a privilege escalation
    # over the read-only reviewer. Auth is GitHub Actions OIDC verified
    # against ``task_oidc_issuer``'s JWKS, authorized on the token's
    # ``repository`` claim; ``task_oidc_audience`` is the ``aud`` serge
    # requires the caller to mint the token with.
    task_api_enabled: bool = False
    task_oidc_issuer: str = "https://token.actions.githubusercontent.com"
    task_oidc_audience: str = "serge"
    # Optional task-only completion-token cap. When unset, tasks use the
    # normal llm_max_tokens value.
    task_llm_max_tokens: Optional[int] = None
    # Optional task-only input-token/tool caps. When unset, tasks use the
    # normal review values.
    task_llm_max_input_tokens: Optional[int] = None
    task_tool_max_iterations: Optional[int] = None
    # Cap on serge-authored commits per fix branch (follow-up loop guard).
    task_max_followups: int = 5

    # --- Task execution backend (per-task-pod model) ------------------
    # How the write-capable /tasks flow executes each task
    # (SERGE_PERTASK_POD_PLAN.md):
    #   "inprocess" (default) — run the agent loop in a serge thread pool
    #       (the legacy path);
    #   "docker"    — launch a per-task runner container
    #       (``reviewbot-task-runner``) that runs the whole loop + normalize
    #       and streams results back over the HTTP callback;
    #   "kubernetes"— launch a per-task runner Job (Phase 3).
    # The flag keeps the pod-per-task rollout reversible.
    task_execution: str = "inprocess"
    # Runner image for the docker/kubernetes backends (reviewbot layered on
    # the repo toolchain — see docker/Dockerfile.task-runner).
    task_runner_image: Optional[str] = None
    # Base URL the runner POSTs events + the terminal result back to
    # (``{base}/internal/tasks/{job_id}/events``). In k8s this is serge's
    # in-cluster Service URL; for docker-on-host it points at the serge host
    # (e.g. http://host.docker.internal:8000, or http://localhost:8000 with
    # ``--network host``).
    task_callback_base_url: Optional[str] = None
    # Wall-clock cap (seconds) on a single runner container/Job.
    task_runner_timeout: int = 3600
    # docker backend egress firewall: the network the runner attaches to (an
    # ``internal`` net in prod, or "host" for local e2e) and the allowlisting
    # forward proxy egress is routed through (see launcher.DockerLaunchOptions).
    task_runner_network: Optional[str] = None
    task_runner_proxy: Optional[str] = None
    # Hosts that bypass the egress proxy — in kubernetes this must include
    # serge's own callback host (the callback goes straight to the serge pod,
    # not through the allowlisting gateway). Comma-separated, as NO_PROXY.
    task_runner_no_proxy: Optional[str] = None
    task_runner_memory: Optional[str] = None

    # --- Post-LLM normalize hook --------------------------------------
    # After the LLM patch is applied to the worktree and before serge
    # commits, optionally run the target repo's own normalizer (e.g. ``make
    # style && make fix-repo``) in a sandbox and fold its edits into the same
    # commit, so the opened PR is already conformant to the repo's standards
    # (no red repo-consistency CI, no follow-up commit). Opt-in: when
    # ``task_normalize_command`` is unset, the hook is skipped entirely and
    # serge stays repo-agnostic.
    #
    # ``task_normalize_command`` is the argv to run (operator/repo config,
    # never request-supplied). ``task_normalize_image`` is the docker image
    # (the repo's toolchain baked in) used by the ``docker`` backend;
    # ``task_sandbox_backend`` selects bwrap | docker | kubernetes | auto. The
    # command always runs network-isolated.
    task_normalize_command: Optional[list[str]] = None
    task_normalize_image: Optional[str] = None
    task_normalize_timeout: int = 1800
    task_normalize_memory: Optional[str] = None
    # How many times the LLM may be asked to correct its patch when the
    # normalizer rejects it (or the patch fails to apply). 0 disables the
    # feedback loop (validate once, accept whatever the model produced). The
    # model gets up to ``task_normalize_max_retries + 1`` patch attempts.
    task_normalize_max_retries: int = 2
    # Optional free-text guidance injected into the task system prompt and the
    # normalize-failure feedback, alongside the normalize command. Use it to
    # encode policy the command itself can't express — e.g. "prefer root-cause
    # fixes over `# noqa`/`# type: ignore` suppressions", or repo-specific
    # conventions. Operator config, never request-supplied.
    task_normalize_guidance: Optional[str] = None
    task_sandbox_backend: str = sandbox.AUTO_BACKEND
    # Kubernetes normalize backend (TASK_SANDBOX_BACKEND=kubernetes). The Job
    # runs the normalizer on the worktree, which serge writes to a shared RWX
    # PVC. ``task_k8s_namespace`` defaults to the in-cluster namespace at
    # runtime; ``task_k8s_worktree_pvc`` is the claim the Job mounts;
    # ``task_k8s_worktree_volume_root`` is where that PVC is mounted in serge
    # (defaults to the clone-cache dir) — the worktree's path *relative* to it
    # becomes the Job's volume subPath, so the Job sees only its own worktree.
    task_k8s_namespace: Optional[str] = None
    task_k8s_worktree_pvc: Optional[str] = None
    task_k8s_worktree_volume_root: Optional[str] = None
    task_k8s_service_account: Optional[str] = None
    # nodeSelector for the normalize Job pods, as "key=value,key2=value2"
    # (e.g. "scheduling.cast.ai/node-template=default-by-castai").
    task_k8s_node_selector: Optional[str] = None
    # Optional Slack notification for PRs created by the /tasks flow.
    # Defaults to the org-level CI feedback Slack secrets; the transformers CI
    # names remain supported as fallbacks.
    slack_bot_token: Optional[str] = None
    slack_report_channel: Optional[str] = None

    @property
    def needs_isolated_checkout(self) -> bool:
        """Whether the /tasks checkout must be a self-contained git clone
        rather than a linked worktree.

        True only when an in-loop normalizer runs inside a container sandbox
        that binds *just* the worktree (``docker``/``kubernetes``, and
        ``auto`` which may resolve to docker), so in-sandbox git works. When
        normalize is unconfigured, or the dev-only ``bwrap`` backend is used,
        the cheaper linked worktree is kept (see
        :meth:`CloneCache.acquire_ref`)."""
        return bool(self.task_normalize_command) and self.task_sandbox_backend in (
            sandbox.DOCKER_BACKEND,
            sandbox.KUBERNETES_BACKEND,
            sandbox.AUTO_BACKEND,
        )

    @classmethod
    def from_env(
        cls,
        *,
        require_app: bool = True,
        require_web: bool = False,
    ) -> "Config":
        app_id = os.environ.get("GITHUB_APP_ID")
        private_key = _load_private_key()
        webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET")

        if require_app or require_web:
            # Web mode also publishes via the App, so it needs the App
            # credentials too — webhook secret is only required for the
            # inbound-events surface.
            required = [
                ("GITHUB_APP_ID", app_id),
                ("GITHUB_PRIVATE_KEY / GITHUB_PRIVATE_KEY_PATH", private_key),
            ]
            if require_app:
                required.append(("GITHUB_WEBHOOK_SECRET", webhook_secret))
            missing = [name for name, val in required if not val]
            if missing:
                mode = "webhook mode" if require_app else "web mode"
                raise RuntimeError(
                    f"Missing required env vars for {mode}: " + ", ".join(missing)
                )

        task_execution = (
            os.environ.get("TASK_EXECUTION") or "inprocess"
        ).strip().lower() or "inprocess"
        if task_execution not in ("inprocess", "docker", "kubernetes"):
            raise RuntimeError(
                "TASK_EXECUTION must be one of inprocess|docker|kubernetes, "
                f"got {task_execution!r}"
            )

        oauth_client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID") or None
        oauth_client_secret = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET") or None
        oauth_callback_url = os.environ.get("GITHUB_OAUTH_CALLBACK_URL") or None
        session_secret = os.environ.get("WEB_SESSION_SECRET") or None
        dev_no_auth = _bool_env("DEV_NO_AUTH", False)
        allowed_users = tuple(
            u.strip().lower()
            for u in (os.environ.get("WEB_ALLOWED_USERS") or "").split(",")
            if u.strip()
        )
        allowed_orgs = tuple(
            o.strip().lower()
            for o in (os.environ.get("WEB_ALLOWED_ORG") or "").split(",")
            if o.strip()
        )
        admin_users = tuple(
            u.strip().lower()
            for u in (os.environ.get("WEB_ADMIN_USERS") or "").split(",")
            if u.strip()
        )

        if require_web and not dev_no_auth:
            missing_web = [
                name
                for name, val in [
                    ("GITHUB_OAUTH_CLIENT_ID", oauth_client_id),
                    ("GITHUB_OAUTH_CLIENT_SECRET", oauth_client_secret),
                    ("WEB_SESSION_SECRET", session_secret),
                ]
                if not val
            ]
            if missing_web:
                raise RuntimeError(
                    "Missing required env vars for web mode "
                    "(set DEV_NO_AUTH=1 to bypass for local testing): "
                    + ", ".join(missing_web)
                )
            if not allowed_users and not allowed_orgs:
                raise RuntimeError(
                    "Web mode requires WEB_ALLOWED_USERS and/or WEB_ALLOWED_ORG "
                    "(comma-separated). Set DEV_NO_AUTH=1 to bypass for local testing."
                )

        # In web mode, per-repo API keys live in the DB (provider_configs)
        # so LLM_API_KEY is no longer required at startup. Action /
        # webhook modes still need it because there's no per-request
        # operator picking a config.
        if require_web:
            llm_api_key = os.environ.get("LLM_API_KEY", "")
        else:
            llm_api_key = os.environ["LLM_API_KEY"]

        return cls(
            github_app_id=app_id,
            github_private_key=private_key,
            github_webhook_secret=webhook_secret,
            llm_api_base=(
                os.environ.get("LLM_BASE_URL")
                or os.environ.get("LLM_API_BASE")
                or "https://api.openai.com/v1"
            ).rstrip("/"),
            llm_api_key=llm_api_key,
            llm_model=os.environ.get("LLM_MODEL") or None,
            llm_bill_to=os.environ.get("LLM_BILL_TO") or None,
            llm_max_tokens=_int_env("LLM_MAX_TOKENS", 4096),
            # Streaming on by default — the web UI's live token counter
            # and reasoning display rely on incremental SSE chunks. Set
            # LLM_STREAM=0 to fall back to the buffered REST path.
            llm_stream=_bool_env("LLM_STREAM", True),
            llm_reasoning_effort=(os.environ.get("LLM_REASONING_EFFORT") or "").strip()
            or None,
            mention_trigger=os.environ.get("MENTION_TRIGGER", "@askserge"),
            review_event=os.environ.get("REVIEW_EVENT", "COMMENT"),
            max_diff_chars=_int_env("MAX_DIFF_CHARS", 200000),
            review_rules_path=os.environ.get(
                "REVIEW_RULES_PATH", ".ai/review-rules.md"
            ),
            helper_tools_path=os.environ.get(
                "HELPER_TOOLS_PATH", ".ai/review-tools.json"
            ),
            default_review_rules=os.environ.get(
                "DEFAULT_REVIEW_RULES",
                "Apply general Python correctness and security standards.",
            ),
            allow_approve=_bool_env("ALLOW_APPROVE", False),
            persona_header=os.environ.get("PERSONA_HEADER", "🤗 **Serge** says:"),
            context_script_path=os.environ.get(
                "CONTEXT_SCRIPT_PATH", ".ai/context-script"
            ),
            context_script_timeout=_int_env("CONTEXT_SCRIPT_TIMEOUT", 30),
            repo_checkout_path=(os.environ.get("REPO_CHECKOUT_PATH") or "").strip(),
            helper_sandbox=normalize_sandbox_mode(os.environ.get("HELPER_SANDBOX")),
            # Set TOOL_MAX_ITERATIONS=0 to disable the cap entirely;
            # otherwise the agentic loop bails out after this many
            # blind tool-call turns and asks for a final answer with
            # tools off. The default is generous so that tool-heavy
            # investigations (browse + grep + helper linter) on large
            # PRs complete without being forced to truncate.
            tool_max_iterations=_int_env("TOOL_MAX_ITERATIONS", 30),
            llm_max_input_tokens=_int_env("LLM_MAX_INPUT_TOKENS", 2_000_000),
            is_staging=_bool_env("STAGING", False),
            github_oauth_client_id=oauth_client_id,
            github_oauth_client_secret=oauth_client_secret,
            github_oauth_callback_url=oauth_callback_url,
            web_session_secret=session_secret,
            web_allowed_users=allowed_users,
            web_allowed_orgs=allowed_orgs,
            web_admin_users=admin_users,
            web_store_path=(os.environ.get("WEB_STORE_PATH") or "jobs.db").strip()
            or "jobs.db",
            web_job_retention=_int_env("WEB_JOB_RETENTION", 25),
            web_dev_no_auth=dev_no_auth,
            web_insecure_cookies=_bool_env("WEB_INSECURE_COOKIES", False),
            web_clone_cache_dir=(os.environ.get("WEB_CLONE_CACHE_DIR") or "").strip()
            or os.path.join(tempfile.gettempdir(), "reviewbot-clones"),
            web_clone_cache_ttl_seconds=_int_env(
                "WEB_CLONE_CACHE_TTL_SECONDS", 7 * 24 * 3600
            ),
            web_clone_depth=_int_env("WEB_CLONE_DEPTH", 50),
            web_github_app_url=(os.environ.get("WEB_GITHUB_APP_URL") or "").strip()
            or "https://github.com/apps/sergereview",
            task_api_enabled=_bool_env("TASK_API_ENABLED", False),
            task_oidc_issuer=(os.environ.get("TASK_OIDC_ISSUER") or "").strip()
            or "https://token.actions.githubusercontent.com",
            task_oidc_audience=(os.environ.get("TASK_OIDC_AUDIENCE") or "").strip()
            or "serge",
            task_llm_max_tokens=(_int_env("TASK_LLM_MAX_TOKENS", 0) or None),
            task_llm_max_input_tokens=(
                _int_env("TASK_LLM_MAX_INPUT_TOKENS", 0) or None
            ),
            task_tool_max_iterations=(_int_env("TASK_TOOL_MAX_ITERATIONS", 0) or None),
            task_max_followups=_int_env("TASK_MAX_FOLLOWUPS", 5),
            task_execution=task_execution,
            task_runner_image=(os.environ.get("TASK_RUNNER_IMAGE") or "").strip()
            or None,
            task_callback_base_url=(os.environ.get("TASK_CALLBACK_BASE_URL") or "")
            .strip()
            .rstrip("/")
            or None,
            task_runner_timeout=_int_env("TASK_RUNNER_TIMEOUT", 3600),
            task_runner_network=(os.environ.get("TASK_RUNNER_NETWORK") or "").strip()
            or None,
            task_runner_proxy=(os.environ.get("TASK_RUNNER_PROXY") or "").strip()
            or None,
            task_runner_no_proxy=(os.environ.get("TASK_RUNNER_NO_PROXY") or "").strip()
            or None,
            task_runner_memory=(os.environ.get("TASK_RUNNER_MEMORY") or "").strip()
            or None,
            task_normalize_command=(
                shlex.split(os.environ.get("TASK_NORMALIZE_COMMAND") or "") or None
            ),
            task_normalize_image=(os.environ.get("TASK_NORMALIZE_IMAGE") or "").strip()
            or None,
            task_normalize_timeout=_int_env("TASK_NORMALIZE_TIMEOUT", 1800),
            task_normalize_memory=(
                os.environ.get("TASK_NORMALIZE_MEMORY") or ""
            ).strip()
            or None,
            task_normalize_max_retries=_int_env("TASK_NORMALIZE_MAX_RETRIES", 2),
            task_normalize_guidance=(
                os.environ.get("TASK_NORMALIZE_GUIDANCE") or ""
            ).strip()
            or None,
            task_sandbox_backend=sandbox.normalize_backend(
                os.environ.get("TASK_SANDBOX_BACKEND")
            ),
            task_k8s_namespace=(os.environ.get("TASK_K8S_NAMESPACE") or "").strip()
            or None,
            task_k8s_worktree_pvc=(
                os.environ.get("TASK_K8S_WORKTREE_PVC") or ""
            ).strip()
            or None,
            task_k8s_worktree_volume_root=(
                os.environ.get("TASK_K8S_WORKTREE_VOLUME_ROOT") or ""
            ).strip()
            or None,
            task_k8s_service_account=(
                os.environ.get("TASK_K8S_SERVICE_ACCOUNT") or ""
            ).strip()
            or None,
            task_k8s_node_selector=(
                os.environ.get("TASK_K8S_NODE_SELECTOR") or ""
            ).strip()
            or None,
            slack_bot_token=(
                os.environ.get("SLACK_CIFEEDBACK_BOT_TOKEN")
                or os.environ.get("CI_SLACK_BOT_TOKEN")
                or ""
            ).strip()
            or None,
            slack_report_channel=(
                os.environ.get("SLACK_CIFEEDBACK_CHANNEL")
                or os.environ.get("SLACK_REPORT_CHANNEL")
                or ""
            ).strip()
            or None,
        )
