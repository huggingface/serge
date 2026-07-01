"""The write-capable /tasks flow.

A GitHub Actions job posts ``{instruction, context, output}`` and serge
produces a contribution to the repo: a new PR (``new_pr``) or a follow-up
commit on an existing serge-authored fix branch (``existing_pr``).

serge stays a **stateless patch producer**: the LLM only proposes a unified
diff (plus a PR title/body). serge applies the patch in a network-isolated
worktree, then uploads the result through the GitHub Git Data API
(``create_blob`` → ``create_tree`` → ``create_commit`` → ``create_ref`` →
``create_pull_request``). The installation token never enters the sandbox
or a git remote. Verification of the fix is done by the caller's CI, not by
serge — serge never runs the test suite.

See ``TASKS_FLOW_PLAN.md`` for the full design.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from . import __version__
from .clone_cache import Checkout, CloneCache, FileChange
from .compression import MessageCompressor
from .config import Config
from .github_client import SERGE_GIT_EMAIL, GitHubClient
from .llm_client import ChatCompletionClient
from .normalize import NormalizeError, run_normalize
from .prompts import build_task_system_prompt, build_task_user_prompt
from .reviewer import (
    _extract_json,
    _format_aggregated_metrics,
    _make_tool_env,
    _run_agentic_loop,
    _UnparseableLLMOutput,
)
from .slack_tool import post_task_pr_created_notification

log = logging.getLogger(__name__)

# Serge only ever writes inside its own branch namespace. ``existing_pr``
# mode is rejected for any head branch outside it, so the OIDC
# ``repository`` claim cannot be leveraged to push to an arbitrary PR.
SERGE_BRANCH_NAMESPACE = "serge/"

_TASK_FORCE_FINAL_MESSAGE = (
    "You have used the available investigation budget. Based only on the "
    "evidence already gathered, produce the final task result immediately. "
    "Do not continue investigating or explain your reasoning. Reply with a "
    "single compact JSON object that starts with `{` and has EXACTLY these keys:\n"
    '  - "title": a concise PR title\n'
    '  - "body": a markdown PR description in at most 12 lines explaining the '
    "failure, root cause, and patch; if no safe fix is possible, explain why\n"
    '  - "patch": a valid unified diff, or an empty string if no safe fix is '
    "possible\n"
    "Return JSON only: no surrounding prose, no code fences, no extra commentary, "
    "and no tool requests."
)
_BRANCH_PREFIX_RE = re.compile(r"^serge/[A-Za-z0-9._/-]+$")
_CANDIDATE_HEADING_RE = re.compile(
    r"(?m)^## Serge candidate failure group \d+/\d+: .+$"
)

VALID_MODES = ("new_pr", "existing_pr")


class TaskError(Exception):
    """A task-level failure (bad request, guard violation, loop cap). The
    message is safe to surface to the caller."""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TaskRequest:
    owner: str
    repo: str
    base_ref: str
    instruction: str
    context: str
    mode: str = "new_pr"
    pr_number: Optional[int] = None
    title: Optional[str] = None
    branch_prefix: str = "serge/fix"
    slack_channel: Optional[str] = None
    slack_notify_pr_created: bool = True
    slack_notify_task_finished: bool = False
    # Resolved during processing (existing_pr): the PR's head branch.
    head_branch: Optional[str] = None

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


@dataclass
class TaskPlan:
    """The LLM's proposed contribution, before serge writes anything."""

    title: str
    body: str
    patch: str
    metrics_line: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: Optional[str] = None
    # True when the patch was validated in-loop (see :func:`_validate_patch`):
    # the worktree already holds the applied + normalized result, so
    # :func:`publish_task` commits it directly instead of re-applying.
    worktree_prepared: bool = False


@dataclass
class TaskResult:
    """The outcome of publishing a task."""

    mode: str
    no_change: bool = False
    message: str = ""
    pr_number: Optional[int] = None
    branch: Optional[str] = None
    url: Optional[str] = None
    commit_sha: Optional[str] = None
    changed_files: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "no_change": self.no_change,
            "message": self.message,
            "pr_number": self.pr_number,
            "branch": self.branch,
            "url": self.url,
            "commit_sha": self.commit_sha,
            "changed_files": self.changed_files,
        }


def task_candidate_requests(req: TaskRequest) -> list[TaskRequest]:
    """Return one request per retryable task candidate.

    The integration triage workflow can send several ordered failure groups in
    a single task context. Each candidate gets an independent LLM cycle while
    preserving the shared preamble before the first candidate heading.
    """
    matches = list(_CANDIDATE_HEADING_RE.finditer(req.context))
    if len(matches) < 2:
        return [req]

    preamble = req.context[: matches[0].start()].strip()
    candidates: list[TaskRequest] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(req.context)
        chunk = req.context[match.start() : end].strip()
        context = f"{preamble}\n\n{chunk}".strip() if preamble else chunk
        candidates.append(dataclasses.replace(req, context=context))
    return candidates


# ---------------------------------------------------------------------------
# Request building / validation
# ---------------------------------------------------------------------------
def build_task_request(
    payload: dict[str, Any],
    *,
    owner: str,
    repo: str,
) -> TaskRequest:
    """Validate an inbound /tasks payload into a TaskRequest.

    ``owner``/``repo`` come from the verified OIDC ``repository`` claim and
    are authoritative; any ``repo`` in the body must match (checked by the
    caller). Raises :class:`TaskError` on malformed input."""
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        raise TaskError("instruction is required")
    context = payload.get("context")
    if context is not None and not isinstance(context, str):
        raise TaskError("context must be a string")
    base_ref = (payload.get("base_ref") or "main").strip() or "main"

    output = payload.get("output") or {}
    if not isinstance(output, dict):
        raise TaskError("output must be an object")
    mode = (output.get("mode") or "new_pr").strip()
    if mode not in VALID_MODES:
        raise TaskError(f"output.mode must be one of {VALID_MODES}")

    title = output.get("title")
    if title is not None:
        title = str(title).strip() or None

    branch_prefix = (output.get("branch_prefix") or "serge/fix").strip()
    if not _BRANCH_PREFIX_RE.match(branch_prefix):
        raise TaskError(
            "output.branch_prefix must live in the 'serge/' namespace "
            "(e.g. 'serge/fix')"
        )

    notifications = payload.get("notifications") or {}
    if not isinstance(notifications, dict):
        raise TaskError("notifications must be an object")
    slack_channel = notifications.get("slack_channel")
    if slack_channel is not None:
        slack_channel = str(slack_channel).strip() or None
        if slack_channel and ("\n" in slack_channel or "\r" in slack_channel):
            raise TaskError("notifications.slack_channel must be a single line")
    slack_notify_pr_created = _notification_bool(
        notifications, "pr_created", default=True
    )
    slack_notify_task_finished = _notification_bool(
        notifications, "task_finished", default=False
    )

    pr_number: Optional[int] = None
    if mode == "existing_pr":
        raw = output.get("pr_number")
        if not isinstance(raw, int) or raw < 1:
            raise TaskError("output.pr_number is required for existing_pr mode")
        pr_number = raw

    return TaskRequest(
        owner=owner,
        repo=repo,
        base_ref=base_ref,
        instruction=instruction,
        context=context or "",
        mode=mode,
        pr_number=pr_number,
        title=title,
        branch_prefix=branch_prefix,
        slack_channel=slack_channel,
        slack_notify_pr_created=slack_notify_pr_created,
        slack_notify_task_finished=slack_notify_task_finished,
    )


def _notification_bool(
    notifications: dict[str, Any], name: str, *, default: bool
) -> bool:
    raw = notifications.get(name)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise TaskError(f"notifications.{name} must be a boolean")


def resolve_existing_pr(gh: GitHubClient, req: TaskRequest, cfg: Config) -> str:
    """For existing_pr mode: look up the PR, enforce the branch-ownership
    guard (head must be a serge-owned branch) and the follow-up loop cap.
    Returns the head branch name. Mutates ``req`` to set base_ref/head_branch.
    Raises :class:`TaskError` on a violation."""
    assert req.pr_number is not None
    pr = gh.get_pr(req.owner, req.repo, req.pr_number)
    head_branch = (pr.get("head") or {}).get("ref") or ""
    if not head_branch.startswith(SERGE_BRANCH_NAMESPACE):
        raise TaskError(
            f"existing_pr mode only targets serge-owned branches "
            f"('{SERGE_BRANCH_NAMESPACE}*'); PR #{req.pr_number} head is "
            f"'{head_branch}'",
            status_code=403,
        )
    base_ref = (pr.get("base") or {}).get("ref") or req.base_ref
    req.base_ref = base_ref
    req.head_branch = head_branch

    if cfg.task_max_followups > 0:
        try:
            existing = gh.count_branch_commits_by_author(
                req.owner, req.repo, head_branch, author_email=SERGE_GIT_EMAIL
            )
        except Exception:  # noqa: BLE001
            log.warning("could not count commits on %s; skipping loop cap", head_branch)
            existing = 0
        if existing >= cfg.task_max_followups:
            raise TaskError(
                f"follow-up loop cap reached: {existing} serge commit(s) on "
                f"'{head_branch}' (max {cfg.task_max_followups})",
                status_code=429,
            )
    return head_branch


# ---------------------------------------------------------------------------
# Agentic loop → patch (with in-loop normalize validation)
# ---------------------------------------------------------------------------
def _validate_patch(
    cfg: Config,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    content: Optional[str],
    emit: Callable[[str, str], None],
) -> tuple[Optional[str], bool]:
    """Validate the model's final answer by applying its patch to a clean
    worktree and running the repo normalizer.

    Returns ``(feedback, prepared)``:

    - ``feedback`` is a non-empty string when the patch should be sent back to
      the model for correction (it didn't apply, or the normalizer rejected
      it); ``None`` when the answer is accepted. The worktree is reset to a
      clean checkout before returning feedback.
    - ``prepared`` is True when the worktree now holds the applied (and, when
      the normalizer ran cleanly, normalized) result, ready for
      :func:`publish_task` to commit directly.

    Only called when ``cfg.task_normalize_command`` is set."""
    command = cfg.task_normalize_command
    assert command is not None

    try:
        result = _extract_json(content)
    except ValueError:
        # Unparseable — not something the normalizer can speak to. Accept here
        # and let prepare_task's own extraction raise the proper error.
        return None, False

    patch = result.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        # No patch to validate (a "no safe fix" answer); accept as-is.
        return None, False

    clone_cache.reset_worktree(checkout)
    try:
        clone_cache.apply_patch(checkout, patch)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:1200]
        return (
            "Your patch was rejected — `git apply` could not apply it to a "
            f"clean checkout:\n\n{stderr}\n\nReturn a corrected unified diff. "
            "Check the file paths and that the hunk context lines match the "
            "current code exactly.",
            False,
        )

    emit("step", "normalize")
    emit("log", f"Validating the patch with `{' '.join(command)}`…")
    try:
        returncode, tail = run_normalize(
            command,
            workdir=checkout.path,
            write_root=checkout.path,
            backend=cfg.task_sandbox_backend,
            image=cfg.task_normalize_image,
            mode=cfg.helper_sandbox,
            timeout=cfg.task_normalize_timeout,
            memory=cfg.task_normalize_memory,
            # k8s wiring is passed as plain values; run_normalize only uses it
            # (and imports the optional k8s_sandbox/kubernetes client) when the
            # kubernetes backend is selected, so non-k8s deploys never touch it.
            k8s_namespace=cfg.task_k8s_namespace,
            k8s_worktree_pvc=cfg.task_k8s_worktree_pvc,
            k8s_worktree_volume_root=(
                cfg.task_k8s_worktree_volume_root or cfg.web_clone_cache_dir or None
            ),
            k8s_service_account=cfg.task_k8s_service_account,
            k8s_node_selector=cfg.task_k8s_node_selector,
        )
    except NormalizeError as exc:
        # Infrastructure problem (sandbox unavailable, timeout) — not the
        # model's fault. Accept the applied patch best-effort rather than
        # blaming the LLM; CI still catches anything the normalizer would have.
        log.warning("normalizer unavailable during validation: %s", exc)
        emit(
            "log", f"Normalizer unavailable ({exc}); accepting the patch un-normalized."
        )
        return None, True

    if returncode != 0:
        clone_cache.reset_worktree(checkout)
        cmd = " ".join(command)
        msg = (
            f"Your patch applied cleanly, but the repository's normalizer "
            f"(`{cmd}`) then failed (exit {returncode}):\n\n{tail}\n\n"
            "Revise the patch so the normalizer passes. Fix the ROOT CAUSE — "
            "suppress a check (`# noqa`, `# type: ignore`, disabling a rule) "
            "only as a last resort, for a deliberate and justified exception, "
            "and explain why in a comment. Common causes: editing an "
            "auto-generated file instead of its modular/source counterpart, "
            "leaving a copied block out of sync, or a lint/format issue the "
            "fixer cannot resolve on its own."
        )
        if cfg.task_normalize_guidance:
            msg += f"\n\n{cfg.task_normalize_guidance.strip()}"
        return msg, False

    emit("log", "Patch validated; normalizer is clean.")
    return None, True


def _read_repo_conventions(cfg: Config, checkout: Checkout) -> str:
    """Read the repo's own conventions file (``cfg.review_rules_path``, e.g.
    ``.ai/review-rules.md``) from the task worktree, falling back to the
    deployment default.

    Safe to read straight from the worktree: a task checks out the repo's own
    trusted branch (base or a serge fix branch), not an untrusted fork PR head,
    so there's no need for the default-branch overlay the review flow uses."""
    rel = (cfg.review_rules_path or "").strip()
    if rel:
        try:
            with open(os.path.join(checkout.path, rel), encoding="utf-8") as fh:
                content = fh.read().strip()
        except OSError:
            content = ""
        if content:
            return content
    return cfg.default_review_rules


def prepare_task(
    cfg: Config,
    req: TaskRequest,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    existing_diff: Optional[str] = None,
    chunk_callback: Optional[Callable[[str, str], None]] = None,
) -> TaskPlan:
    """Run the agentic loop (read-only browse tools rooted at the checkout)
    and return the LLM's proposed patch + PR meta. ``cfg.repo_checkout_path``
    must already point at ``checkout``.

    When ``cfg.task_normalize_command`` is set, the loop also runs an in-loop
    verification gate (see :func:`_validate_patch`): each final patch is
    applied to the worktree and the repo normalizer is run; a failure is fed
    back to the model (up to ``cfg.task_normalize_max_retries`` times) so it can
    correct the patch. On success the worktree holds the applied + normalized
    result and the returned plan has ``worktree_prepared=True``."""

    def _emit(kind: str, text: str) -> None:
        if chunk_callback is not None:
            try:
                chunk_callback(kind, text)
            except Exception:
                log.debug("chunk_callback raised; suppressing", exc_info=True)

    _emit("log", f"Preparing task for {req.repo_full_name} (base={req.base_ref})")
    tool_env = _make_tool_env(cfg, helper_tools=[])

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
        compressor=MessageCompressor.from_env(),
    )
    system_prompt = build_task_system_prompt(
        _read_repo_conventions(cfg, checkout),
        cfg.task_normalize_guidance,
        tools_enabled=tool_env is not None,
    )
    user_prompt = build_task_user_prompt(
        repo_full_name=req.repo_full_name,
        base_ref=req.base_ref,
        instruction=req.instruction,
        context=req.context,
        existing_diff=existing_diff,
    )

    # Wire the normalize verification gate into the loop when configured. The
    # closure records whether the accepted answer left the worktree prepared.
    normalize_configured = bool(cfg.task_normalize_command)
    outcome = {"prepared": False}

    def _validate(chat) -> Optional[str]:
        feedback, prepared = _validate_patch(
            cfg,
            checkout=checkout,
            clone_cache=clone_cache,
            content=chat.content,
            emit=_emit,
        )
        outcome["prepared"] = prepared
        return feedback

    _emit("step", "llm")
    _emit("log", "Calling LLM to produce a patch…")
    chat, metrics = _run_agentic_loop(
        llm,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        cfg=cfg,
        tool_env=tool_env,
        emit=_emit,
        final_force_message=_TASK_FORCE_FINAL_MESSAGE,
        validate=_validate if normalize_configured else None,
        max_validation_retries=cfg.task_normalize_max_retries,
    )
    metrics_line = _format_aggregated_metrics(metrics)
    _emit("log", f"LLM done: {metrics_line}")

    try:
        result = _extract_json(chat.content)
    except ValueError as exc:
        raise _UnparseableLLMOutput(
            content=chat.content or "",
            finish_reason=chat.finish_reason,
            metrics_line=metrics_line,
        ) from exc

    title = (result.get("title") or "").strip() or (req.title or "serge: automated fix")
    body = (result.get("body") or "").strip()
    patch = result.get("patch")
    if not isinstance(patch, str):
        patch = ""

    # If validation never accepted a prepared worktree (retries exhausted, or
    # normalize not configured), make sure the worktree is clean so
    # publish_task's own apply path starts from a pristine checkout.
    if normalize_configured and not outcome["prepared"]:
        clone_cache.reset_worktree(checkout)

    return TaskPlan(
        title=req.title or title,
        body=body,
        patch=patch,
        metrics_line=metrics_line,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        model=llm.model,
        worktree_prepared=outcome["prepared"],
    )


# ---------------------------------------------------------------------------
# Publish: apply patch in the worktree, commit via Git Data API
# ---------------------------------------------------------------------------
def _tree_entries(
    gh: GitHubClient, owner: str, repo: str, changes
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for ch in changes:
        if ch.status == "D":
            entries.append(
                {"path": ch.path, "mode": ch.mode, "type": "blob", "sha": None}
            )
        else:
            blob_sha = gh.create_blob(owner, repo, ch.content or b"")
            entries.append(
                {"path": ch.path, "mode": ch.mode, "type": "blob", "sha": blob_sha}
            )
    return entries


def _failure_blocks(context: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in context.splitlines():
        if line.startswith("- `"):
            if current:
                blocks.append(current)
            current = [line]
        elif current and (line.startswith("  - ") or not line.strip()):
            if line.strip():
                current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _selected_failure_context(req: TaskRequest, plan: TaskPlan) -> str:
    heading = ""
    for line in req.context.splitlines():
        if _CANDIDATE_HEADING_RE.match(line):
            heading = line.removeprefix("## Serge candidate failure group ").strip()
            break

    blocks = _failure_blocks(req.context)
    if not blocks:
        return ""

    haystack = f"{plan.title}\n{plan.body}\n{plan.patch}".lower()

    def _score(block: list[str]) -> int:
        words = set(re.findall(r"[a-z0-9_]{5,}", "\n".join(block).lower()))
        return sum(1 for word in words if word in haystack)

    selected = max(blocks, key=_score)
    lines = ["## Original CI failure", ""]
    if heading:
        lines.append(f"- Failure group: `{heading}`")
    lines.extend(selected)
    return "\n".join(lines)


def _decorate_body(cfg: Config, plan: TaskPlan, req: TaskRequest) -> str:
    body = plan.body or "Automated fix produced by serge."
    failure_context = _selected_failure_context(req, plan)
    if failure_context:
        body = f"{failure_context}\n\n{body}"
    body += (
        "\n\n---\n_This change was produced automatically by serge from a "
        "CI failure report. The patch was generated by an LLM and applied by "
        "serge; review before merging._"
    )
    if cfg.is_staging:
        body += "\n\n_Note: produced by a staging deployment._"
    footer = [f"serge `v{__version__}`"]
    if plan.model:
        footer.append(f"model: `{plan.model}`")
    if plan.metrics_line:
        footer.append(plan.metrics_line)
    if footer:
        body += f"\n\n_{' · '.join(footer)}_"
    return body


def _commit_changes(
    cfg: Config,
    gh: GitHubClient,
    req: TaskRequest,
    *,
    changes: list[FileChange],
    title: str,
    body: str,
    job_id: str,
    emit_fn: Callable[[str, str], None],
) -> TaskResult:
    """Commit a set of worktree changes via the Git Data API and open/update
    a PR. ``changes`` must be non-empty. Never pushes to a non-serge branch."""
    changed_files = [c.path for c in changes]
    emit_fn(
        "log",
        f"Change touches {len(changed_files)} file(s): {', '.join(changed_files)}",
    )

    owner, repo = req.owner, req.repo
    emit_fn("step", "commit")
    entries = _tree_entries(gh, owner, repo, changes)

    if req.mode == "existing_pr":
        head_branch = req.head_branch
        assert head_branch and head_branch.startswith(SERGE_BRANCH_NAMESPACE)
        parent_sha = gh.get_ref_sha(owner, repo, f"heads/{head_branch}")
        base_tree = gh.get_commit_tree_sha(owner, repo, parent_sha)
        tree_sha = gh.create_tree(owner, repo, base_tree, entries)
        commit_sha = gh.create_commit(
            owner,
            repo,
            message=title,
            tree_sha=tree_sha,
            parents=[parent_sha],
        )
        gh.update_ref(owner, repo, f"heads/{head_branch}", commit_sha)
        emit_fn("log", f"Pushed commit {commit_sha[:8]} to {head_branch}")
        return TaskResult(
            mode=req.mode,
            pr_number=req.pr_number,
            branch=head_branch,
            commit_sha=commit_sha,
            changed_files=changed_files,
            message=f"Pushed follow-up commit to PR #{req.pr_number}.",
            url=f"https://github.com/{owner}/{repo}/pull/{req.pr_number}",
        )

    # new_pr
    branch = f"{req.branch_prefix}-{job_id[:8]}"
    parent_sha = gh.get_ref_sha(owner, repo, f"heads/{req.base_ref}")
    base_tree = gh.get_commit_tree_sha(owner, repo, parent_sha)
    tree_sha = gh.create_tree(owner, repo, base_tree, entries)
    commit_sha = gh.create_commit(
        owner,
        repo,
        message=title,
        tree_sha=tree_sha,
        parents=[parent_sha],
    )
    gh.create_ref(owner, repo, f"refs/heads/{branch}", commit_sha)
    emit_fn("log", f"Created branch {branch} at {commit_sha[:8]}")
    # Open as a draft, then immediately mark ready-for-review. The
    # draft->ready transition is what fires the `ready_for_review` webhook that
    # reviewer-assignment workflows (e.g. transformers' assign-reviewers.yml)
    # listen for; a PR born non-draft never emits that event and gets no
    # reviewers routed to it.
    pr = gh.create_pull_request(
        owner,
        repo,
        title=title,
        head=branch,
        base=req.base_ref,
        body=body,
        draft=True,
    )
    gh.mark_pull_request_ready(pr["node_id"])
    emit_fn(
        "log", f"Opened PR #{pr.get('number')} (draft->ready): {pr.get('html_url')}"
    )
    if req.slack_notify_pr_created:
        post_task_pr_created_notification(
            token=cfg.slack_bot_token,
            channel=req.slack_channel or cfg.slack_report_channel,
            repo_full_name=req.repo_full_name,
            pr_number=pr.get("number"),
            pr_url=pr.get("html_url"),
            title=title,
            branch=branch,
            changed_files=changed_files,
        )
    return TaskResult(
        mode=req.mode,
        pr_number=pr.get("number"),
        branch=branch,
        commit_sha=commit_sha,
        changed_files=changed_files,
        message=f"Opened PR #{pr.get('number')}.",
        url=pr.get("html_url"),
    )


def publish_task(
    cfg: Config,
    gh: GitHubClient,
    req: TaskRequest,
    plan: TaskPlan,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    job_id: str,
    emit: Optional[Callable[[str, str], None]] = None,
) -> TaskResult:
    """Commit the task's change via the Git Data API, opening a new PR
    (new_pr) or pushing onto the serge fix branch (existing_pr). Never pushes
    to a non-serge branch.

    When ``plan.worktree_prepared`` is set, the in-loop validation
    (:func:`_validate_patch`) already applied + normalized the worktree, so we
    just stage and commit it. Otherwise we apply ``plan.patch`` here (the path
    taken when no normalizer is configured, or when validation was abandoned
    and left a clean checkout)."""

    def _emit(kind: str, text: str) -> None:
        if emit is not None:
            emit(kind, text)

    if not plan.worktree_prepared and not plan.patch.strip():
        _emit("log", "LLM proposed no patch; nothing to commit")
        return TaskResult(
            mode=req.mode,
            no_change=True,
            message=plan.body or "No fix was proposed.",
        )

    if plan.worktree_prepared:
        _emit("log", "Committing the validated, normalized worktree.")
    else:
        _emit("step", "apply")
        try:
            clone_cache.apply_patch(checkout, plan.patch)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:800]
            raise TaskError(f"patch did not apply cleanly: {stderr}", status_code=422)

    clone_cache.stage_all(checkout)
    changes = clone_cache.collect_changes(checkout)
    if not changes:
        return TaskResult(
            mode=req.mode,
            no_change=True,
            message="Patch applied but produced no file changes.",
        )

    return _commit_changes(
        cfg,
        gh,
        req,
        changes=changes,
        title=plan.title,
        body=_decorate_body(cfg, plan, req),
        job_id=job_id,
        emit_fn=_emit,
    )
