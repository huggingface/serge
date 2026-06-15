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

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .clone_cache import Checkout, CloneCache
from .compression import MessageCompressor
from .config import Config
from .github_client import SERGE_GIT_EMAIL, GitHubClient
from .llm_client import ChatCompletionClient
from .prompts import build_task_system_prompt, build_task_user_prompt
from .reviewer import (
    _extract_json,
    _format_aggregated_metrics,
    _make_tool_env,
    _run_agentic_loop,
    _UnparseableLLMOutput,
)

log = logging.getLogger(__name__)

# Serge only ever writes inside its own branch namespace. ``existing_pr``
# mode is rejected for any head branch outside it, so the OIDC
# ``repository`` claim cannot be leveraged to push to an arbitrary PR.
SERGE_BRANCH_NAMESPACE = "serge/"
_BRANCH_PREFIX_RE = re.compile(r"^serge/[A-Za-z0-9._/-]+$")

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


# ---------------------------------------------------------------------------
# Request building / validation
# ---------------------------------------------------------------------------
def build_task_request(
    payload: dict[str, Any], *, owner: str, repo: str
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
    )


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
# Agentic loop → patch
# ---------------------------------------------------------------------------
def prepare_task(
    cfg: Config,
    req: TaskRequest,
    *,
    existing_diff: Optional[str] = None,
    chunk_callback: Optional[Callable[[str, str], None]] = None,
) -> TaskPlan:
    """Run the agentic loop (read-only browse tools rooted at the checkout)
    and return the LLM's proposed patch + PR meta. ``cfg.repo_checkout_path``
    must already point at the worktree."""

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
    system_prompt = build_task_system_prompt(tools_enabled=tool_env is not None)
    user_prompt = build_task_user_prompt(
        repo_full_name=req.repo_full_name,
        base_ref=req.base_ref,
        instruction=req.instruction,
        context=req.context,
        existing_diff=existing_diff,
    )

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

    return TaskPlan(
        title=req.title or title,
        body=body,
        patch=patch,
        metrics_line=metrics_line,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        model=llm.model,
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


def _decorate_body(cfg: Config, plan: TaskPlan) -> str:
    body = plan.body or "Automated fix produced by serge."
    body += (
        "\n\n---\n_This change was produced automatically by serge from a "
        "CI failure report. The patch was generated by an LLM and applied by "
        "serge; review before merging._"
    )
    if cfg.is_staging:
        body += "\n\n_Note: produced by a staging deployment._"
    footer = []
    if plan.model:
        footer.append(f"model: `{plan.model}`")
    if plan.metrics_line:
        footer.append(plan.metrics_line)
    if footer:
        body += f"\n\n_{' · '.join(footer)}_"
    return body


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
    """Apply the patch in the worktree and commit it via the Git Data API,
    opening a new PR (new_pr) or pushing onto the serge fix branch
    (existing_pr). Never pushes to a non-serge branch."""

    def _emit(kind: str, text: str) -> None:
        if emit is not None:
            emit(kind, text)

    if not plan.patch.strip():
        _emit("log", "LLM proposed no patch; nothing to commit")
        return TaskResult(
            mode=req.mode,
            no_change=True,
            message=plan.body or "No fix was proposed.",
        )

    _emit("step", "apply")
    try:
        clone_cache.apply_patch(checkout, plan.patch)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[:800]
        raise TaskError(f"patch did not apply cleanly: {stderr}", status_code=422)
    changes = clone_cache.collect_changes(checkout)
    if not changes:
        return TaskResult(
            mode=req.mode,
            no_change=True,
            message="Patch applied but produced no file changes.",
        )
    changed_files = [c.path for c in changes]
    _emit(
        "log", f"Patch touches {len(changed_files)} file(s): {', '.join(changed_files)}"
    )

    owner, repo = req.owner, req.repo
    _emit("step", "commit")
    entries = _tree_entries(gh, owner, repo, changes)
    body = _decorate_body(cfg, plan)

    if req.mode == "existing_pr":
        head_branch = req.head_branch
        assert head_branch and head_branch.startswith(SERGE_BRANCH_NAMESPACE)
        parent_sha = gh.get_ref_sha(owner, repo, f"heads/{head_branch}")
        base_tree = gh.get_commit_tree_sha(owner, repo, parent_sha)
        tree_sha = gh.create_tree(owner, repo, base_tree, entries)
        commit_sha = gh.create_commit(
            owner,
            repo,
            message=plan.title,
            tree_sha=tree_sha,
            parents=[parent_sha],
        )
        gh.update_ref(owner, repo, f"heads/{head_branch}", commit_sha)
        _emit("log", f"Pushed commit {commit_sha[:8]} to {head_branch}")
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
        message=plan.title,
        tree_sha=tree_sha,
        parents=[parent_sha],
    )
    gh.create_ref(owner, repo, f"refs/heads/{branch}", commit_sha)
    _emit("log", f"Created branch {branch} at {commit_sha[:8]}")
    pr = gh.create_pull_request(
        owner,
        repo,
        title=plan.title,
        head=branch,
        base=req.base_ref,
        body=body,
    )
    _emit("log", f"Opened PR #{pr.get('number')}: {pr.get('html_url')}")
    return TaskResult(
        mode=req.mode,
        pr_number=pr.get("number"),
        branch=branch,
        commit_sha=commit_sha,
        changed_files=changed_files,
        message=f"Opened PR #{pr.get('number')}.",
        url=pr.get("html_url"),
    )
