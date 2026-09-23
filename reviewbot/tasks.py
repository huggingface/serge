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
import json
import logging
import os
import re
import contextlib
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from . import __version__, anchored_edits, budget, pr_links
from .brevity import BrevityResult, condense_patch_comments
from .clone_cache import Checkout, CloneCache, FileChange
from .commit_scope import describe_dropped, scope_paths
from .compression import MessageCompressor
from .config import Config
from .expectation_guard import PatchClassification, classify_patch
from .github_client import SERGE_GIT_EMAIL, GitHubClient
from .llm_client import ChatCompletionClient
from .normalize import NormalizeError, run_normalize
from .prompts import build_task_system_prompt, build_task_user_prompt
from .relore_tool import (
    culprit_pr_number,
    culprit_thread,
    culprit_thread_note,
    prior_art,
    prior_art_note,
)
from .tools import ToolEnv
from .reviewer import (
    _extract_json,
    _format_aggregated_metrics,
    merge_session_records,
    session_record,
    _make_tool_env,
    _run_agentic_loop,
    _UnparseableLLMOutput,
)
from .classify import ENVIRONMENT_ISSUE, UNCLEAR, ClassifyResult, classify_failure
from .slack_tool import post_task_pr_created_notification
from .verify import (
    NOT_REPRODUCED,
    REPRODUCED,
    VerifyOutcome,
    extract_verify_targets,
    gate_did_not_run,
    patch_needs_collateral,
    run_gpu_reproduce,
    run_gpu_verify,
    should_retry,
)

log = logging.getLogger(__name__)

_NORMALIZE_FEEDBACK_CHARS = 80_000
_REJECTED_PATCH_EVENT_CHARS = 80_000
_VALIDATION_RETRY_CONTEXT_CHARS = 120_000
_VALIDATION_RETRY_OUTPUT_CHARS = 80_000
# `git apply -v` stderr echoes every context line it searched for, so the old
# 1200-char clip cut the diagnostic off mid-block on exactly the multi-hunk
# patches that need it most.
_APPLY_STDERR_CHARS = 12_000
# The real file content sent back with an apply rejection.
_APPLY_WINDOWS_CHARS = 24_000

# The task JSON contract from prompts.py. Passed to `_extract_json` so a stray
# `{...}` in the reply — notably a leaked tool call's own argument object —
# can't be mistaken for the task result.
#
# "edits" is the anchored-edit answer (:mod:`reviewbot.anchored_edits`), which a
# task may send INSTEAD of "patch". It never reaches publish_task: the gate
# applies the edits and has git write the diff, so everything downstream still
# sees a unified diff.
_TASK_JSON_KEYS = ("title", "body", "patch", "edits")

# Serge only ever writes inside its own branch namespace. ``existing_pr``
# mode is rejected for any head branch outside it, so the OIDC
# ``repository`` claim cannot be leveraged to push to an arbitrary PR.
SERGE_BRANCH_NAMESPACE = "serge/"

# The anchored-edit contract, restated wherever the schema is re-asserted
# outside the system prompt. Kept in one place so the three copies cannot drift:
# the rule that does the work is "copy it from the file, exactly once", and a
# retry prompt that quietly dropped it would be asking for the format without
# the reason it applies.
_EDIT_SCHEMA_LINE = (
    '  - "edits": a list of {"path", "old", "new"} objects. "old" must be '
    "copied character-for-character from the file (never from memory, never "
    "from an anchor that was just rejected) and must occur EXACTLY ONCE in it. "
    "Use an empty list if no safe fix is possible\n"
)

_TASK_FORCE_FINAL_MESSAGE = (
    "You have used the available investigation budget. Based only on the "
    "evidence already gathered, produce the final task result immediately. "
    "Do not continue investigating or explain your reasoning. Reply with a "
    "single compact JSON object that starts with `{` and has EXACTLY these keys:\n"
    '  - "title": a concise PR title\n'
    '  - "body": a markdown PR description in at most 12 lines explaining the '
    "failure, root cause, and fix; if no safe fix is possible, explain why\n"
    + _EDIT_SCHEMA_LINE
    + 'Send a unified diff as "patch" instead of "edits" only for a new, '
    "deleted or renamed file.\n"
    "Return JSON only: no surrounding prose, no code fences, no extra commentary, "
    "and no tool requests."
)

_TASK_VALIDATION_RETRY_SYSTEM_MESSAGE = (
    "You are fixing a previously rejected task change. Reply with a single "
    "compact JSON object that starts with `{` and has EXACTLY these keys: "
    '"title", "body", and "edits".\n' + _EDIT_SCHEMA_LINE + "Send a unified "
    'diff as "patch" instead of "edits" only for a new, deleted or renamed '
    "file. This turn has no tools, so you cannot re-read the repository: an "
    "anchor from your rejected output that the validator did NOT complain "
    "about is still good and can be reused verbatim. Return JSON only: no "
    "prose, no code fences, no tool requests."
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
        # Agent-loop counters for the work already done when this was raised.
        # A patch that will not apply is a *terminated session*, and it is one of
        # the outcomes most worth counting; without this the whole round would
        # leave no trace but an error string.
        self.session: dict[str, Any] = {}


class NormalizeGateBroken(TaskError):
    """The repo normalizer fails on the *pristine* checkout, before any patch
    is applied — the gate is broken, so no patch can ever pass it.

    Raised instead of feeding the failure back to the model, because the model
    cannot fix it: every correction is rejected by the same environment error
    and the task ends with a misleading "the patch does not pass the
    normalizer" after burning the whole correction budget. Observed 2026-08-17
    (transformers#48037): transformers' `main` moved its `tokenizers` pin to
    >=0.23.1 while the task-runner image still had 0.22.2, and the gate's
    `uv pip install -e . --no-deps` cannot upgrade a dependency, so every
    checker that imports transformers died on the version guard. Three groups
    reported "no fix" and ~3.5M input tokens were spent on an error no patch
    could address.

    Deliberately NOT status 422: 422 means "this candidate's patch does not
    apply", which makes the runner move on to the next candidate group. A
    broken gate breaks every group, so the task must fail loudly instead."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


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
    # Set by reproduce-first when the group reproduced on GPU: the run that
    # confirmed the failure at the base commit. Surfaced in the PR body as the
    # "failed before" evidence.
    reproduce_run_url: Optional[str] = None
    # Optional, from the /tasks payload: node-id → [{label, url}] links to where
    # the dispatcher observes each failing test (a dashboard, a run page). serge
    # renders them in the PR body and knows nothing about what they point at —
    # see :func:`pr_links.sanitize_test_links`.
    test_links: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    # Optional, from the /tasks payload: GitHub logins to request a review from
    # on a newly opened PR. The dispatcher decides who is relevant — for the
    # integration-failure triage that is the author of the commit its bisect
    # blamed — and serge only forwards the request. See
    # :func:`sanitize_reviewers`.
    reviewers: tuple[str, ...] = ()

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
    # Per-job agent-loop counters (reviewer.session_record): turns, tool calls,
    # re-opened paths, and which guard ended the loop. Kept past the job row's
    # 25-job retention so a change to the loop can be shown to have helped.
    session: dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    # True when the patch was validated in-loop (see :func:`_validate_patch`):
    # the worktree already holds the applied + normalized result, so
    # :func:`publish_task` commits it directly instead of re-applying.
    # NOTE: when it is set, ``patch`` above is the model's own diff and the
    # worktree is what ships. The brevity pass (:mod:`reviewbot.brevity`)
    # shortens comments in the worktree, so the committed comments can be
    # shorter than the ones in ``patch``. Everything reading ``patch`` reads it
    # for the code it touches (``scope_paths``, ``classify_patch``), which a
    # comment rewrite cannot change.
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
    # Set when the GPU verify gate ran (see verify.py). ``verify_verdict`` drives
    # the retry-with-tracebacks loop; ``verify_tracebacks`` is the feedback fed
    # to the next LLM round. Not persisted in ``to_json`` (tracebacks are large).
    verify_verdict: Optional[str] = None
    verify_tracebacks: dict[str, str] = field(default_factory=dict)
    # True when the patch changed only expected values in test files, so the GPU
    # verdict above is circular and must not be reported as confidence — see
    # :mod:`reviewbot.expectation_guard`. Travels in ``to_json`` because the
    # triage recap and the session metrics are where "verify_verdict=fixed" is
    # otherwise read as success.
    expectation_only: bool = False
    expectation_note: str = ""
    # Agent-loop counters accumulated over every round this candidate ran
    # (reviewer.merge_session_records). Small — counters only, no tracebacks —
    # so unlike ``verify_tracebacks`` it does travel in ``to_json``, which is
    # how it reaches serge from a per-task runner pod.
    session: dict[str, Any] = field(default_factory=dict)

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
            "verify_verdict": self.verify_verdict,
            "expectation_only": self.expectation_only,
            "expectation_note": self.expectation_note,
            "session": self.session,
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
        test_links=pr_links.sanitize_test_links(payload.get("test_links")),
        reviewers=sanitize_reviewers(payload.get("reviewers")),
    )


# GitHub caps a single review request at 15 logins; stay under it and never ping
# a crowd because a dispatcher sent a long list.
MAX_REVIEWERS = 10
# A GitHub login: 1-39 chars of alphanumerics and single inner hyphens. Bot
# accounts (`dependabot[bot]`) deliberately fail this — a bot cannot review, and
# requesting one 422s the whole call, taking the valid logins down with it.
_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


def sanitize_reviewers(raw: Any) -> tuple[str, ...]:
    """Validate an inbound ``reviewers`` list of GitHub logins.

    Anything malformed is dropped rather than raising, for the same reason as
    :func:`pr_links.sanitize_test_links`: a review request is a courtesy on top
    of the fix, and a bad entry must not cost a PR. Order is preserved and
    duplicates collapse (case-insensitively — GitHub logins are unique
    case-insensitively, and requesting both spellings is one wasted 422)."""
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            continue
        login = entry.strip().lstrip("@")
        if not _LOGIN_RE.match(login) or login.lower() in seen:
            continue
        seen.add(login.lower())
        out.append(login)
        if len(out) == MAX_REVIEWERS:
            break
    return tuple(out)


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


def format_pr_files_diff(files: list[dict[str, Any]], *, limit: int = 30000) -> str:
    """Concatenate a PR's per-file patches into a single blob used as the
    "prior attempt" context on an existing_pr follow-up. Best-effort and
    purely informational — not meant to be re-applied."""
    parts: list[str] = []
    total = 0
    for f in files:
        patch = f.get("patch") or ""
        chunk = f"--- {f.get('filename')} ---\n{patch}\n"
        parts.append(chunk)
        total += len(chunk)
        if total >= limit:
            parts.append("[... prior attempt truncated ...]")
            break
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agentic loop → patch (with in-loop normalize validation)
# ---------------------------------------------------------------------------
# Env the normalizer's checkers see, and the ref the worktree grows for them.
#
# transformers' `utils/check_noisy_comments.py` — and any checker that reasons
# about "what this patch changed" — answers a different question depending on
# whether it can resolve the patch:
#
#     diff_only         = _running_in_pr() and not args.all_files
#     patch_is_resolved = diff_only and _patch_added_lines() is not None
#     if patch_is_resolved:
#         findings = _filter_findings_to_patch(findings)
#     blocking = args.fail_on_findings or (patch_is_resolved and ...)
#
# A serge worktree resolved NEITHER half, and both failures are silent:
#
# 1. `CloneCache` makes a detached worktree with no branch refs, so `origin/main`
#    does not exist, `merge-base` fails and the patch is unresolved. The checker
#    then scans the whole tree. It is written to degrade to *reporting* there —
#    but `--fail-on-findings` (which transformers' `fix_args` now passes, and
#    `make fix-repo` runs fix mode) sits on the left of that `or` and blocks
#    anyway. Measured on a real worktree: 377 findings, 375 of them comments in
#    files serge never opened. That is what failed 7 of 10 groups on 2026-09-22
#    (transformers#49026).
# 2. The clone is shallow (`--depth 50`), so `git blame` attributes every line
#    older than the graft to the boundary commit. The checker's two noise filters
#    — ignore findings last committed before a cutoff, ignore an owner's own
#    comments — both read blame, so both silently become no-ops. The same scan on
#    a full clone yields 2 findings, not 375.
#
# So: give the worktree the ref, and commit the patch, for the duration of the
# normalizer only. Then `merge_base...HEAD` is exactly serge's patch and the
# checker blocks on serge's own comments and nothing else — verified on the same
# worktree: "Restricting noisy comment scan to 1 Python file(s) changed in this
# patch. Found 2 noisy comment finding(s) on lines this patch adds."
#
# The commit is temporary and undone in a `finally`: `publish_task` still commits
# the worktree itself, exactly as before.
_CHECKER_BASE_REF = "refs/remotes/origin/main"
_CHECKER_ENV = {
    # `_running_in_pr()`. Not a lie: serge is preparing a pull request, and this
    # is the variable that tells a checker to scope to the patch rather than the
    # tree. Without it `diff_only` is False and the scoping never happens.
    "CI_PULL_REQUEST": "1",
}


def _git_in(checkout: Checkout, *args: str) -> tuple[int, str]:
    """One git call in the worktree. Never raises; callers decide what a
    non-zero means, because every use here is best-effort."""
    try:
        done = subprocess.run(
            ["git", "-C", checkout.path, *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:
        log.debug("git %s failed in %s", args, checkout.path, exc_info=True)
        return 1, ""
    return done.returncode, (done.stdout or "").strip()


@contextlib.contextmanager
def _patch_shaped_for_checkers(checkout: Checkout) -> Iterator[None]:
    """Make the worktree look like a pull request while the normalizer runs.

    Best-effort in both directions. If the shaping cannot be done the normalizer
    still runs — on today's behaviour, which is the bug, but a normalizer that
    does not run at all is worse. If it was done it is always undone, including
    on an exception, so the worktree handed back is byte-identical to the one
    handed in: same content, same HEAD, nothing staged that was not staged.
    """
    base_rc, base = _git_in(checkout, "rev-parse", "HEAD")
    if base_rc != 0 or not base:
        yield
        return

    # Only mint the ref when it is missing, and only delete what we minted.
    ref_existed = (
        _git_in(checkout, "rev-parse", "--verify", "-q", _CHECKER_BASE_REF)[0] == 0
    )
    minted_ref = False
    if not ref_existed:
        minted_ref = _git_in(checkout, "update-ref", _CHECKER_BASE_REF, base)[0] == 0

    # `add -A` and not a path list: serge's patch may add files, and a checker
    # that cannot see a new file reports it as unchanged.
    committed = False
    if _git_in(checkout, "add", "-A")[0] == 0:
        committed = (
            _git_in(
                checkout,
                "-c",
                "user.email=serge@huggingface.co",
                "-c",
                "user.name=serge",
                "commit",
                "--quiet",
                "--no-verify",
                "-m",
                "serge: patch under validation",
            )[0]
            == 0
        )
    try:
        yield
    finally:
        if committed:
            # --soft: HEAD goes back to the base commit and every change the
            # normalizer's fixers made stays in the worktree, which is what
            # `publish_task` reads.
            _git_in(checkout, "reset", "--soft", base)
        if minted_ref:
            _git_in(checkout, "update-ref", "-d", _CHECKER_BASE_REF)


def _run_repo_normalizer(
    cfg: Config,
    checkout: Checkout,
    emit: Callable[[str, str], None],
) -> tuple[Optional[int], str]:
    """Run the configured repo normalizer (``cfg.task_normalize_command``) over
    the current worktree with its fixers enabled, writing any regenerated files
    (e.g. transformers' ``modeling_*.py`` from ``modular_*.py``) back in place.

    Returns ``(returncode, output)``. ``returncode`` is ``None`` when the
    normalizer could not run at all (sandbox unavailable / timeout / launch
    failure) — infrastructure, not the patch's fault, which callers treat as a
    best-effort pass. Assumes ``cfg.task_normalize_command`` is set."""
    command = cfg.task_normalize_command
    assert command is not None
    emit("step", "normalize")
    # Clamped to the budget the pod has left, for the same reason the verify
    # poll is: an un-clamped normalize outlives the process waiting on it, and
    # what dies with it is a patch that already applied cleanly. Job `d2c24049`
    # (2026-09-16) entered this step 5m19s before its Job deadline with a
    # 30-minute normalize timeout and was killed mid-run; the triage issue
    # recorded `⚠️ task failed` and the two tests stayed unfixed. 0 means there
    # is not even room to start, which is the same "could not run" outcome the
    # NormalizeError branch below already handles — the patch is accepted
    # un-normalized and CI still catches what the normalizer would have.
    timeout = budget.clamp(cfg.task_normalize_timeout)
    if timeout <= 0:
        log.warning("no budget left to run the normalizer; accepting un-normalized")
        emit(
            "log",
            "Not enough of the runner budget is left to run the normalizer; "
            "accepting the patch un-normalized so it can still be pushed.",
        )
        return None, ""
    if timeout < cfg.task_normalize_timeout:
        emit(
            "log",
            f"Normalizer timeout cut to {timeout}s by the remaining runner "
            f"budget (configured {cfg.task_normalize_timeout}s).",
        )
    emit("log", f"Running the repo normalizer: `{' '.join(command)}`…")
    try:
        with _patch_shaped_for_checkers(checkout):
            return run_normalize(
                command,
                workdir=checkout.path,
                write_root=checkout.path,
                backend=cfg.task_sandbox_backend,
                image=cfg.task_normalize_image,
                mode=cfg.helper_sandbox,
                timeout=timeout,
                memory=cfg.task_normalize_memory,
                extra_env=_CHECKER_ENV,
            )
    except NormalizeError as exc:
        # Infrastructure problem (sandbox unavailable, timeout) — not the
        # model's fault. Signal best-effort with a None returncode; CI still
        # catches anything the normalizer would have.
        log.warning("normalizer unavailable: %s", exc)
        emit(
            "log", f"Normalizer unavailable ({exc}); accepting the patch un-normalized."
        )
        return None, ""


def check_task_preflight(
    cfg: Config,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    emit: Callable[[str, str], None],
) -> None:
    """Prove the normalize gate is passable at all, before any LLM work.

    Runs ``cfg.task_preflight_command`` (operator config, unset = skipped) on
    the pristine checkout. A non-zero exit means the gate rejects the *unpatched*
    base, so no patch this task could write would ever pass it, and the task
    fails immediately with :class:`NormalizeGateBroken`.

    :func:`_check_normalizer_baseline` already draws that distinction, but only
    on the failure path — after a full agent loop has been paid for, once per
    candidate, every night until an operator notices. Two nights in September
    2026 are the whole argument for this function: transformers#48685 moved the
    ``huggingface-hub`` pin from ``>=1.5.0`` to ``>=1.31.0`` on 2026-09-12, the
    task-runner image had been built on 2026-09-09, and the gate's
    ``--no-deps`` install (mandatory — the task pod's egress allowlist has no
    PyPI) could not upgrade it. Every checker importing transformers died on
    ``cannot import name 'httpx' from 'huggingface_hub.utils'``; 5 groups were
    lost over the 09-12 and 09-13 nightlies for ~8.4M input tokens, and the
    probe that would have caught it is an editable install plus one ``import``,
    about ten seconds.

    Leaves the worktree pristine. Fails **open** when the sandbox itself cannot
    run the probe: that is infrastructure, and the real gate is still ahead.
    """
    command = cfg.task_preflight_command
    if not command:
        return
    emit("step", "preflight")
    emit(
        "log",
        f"Preflight — checking the normalize gate is passable: `{' '.join(command)}`…",
    )
    try:
        returncode, output = run_normalize(
            command,
            workdir=checkout.path,
            write_root=checkout.path,
            backend=cfg.task_sandbox_backend,
            image=cfg.task_normalize_image,
            mode=cfg.helper_sandbox,
            timeout=cfg.task_preflight_timeout,
            memory=cfg.task_normalize_memory,
        )
    except NormalizeError as exc:
        # The probe could not run. Not evidence of anything about the gate, and
        # the gate itself still runs later, so do not fail the task over it.
        log.warning("preflight unavailable: %s", exc)
        emit("log", f"Preflight unavailable ({exc}); continuing.")
        return
    finally:
        # The probe installs and may write (build artefacts, __pycache__), and
        # everything downstream assumes an untouched base.
        clone_cache.reset_worktree(checkout)

    if returncode == 0:
        emit("log", "Preflight passed.")
        return

    message = (
        f"The task-runner environment fails the preflight check (`{' '.join(command)}`, "
        f"exit {returncode}) on the unpatched checkout, so the normalize gate cannot be "
        "passed by any patch and no LLM work was started. This needs an operator, not a "
        "retry. Common cause: the task-runner image has drifted from the target repo's "
        "`main` (a dependency pin moved, and the gate's editable install runs with "
        f"`--no-deps`).\n\n{_bounded_normalize_feedback(output)}"
    )
    emit("normalize_error", f"Preflight fails on the pristine checkout:\n{output}")
    raise NormalizeGateBroken(message)


def _check_normalizer_baseline(
    cfg: Config,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    emit: Callable[[str, str], None],
    state: dict,
) -> None:
    """Tell a bad patch apart from a broken gate, after a normalizer failure.

    Resets the worktree to the pristine checkout and runs the normalizer on it.
    A clean exit means the gate works and the patch really is at fault, so this
    returns and the caller feeds the failure back to the model. A non-zero exit
    means the normalizer rejects the *unpatched* base too — no patch can pass —
    so this raises :class:`NormalizeGateBroken`.

    Runs at most once per task: the answer is cached in ``state`` and replayed,
    because the normalizer is the expensive step (transformers' takes minutes).
    Only ever called on the failure path, so a healthy task pays nothing.

    Leaves the worktree pristine either way — the normalizer's fixers rewrite
    files, and a base that is not itself normalizer-clean legitimately produces
    changes here, which is why the *exit code* is the signal and not the diff.
    """
    if state.get("checked"):
        broken = state.get("broken")
        if broken:
            raise NormalizeGateBroken(broken)
        return
    state["checked"] = True

    emit(
        "log",
        "Normalizer rejected the patch; re-running it on the unpatched "
        "checkout to tell a bad patch from a broken gate…",
    )
    clone_cache.reset_worktree(checkout)
    returncode, output = _run_repo_normalizer(cfg, checkout, emit)
    clone_cache.reset_worktree(checkout)
    if returncode in (None, 0):
        return

    cmd = " ".join(cfg.task_normalize_command or [])
    message = (
        f"The repository's normalizer (`{cmd}`) fails on the unpatched "
        f"checkout (exit {returncode}), so no patch can pass it — this is a "
        "broken gate, not a bad patch, and it needs an operator. Common cause: "
        "the task-runner image has drifted from the target repo's `main` (a "
        "dependency pin moved, and the gate's editable install runs with "
        f"`--no-deps`).\n\n{_bounded_normalize_feedback(output)}"
    )
    state["broken"] = message
    emit("normalize_error", f"Normalizer fails on the pristine checkout:\n{output}")
    raise NormalizeGateBroken(message)


def _condense_added_comments(
    cfg: Config,
    *,
    checkout: Checkout,
    patch: str,
    llm: Optional[ChatCompletionClient],
    emit: Callable[[str, str], None],
) -> Optional[BrevityResult]:
    """Run the one-call brevity pass over the comments this patch adds
    (:mod:`reviewbot.brevity`), in the worktree, before the normalizer.

    Deliberately placed *before* the normalizer and not after it: the comments
    that get committed are then the ones the repo's own formatter saw and the
    gate accepted, exactly like the code around them.

    Returns None when the pass is off or unavailable, and swallows anything it
    raises — some models are verbose, but none of that is worth failing a task
    over, and the un-condensed comments are merely the status quo."""
    if not cfg.task_comment_brevity or llm is None or not patch.strip():
        return None
    try:
        return condense_patch_comments(
            llm,
            root=checkout.path,
            patch=patch,
            width=cfg.comment_brevity_width,
            min_chars=cfg.comment_brevity_min_chars,
            max_items=cfg.comment_brevity_max_items,
            max_tokens=cfg.llm_max_tokens,
            reasoning_effort=cfg.llm_reasoning_effort,
            emit=emit,
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "comment brevity pass failed; keeping the original comments", exc_info=True
        )
        return None


def _validate_patch(
    cfg: Config,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    content: Optional[str],
    emit: Callable[[str, str], None],
    baseline_state: Optional[dict] = None,
    llm: Optional[ChatCompletionClient] = None,
    brevity_chats: Optional[list] = None,
    report: Optional[dict[str, Any]] = None,
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

    ``report``, when given, is stamped with ``rejection``: ``"apply"`` when
    ``git apply`` refused the patch, ``"normalize"`` when the normalizer did, and
    ``""`` when the answer is accepted. The caller needs the distinction because
    the two want different correction turns — a normalizer rejection carries its
    own reason in the feedback, an apply rejection needs the repo re-read.

    Only called when ``cfg.task_normalize_command`` is set."""
    command = cfg.task_normalize_command
    assert command is not None

    if report is not None:
        # Cleared per attempt: a patch synthesized from an accepted answer must
        # never outlive the answer it came from.
        report["patch"] = ""

    try:
        result = _extract_json(content, _TASK_JSON_KEYS)
    except ValueError:
        # Unparseable — not something the normalizer can speak to. Accept here
        # and let prepare_task's own extraction raise the proper error.
        if report is not None:
            report["rejection"] = ""
        return None, False

    raw_edits = result.get("edits")
    if isinstance(raw_edits, list) and raw_edits:
        # The anchored-edit answer (:mod:`reviewbot.anchored_edits`). We apply
        # it and let git write the diff, so from the next line on this is the
        # unified-diff path it has always been — the brevity pass, the
        # normalizer, `commit_scope`, `classify_patch` and publish_task all keep
        # reading a diff, and one with correct geometry by construction.
        if isinstance(result.get("patch"), str) and result["patch"].strip():
            emit(
                "log",
                "Answer carried both `edits` and `patch`; applying the edits "
                "and ignoring the diff.",
            )
        patch, edit_feedback = _apply_anchored_edits(
            checkout=checkout,
            clone_cache=clone_cache,
            raw_edits=raw_edits,
            emit=emit,
            report=report,
        )
        if edit_feedback is not None:
            # Same rejection class as a failed `git apply`: the model and the
            # file disagree about what the file says, and only a re-read
            # settles it. The stamp is what keeps the correction turn's tools.
            if report is not None:
                report["rejection"] = "apply"
            return edit_feedback, False
        if report is not None:
            report["patch"] = patch
    else:
        patch = result.get("patch")
        if not isinstance(patch, str) or not patch.strip():
            # Nothing to validate (a "no safe fix" answer); accept as-is.
            if report is not None:
                report["rejection"] = ""
            return None, False

        clone_cache.reset_worktree(checkout)
        try:
            clone_cache.apply_patch(checkout, patch)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[
                :_APPLY_STDERR_CHARS
            ]
            emit("rejected_patch", _bounded_rejected_patch(patch))
            emit(
                "patch_apply_error",
                f"`git apply` rejected the proposed patch:\n{stderr}",
            )
            # The old feedback was the stderr plus "check that the hunk context
            # lines match the current code exactly" — an instruction the model had
            # no way to follow, because it was never shown the current code. It
            # re-guessed instead: on prod job 13387bf45ef6 the same context line
            # came back as "<s><s> Tell me..." twice and "[PAD][PAD]...<s> Tell
            # me..." once across three corrections, and the task died having never
            # applied. So send what the file actually holds around each rejected
            # hunk, and say plainly that it is authoritative.
            if report is not None:
                report["rejection"] = "apply"
            windows = patch_target_windows(patch, _checkout_reader(checkout))
            msg = (
                "Your patch was rejected — `git apply` could not apply it to a "
                f"clean checkout:\n\n{apply_error_for_model(stderr)}\n"
            )
            if windows:
                msg += (
                    "\nThis is what the files ACTUALLY contain right now, around "
                    "each hunk you targeted. It is the source of truth — your "
                    "context lines must match it byte for byte, including trailing "
                    "commas and whitespace. Do not reproduce these lines from "
                    "memory; copy them from here (dropping the line-number "
                    "prefix):\n\n"
                    f"{_bounded_apply_windows(windows)}\n"
                )
            msg += (
                "\nReturn the corrected change as `edits`: for each change, the "
                "`old` text copied from the content above (occurring exactly once) "
                "and the `new` text to put in its place. That format has no hunk "
                "headers and no line numbers, so the only thing that has to be "
                "right is the text itself. If what you wanted to change is not in "
                "the content above, the change itself is wrong — say so in `body` "
                "and return an empty `edits` list rather than guessing at it."
            )
            return msg, False

    # The patch is applied and nothing has reformatted it yet, so the file's
    # line numbers are still the diff's new-side ones: the one point where the
    # comments the patch added can be located exactly. Shorten them here so the
    # normalizer runs over the text that will be committed.
    brevity = _condense_added_comments(
        cfg, checkout=checkout, patch=patch, llm=llm, emit=emit
    )
    if brevity is not None and brevity.chat is not None and brevity_chats is not None:
        brevity_chats.append(brevity.chat)
    condensed = brevity is not None and brevity.condensed > 0

    returncode, output = _run_repo_normalizer(cfg, checkout, emit)
    if returncode is None:
        # Normalizer could not run (infra) — accept the applied patch
        # best-effort rather than blaming the LLM.
        return None, True

    if returncode != 0:
        clone_cache.reset_worktree(checkout)
        cmd = " ".join(command)
        feedback_output = _bounded_normalize_feedback(output)
        emit("rejected_patch", _bounded_rejected_patch(patch))
        emit(
            "normalize_error",
            f"Normalizer failed (exit {returncode}) for `{cmd}`:\n{output}",
        )
        # Before spending a correction on it, make sure the patch is actually
        # what the normalizer is unhappy about (raises when it is not).
        _check_normalizer_baseline(
            cfg,
            checkout=checkout,
            clone_cache=clone_cache,
            emit=emit,
            state=baseline_state if baseline_state is not None else {},
        )
        msg = (
            f"Your patch applied cleanly, but the repository's normalizer "
            f"(`{cmd}`) then failed (exit {returncode}):\n\n{feedback_output}\n\n"
            "Revise the patch so the normalizer passes. Fix the ROOT CAUSE — "
            "suppress a check (`# noqa`, `# type: ignore`, disabling a rule) "
            "only as a last resort, for a deliberate and justified exception, "
            "and explain why in a comment. Common causes: editing an "
            "auto-generated file instead of its modular/source counterpart, "
            "leaving a copied block out of sync, or a lint/format issue the "
            "fixer cannot resolve on its own."
        )
        if condensed:
            # The line numbers in the output above are the worktree's, and the
            # brevity pass moved them: a comment that came in over four lines
            # may now sit on one. Say so, rather than let the model conclude
            # its own diff was misread.
            msg += (
                "\n\nNote: the comment TEXT in your patch was shortened before "
                "this run (your code is untouched), so a line number above can "
                "sit a few lines from where your diff put it."
            )
        if cfg.task_normalize_guidance:
            msg += f"\n\n{cfg.task_normalize_guidance.strip()}"
        if report is not None:
            report["rejection"] = "normalize"
        return msg, False

    emit("log", "Patch validated; normalizer is clean.")
    if report is not None:
        report["rejection"] = ""
    return None, True


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_PATCH_TARGET_RE = re.compile(r"^\+\+\+ b/(.+)$")
# Lines of real file content to show around a rejected hunk. Wide enough that
# the hunk's whole old side plus a few lines either side are visible, so the
# model can read the context off the file instead of recalling it.
_APPLY_WINDOW_PAD = 8
_APPLY_WINDOWS_MAX = 6


def patch_target_windows(
    diff_text: str, read_file: Callable[[str], Optional[list[str]]]
) -> str:
    """The file's ACTUAL content around each hunk a rejected diff targets.

    A rejected patch means the model's idea of the file disagrees with the file.
    `git apply -v` says what it searched for; this says what is really there, so
    the correction turn has both sides and needs to recall neither. Numbered the
    way ``tools.read_file`` numbers its output, so the two agree.

    ``read_file`` maps a repo-relative path to its lines (without newlines), or
    ``None`` when it cannot be read — a patch that creates a file has nothing to
    show, which is not an error.

    Pure apart from ``read_file`` so it is testable without a checkout.
    """
    blocks: list[str] = []
    target: Optional[str] = None
    cache: dict[str, Optional[list[str]]] = {}
    for line in diff_text.splitlines():
        m = _PATCH_TARGET_RE.match(line)
        if m:
            target = m.group(1).strip()
            continue
        h = _HUNK_HEADER_RE.match(line)
        if not h or target is None or len(blocks) >= _APPLY_WINDOWS_MAX:
            continue
        if target not in cache:
            cache[target] = read_file(target)
        lines = cache[target]
        if lines is None:
            blocks.append(f"{target}: not in the checkout (a new file?)")
            continue
        start = int(h.group(1))
        span = int(h.group(2) or 1)
        lo = max(1, start - _APPLY_WINDOW_PAD)
        hi = min(len(lines), start + span + _APPLY_WINDOW_PAD)
        if lo > len(lines):
            blocks.append(
                f"{target} (hunk @@ -{start} is past the end of the file, "
                f"which has {len(lines)} lines)"
            )
            continue
        body = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(lo, hi + 1))
        blocks.append(f"{target} (lines {lo}-{hi}):\n{body}")
    if not blocks:
        return ""
    return "\n\n".join(blocks)


def _apply_anchored_edits(
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    raw_edits: Any,
    emit: Callable[[str, str], None],
    report: Optional[dict[str, Any]] = None,
) -> tuple[str, Optional[str]]:
    """Apply an anchored-edit answer to a pristine worktree.

    Returns ``(patch, feedback)``, exactly one of which is truthy. ``patch`` is
    the unified diff **git** produced from the edited worktree — the point of
    the format is that the model never writes a diff, so nothing downstream has
    to trust one it wrote. On feedback the worktree is left clean, exactly as a
    rejected ``git apply`` leaves it.

    The rejection is deliberately specific ("`old` does not occur", "occurs 3
    times at lines …") where `git apply`'s is not, and carries the real lines
    when the anchor is a near-miss — but it never echoes the anchor back. See
    :func:`apply_error_for_model` for why quoting the model's own wrong text is
    what kept the fabrication alive.
    """
    parsed = anchored_edits.parse_edits(raw_edits)
    if parsed.problems:
        feedback = (
            "Your `edits` were rejected before anything was applied — they do "
            "not match the edit schema:\n\n"
            + "\n".join(parsed.problems)
            + '\n\nEach edit is {"path": "<repo-relative path>", "old": "<text '
            'copied from the file, occurring exactly once>", "new": '
            '"<replacement>"}.'
        )
        _emit_edit_rejection(emit, raw_edits, feedback, parsed.problems, report)
        return "", feedback

    clone_cache.reset_worktree(checkout)
    contents, problems = anchored_edits.apply_edits(
        parsed.edits, read=_checkout_text_reader(checkout)
    )
    if problems:
        feedback = anchored_edits.rejection_feedback(problems)
        _emit_edit_rejection(emit, raw_edits, feedback, problems, report)
        return "", feedback
    if not contents:
        problems = ["the edits left every file exactly as it was"]
        feedback = (
            "Your edits applied but left every file exactly as it was, so there "
            "is no change to commit. Either make the edit that fixes the "
            "failure, or return an empty `edits` list and say in `body` why no "
            "safe fix is possible."
        )
        _emit_edit_rejection(emit, raw_edits, feedback, problems, report)
        return "", feedback

    for rel, text in contents.items():
        full = _resolve_in_checkout(checkout, rel)
        # apply_edits only returns paths its reader accepted, so this cannot be
        # None; the assert states the boundary rather than trusting the chain.
        assert full is not None, rel
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(text)
    paths = sorted(contents)
    clone_cache.stage_paths(checkout, paths)
    patch = clone_cache.staged_diff(checkout)
    emit(
        "log", f"Applied {len(parsed.edits)} anchored edit(s) to {len(paths)} file(s)."
    )
    return patch, None


def _emit_edit_rejection(
    emit: Callable[[str, str], None],
    raw_edits: Any,
    feedback: str,
    problems: list[str],
    report: Optional[dict[str, Any]],
) -> None:
    """Persist a rejected edit set for the operator, and record why.

    ``report["edits_error"]`` is the one-line reason, which is what a caller
    turns into a job error when the correction budget runs out on an edit that
    never applied — the anchored-edit counterpart of "patch did not apply
    cleanly", and just as much a failure the triage issue must show.

    Reuses the ``rejected_patch`` / ``patch_apply_error`` event kinds the diff
    path uses: they are the two kinds ``store.PERSIST_EVENT_KINDS`` keeps and
    the task page renders, and they are the only reason the 2026-09-16/17
    failures were diagnosable at all (``raw_llm_output`` was null on all three).
    A second pair of kinds would have split that history in half.
    """
    if report is not None:
        report["edits_error"] = anchored_edits.summarize(problems)
    try:
        rendered = json.dumps(raw_edits, indent=1)[:_REJECTED_PATCH_EVENT_CHARS]
    except (TypeError, ValueError):
        rendered = repr(raw_edits)[:_REJECTED_PATCH_EVENT_CHARS]
    emit("rejected_patch", rendered)
    emit("patch_apply_error", f"The anchored edits were rejected:\n{feedback}")


def _resolve_in_checkout(checkout: Checkout, rel: str) -> Optional[str]:
    """``rel`` as an absolute path inside the worktree, or ``None``.

    The model chooses the path and the applier writes to it, so the containment
    check is real: ``realpath`` resolves symlinks, which is the one escape a
    purely textual path check (``anchored_edits._safe_path``) cannot see.
    """
    root = os.path.realpath(checkout.path)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        return None
    return full


def _checkout_text_reader(checkout: Checkout) -> Callable[[str], Optional[str]]:
    """An :func:`anchored_edits.apply_edits` reader bound to a worktree."""

    def read(rel: str) -> Optional[str]:
        full = _resolve_in_checkout(checkout, rel)
        if full is None or not os.path.isfile(full):
            return None
        try:
            with open(full, encoding="utf-8") as fh:
                return fh.read()
        except (OSError, UnicodeDecodeError):
            return None

    return read


def _checkout_reader(checkout: Checkout) -> Callable[[str], Optional[list[str]]]:
    """A ``patch_target_windows`` reader bound to a worktree."""

    def read(rel: str) -> Optional[list[str]]:
        full = os.path.join(checkout.path, rel)
        if not os.path.isfile(full):
            return None
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                return fh.read().splitlines()
        except OSError:
            return None

    return read


def _bounded_normalize_feedback(output: str) -> str:
    """Bound normalize feedback sent back into the LLM conversation."""
    if len(output) <= _NORMALIZE_FEEDBACK_CHARS:
        return output
    head = _NORMALIZE_FEEDBACK_CHARS // 2
    tail = _NORMALIZE_FEEDBACK_CHARS - head
    omitted = len(output) - _NORMALIZE_FEEDBACK_CHARS
    return (
        output[:head].rstrip()
        + f"\n\n--- omitted {omitted} chars of normalize output from LLM feedback ---\n\n"
        + output[-tail:].lstrip()
    ).rstrip()


_SEARCHING_RE = re.compile(
    r"^error: while searching for:\n(?:.*\n)*?(?=^error: )", re.MULTILINE
)


def apply_error_for_model(stderr: str) -> str:
    """The apply failure as the MODEL should see it: locations, not echoes.

    `git apply -v` prints an ``error: while searching for:`` block quoting the
    context it looked for — which is the model's own wrong context. Sending that
    back hands the model its fabrication a second time, labelled as output from
    git, right next to the real file content it is supposed to copy instead.
    Measured against the deployed model on prod job 13387bf45ef6: the true line
    appeared 3 times in the prompt and the fabricated one once, and the model
    still reproduced the fabrication in 3 of 3 runs. So the verbose block stays
    on the operator event (where it is the whole diagnosis) and is stripped
    here, leaving the ``error: patch failed: <file>:<line>`` lines that say
    WHERE to look without saying what to look for.
    """
    return _SEARCHING_RE.sub("", stderr).strip()


def _bounded_apply_windows(text: str) -> str:
    """Bound the real-file windows sent back with an apply rejection."""
    return _bound_middle(text, _APPLY_WINDOWS_CHARS, "chars of file content")


def _bounded_rejected_patch(patch: str) -> str:
    """Bound rejected patch events kept for operator diagnosis."""
    return _bound_middle(
        patch,
        _REJECTED_PATCH_EVENT_CHARS,
        "chars of rejected patch",
    )


def _bounded_validation_retry_output(text: str) -> str:
    """Bound rejected assistant JSON resent to the correction turn."""
    return _bound_middle(
        text,
        _VALIDATION_RETRY_OUTPUT_CHARS,
        "chars of rejected assistant output",
    )


def _bounded_validation_retry_context(text: str) -> str:
    """Bound original task context resent to a validation correction turn."""
    return _bound_middle(
        text,
        _VALIDATION_RETRY_CONTEXT_CHARS,
        "chars of task context",
    )


def _bound_middle(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    return (
        text[:head].rstrip()
        + f"\n\n--- omitted {omitted} {label} ---\n\n"
        + text[-tail:].lstrip()
    ).rstrip()


def _task_validation_retry_messages(
    *,
    repo_full_name: str,
    base_ref: str,
    instruction: str,
    context: str,
    existing_diff: Optional[str],
    rejected_content: Optional[str],
    feedback: str,
    retry_number: int,
) -> list[dict[str, Any]]:
    """Build the compact no-tools retry prompt for rejected task patches."""
    body = (
        f"Repository: {repo_full_name}\n"
        f"Base ref: {base_ref}\n"
        f"Validation correction: {retry_number}\n\n"
        "Original task instruction:\n"
        f"{instruction.strip() or '(none)'}\n\n"
        "Original failure context:\n"
        f"{_bounded_validation_retry_context(context.strip()) or '(none)'}\n\n"
    )
    if existing_diff:
        body += f"Existing diff to preserve:\n{existing_diff.strip()}\n\n"
    body += (
        "Validator rejection to fix:\n"
        f"{feedback.strip()}\n\n"
        "Rejected assistant output:\n"
        f"{_bounded_validation_retry_output(rejected_content or '')}\n\n"
        "Return a corrected task JSON object only."
    )
    return [
        {"role": "system", "content": _TASK_VALIDATION_RETRY_SYSTEM_MESSAGE},
        {"role": "user", "content": body},
    ]


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


def prompt_prefix_summary(
    *,
    system_prompt: str,
    user_prompt: str,
    conventions: str = "",
    normalize_guidance: str | None = "",
    context: str = "",
    instruction: str = "",
    existing_diff: str | None = "",
) -> str:
    """One line naming the size of every part of the resent prompt prefix.

    Pure so the wording is testable without a checkout, an LLM or a GPU. Chars
    rather than tokens deliberately: the exact token count already arrives in
    the per-turn ``metrics`` events, and a char count needs no tokenizer for a
    model whose tokenizer we do not ship.
    """
    total = len(system_prompt) + len(user_prompt)
    return (
        f"Prompt prefix {total:,} chars, resent every turn: "
        f"system {len(system_prompt):,} "
        f"[conventions {len(conventions or ''):,}, "
        f"normalize-guidance {len(normalize_guidance or ''):,}] · "
        f"user {len(user_prompt):,} "
        f"[context {len(context or ''):,}, "
        f"instruction {len(instruction or ''):,}, "
        f"existing-diff {len(existing_diff or ''):,}]"
    )


def _failing_node_ids(req: TaskRequest) -> list[str]:
    """The node-ids this task is about, best source first.

    ``test_links`` is the dispatcher's own structured list and needs no parsing;
    the failure report is the fallback for a task dispatched without one.
    """
    from_links = [n for n in (req.test_links or {}) if n and "::" in n]
    if from_links:
        return from_links
    node_ids, _model, _machine = extract_verify_targets(
        (req.context or "").splitlines(), ""
    )
    return node_ids


def _history_notes(
    req: TaskRequest,
    tool_env: Optional[ToolEnv],
    emit: Callable[[str, str], None],
) -> tuple[str, str]:
    """Everything serge looks up in the project history BEFORE the first turn.

    The agent is told to do this itself and, measured over the whole store,
    does not — see the note above :func:`relore_tool.prior_art`. A lookup worth
    having on every task cannot depend on the model choosing to make it, and
    "before diagnosing" is precisely the ordering an instruction cannot buy.

    Two lookups, returned as two blocks because they carry different things and
    must not be merged: the prior-art note is thread *metadata* serge fetched,
    trusted; the culprit block is a page of text GitHub users wrote, inside
    relore's envelope. Putting the second under the first's "serge already
    searched — trusted" header is exactly the mistake that header exists to
    prevent.

    Emits one ``history`` step covering both, so the task page gets a single
    row with an outcome rather than a log line buried in whichever phase
    happened to be open — it is a thing serge does between the reproduce gate
    and the agent loop.

    Fail-soft and non-gating throughout: a relore that is down, slow or
    unindexed costs this task nothing but the empty string.
    """
    if tool_env is None or tool_env.relore is None:
        return "", ""
    node_ids = _failing_node_ids(req)
    culprit = culprit_pr_number(req.context or "")
    if not node_ids and culprit is None:
        return "", ""
    emit("step", "history")
    return (
        _prior_art_note(tool_env, emit, node_ids=node_ids),
        _culprit_thread_note(tool_env, emit, number=culprit),
    )


def _prior_art_note(
    tool_env: ToolEnv,
    emit: Callable[[str, str], None],
    *,
    node_ids: list[str],
) -> str:
    """The failing tests' earlier threads. Step already emitted by the caller."""
    if not node_ids:
        return ""
    try:
        result = prior_art(tool_env.relore, node_ids=node_ids)
    except Exception:
        log.debug("prior-art lookup failed; continuing", exc_info=True)
        emit("log", "Project history: lookup failed; continuing without it.")
        return ""
    if not result.searched_anything:
        return ""

    if result.threads:
        emit(
            "log",
            "Project history: "
            + ", ".join(f"#{t.number}" for t in result.threads)
            + " already discuss these tests",
        )
    elif result.ran:
        emit(
            "log",
            "Project history: no earlier thread matched "
            f"{'; '.join(result.ran)} — the model is told so, so it does not "
            "re-run the search.",
        )
    # Reported separately and never as "found nothing": a query relore did not
    # answer leaves the history unread, and the note tells the model to try it.
    if result.failed:
        emit(
            "log",
            "Project history: relore did not answer "
            f"{'; '.join(result.failed)} — left for the model to retry.",
        )
    return prior_art_note(result)


def _culprit_thread_note(
    tool_env: ToolEnv,
    emit: Callable[[str, str], None],
    *,
    number: Optional[int],
) -> str:
    """The blamed pull request's discussion, on regression clusters only.

    ``number`` is ``None`` on every other group shape, which is most of them —
    only a bisect-attributed cluster knows which change broke the tests. See
    the note above :func:`relore_tool.culprit_thread` for why this is serge's
    call and not the model's.
    """
    if number is None:
        return ""
    try:
        result = culprit_thread(tool_env.relore, number=number)
    except Exception:
        log.debug("culprit-thread lookup failed; continuing", exc_info=True)
        emit("log", f"Project history: lookup failed for culprit PR #{number}.")
        return ""
    if result.page:
        emit(
            "log",
            f"Project history: culprit PR #{number} — its discussion is in the "
            "prompt, so the model does not reconstruct it from the tree.",
        )
    else:
        # Same distinction the prior-art note draws, and the same wording so the
        # task page's `history` rules read it as a warning: a thread relore
        # could not serve is unread, not absent.
        emit(
            "log",
            f"Project history: relore did not answer for culprit PR #{number} "
            f"({result.error}) — left for the model to retry.",
        )
    return culprit_thread_note(result)


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
    tool_env = _make_tool_env(cfg, helper_tools=[], repo_full_name=req.repo_full_name)
    # Before the prompt is built and before the first turn — the whole point.
    history_note, culprit_note = _history_notes(req, tool_env, _emit)

    llm = ChatCompletionClient(
        cfg.llm_api_base,
        cfg.llm_api_key,
        cfg.llm_model,
        bill_to=cfg.llm_bill_to,
        stream=cfg.llm_stream,
        compressor=MessageCompressor.from_env(),
    )
    conventions = _read_repo_conventions(cfg, checkout)
    system_prompt = build_task_system_prompt(
        conventions,
        cfg.task_normalize_guidance,
        tools_enabled=tool_env is not None,
        history_tools=tool_env is not None and tool_env.relore is not None,
    )
    user_prompt = build_task_user_prompt(
        repo_full_name=req.repo_full_name,
        base_ref=req.base_ref,
        instruction=req.instruction,
        context=req.context,
        existing_diff=existing_diff,
        history_note=history_note,
        culprit_note=culprit_note,
    )

    # This prefix is resent on EVERY turn, so log its breakdown once per task —
    # it is the only way the dominant component of a session's bill is a fact in
    # the job row rather than something reached by subtraction.
    #
    # It is no longer the dominant component, and this comment used to say it
    # was. Measured on mm_grounding_dino (2026-08-31, 51 turns), turn 1 cost
    # 25,335 input tokens and the conversation grew only ~510 a turn, so ~63% of
    # the 2M budget went on re-sending this prefix. Re-measured across all 8
    # tasks that hit the cap in the 30 days to 2026-09-08, the prefix is
    # 16.7k–20.2k chars (~5k tokens) and the transcript is what dominates: on
    # d9d4b022 the prefix was 14% of a 2,094,215-token session and 62
    # accumulated tool results were the other 86%, with the widest single
    # request at 56,400 tokens. So shrinking this buys little now, and the
    # lever is `TOOL_RESULT_WINDOW` (:mod:`reviewbot.transcript`). Keep logging
    # it anyway: that conclusion is only checkable because this line exists.
    _emit(
        "log",
        prompt_prefix_summary(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            conventions=conventions,
            normalize_guidance=cfg.task_normalize_guidance,
            context=req.context,
            instruction=req.instruction,
            existing_diff=existing_diff,
        ),
    )

    # Wire the normalize verification gate into the loop when configured. The
    # closure records whether the accepted answer left the worktree prepared.
    normalize_configured = bool(cfg.task_normalize_command)
    outcome = {"prepared": False}
    # Each accepted final answer gets one brevity call (:mod:`reviewbot.brevity`),
    # so a task that spends its correction budget pays for more than one. They
    # are not loop turns, so they are folded into the token totals below and
    # deliberately not into ``session``'s turn count.
    brevity_chats: list = []
    # Shared across corrections so the pristine-checkout baseline (which costs a
    # full normalizer run) is paid at most once per task.
    baseline_state: dict = {}

    def _parses(content: str) -> bool:
        """Whether the final answer is the JSON object this path requires.

        Handed to the agent loop so a reply that is non-empty but incomplete is
        re-asked instead of parsed. Both places that could have caught the
        2026-09-02 phimoe replay declined to: `_final_answer_defect` had no
        shape for "real prose that is not a complete object", and
        `_validate_patch` swallows the ValueError on purpose because the
        normalizer has nothing to say about it. So the job raised
        `_UnparseableLLMOutput` with zero recovery attempts while two salvage
        retries went unused.
        """
        try:
            _extract_json(content, _TASK_JSON_KEYS)
        except ValueError:
            return False
        return True

    def _validate(chat) -> Optional[str]:
        feedback, prepared = _validate_patch(
            cfg,
            checkout=checkout,
            clone_cache=clone_cache,
            content=chat.content,
            emit=_emit,
            baseline_state=baseline_state,
            llm=llm,
            brevity_chats=brevity_chats,
            report=outcome,
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
        validation_retry_messages=(
            lambda chat, feedback, retry_number: (
                # None asks the loop to APPEND the feedback to the live
                # conversation and keep the tools. An apply rejection is a
                # disagreement about what the file contains: the compact
                # no-tools prompt below cannot settle that, because the model
                # can neither read the file nor see the read it already did, so
                # it re-guesses the context it got wrong. Three prod tasks died
                # that way on 2026-09-16/17, each having re-guessed three times.
                None
                if outcome.get("rejection") == "apply"
                else _task_validation_retry_messages(
                    repo_full_name=req.repo_full_name,
                    base_ref=req.base_ref,
                    instruction=req.instruction,
                    context=req.context,
                    existing_diff=existing_diff,
                    rejected_content=chat.content,
                    feedback=feedback,
                    retry_number=retry_number,
                )
            )
        )
        if normalize_configured
        else None,
        parses=_parses,
    )
    for brevity_chat in brevity_chats:
        # Tokens and latency, but not a turn — same reasoning as the review-side
        # brevity pass in reviewer._condense_review.
        metrics.record_usage(brevity_chat)
    metrics_line = _format_aggregated_metrics(metrics)
    _emit("log", f"LLM done: {metrics_line}")

    try:
        result = _extract_json(chat.content, _TASK_JSON_KEYS)
    except ValueError as exc:
        raise _UnparseableLLMOutput(
            content=chat.content or "",
            finish_reason=chat.finish_reason,
            metrics_line=metrics_line,
            session=session_record(metrics),
            salvage_attempts=metrics.truncation_retries,
        ) from exc

    title = (result.get("title") or "").strip() or (req.title or "serge: automated fix")
    body = (result.get("body") or "").strip()
    patch = result.get("patch")
    if not isinstance(patch, str):
        patch = ""

    raw_edits = result.get("edits")
    if isinstance(raw_edits, list) and raw_edits:
        # An anchored-edit answer leaves this function as a unified diff like
        # every other: the gate already applied the edits and had git write the
        # diff, so `plan.patch` is git's, not the model's. Nothing past here —
        # publish_task, commit_scope, the verify gate — knows the format exists.
        patch = outcome.get("patch") or ""
        if not patch and not normalize_configured:
            # No gate ran (no normalizer configured), so nobody has applied the
            # edits yet. Do it here: without this the answer would silently
            # become "no patch proposed".
            patch, edit_feedback = _apply_anchored_edits(
                checkout=checkout,
                clone_cache=clone_cache,
                raw_edits=raw_edits,
                emit=_emit,
                report=outcome,
            )
            outcome["prepared"] = edit_feedback is None
        if not patch:
            # The correction budget ran out on edits that never applied. Fail
            # loudly, the way an unappliable patch does — a silent "no fix was
            # proposed" would take this whole failure mode out of the error
            # counts that made it visible in the first place.
            error = TaskError(
                "the proposed edits did not apply: "
                f"{outcome.get('edits_error') or 'no anchor matched'}",
                status_code=422,
            )
            error.session = session_record(metrics)
            raise error

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
        session=session_record(metrics),
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


def _select_failure_block(req: TaskRequest, plan: TaskPlan) -> list[str]:
    """The single failure block most relevant to the produced patch (scored by
    word overlap with the patch/title/body). Shared by the PR-body decorator and
    the GPU verify gate, so both target the same group's node-ids.

    Scoring reads the bullet *and* the CI traceback rendered under it. The
    traceback is not part of the block (:func:`_failure_blocks` stops at the
    fence, so the verify targets stay the bullets alone), but it is where the
    identifiers a patch echoes actually live — without it every fenced bullet
    scores zero and the group's first failure wins by position."""
    blocks = _failure_blocks(req.context)
    if not blocks:
        return []
    haystack = f"{plan.title}\n{plan.body}\n{plan.patch}".lower()
    tracebacks = pr_links.failure_tracebacks(req.context)

    def _score(block: list[str]) -> int:
        node_ids, _, _ = extract_verify_targets(block, "")
        text = "\n".join(block + [tracebacks.get(nid, "") for nid in node_ids])
        words = set(re.findall(r"[a-z0-9_]{5,}", text.lower()))
        return sum(1 for word in words if word in haystack)

    return max(blocks, key=_score)


def _selected_failure_context(req: TaskRequest, plan: TaskPlan) -> str:
    """The "Original CI failure" header of the PR body: which group this fixes,
    the failing test, the error CI actually raised, and where to watch it.

    The bullet alone says a test failed; the traceback (parsed separately, since
    :func:`_failure_blocks` stops at the fence) says *how*, which is what a
    reviewer needs to judge the patch without opening the CI log."""
    heading = ""
    for line in req.context.splitlines():
        if _CANDIDATE_HEADING_RE.match(line):
            heading = line.removeprefix("## Serge candidate failure group ").strip()
            # Drop the dispatcher's "3/5: " counter — it means nothing to a
            # reviewer, who only sees this one group's PR.
            heading = re.sub(r"^\d+/\d+:\s*", "", heading)
            break

    selected = _select_failure_block(req, plan)
    if not selected:
        return ""

    lines = ["## Original CI failure", ""]
    if heading:
        lines.append(f"- Failure group: `{heading}`")
    lines.extend(selected)

    node_ids, _, _ = extract_verify_targets(selected, "")
    error = pr_links.error_section(req.context, node_ids)
    if error:
        lines += ["", error]
    links = pr_links.test_links_section(req.test_links, node_ids)
    if links:
        lines += ["", links]
    return "\n".join(lines)


def _related_issues_section(
    gh: Optional[GitHubClient], req: TaskRequest, plan: TaskPlan
) -> str:
    """Prior reports of this failure, looked up when the PR is opened so the
    reviewer sees them at their freshest. Never blocks the PR: no client, no
    node-ids or a failing search all render nothing."""
    if gh is None:
        return ""
    selected = _select_failure_block(req, plan)
    if not selected:
        return ""
    node_ids, model, _ = extract_verify_targets(selected, "")
    related = pr_links.find_related_issues(
        gh, req.owner, req.repo, node_ids=node_ids, model=model
    )
    return pr_links.related_section(related, node_ids)


def _decorate_body(
    cfg: Config,
    plan: TaskPlan,
    req: TaskRequest,
    verification_footer: str = "",
    gh: Optional[GitHubClient] = None,
) -> str:
    body = plan.body or "Automated fix produced by serge."
    failure_context = _selected_failure_context(req, plan)
    if failure_context:
        body = f"{failure_context}\n\n{body}"
    # The GPU-verification section belongs with the change it vouches for —
    # directly under the fix, ABOVE the "produced automatically" disclaimer and
    # the serge footer (not tacked on after them).
    if verification_footer:
        body += f"\n{verification_footer}"
    related = _related_issues_section(gh, req, plan)
    if related:
        body += f"\n\n{related}"
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


def _verify_unjudged_message(
    outcome: VerifyOutcome,
    branch: str,
    classification: Optional[PatchClassification] = None,
) -> str:
    """What to report when the gate never ran. Names the branch, because that
    branch is the only surviving artifact of the work and nothing else points
    at it.

    Carries the expectation warning too. This message becomes the Reason cell of
    the triage recap, which then invites a human to "re-run verification against
    it" — and for a patch that rewrote its own assertion, that later verdict will
    come back `fixed` and mean nothing. Whoever picks the branch up has to be
    told before they read the result, not after."""
    where = f" ({outcome.run_url})" if outcome.run_url else ""
    msg = (
        f"A candidate patch was committed to `{branch}` but GPU verification "
        f"could not run (`{outcome.verdict}`){where}, so it is UNVERIFIED and no "
        "PR was opened. The branch is kept: re-run verification against it, or "
        "delete it if the group has moved on."
    )
    if classification is not None and classification.expectation_only:
        msg += (
            " CAUTION: " + classification.reason() + " Re-running the gate on it "
            "cannot confirm it — the tests would be re-run after the patch "
            "rewrote what they assert, so they pass by construction."
        )
    return msg


def _verify_failure_message(outcome: VerifyOutcome) -> str:
    """Human/next-run explanation when GPU verify does not confirm the fix."""
    reasons = {
        "not_fixed": "the patch did not turn the targeted tests green",
        "already_passing": (
            "a targeted test was NOT failing on the pre-patch baseline "
            "(self-healed or flaky) — nothing to fix"
        ),
        "broke_others": "the patch broke other tests in the model's suite",
        "error": "a targeted test could not be verified (missing/skipped/collection error)",
        "no_targets": "no test node-ids were found in the failure group",
        "dispatch_failed": "the GPU verify workflow could not be dispatched",
        "timeout": "the GPU verify workflow did not finish in time",
        "no_result": "the GPU verify workflow produced no result artifact",
    }
    reason = reasons.get(outcome.verdict, outcome.verdict)
    lines = [
        f"GPU verification did not confirm the fix (`{outcome.verdict}`): {reason}.",
        "No PR was opened.",
    ]
    if outcome.run_url:
        lines.append(f"Verify run: {outcome.run_url}")
    if outcome.detail:
        lines.append(outcome.detail)
    for nodeid, tb in list(outcome.tracebacks.items())[:5]:
        lines.append(f"\n### {nodeid}\n```\n{(tb or '')[-1500:]}\n```")
    return "\n".join(lines)


# The least loop time a fresh round is worth starting with, on top of the tail
# it has to leave behind. Below this the round cannot investigate anything — the
# wall-clock guard ends it on iteration 1 and the model answers with no tools —
# while the round already in hand is on a branch and would be replaced by it.
_MIN_ROUND_LOOP_SECONDS = 600


def round_reserve(cfg: Config) -> int:
    """What a *fresh* agent cycle needs before it is worth starting: the tail it
    must leave behind, plus enough loop time to be worth more than the result it
    would replace. Used for both a GPU-verify retry round and the next
    candidate group."""
    return budget.reserve_for(cfg) + _MIN_ROUND_LOOP_SECONDS


def effective_poll_timeout(configured: int, *, now: Optional[float] = None) -> int:
    """The verify poll timeout, clamped to the budget the runner has LEFT.

    `TASK_RUNNER_TIMEOUT` is enforced from outside — in kubernetes it is the
    Job's activeDeadlineSeconds — so a poll that outlives the runner is not a
    slow poll, it is a lost job: no code in this process gets to run, and the
    branch is left with no PR and no verdict.

    Observed 2026-08-31 on job `b228e033`: LLM work finished 78 minutes into a
    120-minute budget, verify was dispatched, and the poll believed it had its
    full 60 minutes when 42 remained. 5 of 79 serge fix branches (6%) are
    orphaned patches with no PR, which is what this shape leaves behind.

    Returns ``configured`` unchanged when the budget is not armed (an unbounded
    or locally-run task). Never returns less than 0; a caller that gets 0 should
    treat the gate as unavailable rather than poll forever.
    """
    return budget.clamp(configured, now=now)


def _make_verify_gate(
    cfg: Config,
    gh: GitHubClient,
    req: TaskRequest,
    plan: TaskPlan,
    job_id: str,
    emit_fn: Callable[[str, str], None],
) -> Optional[Callable[[str, str], VerifyOutcome]]:
    """Build the pre-PR GPU verify gate, or ``None`` when disabled.

    Returns a callback ``(base_sha, candidate_sha) -> VerifyOutcome``: it runs
    the targeted tests on GPU (baseline must be red, candidate should be green).
    The caller (:func:`_commit_changes`) opens the PR / keeps the follow-up
    commit only when ``outcome.is_fixed``. Wired for both ``new_pr`` and
    ``existing_pr``."""
    if not cfg.verify_on_gpu:
        return None
    block = _select_failure_block(req, plan)
    # The gate re-runs the group's own node-ids, which is full coverage for a
    # patch confined to those tests and none at all for a patch that edits the
    # scaffolding they share with the rest of the file. Ask for the collateral
    # suite in the second case even when it is globally off.
    node_ids, _, _ = extract_verify_targets(block, cfg.verify_machine_type)
    collateral = cfg.verify_run_collateral or patch_needs_collateral(
        plan.patch, node_ids
    )
    if collateral and not cfg.verify_run_collateral:
        emit_fn(
            "log",
            "GPU verify: the patch touches shared test scaffolding, so the "
            "collateral suite runs too (the targeted node-ids alone would not "
            "see a sibling test break).",
        )

    def _gate(base_sha: str, candidate_sha: str) -> VerifyOutcome:
        outcome = run_gpu_verify(
            gh,
            owner=req.owner,
            repo=req.repo,
            base_sha=base_sha,
            commit_sha=candidate_sha,
            block_lines=block,
            correlation_id=job_id,
            workflow_file=cfg.verify_workflow_file,
            ref=cfg.verify_ref,
            default_machine_type=cfg.verify_machine_type,
            run_collateral=collateral,
            transformersci_ref=cfg.verify_transformersci_ref,
            poll_timeout=effective_poll_timeout(cfg.verify_poll_timeout),
            poll_interval=cfg.verify_poll_interval,
            emit=emit_fn,
        )
        if outcome.is_fixed:
            emit_fn("log", f"GPU verify: fixed ✓ ({outcome.run_url or 'run'})")
        else:
            emit_fn(
                "log",
                f"GPU verify: {outcome.verdict} — not accepting ({outcome.run_url or ''})",
            )
        return outcome

    return _gate


def _verification_footer(
    verify_run_url: Optional[str],
    reproduce_run_url: Optional[str],
    runs: Optional[int] = None,
    classification: Optional[PatchClassification] = None,
) -> str:
    """Provenance appended to the PR body: serge ran the targeted ``@slow`` tests
    on GPU and opened this PR only after they passed with the patch. Links the
    verify run (green with the patch) and, when reproduce-first ran, the run that
    confirmed the failure at the base commit (red before). ``runs`` (from the
    verify verdict's ``runs`` field) is how many times each targeted test was run
    per tree — surfaced so reviewers know the result was checked for flakiness."""
    if not verify_run_url and not reproduce_run_url:
        return ""
    flakiness = (
        f" Each targeted test was run {runs}× on both the pre-patch and patched "
        "trees to rule out flakiness — the result held on every run."
        if runs and runs > 1
        else ""
    )
    # An expectation-only patch rewrote the assertion the gate then re-ran, so
    # the green run is circular and cannot be offered as confidence. The runs
    # are still linked — a reviewer needs them to see what the model produced —
    # but the heading and the claim are replaced, not decorated. Leaving
    # "✅ Verified on GPU" on such a PR is exactly the reasoning the review
    # prompt tells a human reviewer to reject (prompts.py, CHANGED EXPECTATIONS).
    if classification is not None and classification.expectation_only:
        lines = [
            "",
            "---",
            "### ⚠️ Not verified — this patch changed the expectation",
            classification.reason(),
            "",
            "The GPU run below is **not evidence that this change is correct**: "
            "the tests were re-run after the patch rewrote what they assert, so "
            "they pass by construction. Judge the new value on its merits.",
        ]
    else:
        lines = [
            "",
            "---",
            "### ✅ Verified on GPU",
            "serge ran the targeted `@slow` test(s) on a GPU runner and opened this PR "
            "only after they passed with this patch." + flakiness,
        ]
    if verify_run_url:
        lines.append(f"- **Passes with this patch:** {verify_run_url}")
    if reproduce_run_url:
        lines.append(f"- **Failed before the fix (base commit):** {reproduce_run_url}")
    return "\n".join(lines)


def _commit_changes(
    cfg: Config,
    gh: GitHubClient,
    req: TaskRequest,
    *,
    changes: list[FileChange],
    plan: TaskPlan,
    job_id: str,
    emit_fn: Callable[[str, str], None],
    verify: Optional[Callable[[str, str], VerifyOutcome]] = None,
) -> TaskResult:
    """Commit a set of worktree changes via the Git Data API and open/update
    a PR. ``changes`` must be non-empty. Never pushes to a non-serge branch.

    When ``verify`` is given (opt-in GPU gate), the candidate commit is created
    and made fetchable, the tests are run on GPU, and the PR is opened / the
    follow-up commit is kept only on a ``fixed`` verdict; otherwise the branch is
    torn down (new_pr) or rolled back to its previous head (existing_pr)."""
    changed_files = [c.path for c in changes]
    emit_fn(
        "log",
        f"Change touches {len(changed_files)} file(s): {', '.join(changed_files)}",
    )
    # Decided from the committed file list, not just the proposed diff — the
    # normalizer can add a regenerated source file the patch never mentions.
    classification = classify_patch(plan.patch, changed_files=changed_files)
    if classification.expectation_only:
        emit_fn(
            "log",
            "Expectation-only patch: the GPU verdict is circular here and will "
            "not be reported as verification."
            + (
                f" Degenerate new value(s): {', '.join(classification.degenerate_values)}."
                if classification.degenerate_values
                else ""
            ),
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
            message=plan.title,
            tree_sha=tree_sha,
            parents=[parent_sha],
        )
        # Move the branch forward so the candidate commit is fetchable by the
        # verify workflow, then gate. On a non-fixed verdict, roll the branch
        # back to its previous head so the follow-up commit is undone.
        gh.update_ref(owner, repo, f"heads/{head_branch}", commit_sha)
        emit_fn("log", f"Pushed commit {commit_sha[:8]} to {head_branch}")
        # Recorded on BOTH branches below. A verdict that only survives failure
        # is not a metric: every published job stored `verify_verdict=None`
        # while the ones that produced no PR carried it, so the field read as
        # "the gate never ran" exactly where it had run and passed.
        verdict: Optional[str] = None
        if verify is not None:
            outcome = verify(parent_sha, commit_sha)
            verdict = outcome.verdict
            if not outcome.is_fixed:
                try:
                    gh.update_ref(
                        owner, repo, f"heads/{head_branch}", parent_sha, force=True
                    )
                except Exception:  # noqa: BLE001 — best-effort rollback
                    emit_fn(
                        "log", f"Could not roll {head_branch} back to {parent_sha[:8]}"
                    )
                emit_fn(
                    "log", "GPU verify did not confirm the fix; follow-up reverted."
                )
                return TaskResult(
                    mode=req.mode,
                    no_change=True,
                    pr_number=req.pr_number,
                    commit_sha=commit_sha,
                    message=_verify_failure_message(outcome),
                    verify_verdict=outcome.verdict,
                    verify_tracebacks=outcome.tracebacks,
                    expectation_only=classification.expectation_only,
                    expectation_note=classification.reason(),
                )
        return TaskResult(
            mode=req.mode,
            pr_number=req.pr_number,
            branch=head_branch,
            commit_sha=commit_sha,
            changed_files=changed_files,
            message=f"Pushed follow-up commit to PR #{req.pr_number}.",
            url=f"https://github.com/{owner}/{repo}/pull/{req.pr_number}",
            verify_verdict=verdict,
            expectation_only=classification.expectation_only,
            expectation_note=classification.reason(),
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
    emit_fn("log", f"Created branch {branch} at {commit_sha[:8]}")

    # GPU verify gate (opt-in): run the targeted tests on the candidate before
    # opening the PR. `parent_sha` is the baseline-red reference. On a non-`fixed`
    # verdict, tear the branch down and return without a PR.
    verify_run_url: Optional[str] = None
    verify_runs: Optional[int] = None
    verdict = None
    if verify is not None:
        outcome = verify(parent_sha, commit_sha)
        verdict = outcome.verdict
        if gate_did_not_run(verdict):
            # The gate never judged this patch — no runner picked the job up, the
            # dispatch failed, or no artifact came back. That says nothing about
            # the candidate, so deleting the branch throws away work for a reason
            # that is not about the work. KEEP it, and report it: the branch name
            # travels in TaskResult.to_json(), so the triage reconciler can link
            # it from the tracking issue and say verification could not run.
            #
            # No PR: an unverified patch must not land in the review queue looking
            # like a verified one. 5 of 79 serge fix branches were orphaned this
            # way before this existed — committed patches with no PR and no
            # verdict, invisible unless someone listed the refs.
            emit_fn(
                "log",
                f"GPU verify could not run ({verdict}); keeping branch {branch} "
                "unverified and opening no PR.",
            )
            return TaskResult(
                mode=req.mode,
                no_change=True,
                branch=branch,
                commit_sha=commit_sha,
                message=_verify_unjudged_message(outcome, branch, classification),
                verify_verdict=verdict,
                verify_tracebacks=outcome.tracebacks,
                # The branch survives and the recap invites someone to finish it,
                # so this is the ONE outcome where the flag matters most: whoever
                # picks the branch up has to be told the patch rewrote its own
                # assertion, or they will read a later `fixed` as evidence.
                # Observed 2026-09-01 on job 9a266db2, whose kept patch rewrote
                # EXPECTED_TEXT_COMPLETION and was recorded expectation_only=False.
                expectation_only=classification.expectation_only,
                expectation_note=classification.reason(),
            )
        if not outcome.is_fixed:
            try:
                gh.delete_ref(owner, repo, f"heads/{branch}")
            except Exception:  # noqa: BLE001 — cleanup is best-effort
                emit_fn("log", f"Could not delete branch {branch} after failed verify")
            emit_fn("log", "GPU verify did not confirm the fix; no PR opened.")
            return TaskResult(
                mode=req.mode,
                no_change=True,
                commit_sha=commit_sha,
                message=_verify_failure_message(outcome),
                verify_verdict=outcome.verdict,
                verify_tracebacks=outcome.tracebacks,
                expectation_only=classification.expectation_only,
                expectation_note=classification.reason(),
            )
        verify_run_url = outcome.run_url
        verify_runs = (outcome.result or {}).get("runs")

    # Open as a draft, then immediately mark ready-for-review. The
    # draft->ready transition is what fires the `ready_for_review` webhook that
    # reviewer-assignment workflows (e.g. transformers' assign-reviewers.yml)
    # listen for; a PR born non-draft never emits that event and gets no
    # reviewers routed to it.
    pr = gh.create_pull_request(
        owner,
        repo,
        title=plan.title,
        head=branch,
        base=req.base_ref,
        body=_decorate_body(
            cfg,
            plan,
            req,
            verification_footer=_verification_footer(
                verify_run_url,
                req.reproduce_run_url,
                runs=verify_runs,
                classification=classification,
            ),
            gh=gh,
        ),
        draft=True,
    )
    gh.mark_pull_request_ready(pr["node_id"])
    emit_fn(
        "log", f"Opened PR #{pr.get('number')} (draft->ready): {pr.get('html_url')}"
    )
    # Reviewers the dispatcher named — for the integration-failure triage, the
    # author of the commit its bisect blamed for the regression. They are the one
    # person who knows what the change was meant to do, so a fix PR that touches
    # it should land in their queue rather than waiting for someone to notice.
    #
    # New PRs only: an existing_pr follow-up pushes onto a branch whose review is
    # already requested, and re-requesting resets a review the author may have
    # already given. Requested AFTER draft->ready, so this adds to whatever
    # `assign-reviewers.yml` routes rather than racing it. Fail-soft (see
    # `GitHubClient.request_reviewers`) — never lose a published PR over it.
    if req.reviewers and pr.get("number"):
        requested = gh.request_reviewers(owner, repo, pr["number"], req.reviewers)
        if requested:
            emit_fn("log", f"Requested review from {', '.join(requested)}")
    if req.slack_notify_pr_created:
        post_task_pr_created_notification(
            token=cfg.slack_bot_token,
            channel=req.slack_channel or cfg.slack_report_channel,
            repo_full_name=req.repo_full_name,
            pr_number=pr.get("number"),
            pr_url=pr.get("html_url"),
            title=plan.title,
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
        verify_verdict=verdict,
        expectation_only=classification.expectation_only,
        expectation_note=classification.reason(),
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
    and left a clean checkout) and, if a normalizer *is* configured, re-run it
    so regenerated files ride along — refusing to open a PR it rejects rather
    than committing a raw, repo-inconsistent patch."""

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

        # When a normalizer is configured, the committed tree must satisfy it —
        # most consequentially transformers' modular/generated-file coupling,
        # where editing a ``modular_*.py`` must regenerate its ``modeling_*.py``.
        # The in-loop gate (:func:`_validate_patch`) enforces that, but when its
        # correction budget is exhausted prepare_task resets the worktree and we
        # land here with only the raw LLM patch. Re-run the normalizer so any
        # regenerated files ride along in the commit, and refuse rather than open
        # a PR the repo's own consistency CI would immediately reject.
        if cfg.task_normalize_command:
            returncode, output = _run_repo_normalizer(cfg, checkout, _emit)
            if returncode not in (None, 0):
                clone_cache.reset_worktree(checkout)
                cmd = " ".join(cfg.task_normalize_command)
                _emit(
                    "normalize_error",
                    f"Normalizer failed (exit {returncode}) for `{cmd}`:\n{output}",
                )
                # Same distinction as the in-loop gate: if the normalizer also
                # rejects the pristine checkout this is an operator problem,
                # and reporting it as "the budget was exhausted" hides that.
                _check_normalizer_baseline(
                    cfg,
                    checkout=checkout,
                    clone_cache=clone_cache,
                    emit=_emit,
                    state={},
                )
                _emit("log", "Normalizer rejected the patch; not opening a PR.")
                return TaskResult(
                    mode=req.mode,
                    no_change=True,
                    message=(
                        "The proposed patch does not pass the repository's "
                        f"normalizer (exit {returncode}), so no PR was opened — "
                        "the in-loop correction budget was exhausted before a "
                        f"clean patch was found:\n\n{output}"
                    ),
                )

    clone_cache.stage_all(checkout)
    changes = clone_cache.collect_changes(checkout)
    if not changes:
        return TaskResult(
            mode=req.mode,
            no_change=True,
            message="Patch applied but produced no file changes.",
        )

    # The normalizer fixes the whole worktree, so a base that is not itself
    # normalizer-clean contributes regenerated files the fix never touched (one
    # prod task patched 1 file and committed 32). Drop those — validation above
    # still ran repo-wide; only the commit is scoped.
    if getattr(cfg, "task_scope_commit_to_patch", False):
        keep, dropped = scope_paths(
            [c.path for c in changes],
            plan.patch,
            always_include=getattr(cfg, "task_commit_always_include", None),
        )
        if dropped:
            kept = set(keep)
            changes = [c for c in changes if c.path in kept]
            _emit(
                "log",
                f"Left {len(dropped)} unrelated file(s) out of the commit — "
                f"regenerated from a stale base, not by this fix: "
                f"{describe_dropped(dropped)}",
            )

    return _commit_changes(
        cfg,
        gh,
        req,
        changes=changes,
        plan=plan,
        job_id=job_id,
        emit_fn=_emit,
        verify=_make_verify_gate(cfg, gh, req, plan, job_id, _emit),
    )


# The smallest per-traceback slice worth sending. Below this a traceback is
# noise in both directions, so we would rather show fewer tests in full.
MIN_TB_CHARS = 6000
# How much of the HEAD of each traceback is protected from the middle cut.
# Sized from real reproduce artifacts (see ``_clip_tb``): the assertion summary
# pytest prepends, the failing source line and the ``E`` block together run to
# ~2.4k, so 4k keeps them whole with margin.
TB_HEAD_KEEP_CHARS = 4000


def _clip_tb(tb: Optional[str], max_chars: int) -> str:
    """Clip a traceback to ``max_chars`` from the MIDDLE, keeping both ends.

    A traceback carries the load-bearing ``E`` block at *either* end depending
    on how pytest laid it out, so no single-ended cut can serve both shapes.
    Measured on the 2026-08-31 reproduce artifacts:

    * ``mm_grounding_dino`` — pytest prepends the assertion summary, so the
      ``E`` block sits at offset 1,515 of 25,586 and 2,116 of 45,905: at the
      **head**;
    * ``cwm`` (36,497) and ``kimi_k25`` (117,073 / 118,454) — the test source
      contains a huge inline ``Expectations({...})`` literal, pushing the ``E``
      block to 404 and ~9,640 chars from the **end**.

    The old tail-only cut kept the last ``max_chars`` and therefore deleted the
    entire assertion diff, the ``expected_*`` literal and the actual-vs-expected
    comparison for the first shape — everything the model needs to rewrite a
    stale expectation, leaving only the ``--showlocals`` tensor dump. Job
    ``d2d9c129`` returned ``no_fix`` saying exactly that, and was right to:
    "the actual decoder logits ... is in the elided portion of the traceback,
    so I cannot record a plausible, verified expectation."
    """
    text = tb or ""
    if len(text) <= max_chars:
        return text
    head_len = min(TB_HEAD_KEEP_CHARS, max_chars // 3)
    tail_len = max_chars - head_len
    omitted = len(text) - max_chars
    return (
        "%s\n…(%d chars omitted from the middle of the traceback; the end follows)…\n%s"
        % (
            text[:head_len],
            omitted,
            text[len(text) - tail_len :],
        )
    )


def _tb_budget(count: int, block_chars: int) -> int:
    """Per-traceback slice of the whole-block budget.

    ``block_chars`` bounds the reproduce/verify block as a whole, because that
    is what has to fit inside ``prompts.CONTEXT_TAIL_RESERVE_CHARS``. The old
    per-traceback cap could not: the formatters take up to 5 tracebacks, so a
    12,000 cap allowed a 60,000-char block against a 16,000 reserve, and a
    40,000 cap produced the 121,417-char block observed on job ``f50c5e85``.
    """
    return max(MIN_TB_CHARS, block_chars // max(1, count))


def _format_verify_feedback(result: TaskResult, max_chars: int = 32000) -> str:
    """Markdown feedback for the next LLM round: the failures the previous
    candidate's GPU verify still produced. ``max_chars`` budgets the block as a
    whole and is divided across the tracebacks it shows."""
    lines = [
        "## Your previous patch did NOT fix the tests (GPU verification)",
        "",
        f"A previous candidate was run on GPU; the verdict was "
        f"`{result.verify_verdict}`. With that patch applied, the targeted tests "
        "below still fail. Produce a NEW patch that makes them pass — do not "
        "repeat the same change; use the tracebacks to find the real cause.",
        "",
    ]
    shown = list(result.verify_tracebacks.items())[:5]
    per_tb = _tb_budget(len(shown), max_chars)
    for nodeid, tb in shown:
        lines.append(f"### {nodeid}")
        lines.append("```")
        lines.append(_clip_tb(tb, per_tb))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()


def _with_verify_feedback(req: TaskRequest, feedback: str) -> TaskRequest:
    """A copy of ``req`` with GPU-verify feedback appended to its context, so the
    next LLM round sees the failures its previous patch still produced."""
    return dataclasses.replace(req, context=f"{req.context}\n\n{feedback}")


# ---------------------------------------------------------------------------
# Reproduce-first: confirm the failure on GPU BEFORE investigating, classify it,
# and seed the LLM with the real traceback. See docs/plans/serge-reproduce-first.
# ---------------------------------------------------------------------------
def _candidate_failure_lines(req: TaskRequest) -> list[str]:
    """All failure-bullet lines for this candidate group (flattened), used to
    parse the node-ids to reproduce. The candidate context is already scoped to
    one group by :func:`task_candidate_requests`, so this is that group's tests."""
    lines: list[str] = []
    for block in _failure_blocks(req.context):
        lines.extend(block)
    return lines


def _reproduce_bail_message(outcome: VerifyOutcome) -> str:
    lines = [
        f"GPU reproduce: the targeted tests did NOT fail at the base commit "
        f"(`{outcome.verdict}`) — the failure is stale, already fixed, flaky, or "
        "environment-specific. Skipping this group; no investigation, no PR.",
    ]
    if outcome.run_url:
        lines.append(f"Reproduce run: {outcome.run_url}")
    if outcome.detail:
        lines.append(outcome.detail)
    return "\n".join(lines)


def _environment_bail_message(
    outcome: VerifyOutcome, classification: ClassifyResult
) -> str:
    """Why we stopped after classifying, with no investigation and no PR.

    Written for the nightly triage recap: ``_distill_outcome`` on the dashboard
    side takes the FIRST line as the human reason, so the reason leads."""
    lines = [
        "GPU reproduce + classify: this is an ENVIRONMENT issue "
        f"({classification.reason or 'per the traceback'}) — no source patch can "
        "fix it, so no investigation and no PR.",
        "",
        "The tests do fail on GPU at the base commit, but the traceback points at "
        "the machine or the installed environment (device memory, a missing or "
        "incompatible dependency, a driver mismatch, a checkpoint gone from the "
        "Hub) rather than at the library or the test. This needs a human.",
    ]
    if outcome.run_url:
        lines.append(f"Reproduce run: {outcome.run_url}")
    return "\n".join(lines)


def _format_reproduce_feedback(
    outcome: VerifyOutcome,
    classification: ClassifyResult,
    max_chars: int = 32000,
) -> str:
    """Authoritative reproduction context seeded into the LLM prompt. Supersedes
    any inbound 'do not run the test suite' note: serge already ran the tests on
    GPU and the real tracebacks are below."""
    lines = [
        "## The targeted test(s) were REPRODUCED on GPU (authoritative)",
        "",
        "serge ran these `@slow` tests on a GPU runner at the base commit before "
        "asking you to fix them; they FAIL as shown. These tracebacks are the "
        "real, current failures — base your fix on them. You do NOT need to run "
        "the tests yourself (you cannot); ignore any earlier note about CI "
        "verifying later.",
        "",
    ]
    if classification.label == "test_issue":
        lines += [
            f"**Triage: this looks like a TEST/expectations issue** "
            f"({classification.reason or 'per the traceback'}). Prefer correcting "
            "the test's expected values/tolerances over changing model code — "
            "unless the traceback clearly shows a library bug.",
            "",
        ]
    elif classification.label == "product_issue":
        lines += [
            f"**Triage: this looks like a genuine library/model bug** "
            f"({classification.reason or 'per the traceback'}). Fix it at the "
            "source — edit the `modular_*.py` source (not the generated "
            "`modeling_*.py`), which is regenerated by `make fix-repo`.",
            "",
        ]
    elif classification.label == ENVIRONMENT_ISSUE:
        # Only reachable with CLASSIFY_BAIL_ON_ENVIRONMENT=0 — otherwise
        # `_maybe_reproduce_first` has already bailed. Say what the label means
        # rather than dropping the agent in with no steer.
        lines += [
            f"**Triage: this looks like an ENVIRONMENT/dependency problem** "
            f"({classification.reason or 'per the traceback'}) — device memory, a "
            "missing or incompatible dependency, a driver mismatch, or a "
            "checkpoint gone from the Hub. If that is what the traceback shows, no "
            "source patch fixes it: return an empty patch and say so. Only propose "
            "a change if you can point at concrete code that is actually at fault.",
            "",
        ]
    shown = list(outcome.tracebacks.items())[:5]
    per_tb = _tb_budget(len(shown), max_chars)
    for nodeid, tb in shown:
        lines += [f"### {nodeid}", "```", _clip_tb(tb, per_tb), "```", ""]
    return "\n".join(lines).rstrip()


def _classify_reproduced(
    cfg: Config,
    node_ids: list[str],
    tracebacks: dict[str, str],
    context: str,
    emit: Callable[[str, str], None],
) -> ClassifyResult:
    """Run the cheap product-vs-test classifier over the reproduced traceback.
    Never raises — degrades to ``unclear`` so the flow still investigates."""
    try:
        llm = ChatCompletionClient(
            cfg.llm_api_base,
            cfg.llm_api_key,
            cfg.llm_model,
            bill_to=cfg.llm_bill_to,
            stream=False,
        )
        result = classify_failure(
            llm,
            node_ids=node_ids,
            tracebacks=tracebacks,
            context=context,
            max_tokens=cfg.classify_max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — classification is best-effort
        log.debug("classify failed: %s", exc, exc_info=True)
        return ClassifyResult(UNCLEAR, reason=f"classifier error: {exc}"[:200])
    emit("log", f"GPU reproduce: classified as {result.label} — {result.reason}")
    return result


# OOM shapes the transformers-ci verdict tool reports per node-id (it runs on the
# GPU box, where the whole traceback exists, and applies that repo's `oom_shape`).
# `retention` = a trivial request against a card earlier tests never freed, fixed
# by a tearDown; `load` = an OOM while `from_pretrained` materializes weights,
# fixed by the load pattern. Both are SOURCE patches. `capacity` = one allocation
# that cannot fit however clean the card is, and `unknown` = unparsed.
_FIXABLE_OOM_SHAPES = frozenset({"retention", "load"})


def fixable_oom_shapes(outcome: VerifyOutcome) -> list[str]:
    """The node-ids whose OOM this repo should still try to patch.

    serge's classifier prompt makes ``environment_issue`` cover *any*
    "device/host out-of-memory", so before this every OOM bailed with zero LLM
    turns. transformers-ci's triage has known better since 2026-08-14, when 26
    of 54 persistent OOMs turned out to be retention-shaped and had been
    deferred as "needs capacity" for weeks; its ``env_only_reason`` dispatches a
    group when any test shows the retention or load shape. This mirrors that
    rule exactly, so the two components cannot disagree again.

    They did disagree, and the less-informed one won because it runs later: the
    `phimoe` group (job `c6836491`) is a CUDA OOM on the weight-conversion path
    — ``load``, which triage explicitly dispatches — and serge killed it. Triage
    could not even see it was an OOM: the CI dataset shows only ``RuntimeError:
    We encountered some issues during automatic conversion of the weights``.
    """
    shapes = (outcome.result or {}).get("oom_shapes") or {}
    return sorted(n for n, shape in shapes.items() if shape in _FIXABLE_OOM_SHAPES)


def _maybe_reproduce_first(
    cfg: Config,
    gh: GitHubClient,
    req: TaskRequest,
    job_id: str,
    emit: Callable[[str, str], None],
) -> tuple[Optional[TaskResult], TaskRequest]:
    """Reproduce-first gate. Returns ``(bail_result, req)``:

    - ``(TaskResult, req)`` when the group does NOT reproduce → the caller returns
      it immediately (no LLM, no PR).
    - ``(None, seeded_req)`` otherwise → proceed. ``seeded_req`` carries the real
      traceback + classification when the group reproduced; it is ``req`` unchanged
      when reproduce is disabled or could not run (fail-open — the end verify gate
      still guards the PR)."""
    if not (cfg.verify_on_gpu and cfg.verify_reproduce_first):
        return None, req
    block = _candidate_failure_lines(req)
    node_ids, _model, _machine = extract_verify_targets(block, cfg.verify_machine_type)
    if not node_ids:
        return None, req  # nothing to reproduce → investigate as before

    # Which commit to reproduce AT. For a new PR that is the base branch, but
    # for a follow-up on serge's own existing PR the work already lives on that
    # PR's head — reproducing at `main` asks "is this still broken on main?",
    # which stays true until the fix PR merges, and answers `reproduced` on a
    # branch that already fixes it.
    #
    # That is the stale-group burn, measured: the nightly re-dispatched PR
    # #48414's group on two consecutive nights (`f31aa5d8`, then `9a898563`),
    # each ran a full session — 43 and 60 turns, ~2.5M input tokens between them
    # — and each ended `verify_verdict=already_passing`, because the end gate
    # *does* use the PR head as its baseline. The two gates disagreed about what
    # "the baseline" is, and the expensive one was wrong.
    reproduce_ref = (
        f"heads/{req.head_branch}"
        if req.mode == "existing_pr" and req.head_branch
        else f"heads/{req.base_ref}"
    )
    try:
        base_sha = gh.get_ref_sha(req.owner, req.repo, reproduce_ref)
    except Exception as exc:  # noqa: BLE001 — can't resolve base → fail open
        emit(
            "log",
            f"GPU reproduce: could not resolve {reproduce_ref} ({exc}); proceeding.",
        )
        return None, req

    outcome = run_gpu_reproduce(
        gh,
        owner=req.owner,
        repo=req.repo,
        base_sha=base_sha,
        block_lines=block,
        correlation_id=f"{job_id}-repro",
        workflow_file=cfg.verify_workflow_file,
        ref=cfg.verify_ref,
        default_machine_type=cfg.verify_machine_type,
        transformersci_ref=cfg.verify_transformersci_ref,
        poll_timeout=effective_poll_timeout(cfg.verify_poll_timeout),
        poll_interval=cfg.verify_poll_interval,
        emit=emit,
    )

    if outcome.verdict == NOT_REPRODUCED:
        if req.mode == "existing_pr" and req.head_branch:
            emit(
                "log",
                f"GPU reproduce: the targeted test(s) already PASS on "
                f"{req.head_branch} — the open PR already fixes this group. "
                "Nothing to add; skipping without spending an LLM session.",
            )
        else:
            emit("log", "GPU reproduce: not reproducible — skipping group.")
        return (
            TaskResult(
                mode=req.mode,
                no_change=True,
                pr_number=req.pr_number,
                message=_reproduce_bail_message(outcome),
                verify_verdict=outcome.verdict,
                verify_tracebacks=outcome.tracebacks,
            ),
            req,
        )
    if outcome.verdict != REPRODUCED:
        # Infra error / timeout / no artifact — fail open and investigate.
        emit(
            "log",
            f"GPU reproduce: {outcome.verdict} — proceeding to investigate "
            "(failure unconfirmed).",
        )
        return None, req

    emit("log", f"GPU reproduce: reproduced ✓ ({outcome.run_url or 'run'})")
    classification = _classify_reproduced(
        cfg, node_ids, outcome.tracebacks, req.context, emit
    )
    fixable = fixable_oom_shapes(outcome)
    if classification.is_environment_issue and fixable:
        # Not every OOM is a capacity fact. The verdict tool measured the shape
        # on the GPU box and says this one is patchable, so the flat
        # "any OOM is environment" verdict is overruled rather than obeyed.
        emit(
            "log",
            "GPU reproduce: classified environment, but the OOM shape is "
            f"patchable ({', '.join(fixable)}) — investigating anyway.",
        )
    elif classification.is_environment_issue and cfg.classify_bail_on_environment:
        # The failure is real but not patchable: an investigation could only end
        # `no_fix`, so stop here — one classifier call instead of a full LLM cycle.
        emit("log", "GPU reproduce: environment issue — skipping group (no patch).")
        return (
            TaskResult(
                mode=req.mode,
                no_change=True,
                pr_number=req.pr_number,
                message=_environment_bail_message(outcome, classification),
                verify_verdict=outcome.verdict,
                verify_tracebacks=outcome.tracebacks,
            ),
            req,
        )
    seeded = _with_verify_feedback(
        req,
        _format_reproduce_feedback(
            outcome, classification, max_chars=cfg.reproduce_block_chars
        ),
    )
    seeded = dataclasses.replace(seeded, reproduce_run_url=outcome.run_url)
    return None, seeded


def prepare_and_publish_candidate(
    cfg: Config,
    gh: GitHubClient,
    candidate_req: TaskRequest,
    *,
    checkout: Checkout,
    clone_cache: CloneCache,
    existing_diff: Optional[str],
    job_id: str,
    emit: Callable[[str, str], None],
) -> TaskResult:
    """Prepare (LLM) then publish one candidate group.

    When the opt-in GPU verify gate rejects the patch with a retryable verdict
    (``not_fixed``/``broke_others``), re-prepare with the fresh tracebacks
    appended and re-publish, up to ``cfg.verify_max_rounds`` extra rounds. This
    is the single place the retry loop lives, so both the in-process worker and
    the per-task pod get it by calling this instead of prepare_task+publish_task.

    Publishing exceptions (e.g. a patch that won't apply, ``TaskError`` 422)
    propagate unchanged so the caller's candidate loop can move on.

    When reproduce-first is enabled (``cfg.verify_reproduce_first``), the group's
    targeted tests are first run on GPU at the base commit: if they do not fail
    there the group is skipped without any LLM work, and if they do the real
    traceback + a product/test classification seed the investigation."""
    bail, candidate_req = _maybe_reproduce_first(cfg, gh, candidate_req, job_id, emit)
    if bail is not None:
        return bail

    rounds = cfg.verify_max_rounds if cfg.verify_on_gpu else 0
    req_i = candidate_req
    result: Optional[TaskResult] = None
    # Every round runs a fresh agent loop with its own budget; the job-level
    # bill is their sum, so fold each one in as it completes.
    session: dict[str, Any] = {}
    for attempt in range(rounds + 1):
        if attempt > 0:
            # Start each retry from a pristine worktree.
            clone_cache.reset_worktree(checkout)
        plan = prepare_task(
            cfg,
            req_i,
            checkout=checkout,
            clone_cache=clone_cache,
            existing_diff=existing_diff,
            chunk_callback=emit,
        )
        session = merge_session_records(session, plan.session)
        try:
            result = publish_task(
                cfg,
                gh,
                req_i,
                plan,
                checkout=checkout,
                clone_cache=clone_cache,
                job_id=job_id,
                emit=emit,
            )
        except TaskError as exc:
            # The LLM work is done and paid for even though publishing failed;
            # carry its counters out with the error.
            exc.session = merge_session_records(exc.session, session)
            raise
        result.session = session
        if attempt < rounds and should_retry(result.verify_verdict or ""):
            # A retry round is a whole fresh agent loop. Starting one with no
            # budget left does not produce a better patch — the wall-clock guard
            # would end it on iteration 1 and the model would answer with no
            # tools — and it throws away the round we already have, which is on
            # a branch. Keep that one instead.
            if budget.exhausted(reserve=round_reserve(cfg)):
                emit(
                    "log",
                    f"GPU verify: {result.verify_verdict}, but there is not enough "
                    "runner budget left for another round; keeping this result.",
                )
                return result
            emit(
                "log",
                f"GPU verify: {result.verify_verdict}; re-prompting with tracebacks "
                f"(round {attempt + 2}/{rounds + 1})",
            )
            req_i = _with_verify_feedback(
                candidate_req,
                _format_verify_feedback(result, max_chars=cfg.reproduce_block_chars),
            )
            continue
        return result
    assert result is not None  # rounds >= 0 guarantees at least one iteration
    return result
