"""Close a serge task PR its caller says is no longer needed.

The nightly triage that dispatched a fix is also the only party that knows when
the fix stopped mattering: the target tests pass on ``main`` again (someone else
fixed them), or a newer serge PR covers the same tests. Before this existed
nobody closed those PRs, and on 2026-10-11 19 of the 47 open serge PRs on
huggingface/transformers were exactly that — 14 for tests already passing and
3 duplicates — sitting in the maintainers' review queue.

The caller decides *whether* a PR is stale; serge only checks it is allowed to
close it. Three rules keep that narrow:

* the PR is serge's own: head branch in ``serge/``, head repository the base
  repository (a fork can name a branch ``serge/…`` too), author a bot account;
* the caller's OIDC ``repository`` claim names this repository (checked by the
  route, like every other ``/tasks`` write);
* the reason is one of :data:`CLOSE_REASONS`, and its marker is appended to the
  PR body, so the triage's ledger can tell "closed because stale" apart from
  "closed because a maintainer rejected it" from the PR listing alone.

A PR that is already closed is not an error: two runs racing to close the same
PR both succeed.
"""

from __future__ import annotations

from typing import Any, Protocol

from .tasks import SERGE_BRANCH_NAMESPACE

# reason -> marker appended to the PR body. The triage matches these strings
# (transformersci.agentic.integration_failure_triage), so they are a contract:
# change one only together with its reader.
CLOSE_REASONS = {
    "stale": "<!-- serge-closed:stale -->",
    "superseded": "<!-- serge-closed:superseded -->",
}
MAX_COMMENT_CHARS = 4000
_FOOTER = "\n\n_Closed by serge at the request of the nightly CI triage._"


class ClosePRError(Exception):
    """A refused close; ``status`` and ``detail`` become the HTTP reply."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class PullWrites(Protocol):
    def get_pr(self, owner: str, repo: str, number: int) -> dict: ...

    def post_issue_comment(
        self, owner: str, repo: str, number: int, body: str
    ) -> dict: ...

    def close_pull_request(
        self, owner: str, repo: str, number: int, *, body: str | None = None
    ) -> dict: ...


def parse_close_request(payload: Any) -> tuple[str, str]:
    """Validate the request body; returns ``(reason, comment)``."""
    if not isinstance(payload, dict):
        raise ClosePRError(400, "invalid_json_body")
    reason = str(payload.get("reason") or "").strip()
    if reason not in CLOSE_REASONS:
        raise ClosePRError(
            400, f"reason must be one of {', '.join(sorted(CLOSE_REASONS))}"
        )
    comment = str(payload.get("comment") or "").strip()
    if not comment:
        raise ClosePRError(400, "comment is required")
    if len(comment) > MAX_COMMENT_CHARS:
        raise ClosePRError(400, f"comment exceeds {MAX_COMMENT_CHARS} characters")
    return reason, comment


def close_task_pr(
    client: PullWrites, owner: str, repo: str, number: int, payload: Any
) -> dict:
    """Comment on and close serge's PR ``owner/repo#number``.

    Returns ``{"closed": True, ...}`` when this call closed it, or
    ``{"closed": False, "detail": "not_open"}`` when it already was."""
    reason, comment = parse_close_request(payload)
    pr = client.get_pr(owner, repo, number)
    if pr.get("state") != "open":
        return {"closed": False, "number": number, "detail": "not_open"}

    head = pr.get("head") or {}
    head_ref = head.get("ref") or ""
    if not head_ref.startswith(SERGE_BRANCH_NAMESPACE):
        raise ClosePRError(403, f"not a serge branch: '{head_ref}'")
    head_repo = ((head.get("repo") or {}).get("full_name") or "").lower()
    if head_repo != f"{owner}/{repo}".lower():
        raise ClosePRError(403, "head branch is not in the base repository")
    if ((pr.get("user") or {}).get("type") or "") != "Bot":
        raise ClosePRError(403, "PR was not opened by a bot account")

    client.post_issue_comment(owner, repo, number, comment + _FOOTER)
    marker = CLOSE_REASONS[reason]
    body = pr.get("body") or ""
    if marker not in body:
        body = f"{body.rstrip()}\n\n{marker}".lstrip()
    client.close_pull_request(owner, repo, number, body=body)
    return {"closed": True, "number": number, "reason": reason}
