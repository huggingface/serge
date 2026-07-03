from typing import Optional

from .reviewer import InlineCommentContext, ReviewRequest


def build_review_request(
    event_name: str,
    payload: dict,
    mention_trigger: str,
) -> Optional[ReviewRequest]:
    """Decide whether an incoming event should trigger a review, and build
    the ReviewRequest if so. Returns None when gating conditions don't match.

    The same logic backs both the Flask webhook (app.py) and the GitHub
    Action entry point (action_runner.py), so behaviour is identical in
    both deployment modes.

    For ``pull_request_review_comment`` events the returned request carries
    an ``inline`` context object — the caller dispatches to the follow-up
    flow (a focused reply on the comment thread) instead of a full PR
    review.
    """
    if event_name not in ("issue_comment", "pull_request_review_comment"):
        return None
    if payload.get("action") != "created":
        return None

    comment = payload.get("comment") or {}
    if mention_trigger not in (comment.get("body") or ""):
        return None
    # In App mode the webhook sees every comment on installed repos, including
    # ones the App itself (or any other bot) posts. Never react to a bot's
    # comment — that would let a stray mention in our own output loop forever.
    if (comment.get("user") or {}).get("type") == "Bot":
        return None
    if comment.get("author_association") not in ("MEMBER", "OWNER", "COLLABORATOR"):
        return None

    repo = payload.get("repository") or {}
    full = repo.get("full_name") or ""
    if "/" not in full:
        return None
    owner, name = full.split("/", 1)

    inline: Optional[InlineCommentContext] = None
    if event_name == "issue_comment":
        issue = payload.get("issue") or {}
        if not issue.get("pull_request"):
            return None
        if issue.get("state") != "open":
            return None
        pr_number = issue.get("number")
    else:
        pr = payload.get("pull_request") or {}
        pr_number = pr.get("number")
        if pr.get("state") and pr.get("state") != "open":
            return None
        path = comment.get("path")
        # GitHub nulls out line/side when the comment is "outdated" against
        # the latest commit; the original_* fields keep the anchor that the
        # commenter actually looked at, which is the right thing to reason
        # about for a follow-up question.
        line = comment.get("line") or comment.get("original_line")
        side = comment.get("side") or comment.get("original_side") or "RIGHT"
        if not isinstance(path, str) or not isinstance(line, int):
            return None
        comment_id = comment.get("id")
        if not isinstance(comment_id, int):
            return None
        inline = InlineCommentContext(
            comment_id=comment_id,
            path=path,
            side=side if side in ("RIGHT", "LEFT") else "RIGHT",
            line=line,
            diff_hunk=comment.get("diff_hunk") or "",
            in_reply_to_id=comment.get("in_reply_to_id"),
        )

    if not isinstance(pr_number, int):
        return None

    return ReviewRequest(
        owner=owner,
        repo=name,
        number=pr_number,
        trigger_comment_id=comment.get("id") or 0,
        trigger_comment_body=comment.get("body") or "",
        commenter=(comment.get("user") or {}).get("login") or "unknown",
        inline=inline,
    )
