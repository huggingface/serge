import base64
from typing import Any, Optional

import requests


# Identity stamped on serge-authored commits created through the Git Data
# API (the /tasks flow). Using a noreply address keeps the commits from
# pointing at a real mailbox; the name makes them obvious in `git log`.
SERGE_GIT_NAME = "serge[bot]"
SERGE_GIT_EMAIL = "serge[bot]@users.noreply.github.com"


class GitHubClient:
    """Thin REST wrapper scoped to a single installation token."""

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "serge",
            }
        )

    def get_pr(self, owner: str, repo: str, number: int) -> dict:
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def get_pr_files(self, owner: str, repo: str, number: int) -> list[dict]:
        files: list[dict] = []
        page = 1
        while True:
            r = self.session.get(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/files",
                params={"per_page": 100, "page": page},
                timeout=60,
            )
            r.raise_for_status()
            batch = r.json()
            files.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return files

    def get_file_contents(
        self, owner: str, repo: str, path: str, ref: Optional[str] = None
    ) -> Optional[str]:
        params = {"ref": ref} if ref else None
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            params=params,
            timeout=30,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if data.get("encoding") == "base64":
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return data.get("content")

    def create_review(
        self,
        owner: str,
        repo: str,
        number: int,
        commit_id: str,
        body: str,
        comments: list[dict[str, Any]],
        event: str = "COMMENT",
    ) -> dict:
        payload: dict[str, Any] = {
            "commit_id": commit_id,
            "body": body,
            "event": event,
        }
        if comments:
            payload["comments"] = comments
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/reviews",
            json=payload,
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating review on {owner}/{repo}#{number}: {r.text}",
                response=r,
            )
        return r.json()

    def post_issue_comment(self, owner: str, repo: str, number: int, body: str) -> dict:
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments",
            json={"body": body},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def add_reaction_to_issue_comment(
        self, owner: str, repo: str, comment_id: int, content: str = "eyes"
    ) -> None:
        self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
            json={"content": content},
            timeout=30,
        )

    def add_reaction_to_review_comment(
        self, owner: str, repo: str, comment_id: int, content: str = "eyes"
    ) -> None:
        self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/comments/{comment_id}/reactions",
            json={"content": content},
            timeout=30,
        )

    # -- Git Data + Pulls write API (the /tasks flow) --------------------
    #
    # serge applies the LLM's patch in a network-isolated worktree, then
    # uploads the result through these methods. The installation token
    # never enters the sandbox or a git remote — blobs/trees/commits/refs
    # are created over HTTPS from the main process.

    def get_ref_sha(self, owner: str, repo: str, ref: str) -> str:
        """Return the object SHA a ref points at. ``ref`` is the short
        form without the ``refs/`` prefix, e.g. ``heads/main`` or
        ``heads/serge/fix-abc``."""
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/ref/{ref}",
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} resolving ref {ref} on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()["object"]["sha"]

    def get_commit_tree_sha(self, owner: str, repo: str, commit_sha: str) -> str:
        """Return the tree SHA of a commit (the base tree for create_tree)."""
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/commits/{commit_sha}",
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["tree"]["sha"]

    def create_blob(self, owner: str, repo: str, content: bytes) -> str:
        """Upload a file blob (raw bytes, base64-encoded) and return its SHA."""
        encoded = base64.b64encode(content).decode("ascii")
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/git/blobs",
            json={"content": encoded, "encoding": "base64"},
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating blob on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()["sha"]

    def create_tree(
        self,
        owner: str,
        repo: str,
        base_tree: Optional[str],
        entries: list[dict[str, Any]],
    ) -> str:
        """Create a tree from ``entries`` layered on ``base_tree``.

        Each entry is ``{"path", "mode", "type": "blob", "sha": <blob sha
        or None>}``. A ``None`` sha deletes the path from ``base_tree``."""
        payload: dict[str, Any] = {"tree": entries}
        if base_tree:
            payload["base_tree"] = base_tree
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees",
            json=payload,
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating tree on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()["sha"]

    def create_commit(
        self,
        owner: str,
        repo: str,
        *,
        message: str,
        tree_sha: str,
        parents: list[str],
        author_name: str = SERGE_GIT_NAME,
        author_email: str = SERGE_GIT_EMAIL,
    ) -> str:
        """Create a commit object and return its SHA. Author and committer
        are both stamped with the serge identity so the loop cap can count
        serge-authored commits on a branch."""
        ident = {"name": author_name, "email": author_email}
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/git/commits",
            json={
                "message": message,
                "tree": tree_sha,
                "parents": parents,
                "author": ident,
                "committer": ident,
            },
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating commit on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()["sha"]

    def create_ref(self, owner: str, repo: str, ref: str, sha: str) -> dict:
        """Create a new ref. ``ref`` is the full form, e.g.
        ``refs/heads/serge/fix-abc``."""
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs",
            json={"ref": ref, "sha": sha},
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating ref {ref} on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()

    def update_ref(
        self, owner: str, repo: str, ref: str, sha: str, *, force: bool = False
    ) -> dict:
        """Move an existing ref to ``sha``. ``ref`` is the short form
        without ``refs/``, e.g. ``heads/serge/fix-abc``. ``force`` allows a
        non-fast-forward update; serge's follow-up commits are children of
        the current head so a fast-forward is the norm."""
        r = self.session.patch(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref}",
            json={"sha": sha, "force": force},
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} updating ref {ref} on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str,
        draft: bool = False,
    ) -> dict:
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls",
            json={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "draft": draft,
            },
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} creating pull request on {owner}/{repo}: {r.text}",
                response=r,
            )
        return r.json()

    def mark_pull_request_ready(self, node_id: str) -> None:
        """Transition a draft PR to ready-for-review via the GraphQL
        ``markPullRequestReadyForReview`` mutation. The REST ``PATCH /pulls``
        endpoint cannot flip ``draft``, so this is the only way. The
        draft->ready transition is what fires the ``ready_for_review`` webhook
        that downstream reviewer-assignment workflows listen for. ``node_id``
        is the GraphQL global ID returned in the create-PR response."""
        query = (
            "mutation($id: ID!) { markPullRequestReadyForReview(input: "
            "{pullRequestId: $id}) { pullRequest { id isDraft } } }"
        )
        r = self.session.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"id": node_id}},
            timeout=60,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} marking PR ready for review: {r.text}",
                response=r,
            )
        errors = (r.json() or {}).get("errors")
        if errors:
            raise requests.HTTPError(
                f"GraphQL errors marking PR ready for review: {errors}",
                response=r,
            )

    def count_branch_commits_by_author(
        self, owner: str, repo: str, branch: str, *, author_email: str, cap: int = 100
    ) -> int:
        """Count commits on ``branch`` authored by ``author_email``, up to
        ``cap`` (one page). Used to enforce the follow-up loop cap on a
        serge-owned fix branch so a misconfigured workflow can't burn
        tokens forever."""
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            params={"sha": branch, "per_page": min(cap, 100)},
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} listing commits on {owner}/{repo}@{branch}: {r.text}",
                response=r,
            )
        count = 0
        for commit in r.json():
            author = (commit.get("commit") or {}).get("author") or {}
            if (author.get("email") or "").lower() == author_email.lower():
                count += 1
        return count

    def reply_to_review_comment(
        self,
        owner: str,
        repo: str,
        number: int,
        comment_id: int,
        body: str,
    ) -> dict:
        """Post a threaded reply to an existing PR review comment. The
        endpoint accepts any comment_id in the thread and re-uses the
        thread's commit/path/line anchor, so we don't have to look those
        up ourselves."""
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}/comments/{comment_id}/replies",
            json={"body": body},
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} replying to review comment "
                f"{owner}/{repo}#{number} comment {comment_id}: {r.text}",
                response=r,
            )
        return r.json()
