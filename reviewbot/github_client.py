import base64
from typing import Any, Optional

import requests


class GitHubClient:
    """Thin REST wrapper scoped to a single installation token."""

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "ai-reviewer",
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
