import base64
import logging
import time
from typing import Any, Callable, Optional

import requests


log = logging.getLogger(__name__)


# Identity stamped on serge-authored commits created through the Git Data
# API (the /tasks flow). Using a noreply address keeps the commits from
# pointing at a real mailbox; the name makes them obvious in `git log`.
SERGE_GIT_NAME = "serge[bot]"
SERGE_GIT_EMAIL = "serge[bot]@users.noreply.github.com"


# api.github.com returns these when a gateway or backend hiccups, with no bearing
# on the request itself — a replay usually succeeds. Prod lost a finished,
# GPU-verified fix to a single `502 Bad Gateway` on a *read*
# (``GET /git/commits/<sha>``) at the commit step (transformers#47605).
_TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
_TRANSIENT_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
# POST endpoints where a replay cannot duplicate anything: both are
# content-addressed, so re-sending identical bytes returns the same SHA and
# creates nothing new. Everything else that writes is left alone — see
# :func:`_is_replay_safe`.
_REPLAY_SAFE_POST_PATHS = ("/git/blobs", "/git/trees")
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 3.0)
_MAX_RETRY_AFTER = 30.0


def _is_replay_safe(method: str, url: str) -> bool:
    """Whether replaying this request cannot duplicate a side effect.

    Reads are always safe. Writes are deliberately *not*, because a 5xx is
    ambiguous — the write may well have landed and only the response got lost, so
    a blind replay could double-post a review comment or turn a successful
    ``create_ref`` into a confusing 422. The two exceptions are blobs and trees:
    Git object creation is content-addressed, so replaying returns the identical
    SHA and creates nothing.

    Consequence: a 502 on ``create_ref``/``create_commit``/a comment still fails
    the task. That is the safe direction — a lost run is recoverable, a duplicate
    comment or a mangled branch is noise a human has to clean up."""
    verb = method.upper()
    if verb in ("GET", "HEAD"):
        return True
    if verb == "POST":
        return any(url.endswith(path) for path in _REPLAY_SAFE_POST_PATHS)
    return False


def _retry_delay(response: Optional[requests.Response], attempt: int) -> float:
    """Backoff before attempt ``attempt`` (1-based), honouring ``Retry-After``."""
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), _MAX_RETRY_AFTER)
            except (TypeError, ValueError):
                pass
    index = min(attempt - 1, len(_BACKOFF_SECONDS) - 1)
    return _BACKOFF_SECONDS[index]


class _AuthRetrySession(requests.Session):
    """A session that repairs the two failures that cost prod finished work.

    **Expired token (401).** GitHub App installation tokens are valid for exactly
    one hour. A ``/tasks`` run that spends longer than that in the agentic loop
    plus the repo normalizer reaches the publish step holding a dead token, and
    every write fails ``401 Bad credentials`` — the fix is written, validated and
    GPU-verified, then thrown away. Rather than refresh on a timer (which needs a
    clock nobody owns and still races a request in flight), the request that
    *discovers* the expiry fixes it: ask ``token_provider`` for a fresh token,
    restamp the header, replay. Confirmed firing in prod at 60m01s.

    **Transient gateway error (5xx / dropped connection).** Same shape, different
    cause: a single ``502 Bad Gateway`` from api.github.com killed a finished
    qwen3_omni_moe fix at its commit step (transformers#47605). Retried with a
    short backoff, but *only* where a replay cannot duplicate a side effect (see
    :func:`_is_replay_safe`).

    Overriding :meth:`requests.Session.request` — the single funnel every
    ``get``/``post``/``patch`` helper goes through — covers every call site in
    :class:`GitHubClient`, present and future, with no per-method plumbing.

    The two cases are replayed under deliberately different rules. A **401 is
    unambiguous**: the request was rejected, so nothing happened, so replaying any
    method is safe — including a comment or a ref update. A **5xx is ambiguous**:
    the write may have landed with only the response lost, so replaying is
    restricted to requests where that cannot matter.

    The token is re-minted at most once per request, and only when the fresh one
    actually differs: a 401 from a genuinely bad token (wrong App, revoked
    install) must surface as a 401 rather than double every request."""

    def __init__(self, token_provider: Optional[Callable[[], str]] = None):
        super().__init__()
        self._token_provider = token_provider

    def _refreshed_token(self) -> Optional[str]:
        """A replacement token that differs from the one in use, else ``None``."""
        if self._token_provider is None:
            return None
        try:
            token = self._token_provider()
        except Exception:  # noqa: BLE001
            # Refresh is best-effort: on failure keep the original response so
            # the caller reports the real GitHub error, not a refresh stacktrace.
            log.warning("token refresh after 401 failed", exc_info=True)
            return None
        if not token or self.headers.get("Authorization") == f"Bearer {token}":
            return None
        return token

    def request(  # type: ignore[override]
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        refreshed = False
        attempt = 0
        while True:
            try:
                response: Optional[requests.Response] = super().request(
                    method, url, **kwargs
                )
            except _TRANSIENT_EXCEPTIONS as exc:
                attempt += 1
                if attempt >= _MAX_ATTEMPTS or not _is_replay_safe(method, url):
                    raise
                delay = _retry_delay(None, attempt)
                log.warning(
                    "%s %s failed (%s); retrying in %.1fs (attempt %d/%d)",
                    method,
                    url,
                    exc.__class__.__name__,
                    delay,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(delay)
                continue

            assert response is not None
            if response.status_code == 401 and not refreshed:
                refreshed = True
                token = self._refreshed_token()
                if token is not None:
                    log.info("got 401 from %s; re-minted the token and retrying", url)
                    self.headers["Authorization"] = f"Bearer {token}"
                    continue
                return response

            if response.status_code in _TRANSIENT_STATUSES:
                attempt += 1
                if attempt < _MAX_ATTEMPTS and _is_replay_safe(method, url):
                    delay = _retry_delay(response, attempt)
                    log.warning(
                        "%s %s returned %d; retrying in %.1fs (attempt %d/%d)",
                        method,
                        url,
                        response.status_code,
                        delay,
                        attempt + 1,
                        _MAX_ATTEMPTS,
                    )
                    time.sleep(delay)
                    continue

            return response


class GitHubClient:
    """Thin REST wrapper scoped to a single installation token.

    ``token_provider``, when given, is called to mint a replacement token after
    a ``401`` so long-running tasks survive the one-hour installation-token
    lifetime (see :class:`_AuthRetrySession`)."""

    def __init__(self, token: str, token_provider: Optional[Callable[[], str]] = None):
        self.session = _AuthRetrySession(token_provider)
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

    def delete_ref(self, owner: str, repo: str, ref: str) -> None:
        """Delete a ref. ``ref`` is the short form without ``refs/``, e.g.
        ``heads/serge/fix-abc``. Used to tear down a candidate branch whose GPU
        verification did not confirm the fix, so no dangling branch is left."""
        r = self.session.delete(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref}",
            timeout=30,
        )
        # 404 = already gone; treat as success so cleanup is idempotent.
        if not r.ok and r.status_code != 404:
            raise requests.HTTPError(
                f"{r.status_code} deleting ref {ref} on {owner}/{repo}: {r.text}",
                response=r,
            )

    # --- GitHub Actions (serge GPU verify loop) --------------------------------
    # These require the serge GitHub App to be granted `Actions: read and write`.

    def dispatch_workflow(
        self,
        owner: str,
        repo: str,
        workflow_file: str,
        *,
        ref: str,
        inputs: dict[str, Any],
    ) -> None:
        """Trigger a ``workflow_dispatch``. ``workflow_file`` is the filename on
        ``ref`` (e.g. ``serge-verify-caller.yml``). Returns 204 with no body and
        no run id — correlate the resulting run via a ``run-name`` echoing a
        unique input (see :func:`list_workflow_runs`)."""
        r = self.session.post(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/dispatches",
            json={"ref": ref, "inputs": inputs},
            timeout=30,
        )
        if not r.ok:
            raise requests.HTTPError(
                f"{r.status_code} dispatching {workflow_file} on {owner}/{repo}: {r.text}",
                response=r,
            )

    def list_workflow_runs(
        self,
        owner: str,
        repo: str,
        workflow_file: str,
        *,
        event: Optional[str] = None,
        per_page: int = 30,
    ) -> list[dict]:
        params: dict[str, Any] = {"per_page": per_page}
        if event:
            params["event"] = event
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_file}/runs",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return (r.json() or {}).get("workflow_runs", [])

    def list_run_artifacts(self, owner: str, repo: str, run_id: int) -> list[dict]:
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts",
            params={"per_page": 100},
            timeout=30,
        )
        r.raise_for_status()
        return (r.json() or {}).get("artifacts", [])

    def download_artifact_zip(self, owner: str, repo: str, artifact_id: int) -> bytes:
        """Download an artifact's zip. GitHub returns a 302 to a signed blob
        URL; ``requests`` follows it and the zip bytes come back in ``.content``."""
        r = self.session.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip",
            timeout=120,
        )
        r.raise_for_status()
        return r.content

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

    def search_issues(
        self,
        query: str,
        *,
        sort: str = "updated",
        order: str = "desc",
        per_page: int = 20,
    ) -> list[dict]:
        """Issue/PR search (``GET /search/issues``), newest activity first.

        Used to surface existing reports of a CI failure in a task PR body.
        Search has its own rate budget (30 req/min for an installation token),
        so callers treat any failure as "no hits" rather than retrying.
        ``advanced_search`` opts into the current query parser explicitly."""
        r = self.session.get(
            "https://api.github.com/search/issues",
            params={
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "advanced_search": "true",
            },
            timeout=30,
        )
        r.raise_for_status()
        return (r.json() or {}).get("items", [])

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
