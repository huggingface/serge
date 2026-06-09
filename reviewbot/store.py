"""SQLite-backed persistence for review jobs. Replaces the original
in-memory-with-4h-TTL registry so reviews survive process restarts and
can be re-opened / published after the fact.

Only structural events (log/step/tool/error/done/metrics) are persisted
— the token/reasoning stream is dropped on completion since it can run
to 10^5 entries per huge PR and is not useful after the fact.

The DB is treated as a single-writer, multi-reader resource: the FastAPI
process runs with workers=1, but the SSE worker threads write from
background threads. We serialize writes with a module-level lock and run
the connection with WAL + check_same_thread=False.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Optional

from .patch import DiffSnippetLine
from .reviewer import DraftComment, ReviewDraft

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    user            TEXT NOT NULL,
    target_owner    TEXT NOT NULL,
    target_repo     TEXT NOT NULL,
    target_number   INTEGER NOT NULL,
    trigger_comment TEXT NOT NULL,
    llm_provider    TEXT,
    llm_api_base    TEXT,
    llm_model       TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    status          TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'web',
    error           TEXT,
    raw_llm_output  TEXT,
    draft_json      TEXT,
    history_json    TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_user    ON jobs(user);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);

-- Per-repo provider config: which LLM provider + API key to use when a
-- given user (member of one of the listed orgs / users) asks for a
-- review on a given repository. Replaces the previous env-var-only key
-- selection so a single deployment can serve many keys.
--
-- repo_pattern is either an exact "owner/repo" or an org wildcard
-- "owner/*". allowed_users / allowed_orgs are JSON arrays of
-- lowercased GitHub logins.
CREATE TABLE IF NOT EXISTS provider_configs (
    id             TEXT PRIMARY KEY,
    provider       TEXT NOT NULL,
    api_key        TEXT NOT NULL,
    api_base       TEXT,
    default_model  TEXT,
    repo_pattern   TEXT NOT NULL,
    allowed_users  TEXT NOT NULL DEFAULT '[]',
    allowed_orgs   TEXT NOT NULL DEFAULT '[]',
    created_by     TEXT NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_provider_configs_repo
    ON provider_configs(repo_pattern);
"""


# Kinds of SSE events worth persisting. Token/reasoning chunks blow up
# the DB on big PRs and are useless after the fact.
PERSIST_EVENT_KINDS = frozenset(
    {
        "log",
        "step",
        "tool",
        "error",
        "done",
        "metrics",
    }
)


class JobStore:
    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # check_same_thread=False: worker threads write from background
        # threads. We serialize with self._lock to keep that safe.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._ensure_column("llm_provider", "TEXT")
            self._ensure_column("llm_api_base", "TEXT")
            self._ensure_column("llm_model", "TEXT")
            self._ensure_column("prompt_tokens", "INTEGER")
            self._ensure_column("completion_tokens", "INTEGER")
            self._ensure_column("source", "TEXT NOT NULL DEFAULT 'web'")
            self._conn.commit()
        log.info("Opened job store at %s", path)

    def _ensure_column(self, name: str, sql_type: str) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if name not in columns:
            self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {sql_type}")

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def insert_job(
        self,
        *,
        id: str,
        user: str,
        target_owner: str,
        target_repo: str,
        target_number: int,
        trigger_comment: str,
        llm_provider: str,
        llm_api_base: str,
        llm_model: Optional[str],
        created_at: float,
        status: str,
        source: str = "web",
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, user, target_owner, target_repo, target_number,
                    trigger_comment, llm_provider, llm_api_base, llm_model,
                    created_at, updated_at, status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    user,
                    target_owner,
                    target_repo,
                    target_number,
                    trigger_comment,
                    llm_provider,
                    llm_api_base,
                    llm_model,
                    created_at,
                    now,
                    status,
                    source,
                ),
            )
            self._conn.commit()

    def save_terminal(
        self,
        job_id: str,
        *,
        status: str,
        error: Optional[str],
        raw_llm_output: Optional[str],
        draft: Optional[ReviewDraft],
        history: list[dict[str, Any]],
    ) -> None:
        """Persist the final state of a job (done/error/published/discarded)
        along with its filtered event history and the resulting draft, if any.

        Token counts come from the draft when available; otherwise (error
        path with no draft) we fall back to the latest ``metrics`` event in
        the history so the journal still reports what was consumed before
        the failure."""
        filtered = [e for e in history if e.get("kind") in PERSIST_EVENT_KINDS]
        if draft is not None:
            prompt_tokens: Optional[int] = draft.prompt_tokens or None
            completion_tokens: Optional[int] = draft.completion_tokens or None
        else:
            prompt_tokens, completion_tokens = _latest_token_counts(history)
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?, error = ?, raw_llm_output = ?,
                       draft_json = ?, history_json = ?, updated_at = ?,
                       prompt_tokens = ?, completion_tokens = ?
                 WHERE id = ?
                """,
                (
                    status,
                    error,
                    raw_llm_output,
                    _encode_draft(draft),
                    json.dumps(filtered, ensure_ascii=False),
                    time.time(),
                    prompt_tokens,
                    completion_tokens,
                    job_id,
                ),
            )
            self._conn.commit()

    def update_status(self, job_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), job_id),
            )
            self._conn.commit()

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def mark_running_as_crashed(self) -> int:
        """Called on startup: any job left in 'running' state was killed
        by the restart. Mark it as errored so the UI shows something
        useful instead of a forever-spinning row."""
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE jobs
                   SET status = 'error',
                       error  = COALESCE(error, 'review aborted (server restarted while running)'),
                       updated_at = ?
                 WHERE status = 'running'
                """,
                (time.time(),),
            )
            self._conn.commit()
            return cur.rowcount

    def prune(self, keep: int) -> int:
        """Keep the most recent ``keep`` jobs (globally, by created_at).
        Returns the number of rows deleted. Never prunes jobs in the
        'running' state — in-flight work always survives."""
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM jobs
                 WHERE status != 'running'
                   AND id NOT IN (
                       SELECT id FROM jobs
                        ORDER BY created_at DESC
                        LIMIT ?
                   )
                """,
                (keep,),
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def load(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_dict(row)

    def list_for_user(self, user: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, user, target_owner, target_repo, target_number,
                       status, source, created_at, updated_at,
                       llm_provider, llm_api_base, llm_model
                  FROM jobs
                 WHERE user = ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (user, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # provider configs
    # ------------------------------------------------------------------
    def insert_provider_config(
        self,
        *,
        id: str,
        provider: str,
        api_key: str,
        api_base: Optional[str],
        default_model: Optional[str],
        repo_pattern: str,
        allowed_users: list[str],
        allowed_orgs: list[str],
        created_by: str,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO provider_configs (
                    id, provider, api_key, api_base, default_model,
                    repo_pattern, allowed_users, allowed_orgs,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id,
                    provider,
                    api_key,
                    api_base,
                    default_model,
                    repo_pattern,
                    json.dumps([u.lower() for u in allowed_users]),
                    json.dumps([o.lower() for o in allowed_orgs]),
                    created_by,
                    now,
                    now,
                ),
            )
            self._conn.commit()

    def update_provider_config(
        self,
        config_id: str,
        *,
        provider: str,
        api_base: Optional[str],
        default_model: Optional[str],
        repo_pattern: str,
        allowed_users: list[str],
        allowed_orgs: list[str],
        new_api_key: Optional[str] = None,
    ) -> bool:
        """Update everything except the api_key by default. When
        ``new_api_key`` is provided, the stored secret is replaced too —
        otherwise the existing one is preserved (write-only model).
        Returns True if a row was updated."""
        now = time.time()
        with self._lock:
            if new_api_key is not None:
                cur = self._conn.execute(
                    """
                    UPDATE provider_configs
                       SET provider = ?, api_key = ?, api_base = ?,
                           default_model = ?, repo_pattern = ?,
                           allowed_users = ?, allowed_orgs = ?,
                           updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        provider,
                        new_api_key,
                        api_base,
                        default_model,
                        repo_pattern,
                        json.dumps([u.lower() for u in allowed_users]),
                        json.dumps([o.lower() for o in allowed_orgs]),
                        now,
                        config_id,
                    ),
                )
            else:
                cur = self._conn.execute(
                    """
                    UPDATE provider_configs
                       SET provider = ?, api_base = ?, default_model = ?,
                           repo_pattern = ?, allowed_users = ?,
                           allowed_orgs = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        provider,
                        api_base,
                        default_model,
                        repo_pattern,
                        json.dumps([u.lower() for u in allowed_users]),
                        json.dumps([o.lower() for o in allowed_orgs]),
                        now,
                        config_id,
                    ),
                )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_provider_config(self, config_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM provider_configs WHERE id = ?", (config_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def list_provider_configs(self) -> list[dict[str, Any]]:
        """All configs, newest-updated first. Caller is responsible for
        scrubbing api_key before sending to the UI."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, provider, api_key, api_base, default_model,
                       repo_pattern, allowed_users, allowed_orgs,
                       created_by, created_at, updated_at
                  FROM provider_configs
                 ORDER BY updated_at DESC
                """
            ).fetchall()
        return [_decode_provider_config(r) for r in rows]

    def get_provider_config(self, config_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM provider_configs WHERE id = ?", (config_id,)
            ).fetchone()
        if row is None:
            return None
        return _decode_provider_config(row)

    def allowed_orgs_for_repo(self, owner: str, repo: str) -> list[str]:
        """Union of ``allowed_orgs`` across every provider_config whose
        ``repo_pattern`` could match ``owner/repo`` (exact or wildcard).
        Used at request time to decide which orgs are worth probing via
        the GitHub App when the user's session has no cached org list —
        e.g. SAML-protected memberships that don't appear in
        ``/user/orgs``."""
        exact = f"{owner}/{repo}".lower()
        wildcard = f"{owner}/*".lower()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT allowed_orgs FROM provider_configs
                 WHERE LOWER(repo_pattern) IN (?, ?)
                """,
                (exact, wildcard),
            ).fetchall()
        seen: set[str] = set()
        result: list[str] = []
        for row in rows:
            raw = row["allowed_orgs"] or "[]"
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, str):
                    continue
                lc = item.lower()
                if lc in seen:
                    continue
                seen.add(lc)
                result.append(lc)
        return result

    def find_provider_config(
        self,
        *,
        user: str,
        user_orgs: list[str],
        owner: str,
        repo: str,
        provider: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Pick the provider config that should serve this (user, repo).

        Matches when the user is in ``allowed_users`` or one of the
        ``user_orgs`` is in ``allowed_orgs``, AND ``repo_pattern`` is
        either ``"{owner}/{repo}"`` or ``"{owner}/*"``. When ``provider``
        is given, candidates are further filtered to that LLM provider
        so the form can let the user pick which authorized key to use.

        Exact repo matches win over wildcards; among ties (or among
        candidates of the same specificity), most-recently-updated wins.
        Returns None when no config matches."""
        exact = f"{owner}/{repo}".lower()
        wildcard = f"{owner}/*".lower()
        user_lc = user.lower()
        orgs_lc = {o.lower() for o in user_orgs}
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM provider_configs
                 WHERE LOWER(repo_pattern) IN (?, ?)
                 ORDER BY updated_at DESC
                """,
                (exact, wildcard),
            ).fetchall()
        best_exact: Optional[dict[str, Any]] = None
        best_wild: Optional[dict[str, Any]] = None
        for row in rows:
            cfg = _decode_provider_config(row)
            if provider is not None and cfg["provider"] != provider:
                continue
            allowed_users = {u.lower() for u in cfg["allowed_users"]}
            allowed_orgs = {o.lower() for o in cfg["allowed_orgs"]}
            if user_lc not in allowed_users and not (allowed_orgs & orgs_lc):
                continue
            pattern = cfg["repo_pattern"].lower()
            if pattern == exact and best_exact is None:
                best_exact = cfg
            elif pattern == wildcard and best_wild is None:
                best_wild = cfg
            if best_exact is not None:
                # Rows are ordered updated_at DESC, so the first exact
                # match is the freshest — short-circuit.
                return best_exact
        return best_exact or best_wild

    def find_provider_config_for_repo(
        self,
        *,
        owner: str,
        repo: str,
        provider: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Repo-only provider config lookup for webhook mode.

        Unlike :meth:`find_provider_config`, this ignores
        ``allowed_users`` / ``allowed_orgs``: webhook reviews have no
        logged-in user to gate on, so the GitHub App being installed on
        the repo is treated as sufficient authorization. Matching on
        ``repo_pattern`` is identical — exact ``"{owner}/{repo}"`` wins
        over the ``"{owner}/*"`` wildcard, and among ties the
        most-recently-updated row wins. When ``provider`` is given,
        candidates are filtered to that LLM provider. Returns None when
        no config matches the repo."""
        exact = f"{owner}/{repo}".lower()
        wildcard = f"{owner}/*".lower()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM provider_configs
                 WHERE LOWER(repo_pattern) IN (?, ?)
                 ORDER BY updated_at DESC
                """,
                (exact, wildcard),
            ).fetchall()
        best_exact: Optional[dict[str, Any]] = None
        best_wild: Optional[dict[str, Any]] = None
        for row in rows:
            cfg = _decode_provider_config(row)
            if provider is not None and cfg["provider"] != provider:
                continue
            pattern = cfg["repo_pattern"].lower()
            if pattern == exact and best_exact is None:
                best_exact = cfg
            elif pattern == wildcard and best_wild is None:
                best_wild = cfg
            if best_exact is not None:
                return best_exact
        return best_exact or best_wild

    def list_all_calls(self, limit: int = 500) -> list[dict[str, Any]]:
        """Cross-user journal: every review job ever recorded, newest
        first. Used by the /journal page so any authenticated viewer can
        audit who called what model with how many tokens."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, user, target_owner, target_repo, target_number,
                       status, source, created_at, updated_at,
                       llm_provider, llm_api_base, llm_model,
                       prompt_tokens, completion_tokens
                  FROM jobs
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# (de)serialization helpers
# ---------------------------------------------------------------------------
def _latest_token_counts(
    history: list[dict[str, Any]],
) -> tuple[Optional[int], Optional[int]]:
    """Scan the event history backwards for the most recent ``metrics``
    payload and return its ``in`` / ``out`` totals. Used when a job
    errors out before a ReviewDraft is built but we still want the
    journal to show what was consumed."""
    for event in reversed(history):
        if event.get("kind") != "metrics":
            continue
        text = event.get("text")
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        p_in = payload.get("in")
        p_out = payload.get("out")
        return (
            p_in if isinstance(p_in, int) and p_in > 0 else None,
            p_out if isinstance(p_out, int) and p_out > 0 else None,
        )
    return None, None


def _encode_draft(draft: Optional[ReviewDraft]) -> Optional[str]:
    if draft is None:
        return None
    return json.dumps(
        {
            "owner": draft.owner,
            "repo": draft.repo,
            "number": draft.number,
            "head_sha": draft.head_sha,
            "summary": draft.summary,
            "event": draft.event,
            "rejected_count": draft.rejected_count,
            "metrics_line": draft.metrics_line,
            "model": draft.model,
            "comments": [dataclasses.asdict(c) for c in draft.comments],
        },
        ensure_ascii=False,
    )


def decode_draft(s: Optional[str]) -> Optional[ReviewDraft]:
    if not s:
        return None
    data = json.loads(s)
    comments = [
        DraftComment(
            id=c["id"],
            path=c["path"],
            side=c["side"],
            line=c["line"],
            body=c["body"],
            diff_hunk=[DiffSnippetLine(**dh) for dh in c.get("diff_hunk", [])],
        )
        for c in data.get("comments", [])
    ]
    return ReviewDraft(
        owner=data["owner"],
        repo=data["repo"],
        number=data["number"],
        head_sha=data["head_sha"],
        summary=data["summary"],
        event=data["event"],
        comments=comments,
        rejected_count=data.get("rejected_count", 0),
        metrics_line=data.get("metrics_line", ""),
        model=data.get("model"),
    )


def _decode_provider_config(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("allowed_users", "allowed_orgs"):
        raw = d.get(key)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = []
            d[key] = parsed if isinstance(parsed, list) else []
        elif raw is None:
            d[key] = []
    return d


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    history_raw = d.pop("history_json", None)
    if history_raw:
        try:
            d["history"] = json.loads(history_raw)
        except json.JSONDecodeError:
            d["history"] = []
    else:
        d["history"] = []
    return d
