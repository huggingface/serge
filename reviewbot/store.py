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
    error           TEXT,
    raw_llm_output  TEXT,
    draft_json      TEXT,
    history_json    TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_user    ON jobs(user);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
"""


# Kinds of SSE events worth persisting. Token/reasoning chunks blow up
# the DB on big PRs and are useless after the fact.
PERSIST_EVENT_KINDS = frozenset({
    "log", "step", "tool", "error", "done", "metrics",
})


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
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, user, target_owner, target_repo, target_number,
                    trigger_comment, llm_provider, llm_api_base, llm_model,
                    created_at, updated_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    id, user, target_owner, target_repo, target_number,
                    trigger_comment, llm_provider, llm_api_base, llm_model,
                    created_at, now, status,
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
        along with its filtered event history and the resulting draft, if any."""
        filtered = [e for e in history if e.get("kind") in PERSIST_EVENT_KINDS]
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs
                   SET status = ?, error = ?, raw_llm_output = ?,
                       draft_json = ?, history_json = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    status,
                    error,
                    raw_llm_output,
                    _encode_draft(draft),
                    json.dumps(filtered, ensure_ascii=False),
                    time.time(),
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
                       status, created_at, updated_at,
                       llm_provider, llm_api_base, llm_model
                  FROM jobs
                 WHERE user = ?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (user, limit),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# (de)serialization helpers
# ---------------------------------------------------------------------------
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
    )


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
