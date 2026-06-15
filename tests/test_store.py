import json
import os
import sqlite3
import tempfile
import unittest

from reviewbot.reviewer import ReviewDraft
from reviewbot.store import JobStore, _is_postgres_url


class JobStoreTests(unittest.TestCase):
    def test_persists_llm_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "jobs.db"))
            store.insert_job(
                id="job1",
                user="octocat",
                target_owner="owner",
                target_repo="repo",
                target_number=123,
                trigger_comment="@askserge please review",
                llm_provider="openai",
                llm_api_base="https://api.openai.com/v1",
                llm_model="gpt-4.1",
                created_at=1.0,
                status="running",
            )

            row = store.load("job1")
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["llm_provider"], "openai")
            self.assertEqual(row["llm_api_base"], "https://api.openai.com/v1")
            self.assertEqual(row["llm_model"], "gpt-4.1")

    def test_journal_captures_tokens_from_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "jobs.db"))
            store.insert_job(
                id="j1",
                user="alice",
                target_owner="o",
                target_repo="r",
                target_number=1,
                trigger_comment="x",
                llm_provider="anthropic",
                llm_api_base="https://api.anthropic.com",
                llm_model="claude-opus-4-6",
                created_at=1.0,
                status="running",
            )
            draft = ReviewDraft(
                owner="o",
                repo="r",
                number=1,
                head_sha="abc",
                summary="s",
                event="COMMENT",
                prompt_tokens=12345,
                completion_tokens=678,
            )
            store.save_terminal(
                "j1",
                status="done",
                error=None,
                raw_llm_output=None,
                draft=draft,
                history=[],
            )

            entries = store.list_all_calls()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["user"], "alice")
            self.assertEqual(entries[0]["prompt_tokens"], 12345)
            self.assertEqual(entries[0]["completion_tokens"], 678)

    def test_journal_falls_back_to_history_metrics(self) -> None:
        # No draft (error path) — tokens come from the most recent
        # metrics event in the history.
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "jobs.db"))
            store.insert_job(
                id="j2",
                user="bob",
                target_owner="o",
                target_repo="r",
                target_number=2,
                trigger_comment="x",
                llm_provider="hf",
                llm_api_base="https://router.huggingface.co/v1",
                llm_model="moonshotai/Kimi-K2.6",
                created_at=2.0,
                status="running",
            )
            history = [
                {"kind": "metrics", "text": json.dumps({"in": 100, "out": 50})},
                {"kind": "log", "text": "something happened"},
                {"kind": "metrics", "text": json.dumps({"in": 999, "out": 222})},
            ]
            store.save_terminal(
                "j2",
                status="error",
                error="boom",
                raw_llm_output=None,
                draft=None,
                history=history,
            )

            entries = store.list_all_calls()
            self.assertEqual(entries[0]["prompt_tokens"], 999)
            self.assertEqual(entries[0]["completion_tokens"], 222)

    def test_adds_llm_columns_to_existing_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "jobs.db")
            conn = sqlite3.connect(path)
            conn.executescript(
                """
                CREATE TABLE jobs (
                    id              TEXT PRIMARY KEY,
                    user            TEXT NOT NULL,
                    target_owner    TEXT NOT NULL,
                    target_repo     TEXT NOT NULL,
                    target_number   INTEGER NOT NULL,
                    trigger_comment TEXT NOT NULL,
                    created_at      REAL NOT NULL,
                    updated_at      REAL NOT NULL,
                    status          TEXT NOT NULL,
                    error           TEXT,
                    raw_llm_output  TEXT,
                    draft_json      TEXT,
                    history_json    TEXT
                );
                """
            )
            conn.commit()
            conn.close()

            JobStore(path)
            conn = sqlite3.connect(path)
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            conn.close()

            self.assertIn("llm_provider", columns)
            self.assertIn("llm_api_base", columns)
            self.assertIn("llm_model", columns)


class BackendDetectionTests(unittest.TestCase):
    def test_recognizes_postgres_urls(self) -> None:
        self.assertTrue(_is_postgres_url("postgres://u:p@h/db"))
        self.assertTrue(_is_postgres_url("postgresql://u:p@h:5432/db?sslmode=require"))

    def test_treats_paths_as_sqlite(self) -> None:
        self.assertFalse(_is_postgres_url("jobs.db"))
        self.assertFalse(_is_postgres_url("/var/lib/reviewbot/jobs.db"))


@unittest.skipUnless(
    os.environ.get("TEST_DATABASE_URL"),
    "set TEST_DATABASE_URL to a Postgres DSN to run the Postgres backend tests",
)
class JobStorePostgresTests(unittest.TestCase):
    """Exercises the Postgres backend against a real server. Skipped unless
    TEST_DATABASE_URL is set (e.g. a local docker postgres). The same
    public API as the SQLite path, so these mirror the SQLite tests and
    guard the ?->%s rewrite and the quoted "user" column."""

    def setUp(self) -> None:
        self.store = JobStore(os.environ["TEST_DATABASE_URL"])
        # Start each test from a clean slate (the schema is created once,
        # CREATE TABLE IF NOT EXISTS, so we truncate rather than drop).
        with self.store._lock:
            self.store._conn.execute("TRUNCATE jobs, provider_configs")
            self.store._conn.commit()

    def test_insert_load_and_journal(self) -> None:
        self.store.insert_job(
            id="pg1",
            user="octocat",
            target_owner="owner",
            target_repo="repo",
            target_number=123,
            trigger_comment="@askserge please review",
            llm_provider="openai",
            llm_api_base="https://api.openai.com/v1",
            llm_model="gpt-4.1",
            created_at=1.0,
            status="running",
        )

        row = self.store.load("pg1")
        assert row is not None
        self.assertEqual(row["user"], "octocat")
        self.assertEqual(row["llm_model"], "gpt-4.1")

        listed = self.store.list_for_user("octocat")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], "pg1")

        draft = ReviewDraft(
            owner="owner",
            repo="repo",
            number=123,
            head_sha="abc",
            summary="s",
            event="COMMENT",
            prompt_tokens=42,
            completion_tokens=7,
        )
        self.store.save_terminal(
            "pg1",
            status="done",
            error=None,
            raw_llm_output=None,
            draft=draft,
            history=[],
        )
        entries = self.store.list_all_calls()
        self.assertEqual(entries[0]["prompt_tokens"], 42)
        self.assertEqual(entries[0]["completion_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
