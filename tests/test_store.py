import json
import os
import sqlite3
import tempfile
import unittest

from reviewbot.reviewer import ReviewDraft
from reviewbot.store import JobStore


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
                trigger_comment="@serge please review",
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


if __name__ == "__main__":
    unittest.main()
