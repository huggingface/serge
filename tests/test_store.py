import os
import sqlite3
import tempfile
import unittest

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
