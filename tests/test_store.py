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

    def test_persists_chat_events_but_drops_token_stream(self) -> None:
        # "chat" events (assistant turns + tool I/O) must survive to the
        # stored history so unparseable-output failures stay debuggable;
        # token/reasoning chunks must still be dropped.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "jobs.db")
            store = JobStore(db)
            store.insert_job(
                id="j3",
                user="bob",
                target_owner="o",
                target_repo="r",
                target_number=3,
                trigger_comment="x",
                llm_provider="hf",
                llm_api_base="https://router.huggingface.co/v1",
                llm_model="moonshotai/Kimi-K2.6",
                created_at=3.0,
                status="running",
            )
            history = [
                {"kind": "chat", "text": json.dumps({"role": "assistant"})},
                {"kind": "tool", "text": "read_file(...)"},
                {"kind": "token", "text": "noise"},
                {"kind": "reasoning", "text": "noise"},
            ]
            store.save_terminal(
                "j3",
                status="error",
                error="boom",
                raw_llm_output=None,
                draft=None,
                history=history,
            )
            conn = sqlite3.connect(db)
            raw = conn.execute(
                "SELECT history_json FROM jobs WHERE id = ?", ("j3",)
            ).fetchone()[0]
            kinds = [e["kind"] for e in json.loads(raw)]
            self.assertIn("chat", kinds)
            self.assertIn("tool", kinds)
            self.assertNotIn("token", kinds)
            self.assertNotIn("reasoning", kinds)

    def test_persists_generated_and_published_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "jobs.db"))
            store.insert_job(
                id="j3",
                user="carol",
                target_owner="o",
                target_repo="r",
                target_number=3,
                trigger_comment="@askserge review this",
                llm_provider="hf",
                llm_api_base="https://router.huggingface.co/v1",
                llm_model="model",
                created_at=3.0,
                status="running",
            )
            generated = ReviewDraft(
                owner="o",
                repo="r",
                number=3,
                head_sha="abc",
                summary="generated",
                event="COMMENT",
            )
            published = ReviewDraft(
                owner="o",
                repo="r",
                number=3,
                head_sha="abc",
                summary="edited",
                event="REQUEST_CHANGES",
            )
            store.save_terminal(
                "j3",
                status="done",
                error=None,
                raw_llm_output=None,
                draft=generated,
                history=[],
            )
            store.save_published_review(
                "j3",
                edits={"summary": "edited", "event": "REQUEST_CHANGES"},
                published_draft=published,
            )

            row = store.load("j3")
            self.assertIsNotNone(row)
            assert row is not None
            self.assertIn('"summary": "generated"', row["draft_json"])
            self.assertIn('"summary": "edited"', row["published_draft_json"])
            self.assertIn('"event": "REQUEST_CHANGES"', row["review_edits_json"])

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
