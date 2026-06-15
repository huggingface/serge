"""Tests for the /tasks-related store additions: the jobs.kind / task_spec /
result columns and the provider_configs.task_write_enabled opt-in flag."""

import json
import os
import tempfile
import unittest

from reviewbot.store import JobStore


class StoreTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = JobStore(os.path.join(self._tmp.name, "jobs.db"))

    def test_insert_task_job_and_result(self):
        self.store.insert_job(
            id="j1",
            user="octocat",
            target_owner="acme",
            target_repo="widgets",
            target_number=0,
            trigger_comment="fix the tests",
            llm_provider="hf",
            llm_api_base="https://router/v1",
            llm_model="m",
            created_at=1.0,
            status="running",
            source="task",
            kind="task",
            task_spec_json=json.dumps({"mode": "new_pr"}),
        )
        self.store.save_task_result("j1", json.dumps({"pr_number": 99}))
        row = self.store.load("j1")
        self.assertIsNotNone(row)
        self.assertEqual(row["kind"], "task")
        self.assertEqual(json.loads(row["task_spec_json"])["mode"], "new_pr")
        self.assertEqual(json.loads(row["result_json"])["pr_number"], 99)

    def test_default_kind_is_review(self):
        self.store.insert_job(
            id="j2",
            user="u",
            target_owner="a",
            target_repo="b",
            target_number=1,
            trigger_comment="review",
            llm_provider="hf",
            llm_api_base="x",
            llm_model=None,
            created_at=1.0,
            status="running",
        )
        row = self.store.load("j2")
        self.assertEqual(row["kind"], "review")

    def test_provider_config_task_write_flag(self):
        self.store.insert_provider_config(
            id="c1",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model=None,
            repo_pattern="acme/widgets",
            allowed_users=["octocat"],
            allowed_orgs=[],
            created_by="admin",
            task_write_enabled=True,
        )
        cfg = self.store.find_provider_config_for_repo(owner="acme", repo="widgets")
        self.assertIsNotNone(cfg)
        self.assertTrue(cfg["task_write_enabled"])

        # Default is off.
        self.store.insert_provider_config(
            id="c2",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model=None,
            repo_pattern="other/repo",
            allowed_users=["x"],
            allowed_orgs=[],
            created_by="admin",
        )
        cfg2 = self.store.find_provider_config_for_repo(owner="other", repo="repo")
        self.assertFalse(cfg2["task_write_enabled"])

    def test_update_toggles_task_write(self):
        self.store.insert_provider_config(
            id="c3",
            provider="hf",
            api_key="key",
            api_base=None,
            default_model=None,
            repo_pattern="acme/x",
            allowed_users=["u"],
            allowed_orgs=[],
            created_by="admin",
            task_write_enabled=False,
        )
        self.store.update_provider_config(
            "c3",
            provider="hf",
            api_base=None,
            default_model=None,
            repo_pattern="acme/x",
            allowed_users=["u"],
            allowed_orgs=[],
            task_write_enabled=True,
        )
        cfg = self.store.get_provider_config("c3")
        self.assertTrue(cfg["task_write_enabled"])


if __name__ == "__main__":
    unittest.main()
