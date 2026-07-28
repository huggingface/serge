import os
import tempfile
import time
import unittest

from reviewbot.store import JobStore


class ProviderConfigsTests(unittest.TestCase):
    def _store(self) -> JobStore:
        tmp = tempfile.mkdtemp(prefix="serge-providers-")
        self.addCleanup(self._cleanup, tmp)
        return JobStore(os.path.join(tmp, "jobs.db"))

    def _cleanup(self, tmp: str) -> None:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    def _insert(
        self,
        store: JobStore,
        cid: str,
        *,
        provider: str = "openai",
        api_key: str = "sk-test",
        api_base: str | None = None,
        default_model: str | None = None,
        repo_pattern: str = "huggingface/transformers",
        allowed_users: list[str] | None = None,
        allowed_orgs: list[str] | None = None,
        created_by: str = "admin",
    ) -> None:
        store.insert_provider_config(
            id=cid,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            default_model=default_model,
            repo_pattern=repo_pattern,
            allowed_users=allowed_users or [],
            allowed_orgs=allowed_orgs or [],
            created_by=created_by,
        )

    def test_insert_and_list(self) -> None:
        store = self._store()
        self._insert(store, "c1", default_model="gpt-4.1", allowed_users=["alice"])
        rows = store.list_provider_configs()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["provider"], "openai")
        self.assertEqual(rows[0]["default_model"], "gpt-4.1")
        self.assertEqual(rows[0]["allowed_users"], ["alice"])
        self.assertEqual(rows[0]["allowed_orgs"], [])

    def test_update_preserves_api_key_unless_replaced(self) -> None:
        store = self._store()
        self._insert(store, "c1", api_key="orig", allowed_users=["alice"])
        store.update_provider_config(
            "c1",
            provider="openai",
            api_base=None,
            default_model="gpt-4.1-mini",
            repo_pattern="huggingface/transformers",
            allowed_users=["alice", "bob"],
            allowed_orgs=[],
        )
        row = store.get_provider_config("c1")
        assert row is not None
        self.assertEqual(row["api_key"], "orig")
        self.assertEqual(row["default_model"], "gpt-4.1-mini")
        self.assertEqual(row["allowed_users"], ["alice", "bob"])

        store.update_provider_config(
            "c1",
            provider="openai",
            api_base=None,
            default_model="gpt-4.1-mini",
            repo_pattern="huggingface/transformers",
            allowed_users=["alice", "bob"],
            allowed_orgs=[],
            new_api_key="replaced",
        )
        row = store.get_provider_config("c1")
        assert row is not None
        self.assertEqual(row["api_key"], "replaced")

    def test_delete(self) -> None:
        store = self._store()
        self._insert(store, "c1", allowed_users=["alice"])
        self.assertTrue(store.delete_provider_config("c1"))
        self.assertFalse(store.delete_provider_config("c1"))
        self.assertIsNone(store.get_provider_config("c1"))

    def test_user_must_be_in_allowed_users_or_orgs(self) -> None:
        store = self._store()
        self._insert(store, "c1", allowed_users=["alice"], repo_pattern="hf/repo")

        self.assertIsNotNone(
            store.find_provider_config(
                user="alice",
                user_orgs=[],
                owner="hf",
                repo="repo",
            )
        )
        self.assertIsNone(
            store.find_provider_config(
                user="mallory",
                user_orgs=[],
                owner="hf",
                repo="repo",
            )
        )

    def test_org_membership_grants_access(self) -> None:
        store = self._store()
        self._insert(store, "c1", allowed_orgs=["huggingface"], repo_pattern="hf/repo")
        self.assertIsNotNone(
            store.find_provider_config(
                user="alice",
                user_orgs=["HuggingFace"],
                owner="hf",
                repo="repo",
            )
        )

    def test_exact_repo_beats_wildcard(self) -> None:
        store = self._store()
        # Insert wildcard FIRST so it's older. Even though list ordering
        # is updated_at DESC, the matcher must still prefer the exact
        # pattern regardless of recency.
        self._insert(
            store,
            "wild",
            allowed_orgs=["hf"],
            repo_pattern="hf/*",
            default_model="wild-model",
        )
        time.sleep(0.01)
        self._insert(
            store,
            "exact",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
            default_model="exact-model",
        )
        match = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="transformers",
        )
        assert match is not None
        self.assertEqual(match["id"], "exact")

    def test_exact_wins_even_when_wildcard_is_newer(self) -> None:
        store = self._store()
        self._insert(
            store,
            "exact",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
            default_model="exact-model",
        )
        time.sleep(0.01)
        self._insert(
            store,
            "wild",
            allowed_orgs=["hf"],
            repo_pattern="hf/*",
            default_model="wild-model",
        )
        match = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="transformers",
        )
        assert match is not None
        self.assertEqual(match["id"], "exact")

    def test_wildcard_used_when_no_exact_match(self) -> None:
        store = self._store()
        self._insert(
            store,
            "wild",
            allowed_orgs=["hf"],
            repo_pattern="hf/*",
            default_model="wild-model",
        )
        match = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="datasets",
        )
        assert match is not None
        self.assertEqual(match["id"], "wild")

    def test_ties_break_by_most_recently_updated(self) -> None:
        store = self._store()
        # Two configs with the same repo pattern (same specificity);
        # the most recently updated should win.
        self._insert(
            store,
            "older",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
            default_model="older",
        )
        time.sleep(0.01)
        self._insert(
            store,
            "newer",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
            default_model="newer",
        )
        match = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="transformers",
        )
        assert match is not None
        self.assertEqual(match["id"], "newer")

    def test_provider_filter_narrows_candidates(self) -> None:
        store = self._store()
        self._insert(
            store,
            "openai-cfg",
            provider="openai",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
        )
        self._insert(
            store,
            "anthropic-cfg",
            provider="anthropic",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
        )
        a = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="transformers",
            provider="anthropic",
        )
        assert a is not None
        self.assertEqual(a["id"], "anthropic-cfg")
        o = store.find_provider_config(
            user="alice",
            user_orgs=["hf"],
            owner="hf",
            repo="transformers",
            provider="openai",
        )
        assert o is not None
        self.assertEqual(o["id"], "openai-cfg")

    def test_returns_none_when_no_match(self) -> None:
        store = self._store()
        self._insert(
            store,
            "c1",
            allowed_orgs=["hf"],
            repo_pattern="hf/transformers",
        )
        # Wrong repo entirely.
        self.assertIsNone(
            store.find_provider_config(
                user="alice",
                user_orgs=["hf"],
                owner="other",
                repo="repo",
            )
        )
        # Right repo, wrong user (no orgs match, not in allowed_users).
        self.assertIsNone(
            store.find_provider_config(
                user="alice",
                user_orgs=["other-org"],
                owner="hf",
                repo="transformers",
            )
        )

    def test_provider_lookup_ignores_repo_but_not_authorization(self) -> None:
        store = self._store()
        self._insert(
            store,
            "anthropic-cfg",
            provider="anthropic",
            allowed_users=["alice"],
            repo_pattern="hf/transformers",
        )
        self._insert(
            store,
            "openai-cfg",
            provider="openai",
            allowed_orgs=["hf"],
            repo_pattern="other/repo",
        )

        # No repo involved: any authorized config for the provider serves.
        a = store.find_provider_config_for_provider(
            user="alice", user_orgs=[], provider="anthropic"
        )
        assert a is not None
        self.assertEqual(a["id"], "anthropic-cfg")
        o = store.find_provider_config_for_provider(
            user="bob", user_orgs=["HF"], provider="openai"
        )
        assert o is not None
        self.assertEqual(o["id"], "openai-cfg")

        # Unauthorized users and unconfigured providers get nothing.
        self.assertIsNone(
            store.find_provider_config_for_provider(
                user="mallory", user_orgs=[], provider="anthropic"
            )
        )
        self.assertIsNone(
            store.find_provider_config_for_provider(
                user="alice", user_orgs=[], provider="hf"
            )
        )


if __name__ == "__main__":
    unittest.main()
