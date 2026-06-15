import os
import unittest
from unittest.mock import patch

from reviewbot.config import Config


class ConfigTests(unittest.TestCase):
    def test_defaults_to_serge_trigger(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.mention_trigger, "@askserge")

    def test_respects_explicit_trigger_override(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "token", "MENTION_TRIGGER": "@custom"},
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.mention_trigger, "@custom")

    def test_defaults_helper_tools_path(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.helper_tools_path, ".ai/review-tools.json")

    def test_respects_helper_tools_path_override(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "token", "HELPER_TOOLS_PATH": ".review/helpers.json"},
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.helper_tools_path, ".review/helpers.json")

    def test_staging_defaults_off(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            cfg = Config.from_env(require_app=False)

        self.assertFalse(cfg.is_staging)

    def test_staging_enabled_via_env(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "token", "STAGING": "true"},
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertTrue(cfg.is_staging)

    def test_store_path_defaults_to_sqlite(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.web_store_path, "jobs.db")

    def test_database_url_overrides_store_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "token",
                "WEB_STORE_PATH": "/var/lib/reviewbot/jobs.db",
                "DATABASE_URL": "postgresql://u:p@db.internal:5432/serge",
            },
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        # DATABASE_URL wins over WEB_STORE_PATH so the hosted deployment
        # keeps state off the ephemeral pod filesystem.
        self.assertEqual(cfg.web_store_path, "postgresql://u:p@db.internal:5432/serge")


if __name__ == "__main__":
    unittest.main()
