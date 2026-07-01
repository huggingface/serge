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

    def test_slack_config_prefers_ci_feedback_env_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "token",
                "SLACK_CIFEEDBACK_BOT_TOKEN": "feedback-token",
                "SLACK_CIFEEDBACK_CHANNEL": "#feedback-ci",
                "CI_SLACK_BOT_TOKEN": "legacy-token",
                "SLACK_REPORT_CHANNEL": "#legacy-ci",
            },
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.slack_bot_token, "feedback-token")
        self.assertEqual(cfg.slack_report_channel, "#feedback-ci")

    def test_slack_config_accepts_transformers_ci_env_fallbacks(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "token",
                "CI_SLACK_BOT_TOKEN": "legacy-token",
                "SLACK_REPORT_CHANNEL": "#legacy-ci",
            },
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.slack_bot_token, "legacy-token")
        self.assertEqual(cfg.slack_report_channel, "#legacy-ci")

    def test_needs_isolated_checkout_requires_normalize_and_container_backend(
        self,
    ) -> None:
        import dataclasses

        from reviewbot import sandbox

        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            base = Config.from_env(require_app=False)

        # No normalize command configured -> always the cheap worktree.
        self.assertFalse(
            dataclasses.replace(
                base,
                task_normalize_command=None,
                task_sandbox_backend=sandbox.DOCKER_BACKEND,
            ).needs_isolated_checkout
        )
        # Normalize + a container backend that binds only the worktree.
        for backend in (
            sandbox.DOCKER_BACKEND,
            sandbox.KUBERNETES_BACKEND,
            sandbox.AUTO_BACKEND,
        ):
            self.assertTrue(
                dataclasses.replace(
                    base,
                    task_normalize_command=["make", "style"],
                    task_sandbox_backend=backend,
                ).needs_isolated_checkout,
                backend,
            )
        # bwrap is dev/style-only -> keep the linked worktree even with a command.
        self.assertFalse(
            dataclasses.replace(
                base,
                task_normalize_command=["make", "style"],
                task_sandbox_backend=sandbox.BWRAP_BACKEND,
            ).needs_isolated_checkout
        )


if __name__ == "__main__":
    unittest.main()
