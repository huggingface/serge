import unittest

from reviewbot.prompts import (
    build_followup_system_prompt,
    build_followup_user_prompt,
    build_system_prompt,
)


class PromptTests(unittest.TestCase):
    def test_system_prompt_guides_models_to_use_github_suggestions(self) -> None:
        prompt = build_system_prompt("Review carefully.", tools_enabled=False)

        self.assertIn("```suggestion", prompt)
        self.assertIn("GitHub suggested-change block", prompt)
        self.assertIn("only for confident, minimal fixes", prompt)


class FollowupPromptTests(unittest.TestCase):
    def test_followup_system_prompt_forbids_json_output(self) -> None:
        prompt = build_followup_system_prompt("Be terse.", tools_enabled=True)

        # The reply must be markdown, not the JSON schema used by the
        # full-review flow.
        self.assertIn("ONE GitHub markdown reply", prompt)
        self.assertIn("No JSON", prompt)
        # Tools-enabled section still flows through.
        self.assertIn("BROWSE TOOLS", prompt)

    def test_followup_user_prompt_includes_anchor_and_question(self) -> None:
        prompt = build_followup_user_prompt(
            repo_full_name="acme/project",
            number=9,
            title="Improve cache",
            body="adds an LRU layer",
            author="alice",
            commenter="bob",
            trigger_comment="@askserge could you help me understand this line?",
            path="src/cache.py",
            side="RIGHT",
            line=42,
            diff_hunk="@@ -40,3 +40,3 @@\n-old\n+new line",
        )

        self.assertIn("acme/project#9", prompt)
        self.assertIn("src/cache.py", prompt)
        self.assertIn("42", prompt)
        self.assertIn("new line", prompt)
        self.assertIn("could you help me understand", prompt)
        # No JSON envelope.
        self.assertNotIn('"summary":', prompt)

    def test_followup_user_prompt_handles_missing_diff_hunk(self) -> None:
        prompt = build_followup_user_prompt(
            repo_full_name="a/b",
            number=1,
            title="t",
            body="",
            author="u",
            commenter="u",
            trigger_comment="@askserge ?",
            path="x.py",
            side="RIGHT",
            line=1,
            diff_hunk="",
        )

        self.assertIn("diff hunk unavailable", prompt)


if __name__ == "__main__":
    unittest.main()
