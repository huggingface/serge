import unittest

from reviewbot.prompts import build_system_prompt


class PromptTests(unittest.TestCase):
    def test_system_prompt_guides_models_to_use_github_suggestions(self) -> None:
        prompt = build_system_prompt("Review carefully.", tools_enabled=False)

        self.assertIn("```suggestion", prompt)
        self.assertIn("GitHub suggested-change block", prompt)
        self.assertIn("only for confident, minimal fixes", prompt)


if __name__ == "__main__":
    unittest.main()
