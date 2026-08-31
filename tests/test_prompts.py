import unittest

from reviewbot.prompts import (
    CONTEXT_TAIL_RESERVE_CHARS,
    MAX_CONTEXT_CHARS,
    _truncate_middle,
    build_task_user_prompt,
    build_followup_system_prompt,
    build_followup_user_prompt,
    build_system_prompt,
    build_task_system_prompt,
)


class TaskSystemPromptTests(unittest.TestCase):
    def test_injects_repo_conventions_and_guidance(self) -> None:
        prompt = build_task_system_prompt(
            "Always edit modular_*.py, never the generated modeling file.",
            "Prefer real fixes over `# noqa`.",
            tools_enabled=False,
        )
        self.assertIn("REPO CONVENTIONS", prompt)
        self.assertIn("Always edit modular_*.py", prompt)
        self.assertIn("Prefer real fixes over `# noqa`.", prompt)
        # The standing root-cause / last-resort guidance is always present.
        self.assertIn("ROOT CAUSE", prompt)
        self.assertIn("LAST RESORT", prompt)

    def test_handles_missing_conventions(self) -> None:
        prompt = build_task_system_prompt("", None, tools_enabled=True)
        self.assertIn("no repository conventions file was found", prompt)


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


class ContextTruncationTests(unittest.TestCase):
    """A task context is load-bearing at BOTH ends, so it truncates in the
    middle.

    Head: the `<!-- serge-task:... -->` marker and fingerprint the instruction
    requires in the PR body. Tail: the GPU reproduce/verify block appended by
    `_with_verify_feedback`, which announces itself as authoritative and carries
    the triage steer. Head truncation kept the first and silently dropped the
    second."""

    MARKER = "<!-- serge-task:integration-failure-triage:sha256:abc123 -->"
    REPRODUCE = "## The targeted test(s) were REPRODUCED on GPU (authoritative)"

    def _context(self, filler_chars: int) -> str:
        return (
            f"{self.MARKER}\nSerge task fingerprint: `abc123`.\n\n"
            + ("REPORT " * (filler_chars // 7))[:filler_chars]
            + f"\n\n{self.REPRODUCE}\n\nE AssertionError: Tensor-likes are not close!"
        )

    def _prompt(self, context: str) -> str:
        return build_task_user_prompt(
            repo_full_name="huggingface/transformers",
            base_ref="main",
            instruction="fix it",
            context=context,
            existing_diff="",
        )

    def test_reproduce_block_survives_a_context_over_the_limit(self) -> None:
        # The live 124,011-char case: head truncation dropped 84,011 chars from
        # the tail and the model never saw the reproduce block.
        prompt = self._prompt(self._context(121_000))
        self.assertIn(self.REPRODUCE, prompt)
        self.assertIn("Tensor-likes are not close", prompt)

    def test_marker_also_survives(self) -> None:
        prompt = self._prompt(self._context(121_000))
        self.assertIn(self.MARKER, prompt)
        self.assertIn("Serge task fingerprint", prompt)

    def test_live_sizes_all_keep_both_ends(self) -> None:
        # The three real contexts measured on 2026-08-31.
        for size in (42_815, 62_939, 124_011):
            with self.subTest(context_chars=size):
                prompt = self._prompt(self._context(size))
                self.assertIn(self.MARKER, prompt)
                self.assertIn(self.REPRODUCE, prompt)

    def test_short_context_is_untouched(self) -> None:
        ctx = self._context(100)
        prompt = self._prompt(ctx)
        self.assertNotIn("omitted from the middle", prompt)
        self.assertIn(self.REPRODUCE, prompt)

    def test_says_the_middle_was_dropped(self) -> None:
        prompt = self._prompt(self._context(121_000))
        self.assertIn("omitted from the middle", prompt)


class TruncateMiddleTests(unittest.TestCase):
    def test_keeps_head_and_tail_and_reports_the_gap(self) -> None:
        text = "H" * 100 + "M" * 800 + "T" * 100
        out = _truncate_middle(text, limit=200, tail=100)
        self.assertTrue(out.startswith("H" * 100))
        self.assertTrue(out.endswith("T" * 100))
        self.assertIn("800 chars omitted from the middle", out)

    def test_under_limit_is_returned_verbatim(self) -> None:
        self.assertEqual(_truncate_middle("short", 100, 50), "short")

    def test_tail_larger_than_limit_is_clamped(self) -> None:
        # Degenerate config must not produce a negative head slice.
        out = _truncate_middle("x" * 500, limit=100, tail=999)
        self.assertIn("omitted from the middle", out)
        self.assertTrue(out.endswith("x" * 100))

    def test_zero_tail_behaves_like_head_truncation(self) -> None:
        out = _truncate_middle("A" * 50 + "B" * 50, limit=50, tail=0)
        self.assertTrue(out.startswith("A" * 50))
        self.assertNotIn("B", out)

    def test_reserve_fits_a_full_reproduce_block(self) -> None:
        # The whole point of the constant: a maximal reproduce block must fit
        # inside the reserve. Assert it against the config field rather than a
        # copied literal — the two drifted apart once already (the reserve was
        # sized for one traceback while the formatters take five).
        import dataclasses

        from reviewbot.config import Config

        block = next(
            f.default
            for f in dataclasses.fields(Config)
            if f.name == "reproduce_block_chars"
        )
        self.assertGreaterEqual(CONTEXT_TAIL_RESERVE_CHARS, block)
        self.assertLess(CONTEXT_TAIL_RESERVE_CHARS, MAX_CONTEXT_CHARS)


class ChangedExpectationsRuleTests(unittest.TestCase):
    """serge reviewed its own expectation rewrite on huggingface/transformers
    PR #48437 (job `609a3021`) and half-caught it.

    The diff moved a fill-mask assertion from index 6 to 7 and changed the
    expected token from `"happiness"` to `"<unk>"`. The review never mentioned
    `<unk>`; filed the index change as a maintainability *nit* ("can shift with
    tokenizer changes") when index 6 is in fact a bare `▁` token, so the old
    assertion was reading a non-mask position; and cited the GPU gate — "Serge
    verified the patched test passes on a GPU runner, which gives reasonable
    confidence the new expectation matches the current behavior" — which is
    circular, because the patch rewrote the assertion the gate re-runs.
    """

    def _prompt(self) -> str:
        """Whitespace-collapsed: the section is hard-wrapped for readability, so
        a phrase can straddle a newline. The rule's presence is what matters,
        not where the wrap fell."""
        return " ".join(build_system_prompt("rules").split())

    def test_a_changed_expectation_is_its_own_category(self) -> None:
        p = self._prompt()
        self.assertIn("CHANGED EXPECTATIONS", p)
        self.assertIn("redefining what passing means", p)

    def test_degenerate_values_are_named_so_unk_cannot_pass_unremarked(self) -> None:
        p = self._prompt()
        for token in ("`<unk>`", "empty string", "all-zeros", "NaN"):
            self.assertIn(token, p)

    def test_a_moved_assertion_index_is_a_correctness_finding(self) -> None:
        # The review called this a nit; the old index was reading the wrong token.
        p = self._prompt()
        self.assertIn("correctness finding, not a maintainability nit", p)

    def test_a_passing_run_is_not_evidence_for_a_rewritten_expectation(self) -> None:
        p = self._prompt()
        self.assertIn("passing test run is NOT evidence", p)
        self.assertIn("circular", p)
        self.assertIn("verified on a GPU runner", p)

    def test_the_rule_survives_a_repo_with_no_review_rules(self) -> None:
        self.assertIn("CHANGED EXPECTATIONS", build_system_prompt(""))
        self.assertIn(
            "CHANGED EXPECTATIONS", build_system_prompt("x", tools_enabled=False)
        )

    def test_a_literal_brace_in_the_rule_survives_formatting(self) -> None:
        """The template is rendered with str.format, so the `Expectations({...})`
        example has to be brace-escaped or every review prompt raises."""
        self.assertIn("`Expectations({...})`", self._prompt())
