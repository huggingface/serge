"""The one-call brevity pass (reviewbot/brevity.py).

The pass exists because prompt discipline did not work: the task system
prompt's LENGTH block already forbids comments that restate the code, and
verbose models keep writing them. What matters in these tests is therefore not
just that it shortens prose — it is that it can only ever shorten prose.
"""

import json
import os
import tempfile
import unittest

from reviewbot import brevity
from reviewbot.llm_client import ChatResult


class _FakeLLM:
    """Answers each .complete() with the next queued ChatResult. Records the
    calls so a test can pin how many the pass costs."""

    def __init__(self, results):
        self._results = list(results)
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


class _RaisingLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        raise RuntimeError("provider is down")


def _reply(mapping):
    return ChatResult(
        content=json.dumps({"comments": mapping}),
        usage={"prompt_tokens": 120, "completion_tokens": 30},
    )


def _ids():
    counter = [0]

    def next_id():
        counter[0] += 1
        return f"c{counter[0]}"

    return next_id


class AddedLinesTests(unittest.TestCase):
    """A patch's added lines, in the numbering of the file it produces."""

    PATCH = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,3 +1,5 @@\n"
        " import os\n"
        "-gone = 1\n"
        "+# a new comment\n"
        "+kept = 2\n"
        " tail = 3\n"
        "@@ -20,2 +22,3 @@\n"
        " last = 1\n"
        "+extra = 2\n"
        " end = 3\n"
        "diff --git a/other.txt b/other.txt\n"
        "--- /dev/null\n"
        "+++ b/other.txt\n"
        "@@ -0,0 +1 @@\n"
        "+hello\n"
        "\\ No newline at end of file\n"
    )

    def test_new_side_line_numbers_per_path(self):
        added = brevity.added_new_lines(self.PATCH)
        self.assertEqual(added["pkg/mod.py"], {2, 3, 23})
        self.assertEqual(added["other.txt"], {1})

    def test_empty_and_garbage_patches_add_nothing(self):
        self.assertEqual(brevity.added_new_lines(""), {})
        self.assertEqual(brevity.added_new_lines("not a diff at all"), {})


class CollectCommentsTests(unittest.TestCase):
    """Which comments the pass will even look at."""

    def _collect(self, source, added=None, min_chars=60):
        return brevity.collect_comments(
            "mod.py",
            source,
            added if added is not None else set(range(1, 200)),
            min_chars=min_chars,
            next_id=_ids(),
        )

    def test_block_above_code_is_one_comment_with_its_context(self):
        source = (
            "value = 0\n"
            "\n"
            "# This helper walks the list of items that were passed in and adds\n"
            "# each of their values to the running total, which it then returns.\n"
            "def total(items):\n"
            "    return sum(items)\n"
        )
        (comment,) = self._collect(source)
        self.assertEqual((comment.row, comment.end_row, comment.lines), (3, 4, 2))
        self.assertFalse(comment.trailing)
        self.assertIn("walks the list", comment.text)
        # Joined into one paragraph, so the model is not asked to preserve
        # line breaks it cannot see the width of.
        self.assertNotIn("\n", comment.text)
        self.assertEqual(comment.context[0], "def total(items):")

    def test_hash_inside_a_string_is_not_a_comment(self):
        # The reason this pass runs on the applied worktree and not on the
        # diff text: only a tokenizer can tell these apart.
        source = 'SEP = "value # this is not a comment, it is a separator literal"\n'
        self.assertEqual(self._collect(source), [])

    def test_trailing_comment_is_collected_with_its_code(self):
        source = "total = sum(items)  # add every one of the values in the items list\n"
        (comment,) = self._collect(source, min_chars=40)
        self.assertTrue(comment.trailing)
        self.assertEqual(comment.context, ("total = sum(items)",))

    def test_machine_read_comments_are_never_touched(self):
        for line in (
            "x = call()  # noqa: E501 this line is long for a reason we explain here\n",
            "y: dict = {}  # type: ignore[assignment] because the stub is wrong here\n",
            "# fmt: off  keep this table aligned by hand, the formatter ruins it\n",
            "# Copied from transformers.models.llama.modeling_llama.LlamaAttention\n",
            "# pylint: disable=too-many-locals  this function is a state machine\n",
        ):
            with self.subTest(line=line.strip()):
                self.assertEqual(self._collect(line, min_chars=10), [])

    def test_licence_header_of_a_new_file_survives(self):
        source = (
            "# Copyright 2026 The HuggingFace Team. All rights reserved.\n"
            "#\n"
            "# Licensed under the Apache License, Version 2.0 (the 'License');\n"
            "# you may not use this file except in compliance with the License.\n"
            "import os\n"
        )
        self.assertEqual(self._collect(source, min_chars=10), [])

    def test_a_block_above_the_first_statement_is_still_a_comment(self):
        # Position does not protect a comment — only its content does (the
        # licence test above). A preamble serge wrote is prose like any other.
        source = (
            "# This module holds the helpers that the exporter uses to turn a\n"
            "# span tree into the rows the publisher writes.\n"
            "import os\n"
        )
        (comment,) = self._collect(source)
        self.assertEqual((comment.row, comment.end_row), (1, 2))

    def test_a_comment_the_patch_did_not_add_is_left_alone(self):
        source = (
            "import os\n"
            "# A pre-existing comment that is quite long but is not ours to edit\n"
            "x = 1\n"
        )
        self.assertEqual(self._collect(source, added={3}), [])

    def test_partly_added_block_is_left_alone(self):
        source = (
            "import os\n"
            "# first line of a long explanatory comment block that already existed\n"
            "# second line, which is the one this patch actually added just now\n"
            "x = 1\n"
        )
        self.assertEqual(self._collect(source, added={3, 4}), [])

    def test_short_comments_are_not_worth_a_pass(self):
        source = "import os\nx = 1  # why not\n"
        self.assertEqual(self._collect(source, min_chars=100), [])

    def test_unparseable_source_yields_nothing(self):
        self.assertEqual(self._collect("def broken(:\n"), [])


class RewriteSourceTests(unittest.TestCase):
    """Condensed text → source. The model supplies text; the marker, the
    indent and the wrapping are ours."""

    def _one(self, source, min_chars=40):
        return brevity.collect_comments(
            "mod.py",
            source,
            set(range(1, 200)),
            min_chars=min_chars,
            next_id=_ids(),
        )

    def test_block_is_rewritten_at_its_own_indent(self):
        source = (
            "def total(items):\n"
            "    # This walks the list of items passed in and adds each of their\n"
            "    # values to a running total, and then returns that total value.\n"
            "    return sum(items)\n"
        )
        comments = self._one(source)
        new, applied, dropped = brevity.rewrite_source(
            source,
            comments,
            {comments[0].id: "Items may be a generator, so sum() not len()."},
            width=88,
        )
        self.assertEqual(applied, [comments[0].id])
        self.assertEqual(dropped, 0)
        self.assertEqual(
            new,
            "def total(items):\n"
            "    # Items may be a generator, so sum() not len().\n"
            "    return sum(items)\n",
        )

    def test_long_replacement_wraps_at_the_width(self):
        source = "".join(f"# {'x' * 200}\n" for _ in range(8)) + "y = 1\n"
        comments = self._one(source)
        text = " ".join(["word"] * 40)
        new, applied, _ = brevity.rewrite_source(
            source, comments, {comments[0].id: text}, width=40
        )
        self.assertEqual(applied, [comments[0].id])
        body = [line for line in new.split("\n") if line.startswith("#")]
        self.assertTrue(all(len(line) <= 40 for line in body), body)
        # Wrapped, not truncated: every word survives.
        self.assertEqual(" ".join(line[2:] for line in body), text)

    def test_a_replacement_needing_more_lines_than_it_replaces_is_refused(self):
        # Wrapping 40 words at width 40 takes ~5 lines; a 2-line comment that
        # comes back needing 5 is not a shortening, so the original stands.
        source = f"# {'x' * 60}\n# {'y' * 60}\ny = 1\n"
        comments = self._one(source)
        new, applied, _ = brevity.rewrite_source(
            source, comments, {comments[0].id: " ".join(["word"] * 40)}, width=40
        )
        self.assertIsNone(new)
        self.assertEqual(applied, [])

    def test_empty_answer_deletes_a_comment_only_block(self):
        source = (
            "# Increment the counter by one so that the counter is one larger.\n"
            "counter += 1\n"
        )
        comments = brevity.collect_comments(
            "mod.py", source, {1, 2}, min_chars=40, next_id=_ids()
        )
        new, applied, dropped = brevity.rewrite_source(
            source, comments, {comments[0].id: ""}, width=88
        )
        self.assertEqual(new, "counter += 1\n")
        self.assertEqual((len(applied), dropped), (1, 1))

    def test_empty_answer_on_a_trailing_comment_keeps_the_code(self):
        source = "counter += 1  # add one to the counter, making it one larger\n"
        comments = self._one(source)
        new, applied, _ = brevity.rewrite_source(
            source, comments, {comments[0].id: ""}, width=88
        )
        self.assertEqual(new, "counter += 1\n")
        self.assertEqual(len(applied), 1)

    def test_a_replacement_that_does_not_fit_is_refused(self):
        # One line in, three lines out is not a shortening.
        source = "x = 1  # the reason for this line, briefly stated right here\n"
        comments = self._one(source)
        new, applied, _ = brevity.rewrite_source(
            source, comments, {comments[0].id: "word " * 60}, width=88
        )
        self.assertIsNone(new)
        self.assertEqual(applied, [])

    def test_unchanged_text_is_not_an_edit(self):
        source = "x = 1  # a comment the pass decided is already as short as it gets\n"
        comments = self._one(source)
        new, applied, _ = brevity.rewrite_source(
            source, comments, {comments[0].id: comments[0].text}, width=88
        )
        self.assertIsNone(new)
        self.assertEqual(applied, [])

    def test_model_cannot_smuggle_code_into_the_file(self):
        # The reply is text, and text is what gets emitted: newlines collapse
        # and the marker is ours, so "code" in an answer stays a comment.
        source = (
            "def total(items):\n"
            "    # This walks the items and adds up every one of their values.\n"
            "    return sum(items)\n"
        )
        comments = self._one(source)
        new, applied, _ = brevity.rewrite_source(
            source,
            comments,
            {comments[0].id: "why\nimport os\nos.system('rm -rf /')"},
            width=200,
        )
        self.assertEqual(len(applied), 1)
        self.assertIn("    # why import os os.system('rm -rf /')\n", new)
        self.assertNotIn("\n    import os", new)

    def test_code_token_guard_rejects_a_rewrite_that_moved_code(self):
        source = "x = 1\ny = 2\n"
        self.assertFalse(brevity._code_preserved(source, "x = 1\ny = 3\n"))
        self.assertTrue(brevity._code_preserved(source, "x = 1\n# hi\ny = 2\n"))
        self.assertFalse(brevity._code_preserved(source, "x = 1\n  y = 2\n"))


class ParseCondensedTests(unittest.TestCase):
    """Models drift between reply shapes; a drifted shape is not a reason to
    lose the pass."""

    IDS = {"c1", "c2"}

    def test_documented_shape(self):
        self.assertEqual(
            brevity.parse_condensed('{"comments": {"c1": "short"}}', self.IDS),
            {"c1": "short"},
        )

    def test_bare_mapping(self):
        self.assertEqual(
            brevity.parse_condensed('{"c1": "short", "c2": ""}', self.IDS),
            {"c1": "short", "c2": ""},
        )

    def test_list_of_objects(self):
        self.assertEqual(
            brevity.parse_condensed(
                '{"comments": [{"id": "c2", "text": "short"}]}', self.IDS
            ),
            {"c2": "short"},
        )

    def test_prose_and_a_code_fence_around_the_object(self):
        content = 'Sure!\n```json\n{"comments": {"c1": "short"}}\n```\nDone.'
        self.assertEqual(brevity.parse_condensed(content, self.IDS), {"c1": "short"})

    def test_ids_we_never_asked_about_are_dropped(self):
        self.assertEqual(
            brevity.parse_condensed('{"comments": {"c9": "invented"}}', self.IDS), {}
        )

    def test_unparseable_replies_yield_nothing(self):
        for content in ("", None, "no json here", "{not json}", "[1, 2]"):
            with self.subTest(content=content):
                self.assertEqual(brevity.parse_condensed(content, self.IDS), {})


class PatchCommentsTests(unittest.TestCase):
    """End to end over a worktree, which is how the task flow calls it."""

    PATCH = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,6 @@\n"
        " import os\n"
        "+\n"
        "+# We keep the retry count at three because the upstream API rate limit\n"
        "+# window is sixty seconds, and three attempts is what fits inside it.\n"
        "+RETRIES = 3\n"
        " x = 1\n"
    )
    SOURCE = (
        "import os\n"
        "\n"
        "# We keep the retry count at three because the upstream API rate limit\n"
        "# window is sixty seconds, and three attempts is what fits inside it.\n"
        "RETRIES = 3\n"
        "x = 1\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self._write("mod.py", self.SOURCE)

    def _write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(
            rel
        ) else None
        with open(path, "w") as fh:
            fh.write(text)

    def _read(self, rel):
        with open(os.path.join(self.root, rel)) as fh:
            return fh.read()

    def test_one_call_shortens_the_comment_in_place(self):
        llm = _FakeLLM([_reply({"c1": "Three retries fit in the API's 60s window."})])
        result = brevity.condense_patch_comments(
            llm, root=self.root, patch=self.PATCH, min_chars=40
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual((result.considered, result.condensed), (1, 1))
        self.assertEqual(result.paths, ["mod.py"])
        self.assertGreater(result.saved, 0)
        self.assertEqual(
            self._read("mod.py"),
            "import os\n"
            "\n"
            "# Three retries fit in the API's 60s window.\n"
            "RETRIES = 3\n"
            "x = 1\n",
        )
        # No tools: this is a rewrite, not a second agentic loop.
        self.assertNotIn("tools", llm.calls[0])

    def test_nothing_long_enough_costs_no_llm_call_but_still_says_so(self):
        llm = _FakeLLM([_reply({})])
        events = []
        result = brevity.condense_patch_comments(
            llm,
            root=self.root,
            patch=self.PATCH,
            min_chars=10_000,
            emit=lambda kind, text: events.append((kind, text)),
        )
        self.assertEqual(llm.calls, [])
        self.assertEqual((result.considered, result.condensed), (0, 0))
        self.assertEqual(self._read("mod.py"), self.SOURCE)
        # "Found nothing" must not look like "never ran" in the job log.
        self.assertEqual(len(events), 1)
        self.assertIn("nothing to shorten", events[0][1])

    def test_a_patch_with_no_comments_at_all_still_says_so(self):
        # The shape a live replay actually produced: the model changed two
        # numbers and added a method, and wrote no comment anywhere.
        self._write("plain.py", "x = 1\ny = 2\n")
        patch = (
            "diff --git a/plain.py b/plain.py\n"
            "--- a/plain.py\n"
            "+++ b/plain.py\n"
            "@@ -1 +1,2 @@\n"
            " x = 1\n"
            "+y = 2\n"
        )
        events = []
        llm = _FakeLLM([_reply({})])
        brevity.condense_patch_comments(
            llm,
            root=self.root,
            patch=patch,
            min_chars=40,
            emit=lambda kind, text: events.append((kind, text)),
        )
        self.assertEqual(llm.calls, [])
        self.assertEqual(
            [text for _, text in events][0][:30], "Comment brevity: nothing to sh"
        )

    def test_a_provider_failure_leaves_the_patch_exactly_as_written(self):
        llm = _RaisingLLM()
        result = brevity.condense_patch_comments(
            llm, root=self.root, patch=self.PATCH, min_chars=40
        )
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.condensed, 0)
        self.assertIsNone(result.chat)
        self.assertEqual(self._read("mod.py"), self.SOURCE)

    def test_an_unusable_reply_leaves_the_patch_exactly_as_written(self):
        llm = _FakeLLM([ChatResult(content="I would rather not.")])
        result = brevity.condense_patch_comments(
            llm, root=self.root, patch=self.PATCH, min_chars=40
        )
        self.assertEqual(result.condensed, 0)
        self.assertEqual(self._read("mod.py"), self.SOURCE)

    def test_non_python_files_are_skipped(self):
        self._write(
            "notes.md", "<!-- a long html comment that we cannot tokenize -->\n"
        )
        patch = (
            "diff --git a/notes.md b/notes.md\n"
            "--- a/notes.md\n"
            "+++ b/notes.md\n"
            "@@ -0,0 +1 @@\n"
            "+<!-- a long html comment that we cannot tokenize -->\n"
        )
        llm = _FakeLLM([_reply({})])
        result = brevity.condense_patch_comments(
            llm, root=self.root, patch=patch, min_chars=10
        )
        self.assertEqual(llm.calls, [])
        self.assertEqual(result.considered, 0)

    def test_a_path_outside_the_worktree_is_refused(self):
        outside = os.path.join(self.root, "..", "escape.py")
        self.assertIsNone(brevity._safe_path(self.root, "../escape.py"))
        self.assertIsNone(brevity._safe_path(self.root, "/etc/passwd"))
        self.assertFalse(os.path.exists(outside))
        patch = (
            "diff --git a/../escape.py b/../escape.py\n"
            "--- a/../escape.py\n"
            "+++ b/../escape.py\n"
            "@@ -0,0 +1 @@\n"
            "+# a very long comment that we would have to read the file to shorten\n"
        )
        llm = _FakeLLM([_reply({})])
        brevity.condense_patch_comments(llm, root=self.root, patch=patch, min_chars=10)
        self.assertEqual(llm.calls, [])

    def test_the_cap_asks_about_the_longest_comments(self):
        source = ["import os\n"]
        for i in range(4):
            source.append(f"# {'word ' * (10 + i * 10)}\n")
            source.append(f"x{i} = {i}\n")
        self._write("many.py", "".join(source))
        patch_lines = [
            "diff --git a/many.py b/many.py",
            "--- a/many.py",
            "+++ b/many.py",
            f"@@ -1 +1,{len(source)} @@",
            " import os",
        ]
        patch_lines += [f"+{line.rstrip()}" for line in source[1:]]
        llm = _FakeLLM([_reply({})])
        result = brevity.condense_patch_comments(
            llm,
            root=self.root,
            patch="\n".join(patch_lines) + "\n",
            min_chars=10,
            max_items=2,
        )
        self.assertEqual(result.considered, 2)
        asked = llm.calls[0]["messages"][1]["content"]
        # The two longest of the four (rows 6 and 8), and neither shorter one.
        self.assertIn("many.py:6", asked)
        self.assertIn("many.py:8", asked)
        self.assertNotIn("many.py:2", asked)
        self.assertNotIn("many.py:4", asked)


class ReviewBodiesTests(unittest.TestCase):
    """The review flow: shorten what we are about to publish, and nothing else."""

    SUMMARY = (
        "I reviewed this pull request and overall it looks quite good to me. "
        "In this PR the author changes the retry handling in the client module, "
        "and as we can see from the diff the timeout is also adjusted.\n\n"
        "**Correctness**\n- The retry loop can spin forever when the server "
        "keeps answering 429, because nothing bounds the total attempts."
    )
    INLINE = (
        "This line reads the timeout from the environment every single call, "
        "which as we can see means the environment lookup happens on the hot "
        "path of every request that this client makes. Consider hoisting it.\n\n"
        "```suggestion\n        timeout = self._timeout\n```"
    )

    def test_bodies_are_shortened_and_the_suggestion_survives(self):
        llm = _FakeLLM(
            [
                _reply(
                    {
                        "r0": "**Correctness**\n- The retry loop is unbounded on "
                        "repeated 429s.",
                        "r1": "The env lookup is on the hot path; hoist it.\n\n"
                        "<<<BLOCK1>>>",
                    }
                )
            ]
        )
        out, result = brevity.condense_review_bodies(
            llm, [self.SUMMARY, self.INLINE], min_chars=40
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result.condensed, 2)
        self.assertLess(len(out[0]), len(self.SUMMARY))
        self.assertIn("unbounded on repeated 429s", out[0])
        # The ```suggestion block comes back byte-identical: it is applied by a
        # click, so a rewritten one is a wrong patch on someone's PR.
        self.assertIn("```suggestion\n        timeout = self._timeout\n```", out[1])
        self.assertNotIn("<<<BLOCK1>>>", out[1])

    def test_a_lost_suggestion_block_keeps_the_original_body(self):
        llm = _FakeLLM([_reply({"r0": "The env lookup is on the hot path."})])
        out, result = brevity.condense_review_bodies(llm, [self.INLINE], min_chars=40)
        self.assertEqual(out, [self.INLINE])
        self.assertEqual(result.condensed, 0)

    def test_an_empty_answer_never_deletes_a_finding(self):
        llm = _FakeLLM([_reply({"r0": "", "r1": "   "})])
        out, _ = brevity.condense_review_bodies(
            llm, [self.SUMMARY, self.INLINE], min_chars=40
        )
        self.assertEqual(out, [self.SUMMARY, self.INLINE])

    def test_a_longer_rewrite_is_refused(self):
        llm = _FakeLLM([_reply({"r0": self.SUMMARY + " And one more thing."})])
        out, result = brevity.condense_review_bodies(llm, [self.SUMMARY], min_chars=40)
        self.assertEqual(out, [self.SUMMARY])
        self.assertEqual(result.condensed, 0)

    def test_short_bodies_are_not_sent_at_all(self):
        llm = _FakeLLM([_reply({})])
        out, result = brevity.condense_review_bodies(llm, ["Typo.", "Nit: naming."])
        self.assertEqual(llm.calls, [])
        self.assertEqual(out, ["Typo.", "Nit: naming."])
        self.assertEqual(result.considered, 0)

    def test_a_provider_failure_publishes_the_review_as_written(self):
        llm = _RaisingLLM()
        out, result = brevity.condense_review_bodies(
            llm, [self.SUMMARY, self.INLINE], min_chars=40
        )
        self.assertEqual(out, [self.SUMMARY, self.INLINE])
        self.assertEqual(result.condensed, 0)

    def test_a_body_that_looks_like_a_placeholder_is_left_alone(self):
        # Restoring would be ambiguous, so this body is never sent.
        body = "The template emits <<<BLOCK1>>> here, which is wrong " * 3
        llm = _FakeLLM([_reply({})])
        out, result = brevity.condense_review_bodies(llm, [body], min_chars=40)
        self.assertEqual(llm.calls, [])
        self.assertEqual(out, [body])
        self.assertEqual(result.considered, 0)

    def test_the_label_tells_the_model_what_each_body_is(self):
        llm = _FakeLLM([_reply({})])
        brevity.condense_review_bodies(
            llm,
            [self.SUMMARY, self.INLINE],
            labels=["the PR-level review summary", "inline comment on client.py:42"],
            min_chars=40,
        )
        asked = llm.calls[0]["messages"][1]["content"]
        self.assertIn("[r0] the PR-level review summary", asked)
        self.assertIn("[r1] inline comment on client.py:42", asked)


if __name__ == "__main__":
    unittest.main()
