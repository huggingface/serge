"""Tests for the repeated-tool-call guard.

Reference failure: prod task 433e8274 issued the same two greps 21 and 44 times,
exhausted the 2M input-token cap in ~55 turns without ever reasoning about the
fix, and produced a patch that failed GPU verification — three rounds running.
"""

import unittest

from reviewbot.tool_repeat import ToolRepeatGuard, normalize_arguments, path_argument


GREP_A = '{"pattern": "GlmOcrProcessor|Glm46VProcessor", "path": "src/transformers/models/glm_ocr", "max_results": 50}'
GREP_B = '{"pattern": "image_token_id|video_token_id", "path": "src/transformers"}'


class NormalizeArgumentsTests(unittest.TestCase):
    def test_key_order_does_not_disguise_a_repeat(self):
        self.assertEqual(
            normalize_arguments('{"a": 1, "b": 2}'),
            normalize_arguments('{"b": 2, "a": 1}'),
        )

    def test_whitespace_does_not_disguise_a_repeat(self):
        self.assertEqual(
            normalize_arguments('{"path":"x"}'),
            normalize_arguments('{  "path"  :  "x"  }'),
        )

    def test_different_arguments_stay_different(self):
        self.assertNotEqual(
            normalize_arguments('{"path": "a"}'),
            normalize_arguments('{"path": "b"}'),
        )

    def test_non_json_falls_back_to_the_raw_string(self):
        """Some providers emit partial/non-JSON arguments; two identical broken
        strings are still a repeat."""
        self.assertEqual(
            normalize_arguments("not json  "), normalize_arguments("not json")
        )
        self.assertNotEqual(
            normalize_arguments("not json"), normalize_arguments("other")
        )

    def test_empty_and_missing_arguments_are_equal(self):
        self.assertEqual(normalize_arguments(""), normalize_arguments("{}"))


class ToolRepeatGuardTests(unittest.TestCase):
    def test_first_call_is_free(self):
        guard = ToolRepeatGuard(6)
        self.assertIsNone(guard.observe("grep", GREP_A))
        self.assertEqual(guard.repeats, 0)
        self.assertFalse(guard.tripped)

    def test_distinct_calls_never_count_as_repeats(self):
        """Legitimate investigation makes many different calls."""
        guard = ToolRepeatGuard(6)
        for i in range(50):
            self.assertIsNone(guard.observe("read_file", '{"path": "f%d.py"}' % i))
        self.assertEqual(guard.repeats, 0)
        self.assertFalse(guard.tripped)

    def test_repeat_is_told_it_is_repeating(self):
        guard = ToolRepeatGuard(6)
        guard.observe("grep", GREP_A)
        note = guard.observe("grep", GREP_A)
        self.assertIsNotNone(note)
        self.assertIn("already made this exact grep call 2 times", note)
        # Both escape hatches are spelled out.
        self.assertIn("different pattern", note)
        self.assertIn("final answer", note)

    def test_repeat_count_climbs(self):
        guard = ToolRepeatGuard(0)  # disabled; count via a live one below
        self.assertIsNone(guard.observe("grep", GREP_A))
        guard = ToolRepeatGuard(99)
        for _ in range(4):
            guard.observe("grep", GREP_A)
        note = guard.observe("grep", GREP_A)
        self.assertIn("5 times", note)
        self.assertEqual(guard.repeats, 4)

    def test_trips_after_the_limit(self):
        guard = ToolRepeatGuard(3)
        guard.observe("grep", GREP_A)  # first — free
        guard.observe("grep", GREP_A)  # repeat 1
        guard.observe("grep", GREP_A)  # repeat 2
        self.assertFalse(guard.tripped)
        note = guard.observe("grep", GREP_A)  # repeat 3 → trip
        self.assertTrue(guard.tripped)
        self.assertIn("You are in a loop", note)

    def test_repeats_are_counted_across_signatures(self):
        """The prod shape: the model alternated between two identical greps. A
        per-signature limit would have allowed twice the waste."""
        guard = ToolRepeatGuard(4)
        guard.observe("grep", GREP_A)
        guard.observe("grep", GREP_B)
        for _ in range(2):
            guard.observe("grep", GREP_A)
            guard.observe("grep", GREP_B)
        self.assertTrue(guard.tripped)
        self.assertEqual(guard.repeats, 4)

    def test_same_arguments_different_tool_is_not_a_repeat(self):
        guard = ToolRepeatGuard(6)
        self.assertIsNone(guard.observe("grep", '{"path": "x"}'))
        self.assertIsNone(guard.observe("list_dir", '{"path": "x"}'))

    def test_disabled_guard_never_fires(self):
        guard = ToolRepeatGuard(0)
        self.assertFalse(guard.enabled)
        for _ in range(100):
            self.assertIsNone(guard.observe("grep", GREP_A))
        self.assertFalse(guard.tripped)
        self.assertEqual(guard.repeats, 0)

    def test_summary_names_the_worst_offenders(self):
        guard = ToolRepeatGuard(999)
        for _ in range(21):
            guard.observe("grep", GREP_A)
        for _ in range(3):
            guard.observe("list_dir", '{"path": "src"}')
        summary = guard.summary()
        self.assertIn("22 repeated tool call(s)", summary)
        self.assertIn("grep×21", summary)
        self.assertIn("list_dir×3", summary)

    def test_prod_shape_stops_early(self):
        """The real regression: 21 + 44 identical greps must not all get through.

        With the default limit the loop is cut off after a handful of repeats
        instead of ~55 turns and 2M input tokens."""
        guard = ToolRepeatGuard(6)
        served = 0
        for _ in range(21):
            guard.observe("grep", GREP_A)
            served += 1
            if guard.tripped:
                break
        self.assertTrue(guard.tripped)
        self.assertLessEqual(served, 8)


class PathArgumentTests(unittest.TestCase):
    def test_only_the_opening_tools_name_a_path(self):
        self.assertEqual(path_argument("read_file", '{"path": "a/b.py"}'), "a/b.py")
        self.assertEqual(path_argument("list_dir", '{"path": "a"}'), "a")
        # A grep of the same path with a different pattern is a new search,
        # not a re-read, so it is not counted as a visit.
        self.assertIsNone(path_argument("grep", '{"pattern": "x", "path": "a"}'))
        self.assertIsNone(path_argument("fetch_url", '{"url": "https://x"}'))

    def test_spellings_of_the_same_path_collapse(self):
        for spelling in ("a/b.py", "./a/b.py", "/a/b.py", "  a/b.py  "):
            self.assertEqual(
                path_argument("read_file", '{"path": "%s"}' % spelling.strip()),
                "a/b.py",
                spelling,
            )

    def test_list_dir_without_a_path_is_still_a_visit(self):
        """Its path is optional and defaults to the repo root."""
        self.assertEqual(path_argument("list_dir", "{}"), ".")
        self.assertEqual(path_argument("list_dir", '{"path": "/"}'), ".")

    def test_unusable_arguments_name_no_path(self):
        self.assertIsNone(path_argument("read_file", "not json"))
        self.assertIsNone(path_argument("read_file", "[1, 2]"))
        self.assertIsNone(path_argument("read_file", '{"start_line": 1}'))


class PathVisitTests(unittest.TestCase):
    """The waste the exact-signature counter cannot see.

    Job 9d210794 made 153 tool calls, 137 of them re-openings of a path it had
    already read — ``modular_blt.py`` 53 times, at different line ranges. Every
    one of those is a distinct signature, so ``repeats`` stays 0 while the
    budget drains."""

    def test_rereads_of_one_file_are_revisits_but_not_exact_repeats(self):
        guard = ToolRepeatGuard(6)
        for start in (1, 180, 400, 620):
            guard.observe(
                "read_file",
                '{"path": "src/m/modular_blt.py", "start_line": %d}' % start,
            )
        self.assertEqual(guard.repeats, 0)
        self.assertFalse(guard.tripped)
        self.assertEqual(guard.distinct_paths, 1)
        self.assertEqual(guard.path_revisits, 3)

    def test_a_file_read_once_is_not_waste(self):
        guard = ToolRepeatGuard(6)
        for i in range(10):
            guard.observe("read_file", '{"path": "f%d.py"}' % i)
        self.assertEqual(guard.distinct_paths, 10)
        self.assertEqual(guard.path_revisits, 0)

    def test_counting_survives_a_disabled_guard(self):
        """The nudge is configurable; the observability record is not."""
        guard = ToolRepeatGuard(0)
        for _ in range(5):
            self.assertIsNone(guard.observe("read_file", '{"path": "a.py"}'))
        self.assertEqual(guard.repeats, 0)
        self.assertEqual(guard.path_revisits, 4)

    def test_worst_paths_ranks_the_offenders(self):
        guard = ToolRepeatGuard(999)
        for _ in range(53):
            guard.observe("read_file", '{"path": "modular_blt.py"}')
        for _ in range(21):
            guard.observe("read_file", '{"path": "configuration_utils.py"}')
        guard.observe("read_file", '{"path": "once.py"}')
        self.assertEqual(
            guard.worst_paths(),
            [("modular_blt.py", 53), ("configuration_utils.py", 21)],
        )

    def test_stats_is_the_flat_record(self):
        guard = ToolRepeatGuard(6)
        guard.observe("read_file", '{"path": "a.py"}')
        guard.observe("read_file", '{"path": "a.py"}')
        self.assertEqual(
            guard.stats(),
            {"repeats": 1, "distinct_paths": 1, "path_revisits": 1},
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
