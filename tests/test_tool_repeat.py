"""Tests for the repeated-tool-call guard.

Reference failure: prod task 433e8274 issued the same two greps 21 and 44 times,
exhausted the 2M input-token cap in ~55 turns without ever reasoning about the
fix, and produced a patch that failed GPU verification — three rounds running.
"""

import unittest

from reviewbot.tool_repeat import ToolRepeatGuard, normalize_arguments


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
