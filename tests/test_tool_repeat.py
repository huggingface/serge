"""Tests for the repeated-tool-call guard.

Reference failure: prod task 433e8274 issued the same two greps 21 and 44 times,
exhausted the 2M input-token cap in ~55 turns without ever reasoning about the
fix, and produced a patch that failed GPU verification — three rounds running.
"""

import unittest

from reviewbot.tool_repeat import (
    ToolRepeatGuard,
    normalize_arguments,
    path_argument,
    path_visit,
)


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

    def test_counting_survives_both_guards_being_disabled(self):
        """The nudges are configurable; the observability record is not."""
        guard = ToolRepeatGuard(0, path_revisit_limit=0, path_trip_after=0)
        for _ in range(5):
            self.assertIsNone(guard.observe("read_file", '{"path": "a.py"}'))
        self.assertEqual(guard.repeats, 0)
        self.assertEqual(guard.path_revisits, 4)
        self.assertFalse(guard.tripped)

    def test_the_two_allowances_are_independent(self):
        """Turning off the exact-repeat guard must not turn off the path nudge:
        they catch different failures and have separate budgets."""
        guard = ToolRepeatGuard(0, path_revisit_limit=2)
        self.assertFalse(guard.enabled)  # exact-repeat guard off
        for _ in range(2):
            self.assertIsNone(guard.observe("read_file", '{"path": "a.py"}'))
        self.assertIsNotNone(guard.observe("read_file", '{"path": "a.py"}'))
        # …and vice versa: the path nudge off leaves exact repeats caught.
        guard = ToolRepeatGuard(2, path_revisit_limit=0)
        guard.observe("read_file", '{"path": "a.py"}')
        self.assertIn(
            "already made this exact", guard.observe("read_file", '{"path": "a.py"}')
        )

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


class PathNudgeTests(unittest.TestCase):
    """The correction for re-opening a path, which the exact-match nudge misses."""

    FILE = "src/transformers/models/blt/modular_blt.py"

    def _read(self, guard, start=None, end=None, path=None):
        args = {"path": path or self.FILE}
        if start is not None:
            args["start_line"] = start
        if end is not None:
            args["end_line"] = end
        import json as _json

        return guard.observe("read_file", _json.dumps(args))

    def test_the_allowance_is_spent_before_nudging(self):
        guard = ToolRepeatGuard(6, path_revisit_limit=3)
        for i in range(3):
            self.assertIsNone(self._read(guard, i * 100, i * 100 + 99))
        self.assertIsNotNone(self._read(guard, 400, 500))

    def test_the_nudge_names_the_file_and_what_was_already_served(self):
        """ "You are repeating" is not actionable; "you already have lines
        1-200, 180-420" is."""
        guard = ToolRepeatGuard(6, path_revisit_limit=2)
        self._read(guard, 1, 200)
        self._read(guard, 180, 420)
        note = self._read(guard, 400, 650)
        self.assertIn(f"`{self.FILE}`", note)
        self.assertIn("3 times", note)
        self.assertIn("lines 1-200", note)
        self.assertIn("lines 180-420", note)
        # Not the range being served right now — that one is new to the model.
        self.assertNotIn("lines 400-650", note)
        self.assertIn("cannot tell you anything new", note)

    def test_repeated_identical_ranges_are_described_once(self):
        guard = ToolRepeatGuard(0, path_revisit_limit=1)
        for _ in range(4):
            note = self._read(guard, 1, 200)
        self.assertEqual(note.count("lines 1-200"), 1)

    def test_a_long_history_is_elided(self):
        guard = ToolRepeatGuard(0, path_revisit_limit=1)
        for i in range(9):
            note = self._read(guard, i * 100, i * 100 + 50)
        self.assertIn("…", note)
        self.assertLessEqual(note.count("lines "), 5)

    def test_a_whole_file_read_says_so(self):
        guard = ToolRepeatGuard(0, path_revisit_limit=1)
        self._read(guard)
        self.assertIn("the whole file", self._read(guard))

    def test_list_dir_is_phrased_as_listing(self):
        guard = ToolRepeatGuard(0, path_revisit_limit=1)
        guard.observe("list_dir", '{"path": "src/transformers"}')
        note = guard.observe("list_dir", '{"path": "src/transformers"}')
        self.assertIn("listed", note)
        self.assertNotIn("Re-reading", note)
        self.assertIn("Listing a directory", note)

    def test_different_files_never_nudge(self):
        """A genuine investigation opens many files once each."""
        guard = ToolRepeatGuard(6, path_revisit_limit=3)
        for i in range(40):
            self.assertIsNone(self._read(guard, path=f"m{i}.py"))
        self.assertFalse(guard.tripped)

    def test_grep_is_not_a_revisit(self):
        """Same path, different pattern is a new search, not a re-read."""
        guard = ToolRepeatGuard(0, path_revisit_limit=1)
        for pat in ("Alpha", "Beta", "Gamma", "Delta"):
            note = guard.observe("grep", '{"pattern": "%s", "path": "src"}' % pat)
            self.assertIsNone(note)
        self.assertEqual(guard.path_revisits, 0)


class PathTripTests(unittest.TestCase):
    """The cut-off, which is a separate budget from the exact-repeat one."""

    def test_the_prod_shape_is_cut_off(self):
        """Task 9d210794: 153 calls, 137 of them re-opening an already-opened
        path, `modular_blt.py` 53 times — and the exact-match counter saw none
        of it because every read used a different line range."""
        guard = ToolRepeatGuard(6, path_revisit_limit=3, path_trip_after=40)
        served = 0
        for i in range(153):
            guard.observe(
                "read_file",
                '{"path": "modular_blt.py", "start_line": %d, "end_line": %d}'
                % (i * 10, i * 10 + 200),
            )
            served += 1
            if guard.tripped:
                break
        self.assertTrue(guard.tripped)
        self.assertLess(served, 60)
        # The exact-argument counter never fired — this is the whole point.
        self.assertEqual(guard.repeats, 0)

    def test_the_trip_note_tells_the_model_to_stop(self):
        guard = ToolRepeatGuard(0, path_revisit_limit=1, path_trip_after=3)
        for i in range(5):
            note = guard.observe("read_file", '{"path": "a.py", "start_line": %d}' % i)
        self.assertTrue(guard.tripped)
        self.assertIn("no further tool calls will be answered", note)

    def test_a_zero_trip_nudges_forever_without_cutting_off(self):
        """The escape hatch: keep the correction, never end the session on it."""
        guard = ToolRepeatGuard(0, path_revisit_limit=2, path_trip_after=0)
        for i in range(100):
            note = guard.observe("read_file", '{"path": "a.py", "start_line": %d}' % i)
        self.assertIsNotNone(note)
        self.assertFalse(guard.tripped)
        self.assertEqual(guard.path_revisits, 99)

    def test_the_two_trips_are_separate_budgets(self):
        """A session can exhaust one without touching the other."""
        guard = ToolRepeatGuard(3, path_revisit_limit=99, path_trip_after=99)
        for _ in range(4):
            guard.observe("grep", '{"pattern": "x"}')
        self.assertTrue(guard.tripped)  # exact repeats only
        self.assertEqual(guard.path_revisits, 0)


class PathVisitRangeTests(unittest.TestCase):
    def test_reports_path_and_range(self):
        self.assertEqual(
            path_visit("read_file", '{"path": "a.py", "start_line": 5, "end_line": 9}'),
            ("a.py", "lines 5-9"),
        )
        self.assertEqual(
            path_visit("read_file", '{"path": "a.py"}'), ("a.py", "the whole file")
        )
        self.assertEqual(
            path_visit("read_file", '{"path": "a.py", "start_line": 5}'),
            ("a.py", "from line 5"),
        )
        self.assertEqual(
            path_visit("read_file", '{"path": "a.py", "end_line": 9}'),
            ("a.py", "up to line 9"),
        )
        self.assertEqual(
            path_visit("list_dir", '{"path": "src"}'), ("src", "the directory")
        )
        self.assertIsNone(path_visit("grep", '{"pattern": "x", "path": "src"}'))

    def test_a_non_integer_range_does_not_break_the_label(self):
        self.assertEqual(
            path_visit("read_file", '{"path": "a.py", "start_line": "five"}'),
            ("a.py", "the whole file"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
