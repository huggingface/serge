"""Tests for the browse-transcript window.

The window exists to stop the input-token cap being spent re-sending tool
results (86% of a measured 2M-token task), so the numbers it reports have to be
right or the change cannot be shown to have helped. Its dangerous failure is
structural, not arithmetic: a request whose assistant `tool_calls` entry has no
matching `role: "tool"` reply is a 400 from every provider serge talks to, so
the shape assertions here matter more than the savings ones.
"""

import unittest

from reviewbot.transcript import elide_old_tool_results, elided_stub


def _tool(call_id: str, name: str = "read_file", content: str = "") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content or ("x" * 5_000),
    }


def _assistant(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        ],
    }


def _transcript(count: int) -> list[dict]:
    """A prefix plus ``count`` assistant/tool exchanges, as the loop builds it."""
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for index in range(count):
        messages.append(_assistant(f"c{index}"))
        messages.append(_tool(f"c{index}", content=f"result {index} " + "y" * 5_000))
    return messages


class WindowOffTests(unittest.TestCase):
    def test_zero_keeps_everything_and_does_not_copy(self) -> None:
        """0 is the default, so this is the shipped behaviour: inert."""
        messages = _transcript(30)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=0)
        self.assertIs(windowed, messages)
        self.assertEqual((elided, saved), (0, 0))

    def test_negative_is_treated_as_off(self) -> None:
        messages = _transcript(5)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=-3)
        self.assertIs(windowed, messages)
        self.assertEqual((elided, saved), (0, 0))

    def test_transcript_shorter_than_the_window_is_untouched(self) -> None:
        messages = _transcript(4)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=10)
        self.assertIs(windowed, messages)
        self.assertEqual((elided, saved), (0, 0))

    def test_exactly_the_window_is_untouched(self) -> None:
        messages = _transcript(6)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=6)
        self.assertIs(windowed, messages)
        self.assertEqual(elided, 0)


class ElisionTests(unittest.TestCase):
    def test_only_the_oldest_results_are_elided(self) -> None:
        messages = _transcript(10)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=3)
        self.assertEqual(elided, 7)
        self.assertGreater(saved, 0)
        tools = [m for m in windowed if m["role"] == "tool"]
        self.assertEqual(len(tools), 10)
        for message in tools[:7]:
            self.assertIn("has been removed from this transcript", message["content"])
        for index, message in enumerate(tools[7:], start=7):
            self.assertTrue(message["content"].startswith(f"result {index} "))

    def test_every_tool_reply_keeps_its_position_and_call_id(self) -> None:
        """The 400-shaped failure: a tool_calls entry with no matching reply."""
        messages = _transcript(8)
        windowed, _, _ = elide_old_tool_results(messages, keep_recent=2)
        self.assertEqual(len(windowed), len(messages))
        for before, after in zip(messages, windowed):
            self.assertEqual(before["role"], after["role"])
            self.assertEqual(before.get("tool_call_id"), after.get("tool_call_id"))
            self.assertEqual(before.get("name"), after.get("name"))
            self.assertEqual(before.get("tool_calls"), after.get("tool_calls"))

    def test_the_prefix_and_assistant_turns_are_never_touched(self) -> None:
        messages = _transcript(9)
        windowed, _, _ = elide_old_tool_results(messages, keep_recent=1)
        self.assertEqual(windowed[0], {"role": "system", "content": "sys"})
        self.assertEqual(windowed[1], {"role": "user", "content": "task"})
        self.assertEqual(
            [m for m in windowed if m["role"] == "assistant"],
            [m for m in messages if m["role"] == "assistant"],
        )

    def test_the_input_list_is_not_mutated(self) -> None:
        """Non-destructive by contract: the caller re-windows the full
        transcript every turn, which is what keeps the function pure and an
        elided message from being elided twice."""
        messages = _transcript(10)
        originals = [m["content"] for m in messages if m["role"] == "tool"]
        elide_old_tool_results(messages, keep_recent=2)
        self.assertEqual(
            [m["content"] for m in messages if m["role"] == "tool"], originals
        )

    def test_re_windowing_the_same_transcript_is_stable(self) -> None:
        messages = _transcript(12)
        first, elided_a, saved_a = elide_old_tool_results(messages, keep_recent=4)
        second, elided_b, saved_b = elide_old_tool_results(messages, keep_recent=4)
        self.assertEqual((elided_a, saved_a), (elided_b, saved_b))
        self.assertEqual(first, second)

    def test_saved_is_the_characters_actually_removed(self) -> None:
        messages = _transcript(6)
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=2)
        before = sum(len(m["content"]) for m in messages if m["role"] == "tool")
        after = sum(len(m["content"]) for m in windowed if m["role"] == "tool")
        self.assertEqual(saved, before - after)
        self.assertEqual(elided, 4)


class StubTests(unittest.TestCase):
    def test_the_stub_tells_the_model_it_can_rerun_the_call(self) -> None:
        """A bare "[elided]" invites the model to conclude the file is gone and
        abandon the path; it has to read as "this happened, do it again"."""
        stub = elided_stub("read_file", 5_000)
        self.assertIn("read_file", stub)
        self.assertIn("5,000 characters", stub)
        self.assertIn("Run the call again", stub)

    def test_the_tool_name_travels_into_the_stub(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            _tool("a", name="grep", content="g" * 4_000),
            _tool("b", name="read_file", content="r" * 4_000),
            _tool("c", name="read_file", content="keep"),
        ]
        windowed, _, _ = elide_old_tool_results(messages, keep_recent=1)
        self.assertIn("grep", windowed[1]["content"])
        self.assertIn("read_file", windowed[2]["content"])

    def test_a_nameless_reply_still_gets_a_stub(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "a", "content": "z" * 4_000},
            _tool("b", content="keep"),
        ]
        windowed, elided, _ = elide_old_tool_results(messages, keep_recent=1)
        self.assertEqual(elided, 1)
        self.assertIn("tool", windowed[0]["content"])


class SkipTests(unittest.TestCase):
    def test_a_result_is_never_made_longer_than_it_was(self) -> None:
        """The stub is ~200 chars, so eliding a two-line grep hit would spend
        tokens rather than save them."""
        messages = [
            {"role": "user", "content": "task"},
            _tool("a", content="one hit"),
            _tool("b", content="another hit"),
            _tool("c", content="x" * 9_000),
            _tool("d", content="keep"),
        ]
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=1)
        self.assertEqual(elided, 1)
        self.assertGreater(saved, 0)
        self.assertEqual(windowed[1]["content"], "one hit")
        self.assertEqual(windowed[2]["content"], "another hit")

    def test_min_chars_leaves_smaller_results_verbatim(self) -> None:
        messages = _transcript(3) + [_tool("small", content="s" * 3_000)]
        messages.append(_tool("newest", content="n" * 3_000))
        windowed, elided, _ = elide_old_tool_results(
            messages, keep_recent=1, min_chars=4_000
        )
        self.assertEqual(elided, 3)
        self.assertEqual(windowed[-2]["content"], "s" * 3_000)

    def test_non_string_content_is_skipped_not_crashed(self) -> None:
        messages = [
            {"role": "tool", "tool_call_id": "a", "name": "grep", "content": None},
            _tool("b", content="x" * 9_000),
            _tool("c", content="keep"),
        ]
        windowed, elided, _ = elide_old_tool_results(messages, keep_recent=1)
        self.assertEqual(elided, 1)
        self.assertIsNone(windowed[0]["content"])

    def test_nothing_elidable_returns_the_input_list(self) -> None:
        messages = [
            {"role": "user", "content": "task"},
            _tool("a", content="tiny"),
            _tool("b", content="tiny"),
            _tool("c", content="tiny"),
        ]
        windowed, elided, saved = elide_old_tool_results(messages, keep_recent=1)
        self.assertIs(windowed, messages)
        self.assertEqual((elided, saved), (0, 0))


if __name__ == "__main__":
    unittest.main()
