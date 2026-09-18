"""The anchored-edit answer format (:mod:`reviewbot.anchored_edits`).

A unified diff makes the model responsible for line numbers, hunk counts and
context it has to re-type. That bookkeeping is what killed three prod tasks on
2026-09-16/17: `zamba` lost on one trailing comma inside a tensor it had to
transcribe, `qwen3_omni_moe` on an `@@` count that did not match its own hunk
body. An anchored edit asks only for the text: replace `old` with `new`, where
`old` occurs exactly once.

The two properties that have to hold, and are what these tests are for:

* **exactly once, or refuse** — no fuzz, no offset search, no nearest match. An
  ambiguous anchor must be a rejection, never a silent edit somewhere else.
* **all or nothing** — one bad anchor leaves the whole answer unapplied, so a
  worktree never holds half of a rejected fix.
"""

from reviewbot.anchored_edits import (
    Edit,
    apply_edits,
    parse_edits,
    rejection_feedback,
    summarize,
)

_FILE = "alpha\nbeta\ngamma\nbeta\ndelta\n"


def _reader(files):
    return lambda path: files.get(path)


# ── the schema ──────────────────────────────────────────────────────────────


def test_a_well_formed_list_parses():
    parsed = parse_edits([{"path": "a/f.py", "old": "x", "new": "y"}])
    assert parsed.problems == []
    assert parsed.edits == [Edit(path="f.py", old="x", new="y")]


def test_a_non_list_is_a_shape_problem():
    assert parse_edits({"path": "f.py"}).problems


def test_an_empty_old_is_rejected():
    """An empty anchor matches everywhere, so it identifies nothing."""
    parsed = parse_edits([{"path": "f.py", "old": "", "new": "y"}])
    assert parsed.edits == []
    assert '"old" must be a non-empty string' in parsed.problems[0]


def test_an_edit_that_changes_nothing_is_rejected():
    parsed = parse_edits([{"path": "f.py", "old": "x", "new": "x"}])
    assert parsed.edits == []
    assert "changes nothing" in parsed.problems[0]


def test_too_many_edits_is_rejected_as_a_whole():
    parsed = parse_edits(
        [{"path": "f.py", "old": f"x{i}", "new": "y"} for i in range(41)]
    )
    assert parsed.edits == []
    assert "more than the 40 allowed" in parsed.problems[0]


def test_a_path_escape_is_refused():
    """The applier writes files, so the path the model chose is checked before
    anything is read or written — the one problem here that is a security
    boundary and not a quality one."""
    for bad in ("../../etc/passwd", "/etc/passwd", "a/../../x", "C:/windows/x"):
        parsed = parse_edits([{"path": bad, "old": "x", "new": "y"}])
        assert parsed.edits == [], bad
        assert "repository-relative" in parsed.problems[0], bad


def test_diff_style_prefixes_are_stripped():
    """The model has spent the whole prompt reading `a/`-prefixed diffs."""
    parsed = parse_edits([{"path": "b/src/f.py", "old": "x", "new": "y"}])
    assert parsed.edits[0].path == "src/f.py"


# ── exactly once, or refuse ─────────────────────────────────────────────────


def test_a_unique_anchor_applies():
    contents, problems = apply_edits(
        [Edit("f.py", "gamma", "GAMMA")], read=_reader({"f.py": _FILE})
    )
    assert problems == []
    assert contents == {"f.py": "alpha\nbeta\nGAMMA\nbeta\ndelta\n"}


def test_an_ambiguous_anchor_is_refused_with_its_locations():
    contents, problems = apply_edits(
        [Edit("f.py", "beta", "BETA")], read=_reader({"f.py": _FILE})
    )
    assert contents == {}
    assert "occurs 2 times" in problems[0]
    # The locations are what tells the model how to extend the anchor.
    assert "line 2" in problems[0] and "line 4" in problems[0]


def test_a_missing_anchor_is_refused():
    contents, problems = apply_edits(
        [Edit("f.py", "nowhere", "x")], read=_reader({"f.py": _FILE})
    )
    assert contents == {}
    assert "does not occur" in problems[0]


def test_a_missing_file_is_refused():
    contents, problems = apply_edits([Edit("f.py", "x", "y")], read=_reader({}))
    assert contents == {}
    assert "no such file" in problems[0]


def test_one_bad_anchor_drops_the_whole_answer():
    """All or nothing: a worktree must never hold half of a rejected fix."""
    contents, problems = apply_edits(
        [Edit("f.py", "gamma", "GAMMA"), Edit("f.py", "nowhere", "x")],
        read=_reader({"f.py": _FILE}),
    )
    assert contents == {}
    assert len(problems) >= 1


def test_every_edit_is_reported_not_just_the_first():
    """One round trip per rejection, not one per bad anchor."""
    _, problems = apply_edits(
        [Edit("f.py", "nowhere", "x"), Edit("f.py", "beta", "y")],
        read=_reader({"f.py": _FILE}),
    )
    reasons = [p for p in problems if p.startswith("edit ")]
    assert len(reasons) == 2
    assert "edit 1" in reasons[0] and "edit 2" in reasons[1]


def test_edits_compose_in_order_on_one_file():
    contents, problems = apply_edits(
        [Edit("f.py", "alpha", "one"), Edit("f.py", "delta", "four")],
        read=_reader({"f.py": _FILE}),
    )
    assert problems == []
    assert contents["f.py"] == "one\nbeta\ngamma\nbeta\nfour\n"


def test_an_anchor_may_be_created_by_an_earlier_edit():
    contents, _ = apply_edits(
        [Edit("f.py", "alpha", "zeta"), Edit("f.py", "zeta", "omega")],
        read=_reader({"f.py": _FILE}),
    )
    assert contents["f.py"].startswith("omega\n")


def test_an_anchor_made_ambiguous_by_an_earlier_edit_is_refused():
    """Uniqueness is checked against the text as it stands, not as it arrived —
    otherwise composing edits could reintroduce exactly the ambiguity the
    format exists to rule out."""
    contents, problems = apply_edits(
        [Edit("f.py", "alpha", "beta"), Edit("f.py", "beta", "x")],
        read=_reader({"f.py": _FILE}),
    )
    assert contents == {}
    assert "occurs 3 times" in problems[0]


def test_an_empty_new_deletes():
    contents, _ = apply_edits(
        [Edit("f.py", "gamma\n", "")], read=_reader({"f.py": _FILE})
    )
    assert contents["f.py"] == "alpha\nbeta\nbeta\ndelta\n"


def test_a_file_whose_edits_cancel_out_is_not_a_change():
    contents, problems = apply_edits(
        [Edit("f.py", "alpha", "zeta"), Edit("f.py", "zeta", "alpha")],
        read=_reader({"f.py": _FILE}),
    )
    assert problems == []
    assert contents == {}


# ── what the model is told ──────────────────────────────────────────────────


def test_a_near_miss_is_answered_with_the_real_line():
    """The zamba shape: the model asserts a value the file does not hold, and
    asserting it again is exactly what it will do next unless it is shown the
    line that is really there."""
    text = (
        "    self.assertEqual(\n"
        '        "[PAD][PAD][PAD][PAD][PAD][PAD]<s> Tell me a story",\n'
        "        decoded,\n"
        "    )\n"
    )
    _, problems = apply_edits(
        [Edit("t.py", '        "<s><s> Tell me a story",\n', "x")],
        read=_reader({"t.py": text}),
    )
    hint = "\n".join(problems)
    assert "does not occur" in hint
    assert "[PAD][PAD]" in hint, "the real line has to be quoted back"
    # Numbered the way tools.read_file numbers its output, so the model is not
    # asked to reconcile two renderings of the same file.
    assert "     2\t" in hint


def test_the_hint_probes_the_longest_anchor_line_not_the_first():
    """A multi-line anchor opens on boilerplate — `self.assertEqual(` matches
    every assertion in the file — and the line that disagrees is the long one
    carrying the value. Probing the first line quoted back two unrelated
    assertions on the real zamba reply and never reached the right place."""
    text = (
        "        self.assertEqual(\n"
        "            first,\n"
        '            "<s> Hey how are you doing on this lovely evening?",\n'
        "        )\n"
        "        self.assertEqual(\n"
        "            second,\n"
        '            "[PAD][PAD]<s> Tell me a story about a difficult situation",\n'
        "        )\n"
    )
    _, problems = apply_edits(
        [
            Edit(
                "t.py",
                "        self.assertEqual(\n            second,\n"
                '            "<s> Tell me a story about a difficult situation",\n',
                "x",
            )
        ],
        read=_reader({"t.py": text}),
    )
    hint = "\n".join(problems)
    # One window, centred on the line that actually disagrees. Probing
    # `self.assertEqual(` would have quoted both assertions and neither centre.
    assert "[PAD][PAD]" in hint
    assert hint.count("The file does contain") == 1
    assert "     7\t" in hint


def test_an_anchor_from_another_file_gets_no_invented_hint():
    _, problems = apply_edits(
        [Edit("f.py", "something else entirely", "x")], read=_reader({"f.py": _FILE})
    )
    assert len(problems) == 1, "no candidate is better than a wrong candidate"


def test_the_feedback_never_echoes_the_rejected_anchor():
    """`apply_error_for_model` strips `git apply -v`'s search block for the same
    reason: quoting the model's own wrong text next to the real content is what
    measurably kept the fabrication alive (3 of 3 runs)."""
    _, problems = apply_edits(
        [Edit("f.py", "nowhere at all", "x")], read=_reader({"f.py": _FILE})
    )
    assert "nowhere at all" not in rejection_feedback(problems)


def test_the_feedback_asks_for_a_decline_over_an_invention():
    feedback = rejection_feedback(["edit 1 (f.py): `old` does not occur in the file."])
    assert "empty `edits` list" in feedback
    assert "rather than inventing an anchor" in feedback


def test_the_summary_is_one_line_of_reasons():
    problems = [
        "edit 1 (f.py): `old` does not occur in the file.",
        "  The file does contain this, which is close",
        "edit 2 (g.py): `old` occurs 3 times",
    ]
    line = summarize(problems)
    assert "edit 1" in line and "edit 2" in line
    assert "\n" not in line
    assert "The file does contain" not in line, "quoted content is not an error string"
