"""§3.4: the patch against what maintainers already decided about those lines.

Four things carry the check and are what these tests pin:

1. **The anchors.** Old-side line numbers, modified lines before insertion
   points, created files skipped. Get this wrong and every later step asks
   about the wrong line while looking exactly as confident.
2. **The trust filter.** Only ``authoritative`` is maintainer guidance. A
   relore that stops labelling, or starts serving a bot comment as evidence,
   must produce SILENCE here rather than an unsourced warning on a public PR.
3. **The verdict.** A grader that answers in prose, or claims a contradiction
   with nothing to point at, is a non-answer — not a warning.
4. **The silence.** No guidance found means no PR section at all. "Checked,
   found nothing" on lines nobody ever reviewed is a clean bill of health that
   was never issued.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from reviewbot import guidance
from reviewbot.guidance import (
    Anchor,
    GuidanceResult,
    Quote,
    Rationale,
    build_user_prompt,
    check_patch,
    collect_rationale,
    patch_anchors,
)
from reviewbot.relore_tool import ReloreEnv


@pytest.fixture
def env() -> ReloreEnv:
    return ReloreEnv(repo="huggingface/transformers", api="https://relore.example")


MODIFY = """\
--- a/src/transformers/models/blt/modeling_blt.py
+++ b/src/transformers/models/blt/modeling_blt.py
@@ -120,7 +120,7 @@ class BltAttention(nn.Module):
     def forward(self, hidden_states):
         query = self.q_proj(hidden_states)
-        key = self.k_proj(hidden_states)
+        key = self.k_proj(hidden_states).to(query.dtype)
         value = self.v_proj(hidden_states)
         return query, key, value
"""


def test_a_rewritten_line_is_anchored_at_its_old_number():
    anchors = patch_anchors(MODIFY)
    assert anchors == [
        Anchor(
            path="src/transformers/models/blt/modeling_blt.py",
            line=122,
            modified=True,
        )
    ]


def test_an_insertion_anchors_on_the_line_it_follows():
    """A line the patch only inserts did not exist when reviewers spoke, so the
    question we can actually ask is about the line it lands after."""
    patch = """\
--- a/a.py
+++ b/a.py
@@ -10,3 +11,4 @@
 first
 second
+inserted
 third
"""
    assert patch_anchors(patch) == [Anchor(path="a.py", line=11, modified=False)]


def test_modified_lines_are_asked_about_before_insertion_points():
    patch = """\
--- a/a.py
+++ b/a.py
@@ -1,3 +1,4 @@
 keep
+added
@@ -50,3 +51,3 @@
-gone
 tail
"""
    assert [(a.line, a.modified) for a in patch_anchors(patch)] == [
        (50, True),
        (1, False),
    ]


def test_a_created_file_has_no_history_to_check():
    patch = """\
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1,2 @@
+def test_x():
+    assert True
"""
    assert patch_anchors(patch) == []


def test_a_deleted_file_is_skipped_too():
    patch = """\
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def gone():
-    pass
"""
    assert patch_anchors(patch) == []


def test_a_removed_line_that_looks_like_a_header_does_not_retarget_the_patch():
    """The parser counts hunk lines rather than trusting `---`/`+++`.

    A patch that deletes a diff fragment from a docstring or a test fixture
    carries `--- a/…` INSIDE a hunk. Read as a header it silently re-points
    every following anchor at a file the patch never touched, and nothing
    downstream can tell.
    """
    patch = """\
--- a/docs/example.py
+++ b/docs/example.py
@@ -5,4 +5,3 @@
 SAMPLE = '''
---- a/other/file.py
-+++ b/other/file.py
 '''
"""
    assert [(a.path, a.line) for a in patch_anchors(patch)] == [
        ("docs/example.py", 6),
        ("docs/example.py", 7),
    ]


def test_one_file_cannot_spend_the_whole_budget():
    hunks = "".join(
        f"@@ -{10 * n},3 +{10 * n},3 @@\n-old{n}\n+new{n}\n context\n"
        for n in range(1, 8)
    )
    patch = f"--- a/a.py\n+++ b/a.py\n{hunks}"
    anchors = patch_anchors(patch, max_anchors=8, max_per_file=3)
    assert len(anchors) == 3


def test_the_total_budget_is_a_hard_cap():
    files = "".join(
        f"--- a/f{n}.py\n+++ b/f{n}.py\n@@ -1,2 +1,2 @@\n-old\n+new\n ctx\n"
        for n in range(10)
    )
    assert len(patch_anchors(files, max_anchors=4)) == 4


# -- what relore gives back --------------------------------------------------


def _payload(*comments, number=48000, key="anchored"):
    return {
        "number": number,
        "thread": {
            "title": "Fix Inkling inputs_embeds",
            "url": f"https://github.com/huggingface/transformers/pull/{number}",
            "state": "merged",
        },
        key: list(comments),
    }


def _comment(trust="authoritative", text="keep this on the CPU path", author="cyril"):
    return {"trust": trust, "text": text, "author": author, "age": "3 months ago"}


class _FakeRun:
    """``subprocess.run`` for the ``relore --json why`` calls, in order."""

    def __init__(self, payloads, returncode=0):
        self.payloads = list(payloads)
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        import json

        payload = self.payloads.pop(0) if self.payloads else {}
        return SimpleNamespace(
            returncode=self.returncode, stdout=json.dumps(payload), stderr=""
        )


def test_only_the_authoritative_tier_is_quoted(monkeypatch, env):
    """A contributor's claim and a bot's comment are not maintainer guidance.

    Both are counted, because "no guidance here" and "guidance we filtered out"
    are different facts about a line.
    """
    monkeypatch.setattr(
        guidance.subprocess,
        "run",
        _FakeRun(
            [
                _payload(
                    _comment(),
                    _comment(trust="reported", text="I think this is wrong"),
                    _comment(trust="machine", text="LGTM from a bot"),
                )
            ]
        ),
    )
    [result] = collect_rationale(env, [Anchor("a.py", 10, True)])
    assert [q.text for q in result.quotes] == ["keep this on the CPU path"]
    assert result.skipped == {"reported": 1, "machine": 1}


def test_an_unlabelled_comment_is_never_promoted_to_guidance(monkeypatch, env):
    """If relore stops serving `trust`, this check must go quiet.

    The alternative — treating an unlabelled comment as authoritative — puts a
    warning citing "a maintainer" on a public pull request on the strength of a
    field that is no longer there.
    """
    monkeypatch.setattr(
        guidance.subprocess,
        "run",
        _FakeRun([_payload({"text": "some comment", "author": "someone"})]),
    )
    assert collect_rationale(env, [Anchor("a.py", 10, True)]) == []


def test_a_quote_is_labelled_by_where_it_came_from_and_review_summaries_are_dropped(
    monkeypatch, env
):
    """Measured on the 10 most recent serge fix patches (2026-09-25): the
    `reviews` group was 34 of 67 retrieved quotes and most of its distinct texts
    were pleasantries — "Thanks 🫡", "Thank you", "OK, let see". A verdict on a
    whole pull request cannot say whether one line contradicts anything, and it
    attaches to every anchor inside that pull request, so it arrives once per
    hunk."""
    monkeypatch.setattr(
        guidance.subprocess,
        "run",
        _FakeRun(
            [
                {
                    "number": 1,
                    "thread": {"title": "t", "url": "u", "state": "merged"},
                    "anchored": [_comment(text="on the line")],
                    "on_file": [_comment(text="elsewhere in the file")],
                    "reviews": [_comment(text="Thanks for your hard work!")],
                }
            ]
        ),
    )
    [result] = collect_rationale(env, [Anchor("a.py", 10, True)])
    assert [q.where for q in result.quotes] == [
        "on this line",
        "elsewhere in the same pull request",
    ]


def test_one_comment_is_quoted_once_however_many_hunks_reach_it(monkeypatch, env):
    """Three hunks in one file resolve to one pull request and get its
    discussion three times — 24 of 67 quotes (36%) in the 2026-09-25
    measurement. A grader shown the same sentence three times is being told,
    wrongly, that three reviewers said it."""
    payload = _payload(_comment(text="keep the cast on CPU"))
    monkeypatch.setattr(
        guidance.subprocess, "run", _FakeRun([payload, dict(payload), dict(payload)])
    )
    results = collect_rationale(
        env,
        [Anchor("a.py", 10, True), Anchor("a.py", 20, True), Anchor("a.py", 30, True)],
    )
    assert sum(len(item.quotes) for item in results) == 1
    # And an anchor left with nothing new to say does not become a third
    # "line with guidance" in the count the PR section prints.
    assert len(results) == 1
    assert results[0].anchor.line == 10


def test_a_daemon_that_is_down_costs_the_check_its_evidence_and_nothing_else(
    monkeypatch, env
):
    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="relore", timeout=45)

    monkeypatch.setattr(guidance.subprocess, "run", boom)
    assert collect_rationale(env, [Anchor("a.py", 10, True)]) == []


def test_a_long_comment_is_trimmed_and_marked(monkeypatch, env):
    monkeypatch.setattr(
        guidance.subprocess,
        "run",
        _FakeRun([_payload(_comment(text="x" * (guidance.MAX_QUOTE_CHARS + 500)))]),
    )
    [result] = collect_rationale(env, [Anchor("a.py", 10, True)])
    assert result.quotes[0].text.endswith("[…]")
    assert len(result.quotes[0].text) <= guidance.MAX_QUOTE_CHARS + 8


# -- the prompt --------------------------------------------------------------


def _rationale(n=1, quotes=1, chars=100):
    return Rationale(
        anchor=Anchor(f"f{n}.py", 10 * n, True),
        number=40000 + n,
        title=f"PR {n}",
        url="https://example/1",
        state="merged",
        quotes=[
            Quote(
                author="cyril",
                age="2 months ago",
                text="g" * chars,
                url="https://example/c",
                where="on this line",
            )
            for _ in range(quotes)
        ],
    )


def test_the_prompt_quotes_every_comment_and_names_its_line():
    body, dropped = build_user_prompt("the patch", [_rationale(1), _rationale(2)])
    assert dropped == 0
    assert "f1.py:10 was last changed in #40001" in body
    assert "f2.py:20 was last changed in #40002" in body
    assert body.count("> " + "g" * 100) == 2


def test_the_patch_is_never_trimmed_to_fit_the_evidence():
    """A grader shown half a patch judges a change it cannot see. Quotes are
    what gives, and the count of what went is returned so the log can say so."""
    patch = "p" * 4000
    body, dropped = build_user_prompt(
        patch,
        [_rationale(n, quotes=2, chars=1000) for n in range(1, 6)],
        max_chars=6000,
    )
    assert patch in body
    assert dropped > 0
    assert len(body) <= 6000


# -- the verdict -------------------------------------------------------------


def test_a_verdict_is_read_out_of_prose_and_a_code_fence():
    content = 'Sure!\n```json\n{"contradicts": true, "note": "moves it to GPU", "citations": ["#1", "#1", "#2"]}\n```'
    contradicts, note, citations = guidance._verdict(content)
    assert contradicts is True
    assert note == "moves it to GPU"
    assert citations == ["#1", "#2"]


def test_a_contradiction_with_nothing_to_point_at_is_not_a_warning():
    """A ⚠️ a reviewer cannot act on is worse than none: it is the thing that
    teaches them to skip the section."""
    contradicts, note, _ = guidance._verdict('{"contradicts": true, "note": "  "}')
    assert contradicts is False
    assert note == ""


def test_an_unparseable_answer_establishes_nothing():
    assert guidance._verdict("I could not decide.") == (False, "", [])
    assert guidance._verdict(None) == (False, "", [])


def test_a_narrating_grader_is_cut_off_before_it_reaches_the_pr_body():
    long = "n" * (guidance.MAX_NOTE_CHARS + 400)
    _, note, _ = guidance._verdict('{"contradicts": true, "note": "%s"}' % long)
    assert note.endswith("[…]")
    assert len(note) <= guidance.MAX_NOTE_CHARS + 8


# -- what reaches the pull request -------------------------------------------


def test_no_guidance_means_no_section_at_all():
    result = GuidanceResult(anchors=4, with_guidance=0, asked=False)
    assert result.pr_section() == ""
    assert "no maintainer guidance" in result.log_line()


def test_a_clean_check_is_one_line_and_says_what_it_read():
    result = GuidanceResult(
        anchors=3, with_guidance=2, quotes=3, asked=True, citations=["#40001"]
    )
    section = result.pr_section()
    assert section.count("\n") <= 3
    assert "no contradiction found" in section
    assert "#40001" in section
    assert "⚠️" not in section


def test_a_contradiction_is_a_heading_that_names_what_it_is():
    result = GuidanceResult(
        anchors=3,
        with_guidance=1,
        quotes=2,
        asked=True,
        contradicts=True,
        note="The patch moves the cast the reviewer asked to keep on CPU.",
        citations=["#40001"],
    )
    section = result.pr_section()
    assert "### ⚠️ May contradict maintainer guidance" in section
    assert "not** a test result" in section
    assert result.note in section


# -- the whole check ---------------------------------------------------------


class _FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls: list[dict] = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(content=self.content, usage={})


def test_the_check_does_not_run_without_relore_or_a_model(env):
    assert check_patch(None, _FakeLLM("{}"), patch=MODIFY) is None
    assert check_patch(env, None, patch=MODIFY) is None
    assert check_patch(env, _FakeLLM("{}"), patch="  ") is None


def test_the_model_is_never_asked_when_no_maintainer_spoke(monkeypatch, env):
    monkeypatch.setattr(guidance.subprocess, "run", _FakeRun([{"number": None}]))
    llm = _FakeLLM('{"contradicts": true, "note": "should not happen"}')
    result = check_patch(env, llm, patch=MODIFY)
    assert llm.calls == []
    assert result is not None and result.with_guidance == 0
    assert result.asked is False
    assert result.pr_section() == ""


def test_the_happy_path_asks_once_and_reports_the_threads_it_read(monkeypatch, env):
    monkeypatch.setattr(
        guidance.subprocess, "run", _FakeRun([_payload(_comment(), number=47827)])
    )
    llm = _FakeLLM('{"contradicts": false, "note": "", "citations": []}')
    result = check_patch(env, llm, patch=MODIFY)
    assert len(llm.calls) == 1
    assert result is not None
    assert result.asked is True and result.contradicts is False
    # The grader listed nothing, so the section still names what was read.
    assert result.citations == ["#47827"]
    assert "no contradiction found" in result.pr_section()


def test_the_repo_is_serges_fact_on_every_call(monkeypatch, env):
    run = _FakeRun([_payload(_comment())])
    monkeypatch.setattr(guidance.subprocess, "run", run)
    check_patch(env, _FakeLLM('{"contradicts": false}'), patch=MODIFY)
    assert run.calls[0][-2:] == ["--repo", "huggingface/transformers"]
    assert "--json" in run.calls[0]


def test_a_model_that_raises_never_costs_the_task_its_pull_request(monkeypatch, env):
    monkeypatch.setattr(guidance.subprocess, "run", _FakeRun([_payload(_comment())]))

    class _Boom:
        def complete(self, *_args, **_kwargs):
            raise RuntimeError("provider is down")

    result = check_patch(env, _Boom(), patch=MODIFY)
    assert result is not None
    assert result.asked is False
    assert result.pr_section() == ""


def test_the_log_line_says_which_of_the_three_outcomes_happened():
    assert "no line with a history" in GuidanceResult().log_line()
    assert "no maintainer guidance" in GuidanceResult(anchors=2).log_line()
    assert (
        "no contradiction"
        in GuidanceResult(anchors=2, with_guidance=1, quotes=1, asked=True).log_line()
    )
    assert (
        "⚠️"
        in GuidanceResult(
            anchors=2, with_guidance=1, quotes=1, asked=True, contradicts=True
        ).log_line()
    )


def test_the_rewritten_half_of_a_replacement_is_not_a_second_question():
    """`-old` / `+new` is ONE change. Anchoring the `+` half as well asks
    relore about the line above it, which is a second daemon call for a
    question the `-` half already asked better."""
    patch = """\
--- a/a.py
+++ b/a.py
@@ -30,3 +30,3 @@
 ctx
-old
+new
"""
    assert patch_anchors(patch) == [Anchor(path="a.py", line=31, modified=True)]


# -- the wiring --------------------------------------------------------------


def test_the_check_is_off_unless_the_operator_turned_it_on():
    """Default OFF, unlike the brevity passes: this one publishes a judgement
    about a named maintainer's review on a public pull request."""
    import os
    from unittest.mock import patch as patch_env

    from reviewbot.config import Config

    with patch_env.dict(os.environ, {"LLM_API_KEY": "t"}, clear=True):
        assert Config.from_env(require_app=False).task_guidance_check is False
    with patch_env.dict(
        os.environ,
        {
            "LLM_API_KEY": "t",
            "TASK_GUIDANCE_CHECK": "1",
            "GUIDANCE_MAX_ANCHORS": "3",
        },
        clear=True,
    ):
        cfg = Config.from_env(require_app=False)
    assert cfg.task_guidance_check is True
    assert cfg.guidance_max_anchors == 3


def test_the_section_rides_under_the_gpu_verdict_not_instead_of_it():
    """Both sections are evidence about the change, and this is the weaker of
    the two: an LLM judgement about review prose, next to a test result."""
    from reviewbot.tasks import TaskPlan, _decorate_body

    plan = TaskPlan(
        title="t",
        body="what the fix does",
        patch="",
        guidance_note="\n---\n### ⚠️ May contradict maintainer guidance\nbecause X",
    )
    req = SimpleNamespace(
        owner="huggingface",
        repo="transformers",
        repo_full_name="huggingface/transformers",
        instruction="",
        context="",
        mode="new_pr",
    )
    body = _decorate_body(
        SimpleNamespace(is_staging=False),
        plan,
        req,
        verification_footer="\n---\n### ✅ Verified on GPU\nran the tests",
    )
    assert body.index("✅ Verified on GPU") < body.index("May contradict")
    assert body.index("May contradict") < body.index("produced automatically")


def test_a_plan_without_the_check_adds_nothing_to_the_body():
    from reviewbot.tasks import TaskPlan, _decorate_body

    plan = TaskPlan(title="t", body="what the fix does", patch="")
    req = SimpleNamespace(
        owner="huggingface",
        repo="transformers",
        repo_full_name="huggingface/transformers",
        instruction="",
        context="",
        mode="new_pr",
    )
    body = _decorate_body(SimpleNamespace(is_staging=False), plan, req)
    assert "maintainer guidance" not in body.lower()


def test_a_lookup_that_did_not_answer_is_never_read_as_no_guidance(monkeypatch, env):
    """A daemon that is down and a line nobody reviewed both produce no quotes.

    Reading the first as the second turns "relore was unreachable" into "this
    patch contradicts nothing" — which is the sentence a CLEAN check prints, on
    a public pull request, with no evidence behind it.
    """

    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="relore", timeout=45)

    monkeypatch.setattr(guidance.subprocess, "run", boom)
    result = check_patch(env, _FakeLLM("{}"), patch=MODIFY)
    assert result is not None
    assert result.lookups_failed == result.anchors == 1
    assert "did not answer" in result.log_line()
    assert result.pr_section() == ""


def test_a_partial_outage_is_counted_in_the_log_line(monkeypatch, env):
    run = _FakeRun([_payload(_comment())])
    real = run.__call__

    calls = {"n": 0}

    def flaky(argv, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(returncode=1, stdout="", stderr="429 slow down")
        return real(argv, **kwargs)

    monkeypatch.setattr(guidance.subprocess, "run", flaky)
    patch = """\
--- a/a.py
+++ b/a.py
@@ -10,3 +10,3 @@
-one
 ctx
@@ -40,3 +40,3 @@
-two
 ctx
"""
    result = check_patch(env, _FakeLLM('{"contradicts": false}'), patch=patch)
    assert result is not None
    assert result.anchors == 2 and result.lookups_failed == 1
    assert "1 of 2 lookup(s) did not answer" in result.log_line()
