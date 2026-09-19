"""The task page's report derivation: phases, tool usage, instruction sections.

These are parsers over serge's own event vocabulary, so the fixtures below are
transcribed from a real production task (824a0f5c, transformers#48945) rather
than invented — a phase table that only recognises strings this repo no longer
emits is worse than no phase table, because it reads as "nothing happened".
"""

import json

import pytest

from reviewbot.task_report import build_steps, build_tool_usage, split_instruction


def ev(kind, text, ts=0.0):
    return {"kind": kind, "text": text, "ts": ts}


def metrics(**kw):
    payload = {"in": 0, "out": 0, "turns": 0, "tools": 0}
    payload.update(kw)
    return json.dumps(payload)


# Abridged from the real job: every log line below is verbatim.
REAL_HISTORY = [
    ev("step", "launch", 1.0),
    ev(
        "log",
        "Launching task runner (kubernetes: ghcr.io/huggingface/serge-task-runner:sha-3d4b2cc)…",
        1.1,
    ),
    ev(
        "log",
        "Runner pod serge-task-824a0f5c-b2ba0cd6 created; waiting for it to schedule…",
        1.2,
    ),
    ev("step", "clone", 2.0),
    ev("log", "Checking out huggingface/transformers@main…", 2.1),
    ev("log", "Checkout ready in 9.3s", 11.3),
    ev("step", "preflight", 12.0),
    ev(
        "log",
        "Preflight — checking the normalize gate is passable: `bash -lc ...`…",
        12.1,
    ),
    ev("log", "Preflight passed.", 22.4),
    ev(
        "log",
        "GPU reproduce: dispatching serge-verify-caller.yml on aws-g5-12xlarge-cache for 1 test(s) [nemotron]",
        23.0,
    ),
    ev(
        "log",
        "GPU reproduce: reproduced ✓ (https://github.com/huggingface/transformers/actions/runs/35404303883)",
        276.0,
    ),
    ev("step", "llm", 277.0),
    ev("metrics", metrics(**{"in": 15053, "out": 41, "turns": 1}), 278.0),
    ev("step", "llm:0/80", 279.0),
    ev("tool", 'grep({"pattern": "Expectations"})', 280.0),
    ev(
        "chat",
        json.dumps(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "name": "grep",
                        "arguments": '{"pattern": "Expectations"}',
                        "argument_chars": 28,
                    }
                ],
            }
        ),
        280.1,
    ),
    ev(
        "chat",
        json.dumps(
            {"role": "tool", "name": "grep", "content": "x" * 40, "content_chars": 40}
        ),
        280.2,
    ),
    ev(
        "log",
        "Stuck in a tool-call loop (3 repeated tool call(s)); asking for a final answer without tools",
        300.0,
    ),
    ev(
        "metrics",
        metrics(**{"in": 462220, "out": 2334, "turns": 22, "tools": 21}),
        317.0,
    ),
    ev("step", "normalize", 318.0),
    ev("log", "Patch validated; normalizer is clean.", 1565.0),
    ev("step", "commit", 1566.0),
    ev("log", "Created branch serge/fix/itf-60177e5bdc1a-824a0f5c at b03feeac", 1568.0),
    ev(
        "log",
        "GPU verify: dispatching serge-verify-caller.yml on aws-g5-12xlarge-cache for 1 test(s) [nemotron]",
        1569.0,
    ),
    ev(
        "verify_result",
        json.dumps(
            {
                "mode": "verify",
                "verdict": "fixed",
                "runs": 5,
                "targeted": [
                    {
                        "nodeid": "tests/models/nemotron/test_modeling_nemotron.py::T::test_a",
                        "baseline": "failed",
                        "patched": "green",
                    }
                ],
            }
        ),
        3535.0,
    ),
    ev(
        "log",
        "GPU verify: fixed ✓ (https://github.com/huggingface/transformers/actions/runs/35406012561)",
        3536.0,
    ),
    ev(
        "log",
        "Opened PR #48945 (draft->ready): https://github.com/huggingface/transformers/pull/48945",
        3537.0,
    ),
    ev("step", "done", 3538.0),
]


def by_key(steps):
    return {s["key"]: s for s in steps}


class TestBuildSteps:
    def test_phase_order_follows_the_run_not_the_table(self):
        keys = [s["key"] for s in build_steps(REAL_HISTORY)]
        assert keys == [
            "launch",
            "clone",
            "preflight",
            "gpu_reproduce",
            "llm",
            "normalize",
            "commit",
            "gpu_verify",
            "pr",
        ]

    def test_gpu_gates_become_phases_though_they_emit_no_step(self):
        # The two gates run *between* steps; without the log-opened phases they
        # would be folded into preflight and commit, which is where every
        # "serge did nothing for four minutes" reading comes from.
        steps = by_key(build_steps(REAL_HISTORY))
        assert steps["gpu_reproduce"]["status"] == "ok"
        assert steps["gpu_reproduce"]["seconds"] == pytest.approx(253.0)
        assert steps["gpu_verify"]["run_url"].endswith("/35406012561")

    def test_the_history_lookup_is_its_own_phase(self):
        """It is a stage serge runs, with an outcome, before the first turn.

        As a bare log line it landed in whichever phase happened to be open —
        usually the GPU gate before it — which reads as the gate having done it.
        """
        history = [
            ev("log", "GPU reproduce: reproduced \u2713 (url)", 1.0),
            ev("step", "history", 2.0),
            ev("log", "Project history: #48750 already discuss these tests", 2.4),
            ev("step", "llm", 3.0),
            ev("log", "Calling LLM to produce a patch\u2026", 3.1),
        ]
        steps = build_steps(history)
        assert [s["key"] for s in steps] == ["start", "history", "llm"]
        hist = by_key(steps)["history"]
        assert hist["status"] == "ok"
        assert hist["lines"] == ["Project history: #48750 already discuss these tests"]

    def test_a_search_that_matched_nothing_is_a_pass_not_a_miss(self):
        # relore answered; an empty answer is evidence, not a failure.
        history = [
            ev("step", "history", 1.0),
            ev("log", "Project history: no earlier thread matched `x`", 1.2),
        ]
        assert build_steps(history)[0]["status"] == "ok"

    def test_a_relore_that_did_not_answer_warns(self):
        history = [
            ev("step", "history", 1.0),
            ev("log", "Project history: relore did not answer `x` \u2014 left", 1.2),
        ]
        assert build_steps(history)[0]["status"] == "warn"

    def test_a_guard_cutting_the_loop_off_is_a_warning_not_a_pass(self):
        llm = by_key(build_steps(REAL_HISTORY))["llm"]
        assert llm["status"] == "warn"

    def test_tokens_are_attributed_to_the_phase_that_spent_them(self):
        steps = by_key(build_steps(REAL_HISTORY))
        assert steps["llm"]["tokens_in"] == 462220
        assert steps["llm"]["tokens_out"] == 2334
        assert steps["llm"]["turns"] == 22
        assert steps["llm"]["tool_calls"] == 1
        # Phases that ran no LLM call report no token figure at all, rather
        # than a zero that reads as "measured, and it was free".
        assert "tokens_in" not in steps["normalize"]

    def test_a_second_round_is_charged_to_the_second_round(self):
        # A GPU-verify retry runs the loop again; the counters are cumulative,
        # so a phase's spend is a difference, not the latest reading.
        history = [
            ev("step", "llm", 1.0),
            ev("metrics", metrics(**{"in": 100, "out": 10, "turns": 2}), 2.0),
            ev("step", "normalize", 3.0),
            ev("step", "llm", 4.0),
            ev("metrics", metrics(**{"in": 250, "out": 30, "turns": 5}), 5.0),
        ]
        rounds = [s for s in build_steps(history) if s["key"] == "llm"]
        assert [s["tokens_in"] for s in rounds] == [100, 150]
        assert [s["turns"] for s in rounds] == [2, 3]

    def test_turn_markers_do_not_open_a_phase_each(self):
        history = [
            ev("step", "llm", 1.0),
            ev("log", "Calling LLM to produce a patch…", 1.5),
        ] + [ev("step", f"llm:{i}/80", 2.0 + i) for i in range(30)]
        assert [s["key"] for s in build_steps(history)] == ["llm"]

    def test_an_error_event_fails_its_phase(self):
        history = [
            ev("step", "normalize", 1.0),
            ev("normalize_error", "3 failed: check_repo", 2.0),
            ev("log", "Patch validated; normalizer is clean.", 3.0),
        ]
        # Even with a success line present: the error is the newer fact and the
        # page must not report a clean normalize on a job that errored.
        assert build_steps(history)[0]["status"] == "failed"

    def test_per_test_outcomes_ride_with_their_gate(self):
        verify = by_key(build_steps(REAL_HISTORY))["gpu_verify"]
        assert verify["tests"] == [
            {
                "nodeid": "tests/models/nemotron/test_modeling_nemotron.py::T::test_a",
                "baseline": "failed",
                "patched": "green",
            }
        ]

    def test_a_job_with_no_verify_result_event_still_renders(self):
        # Every job that finished before verify_result existed. The phase keeps
        # its status; it just has no per-test table.
        history = [e for e in REAL_HISTORY if e["kind"] != "verify_result"]
        verify = by_key(build_steps(history))["gpu_verify"]
        assert verify["status"] == "ok"
        assert "tests" not in verify

    def test_turn_noise_is_not_step_detail(self):
        history = (
            [ev("step", "llm", 1.0)]
            + [ev("log", f"LLM turn (blind={i}/80)", 2.0 + i) for i in range(20)]
            + [ev("metrics", metrics(**{"in": 9, "turns": 20}), 30.0)]
        )
        assert build_steps(history)[0]["lines"] == []

    def test_a_fact_logged_twice_is_one_line(self):
        # The PR is announced where it is opened and again by the caller
        # reporting the result; two rows read as two pull requests.
        history = [
            ev("log", "Opened PR #48945 (draft->ready): https://x/pull/48945", 1.0),
            ev("log", "Opened PR #48945.", 2.0),
        ]
        assert build_steps(history)[0]["lines"] == [
            "Opened PR #48945 (draft->ready): https://x/pull/48945"
        ]

    def test_a_phase_that_recorded_nothing_is_not_a_row(self):
        # `done` is bookkeeping: it fires after the PR line and carries nothing.
        # Rendered, it reads as a stage that ran and did nothing.
        history = [
            ev("log", "Opened PR #48945.", 1.0),
            ev("step", "done", 2.0),
        ]
        assert [s["key"] for s in build_steps(history)] == ["pr"]

    def test_empty_history(self):
        assert build_steps([]) == []


class TestBuildToolUsage:
    def test_calls_and_sizes_per_tool(self):
        rows = build_tool_usage(REAL_HISTORY)
        assert rows == [
            {
                "name": "grep",
                "calls": 1,
                "chars_in": 28,
                "chars_out": 40,
                "est_tokens_in": 7,
                "est_tokens_out": 10,
            }
        ]

    def test_true_result_size_survives_the_log_truncation(self):
        # _emit_chat_message caps a stored tool result at 2,000 chars. Reporting
        # the capped length would make every large grep look identical and hide
        # the one number that says what the transcript actually carries.
        history = [
            ev(
                "chat",
                json.dumps(
                    {
                        "role": "tool",
                        "name": "grep",
                        "content": "x" * 2000 + "… [+48000 chars truncated]",
                    }
                ),
            )
        ]
        assert build_tool_usage(history)[0]["chars_out"] == 50000

    def test_exact_size_wins_over_the_recovered_one(self):
        history = [
            ev(
                "chat",
                json.dumps(
                    {
                        "role": "tool",
                        "name": "grep",
                        "content": "xx",
                        "content_chars": 91234,
                    }
                ),
            )
        ]
        assert build_tool_usage(history)[0]["chars_out"] == 91234

    def test_busiest_tool_first(self):
        history = [
            ev(
                "chat",
                json.dumps(
                    {
                        "role": "assistant",
                        "tool_calls": [{"name": "read_file", "arguments": "{}"}],
                    }
                ),
            ),
            ev(
                "chat",
                json.dumps(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"name": "grep", "arguments": "{}"},
                            {"name": "grep", "arguments": "{}"},
                        ],
                    }
                ),
            ),
        ]
        assert [r["name"] for r in build_tool_usage(history)] == ["grep", "read_file"]

    def test_history_tools_are_counted_like_any_other(self):
        # relore's four verbs are ordinary function calls; the point of the
        # table is being able to see at a glance that they were never used.
        history = [
            ev(
                "chat",
                json.dumps(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "name": "history_search",
                                "arguments": '{"query": "nemotron"}',
                            }
                        ],
                    }
                ),
            )
        ]
        assert build_tool_usage(history)[0]["name"] == "history_search"

    def test_malformed_chat_payloads_are_skipped(self):
        history = [ev("chat", "not json"), ev("chat", "[]"), ev("chat", "null")]
        assert build_tool_usage(history) == []


class TestSplitInstruction:
    TRUNK = "Fix the failing tests.\nApply the repo's conventions."
    ADDENDUM = "The test ran to completion and its assertion failed."

    def test_the_per_group_block_is_the_primary_section(self):
        text = (
            f"{self.TRUNK}\n\n"
            "── This group's failure mode: `output_mismatch` ──\n"
            f"{self.ADDENDUM}\n"
        )
        sections = split_instruction(text)
        assert [s["primary"] for s in sections] == [False, True]
        assert sections[0]["title"] is None
        assert sections[1]["title"] == "This group's failure mode: `output_mismatch`"
        assert sections[1]["body"] == self.ADDENDUM

    def test_a_task_with_no_rule_is_one_section_and_nothing_is_collapsed(self):
        # A hand-dispatched task is short and entirely specific: folding any of
        # it away would hide the whole instruction.
        sections = split_instruction("Please bump the pinned torch version.")
        assert len(sections) == 1
        assert sections[0]["primary"] is False

    def test_empty(self):
        assert split_instruction("") == []
        assert split_instruction(None) == []
