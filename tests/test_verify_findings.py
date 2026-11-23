"""reviewbot.verify_findings and its wiring into reviews."""

import json
import unittest

from reviewbot.llm_client import ChatResult
from reviewbot.reviewer import _AggregateMetrics, _verify_review_findings
from reviewbot.tools import ToolEnv
from reviewbot.verify_findings import (
    Finding,
    Verdict,
    drop_refuted,
    parse_verdict,
    second_look,
    split_findings,
    verify_all,
    verify_user_prompt,
)

from tests.test_reviewer import _CfgStub

SUMMARY = (
    "EWoRA is wired in, but not mergeable as-is.\n\n**Correctness**\n"
    "- `Linear.forward` feeds the router a bf16 input under a bf16 base model.\n"
    "- The `self.merged` branch in `forward` is dead code.\n\n**Style**\n"
    "- Use `torch_dtype=` in the example."
)
COMMENTS = [
    {"path": "layer.py", "side": "RIGHT", "line": 159, "body": "dtype mismatch here."},
    {"path": "layer.py", "side": "RIGHT", "line": 130, "body": "Dead branch."},
]


class SplitAndDropTests(unittest.TestCase):
    def test_bullets_and_comments_are_findings_prose_is_not(self):
        ids = [f.id for f in split_findings({"summary": SUMMARY, "comments": COMMENTS})]
        self.assertEqual(
            ids, ["summary:3", "summary:4", "summary:7", "comment:0", "comment:1"]
        )

    def test_refuted_with_evidence_is_dropped_whole(self):
        review = {"summary": SUMMARY, "comments": COMMENTS}
        verdicts = {
            "summary:3": Verdict(
                "refuted",
                "tuners_utils.py:2843 casts ModuleDict and ParameterDict alike",
            ),
            "summary:4": Verdict("confirmed", "layer.py:102 merge always raises"),
            "summary:7": Verdict("refuted", ""),  # no evidence: kept
            "comment:0": Verdict("refuted", "tuners_utils.py:2843"),
            "comment:1": Verdict("unverifiable"),
        }
        out, dropped = drop_refuted(review, verdicts)
        self.assertEqual(dropped, ["comment:0", "summary:3"])
        self.assertNotIn("bf16 input", out["summary"])
        self.assertIn("dead code", out["summary"])
        self.assertIn("torch_dtype", out["summary"])
        self.assertIn("\n\n**Style**", out["summary"])
        self.assertEqual([c["line"] for c in out["comments"]], [130])

    def test_inline_prompt_names_the_location(self):
        f = Finding(
            id="comment:0", text="dtype mismatch here.", path="layer.py", line=159
        )
        text = verify_user_prompt(
            f, repo="o/r", number=1, title="t", changed_files=["layer.py"]
        )
        self.assertIn("inline comment on layer.py line 159", text)
        self.assertIn("<finding>\ndtype mismatch here.\n</finding>", text)


class SecondLookTests(unittest.TestCase):
    def test_sibling_of_a_refuted_finding_is_rechecked_with_its_evidence(self):
        bullet = Finding(id="summary:1", text="- Use `torch_dtype=` in the example.")
        comment = Finding(
            id="comment:0",
            text="`torch_dtype=` is required for ~4.33.",
            path="ex.py",
            line=69,
        )
        other = Finding(id="summary:2", text="- The `self.merged` branch is dead.")
        verdicts = {
            "summary:1": Verdict("confirmed", "ex.py:69"),
            "comment:0": Verdict(
                "refuted", "other.py:1808 requires transformers>=4.53"
            ),
            "summary:2": Verdict("confirmed", "layer.py:102"),
        }
        (again,) = second_look([bullet, comment, other], verdicts)
        # A span many findings share (the new class's name) links nothing.
        common = [
            Finding(id=f"summary:{i}", text=f"- `EworaConfig` point {i}.")
            for i in range(3)
        ]
        common_verdicts = {"summary:0": Verdict("refuted", "a.py:1")}
        self.assertEqual(second_look(common, common_verdicts), [])
        self.assertEqual(again.id, "summary:1")
        self.assertIn("requires transformers>=4.53", again.hint)
        prompt = verify_user_prompt(
            again, repo="o/r", number=1, title="t", changed_files=[]
        )
        self.assertIn("already refuted", prompt)


class ParseVerdictTests(unittest.TestCase):
    def test_parses_fenced_and_bare(self):
        self.assertEqual(
            parse_verdict(
                '```json\n{"verdict": "Refuted", "evidence": "a.py:1"}\n```'
            ).verdict,
            "refuted",
        )
        self.assertEqual(
            parse_verdict('{"verdict": "confirmed", "evidence": "x"}').evidence, "x"
        )

    def test_rejects_unknown_verdicts(self):
        self.assertIsNone(parse_verdict('{"verdict": "probably"}'))
        self.assertIsNone(parse_verdict("it is wrong"))


class VerifyAllTests(unittest.TestCase):
    def test_a_failing_check_keeps_the_finding(self):
        def check(f):
            raise RuntimeError("provider down")

        out = verify_all([Finding(id="summary:1", text="x")], check)
        self.assertEqual(out["summary:1"].verdict, "unverifiable")


class _VerdictLLM:
    """Answers each verify loop from the finding text in its prompt."""

    def __init__(self, refute: str):
        self.refute = refute
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        user = messages[1]["content"]
        verdict = "refuted" if self.refute in user else "confirmed"
        return ChatResult(
            content=json.dumps(
                {"verdict": verdict, "evidence": "tuners_utils.py:2843"}
            ),
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        )


class WiringTests(unittest.TestCase):
    def test_refuted_findings_leave_the_review(self):
        llm = _VerdictLLM(refute="bf16")
        logs = []
        metrics = _AggregateMetrics()
        cfg = _CfgStub()
        cfg.review_verify_max_findings = 25
        summary, comments = _verify_review_findings(
            cfg,  # type: ignore[arg-type]
            llm,  # type: ignore[arg-type]
            ToolEnv(repo_root="/tmp"),
            SUMMARY,
            [dict(c) for c in COMMENTS],
            repo="o/r",
            number=1,
            title="t",
            changed_files=["layer.py"],
            metrics=metrics,
            emit=lambda kind, text: logs.append(text),
        )
        self.assertEqual(llm.calls, 5)
        self.assertNotIn("bf16", summary)
        self.assertEqual(len(comments), 2)
        self.assertEqual(metrics.prompt_tokens, 500)
        self.assertIn(
            "Findings: 4 confirmed, 1 refuted, 0 unverifiable; dropped 1", logs
        )


if __name__ == "__main__":
    unittest.main()
