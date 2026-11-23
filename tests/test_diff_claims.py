"""reviewbot.diff_claims and its wiring into the review loop.

The sentences below are taken from the published review of huggingface/peft#3401
(2026-11-23), where every file the review called absent was in the diff.
"""

import json
import unittest

from reviewbot.diff_claims import (
    DiffInventory,
    absence_feedback,
    find_absence_claims,
    strip_absence_claims,
)
from reviewbot.llm_client import ChatResult
from reviewbot.reviewer import (
    _extract_json,
    _parses_as_review,
    _review_absence_feedback,
    _run_agentic_loop,
)

from tests.test_reviewer import _CfgStub, _FakeLLM

FILES = [
    {
        "filename": "src/peft/__init__.py",
        "patch": "@@ -70,6 +70,8 @@\n     EvaConfig,\n+    EworaConfig,\n+    EworaModel,\n",
    },
    {
        "filename": "src/peft/tuners/__init__.py",
        "patch": "@@ -22,6 +22,7 @@\n+from .ewora import EworaConfig, EworaModel\n",
    },
    {
        "filename": "src/peft/tuners/ewora/config.py",
        "patch": "@@ -0,0 +1,3 @@\n+@dataclass\n+class EworaConfig(PeftConfig):\n+    r: int = 8\n",
    },
    {
        "filename": "src/peft/tuners/ewora/layer.py",
        "patch": "@@ -0,0 +1,4 @@\n+class EworaLayer(BaseTunerLayer):\n"
        "+    def __init__(self, base_layer):\n+class Linear(nn.Module, EworaLayer):\n"
        "+    def forward(self, x):\n",
    },
]

PROD_SUMMARY = (
    "The EWoRA tuner is wired into the registry.\n\n**Correctness**\n"
    "- `get_delta_weight` raises `NotImplementedError`; add an override.\n"
    "- `Linear.__init__` calls `super(nn.Linear, self).__init__()` then "
    "`EworaLayer.__init__`, which is not in the diff; if it does not call "
    "`super().__init__()`, `nn.ParameterDict()` init fails. Confirm the layer "
    "file is complete and included.\n\n**Integration**\n"
    "- `src/peft/tuners/__init__.py` is untouched; without registering "
    "`EworaConfig`/`EworaModel` there, imports will not resolve. The "
    "`peft/__init__.py` hunk is also truncated — `EworaModel` is not added "
    "next to `EworaConfig`.\n"
    "- `EworaConfig` is not in the diff at all. It must subclass `PeftConfig`.\n"
    "- The mapping is fine, but the resolution code is not in the diff."
)


class FindAbsenceClaimsTests(unittest.TestCase):
    def setUp(self):
        self.inv = DiffInventory.from_files(FILES)

    def test_catches_every_false_claim_from_the_peft_review(self):
        claims = find_absence_claims(
            {"summary": PROD_SUMMARY, "comments": []}, self.inv
        )
        self.assertEqual(
            {c.evidence for c in claims},
            {
                "src/peft/tuners/ewora/layer.py",
                "src/peft/tuners/__init__.py",
                "src/peft/__init__.py",
                "src/peft/tuners/ewora/config.py",
            },
        )

    def test_true_absence_claims_are_left_alone(self):
        text = (
            "`tuners_utils.py` is untouched, so `_create_and_replace` keeps the "
            "default. The resolution code is not in the diff. "
            "`src/peft/mapping.py` is not in this PR."
        )
        self.assertEqual(find_absence_claims({"summary": text}, self.inv), [])

    def test_subject_is_the_nearest_name(self):
        # layer.py is changed, but the sentence says tuners_utils.py is untouched.
        text = "`layer.py` looks right, but `tuners_utils.py` is untouched."
        self.assertEqual(find_absence_claims({"summary": text}, self.inv), [])

    def test_untouched_needs_a_file_subject(self):
        # `Linear` is a class this PR adds; "untouched" here is about runtime.
        text = "`EworaLayer` wraps `nn.Linear` and the base layer's bias is left untouched."
        self.assertEqual(find_absence_claims({"summary": text}, self.inv), [])

    def test_claims_about_the_code_are_not_diff_claims(self):
        text = "`EworaModel` does not override `merge_and_unload`. `EworaConfig` is missing a `bias` field."
        self.assertEqual(find_absence_claims({"summary": text}, self.inv), [])

    def test_inline_comment_bodies_are_checked(self):
        review = {
            "summary": "ok",
            "comments": [
                {
                    "path": "a.py",
                    "line": 3,
                    "body": "`EworaConfig` is missing from the diff.",
                }
            ],
        }
        (claim,) = find_absence_claims(review, self.inv)
        self.assertEqual(claim.where, "a.py:3")

    def test_feedback_names_the_file_that_contradicts_it(self):
        claims = find_absence_claims(
            {"summary": "`EworaConfig` is not in the diff."}, self.inv
        )
        text = absence_feedback(claims)
        self.assertIn("src/peft/tuners/ewora/config.py", text)
        self.assertIn("JSON review only", text)


class StripAbsenceClaimsTests(unittest.TestCase):
    def test_drops_whole_bullets_and_keeps_the_rest(self):
        inv = DiffInventory.from_files(FILES)
        review = {
            "summary": PROD_SUMMARY,
            "comments": [
                {"path": "x.py", "line": 1, "body": "`EworaLayer` is not in the diff."},
                {"path": "x.py", "line": 2, "body": "Rename this."},
            ],
        }
        out = strip_absence_claims(review, find_absence_claims(review, inv))
        self.assertEqual(find_absence_claims(out, inv), [])
        self.assertNotIn(
            "It must subclass", out["summary"]
        )  # argued from the false premise
        self.assertNotIn("Confirm the layer file", out["summary"])
        self.assertIn("- `get_delta_weight` raises", out["summary"])
        self.assertIn("the resolution code is not in the diff", out["summary"])
        self.assertIn("\n\n**Integration**", out["summary"])
        self.assertEqual([c["line"] for c in out["comments"]], [2])


class ReviewJsonTests(unittest.TestCase):
    def test_raw_newline_inside_a_string_still_parses(self):
        # GLM-5.2 on peft#3401: a complete review rejected over this.
        raw = (
            '```json\n{\n  "summary": "Line one.\nLine two.",\n  "comments": []\n}\n```'
        )
        self.assertEqual(
            _extract_json(raw, ("summary",))["summary"], "Line one.\nLine two."
        )
        self.assertTrue(_parses_as_review(raw))

    def test_prose_is_not_a_review(self):
        self.assertFalse(_parses_as_review("I looked at the diff and it is fine."))


class ReviewLoopWiringTests(unittest.TestCase):
    def test_a_false_claim_is_sent_back_once_then_accepted(self):
        inv = DiffInventory.from_files(FILES)
        bad = json.dumps(
            {"summary": "`EworaConfig` is not in the diff at all.", "comments": []}
        )
        good = json.dumps(
            {"summary": "Config validation is missing a check.", "comments": []}
        )
        llm = _FakeLLM([ChatResult(content=bad), ChatResult(content=good)])
        logs = []
        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=_CfgStub(),  # type: ignore[arg-type]
            tool_env=None,
            emit=lambda kind, text: logs.append(text),
            parses=_parses_as_review,
            validate=lambda c: _review_absence_feedback(c, inv),
            max_validation_retries=1,
            validation_label="Review check failed",
        )
        self.assertEqual(chat.content, good)
        self.assertEqual(metrics.validation_retries, 1)
        self.assertIn(
            "src/peft/tuners/ewora/config.py", llm.calls[1]["messages"][-1]["content"]
        )
        self.assertTrue(
            any(t.startswith("Review check failed (correction 1/1)") for t in logs)
        )

    def test_unparseable_review_is_reasked_instead_of_failing(self):
        llm = _FakeLLM(
            [
                ChatResult(content="Here is my review: looks good overall."),
                ChatResult(content='{"summary": "ok", "comments": []}'),
            ]
        )
        chat, metrics = _run_agentic_loop(
            llm,  # type: ignore[arg-type]
            [{"role": "user", "content": "review this"}],
            cfg=_CfgStub(),  # type: ignore[arg-type]
            tool_env=None,
            parses=_parses_as_review,
        )
        self.assertEqual(json.loads(chat.content)["summary"], "ok")
        self.assertEqual(metrics.truncation_retries, 1)


if __name__ == "__main__":
    unittest.main()
