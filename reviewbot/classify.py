"""Classify a reproduced CI failure as a product bug vs. a test/expectation issue.

Runs AFTER :func:`verify.run_gpu_reproduce` has confirmed the failure is real on
GPU and captured the baseline traceback, and BEFORE serge investigates. Grounding
the routing on the *real* traceback (rather than a blind read of the source) is
the whole point: transformers#47281 diagnosed the bug correctly but patched the
wrong file, having never run the test — the reproduction traceback pins the crash
to an exact frame, and this classifier decides whether the fix belongs in library
code or in the test's expectations.

Labels:
  product_issue  the failure is a genuine bug in library/model code (a crash, a
                 real regression) — investigate and fix the source
  test_issue     the test itself is wrong (stale expected values, a bad skip
                 condition, an over-tight tolerance) — prefer correcting the test
  unclear        cannot tell from the traceback alone — serge proceeds as if
                 product_issue but flags it

The classifier is a single cheap, non-tool ``response_format=json_object`` call.
It never raises: any transport/parse failure defaults to ``unclear`` so the flow
degrades to today's behaviour (investigate) rather than blocking a fix.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .llm_client import ChatCompletionClient

log = logging.getLogger(__name__)

PRODUCT_ISSUE = "product_issue"
TEST_ISSUE = "test_issue"
UNCLEAR = "unclear"

_LABELS = frozenset({PRODUCT_ISSUE, TEST_ISSUE, UNCLEAR})

# Per-test traceback budget fed to the classifier. The tail of a pytest traceback
# (the failing frame + the exception) is the discriminating part, so we keep the
# END of each traceback rather than the head.
_TRACEBACK_TAIL_CHARS = 4000

_SYSTEM_PROMPT = """\
You triage a failing transformers integration test that has just been REPRODUCED \
on GPU. Decide whether the fix belongs in library/model code or in the test.

Reply with a JSON object: {"label": <label>, "reason": <one sentence>} where label is:
- "product_issue": the traceback shows a genuine bug in library/model code — a \
crash (TypeError/RuntimeError/shape or dtype error), an exception raised inside \
`src/transformers/…`, or a real numerical regression. The source should be fixed.
- "test_issue": the failure is the test's own fault — an assertion on stale \
expected values/tensors that legitimately changed, an over-tight tolerance, or a \
bad skip/guard condition. The test (its expectations) should be corrected.
- "unclear": you genuinely cannot tell from the traceback which it is.

Judge only from the traceback and node-ids given. Do not guess beyond them. \
Prefer "product_issue" for hard crashes; prefer "test_issue" when the ONLY failure \
is an assertion comparing expected constants to fresh outputs.\
"""


@dataclass
class ClassifyResult:
    label: str
    reason: str = ""
    # Raw model content, kept for the tracking-issue log / debugging.
    raw: str = ""

    @property
    def is_test_issue(self) -> bool:
        return self.label == TEST_ISSUE

    @property
    def is_product_issue(self) -> bool:
        # unclear routes with product_issue (investigate), so callers usually
        # branch on is_test_issue; this is provided for symmetry / logging.
        return self.label == PRODUCT_ISSUE


def _tail(text: str, limit: int = _TRACEBACK_TAIL_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return "…(truncated)…\n" + text[-limit:]


def build_classify_messages(
    node_ids: list[str], tracebacks: dict[str, str], context: str = ""
) -> list[dict[str, str]]:
    """The system+user messages for one classification. Pure — no I/O — so the
    prompt assembly is unit-testable without an LLM."""
    parts: list[str] = []
    if context.strip():
        parts.append("Failure group (from the CI triage report):\n" + context.strip())
    parts.append("Targeted tests:\n" + "\n".join(f"- {n}" for n in node_ids))
    if tracebacks:
        blocks = [
            f"### {nid}\n{_tail(tb)}" for nid, tb in tracebacks.items() if tb.strip()
        ]
        if blocks:
            parts.append("Reproduced baseline traceback(s):\n\n" + "\n\n".join(blocks))
    else:
        parts.append("(No traceback text was captured for these tests.)")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_classify_response(content: str) -> ClassifyResult:
    """Parse the model's JSON reply into a :class:`ClassifyResult`. Tolerant: an
    unknown or missing label, or non-JSON content, degrades to ``unclear`` (so the
    caller investigates) rather than raising."""
    raw = content or ""
    obj: Any = None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        # Some models wrap the object in prose/fences — grab the first {...} span.
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(raw[start : end + 1])
            except (ValueError, TypeError):
                obj = None
    if not isinstance(obj, dict):
        return ClassifyResult(UNCLEAR, reason="unparseable classifier output", raw=raw)
    label = str(obj.get("label", "")).strip().lower()
    if label not in _LABELS:
        return ClassifyResult(UNCLEAR, reason=f"unknown label {label!r}", raw=raw)
    return ClassifyResult(label, reason=str(obj.get("reason", "")).strip(), raw=raw)


def classify_failure(
    llm: ChatCompletionClient,
    *,
    node_ids: list[str],
    tracebacks: dict[str, str],
    context: str = "",
    max_tokens: int = 300,
) -> ClassifyResult:
    """Classify one reproduced failure group. Never raises — any error becomes an
    ``unclear`` verdict so the surrounding flow proceeds to investigate."""
    if not node_ids:
        return ClassifyResult(UNCLEAR, reason="no node-ids to classify")
    messages = build_classify_messages(node_ids, tracebacks, context)
    try:
        result = llm.complete(
            messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to unclear, never crash the task
        log.warning(
            "classify_failure: LLM call failed (%s); defaulting to unclear", exc
        )
        return ClassifyResult(UNCLEAR, reason=f"classifier call failed: {exc}"[:200])
    return parse_classify_response(result.content)
