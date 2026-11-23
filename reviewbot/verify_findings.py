"""Check each finding of a drafted review against the code, and drop the refuted.

A review of huggingface/peft#3401 (2026-11-23), after the "not in the diff"
gate, still carried three findings that one tool call each would have refuted:
a dtype mismatch `_cast_adapter_dtype` rules out, "use `torch_dtype=`" in a
repo whose examples use `dtype=` 23 times to 5, and a `merge_and_unload` that
"has already started unloading layers" when the first `merge()` raises before
anything is replaced. The review's final answer is written by the salvage
re-ask whenever the model loops in its reasoning — thinking off, no tools — so
nothing in the loop ever checked them.

This pass asks one small, tool-enabled question per finding: does the code
confirm it, refute it, or can it not be checked (an opinion, a convention
outside the repo)? Small on purpose: the model that loops while composing a
whole review answers a one-finding verdict in a few turns. Only ``refuted``
with evidence drops a finding, and it drops it whole — the summary bullet or
the inline comment — so nothing is rewritten by a model here.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

VERDICTS = ("confirmed", "refuted", "unverifiable")
_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+\S")


@dataclass(frozen=True)
class Finding:
    id: str  # "summary:<line index>" or "comment:<index>"
    text: str
    path: Optional[str] = None
    line: Optional[int] = None
    hunk: str = ""
    # Set on a second look: a related finding that was refuted, and why.
    hint: str = ""


@dataclass(frozen=True)
class Verdict:
    verdict: str  # one of VERDICTS
    evidence: str = ""


def split_findings(review: dict[str, Any]) -> list[Finding]:
    """Every summary bullet and every inline comment. Prose paragraphs (the
    overview) are not findings: they restate bullets, and dropping a sentence
    of one would need a rewrite."""
    findings: list[Finding] = []
    for i, line in enumerate((review.get("summary") or "").split("\n")):
        if _BULLET_RE.match(line):
            findings.append(Finding(id=f"summary:{i}", text=line.strip()))
    for i, c in enumerate(review.get("comments") or []):
        if isinstance(c, dict) and (c.get("body") or "").strip():
            findings.append(
                Finding(
                    id=f"comment:{i}",
                    text=c["body"].strip(),
                    path=c.get("path"),
                    line=c.get("line"),
                    hunk=_hunk_text(c.get("diff_hunk")),
                )
            )
    return findings


def _hunk_text(hunk: Any) -> str:
    if isinstance(hunk, str):
        return hunk
    if isinstance(hunk, list):
        return "\n".join(
            f"{h.get('op', ' ')}{h.get('text', '')}"
            for h in hunk
            if isinstance(h, dict)
        )
    return ""


VERIFY_SYSTEM_PROMPT = """\
You check ONE finding from a code review of a pull request against the code.
The repository is checked out at the pull request's head commit: what you read
with the tools is the code as the PR leaves it. Use the tools; do not answer
from memory.

Classify the finding:
- "refuted": the code contradicts the finding's factual premise. You must have
  read the line that contradicts it; quote its path and line in the evidence.
- "confirmed": you read the code that shows the premise is true.
- "unverifiable": the finding is an opinion, a style preference or a request
  (more tests, better docs), or it depends on something outside this repo.

Judge the premise, not the advice: "X is hard to read, rename it" is
unverifiable; "X is never called" is refuted by one call site. A finding that
is partly wrong in a way that changes its conclusion is refuted.

Reply with ONLY a JSON object, no fences:
{"verdict": "confirmed" | "refuted" | "unverifiable", "evidence": "<path:line — one sentence>"}
"""


def verify_user_prompt(
    finding: Finding, *, repo: str, number: int, title: str, changed_files: list[str]
) -> str:
    parts = [
        f"Pull request: {repo}#{number} — {title}",
        "Files changed by the PR:\n" + "\n".join(f"- {p}" for p in changed_files),
    ]
    if finding.path:
        parts.append(
            f"The finding is an inline comment on {finding.path} line {finding.line}."
        )
        if finding.hunk:
            parts.append(f"Diff around that line:\n{finding.hunk}")
    else:
        parts.append("The finding is a bullet from the review's summary.")
    parts.append(f"Finding:\n<finding>\n{finding.text}\n</finding>")
    if finding.hint:
        parts.append(
            "A related finding from the same review was already refuted:\n"
            f"{finding.hint}\nCheck whether that evidence also refutes this one; "
            "it may not — judge this finding on its own premise."
        )
    return "\n\n".join(parts)


def parse_verdict(text: str) -> Optional[Verdict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return None
    return Verdict(verdict, str(obj.get("evidence") or "").strip())


def verify_all(
    findings: list[Finding],
    check: Callable[[Finding], Optional[Verdict]],
    *,
    max_workers: int = 4,
) -> dict[str, Verdict]:
    """Run ``check`` on every finding; a failed or unparseable check keeps the
    finding (``unverifiable``) — this pass may only ever remove on evidence."""

    def safe(f: Finding) -> tuple[str, Verdict]:
        try:
            v = check(f)
        except Exception as exc:  # noqa: BLE001 — a broken check must not cost the review
            return f.id, Verdict("unverifiable", f"check failed: {exc!r}"[:200])
        return f.id, v or Verdict("unverifiable", "no verdict")

    with ThreadPoolExecutor(max(1, max_workers)) as ex:
        return dict(ex.map(safe, findings))


_CODE_SPAN_RE = re.compile(r"`([^`\n]{4,120})`")


def _spans(text: str) -> set[str]:
    return {m.group(1).strip() for m in _CODE_SPAN_RE.finditer(text)}


def second_look(
    findings: list[Finding], verdicts: dict[str, Verdict], *, limit: int = 10
) -> list[Finding]:
    """Findings worth checking again: not refuted, but sharing a code span with
    one that was. A review states most findings twice, as a summary bullet and
    as an inline comment, and the two checks disagreed: on peft#3401 the
    `torch_dtype` comment was refuted 3 times in 6 and its bullet 0 times, the
    `@UNKNOWN` comment 6 in 6 and its bullet 0. Each gets the refuted
    sibling's evidence and is judged again on its own premise.

    Only spans at most two findings share count: a sibling pair. A span in
    many findings (the new class's name) links unrelated ones, and the second
    look then refuted an opinion on the strength of a neighbour's evidence.
    """
    counts: dict[str, int] = {}
    for f in findings:
        for span in _spans(f.text):
            counts[span] = counts.get(span, 0) + 1
    refuted = [
        f
        for f in findings
        if verdicts.get(f.id, Verdict("")).verdict == "refuted"
        and verdicts[f.id].evidence
    ]
    out: list[Finding] = []
    for f in findings:
        if f in refuted or len(out) >= limit:
            continue
        mine = {span for span in _spans(f.text) if counts[span] <= 2}
        for r in refuted:
            if mine & _spans(r.text):
                hint = f"{r.text}\nRefuted because: {verdicts[r.id].evidence}"
                out.append(replace(f, hint=hint))
                break
    return out


def drop_refuted(
    review: dict[str, Any], verdicts: dict[str, Verdict]
) -> tuple[dict[str, Any], list[str]]:
    """The review without its refuted findings, and the ids that went."""
    refuted = {
        fid for fid, v in verdicts.items() if v.verdict == "refuted" and v.evidence
    }
    lines = (review.get("summary") or "").split("\n")
    kept_lines = [ln for i, ln in enumerate(lines) if f"summary:{i}" not in refuted]
    summary = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    comments = [
        c
        for i, c in enumerate(review.get("comments") or [])
        if f"comment:{i}" not in refuted
    ]
    return {**review, "summary": summary, "comments": comments}, sorted(refuted)
