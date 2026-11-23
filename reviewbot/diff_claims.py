"""Catch a review that says something is absent from the diff when it is not.

A review of huggingface/peft#3401 (2026-11-23) told the author that
``EworaConfig`` "is not in the diff at all", that ``src/peft/tuners/__init__.py``
"is untouched" and that the ``peft/__init__.py`` hunk "is truncated". All three
files were in the single 62 KB chunk the model was given. The review was
written by the salvage re-ask (thinking off, the tail of a looping draft quoted
back), and nothing between the model and GitHub could tell.

This check is deterministic and narrow on purpose. It flags only a sentence
that names, *right before* an absence-from-the-diff phrase, a path this PR
changes or a class/function this PR adds. "X does not override Y" or "there is
no test for Z" are claims about the code, not about the diff, and are left
alone: this module has no way to know whether they are true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Phrases that say something is missing from the *change*, not from the code.
_ABSENT_RE = re.compile(
    r"\bnot (?:in|shown in|part of|included in|present in|touched (?:by|in)|"
    r"modified (?:by|in)|changed (?:by|in)|added (?:in|by)) "
    r"(?:the|this) (?:diff|PR|pull request|patch|change(?:set)?)\b"
    r"|\b(?:missing|absent) from (?:the|this) (?:diff|PR|pull request|patch)\b"
    r"|\b(?:hunk|diff|patch) is (?:also )?(?:truncated|cut off|incomplete)\b",
    re.IGNORECASE,
)
# "untouched" says nothing about the diff unless its subject is a file: "the
# base layer's bias is left untouched" is about runtime behaviour (a replayed
# review once read it as `nn.Linear`, a class the PR adds, being absent).
_UNTOUCHED_RE = re.compile(r"\bis (?:left )?untouched\b", re.IGNORECASE)
# A name must end this close to the phrase to be its subject. Keeps "`a.py` is
# fine, but `b.py` is untouched" from blaming `a.py`.
_SUBJECT_WINDOW = 80
_CODE_SPAN_RE = re.compile(r"`([^`\n]{1,200})`")
_BARE_PATH_RE = re.compile(r"(?<![\w/.])((?:[\w.-]+/)+[\w.-]+\.\w+)")
_ADDED_DEF_RE = re.compile(r"^\+\s*(?:async\s+)?(?:class|def)\s+([A-Za-z_]\w*)")
# Sentence ends are punctuation followed by whitespace, so `__init__.py` and
# `self.foo` stay whole.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Too generic to identify anything on their own.
_GENERIC_NAMES = {"__init__.py", "__init__", "forward", "main", "setup", "test"}


@dataclass(frozen=True)
class AbsenceClaim:
    where: str  # "summary" or "<path>:<line>"
    name: str  # the name the review says is absent
    evidence: str  # the changed path that contradicts it
    sentence: str


@dataclass(frozen=True)
class DiffInventory:
    paths: frozenset[str]
    symbols: dict[str, str]  # added class/def name -> path that adds it

    @classmethod
    def from_files(cls, files: Iterable[dict[str, Any]]) -> "DiffInventory":
        paths: set[str] = set()
        symbols: dict[str, str] = {}
        for f in files:
            path = f.get("filename")
            if not path:
                continue
            paths.add(path)
            for line in (f.get("patch") or "").splitlines():
                m = _ADDED_DEF_RE.match(line)
                if m and m.group(1) not in _GENERIC_NAMES:
                    symbols.setdefault(m.group(1), path)
        return cls(frozenset(paths), symbols)

    def evidence_for(self, name: str) -> str | None:
        """The changed path that shows ``name`` is in the diff, or None."""
        name = name.strip().rstrip("()").strip()
        if not name:
            return None
        if _looks_like_path(name):
            candidate = name.lstrip("./")
            if candidate in self.paths:
                return candidate
            if "/" in candidate:
                hits = [p for p in self.paths if p.endswith("/" + candidate)]
                if len(hits) == 1:
                    return hits[0]
            elif candidate not in _GENERIC_NAMES:
                hits = [p for p in self.paths if p.rsplit("/", 1)[-1] == candidate]
                if len(hits) == 1:
                    return hits[0]
        for part in re.split(r"[.\s]", name):
            if part in self.symbols:
                return self.symbols[part]
        return None


def _looks_like_path(name: str) -> bool:
    return "/" in name or re.search(r"\.\w{1,5}$", name.strip()) is not None


def _subject(text_before: str) -> str | None:
    """The last name written in the window right before an absence phrase."""
    # Match on the whole prefix, then keep what ends inside the window: cutting
    # first can start the window inside a code span and pair the wrong backticks.
    start = len(text_before) - _SUBJECT_WINDOW
    candidates: list[tuple[int, str]] = []
    for rx in (_CODE_SPAN_RE, _BARE_PATH_RE):
        for m in rx.finditer(text_before):
            if m.end() >= start:
                candidates.append((m.end(), m.group(1)))
    if not candidates:
        return None
    return max(candidates)[1]


def _review_texts(review: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "summary", review.get("summary") or ""
    for c in review.get("comments") or []:
        if isinstance(c, dict):
            yield f"{c.get('path')}:{c.get('line')}", c.get("body") or ""


def find_absence_claims(
    review: dict[str, Any], inventory: DiffInventory
) -> list[AbsenceClaim]:
    claims: list[AbsenceClaim] = []
    for where, text in _review_texts(review):
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            matches = [(m, False) for m in _ABSENT_RE.finditer(sentence)]
            matches += [(m, True) for m in _UNTOUCHED_RE.finditer(sentence)]
            for am, paths_only in sorted(matches, key=lambda x: x[0].start()):
                name = _subject(sentence[: am.start()])
                if name is None or (paths_only and not _looks_like_path(name)):
                    continue
                evidence = inventory.evidence_for(name)
                if evidence is not None:
                    claims.append(AbsenceClaim(where, name, evidence, sentence.strip()))
                    break
    return claims


def absence_feedback(claims: list[AbsenceClaim]) -> str:
    lines = [
        "Your review says these are not in the PR's diff, but they are:",
    ]
    for c in claims:
        lines.append(
            f'- `{c.name}` — changed in `{c.evidence}`. You wrote: "{c.sentence}"'
        )
    lines.append(
        "Re-read those files in the diff above, then reply with the corrected "
        "JSON review only. Drop or rewrite every finding that depended on "
        "them being absent; keep the rest of the review as it was."
    )
    return "\n".join(lines)


def strip_absence_claims(
    review: dict[str, Any], claims: list[AbsenceClaim]
) -> dict[str, Any]:
    """Remove the findings built on a flagged sentence. The last resort after
    the model was asked to fix them and did not: a known-false finding costs
    the author more than the gap it leaves.

    A finding goes whole — the summary bullet holding the sentence, or the
    inline comment — because the sentences around a false premise argue from
    it ("It must subclass `PeftConfig`…" after "`EworaConfig` is not in the
    diff"). A sentence outside any bullet is removed on its own.
    """
    by_where: dict[str, list[str]] = {}
    for c in claims:
        by_where.setdefault(c.where, []).append(c.sentence)

    def scrub(text: str, sentences: list[str]) -> str:
        lines = text.split("\n")
        for s in sentences:
            core = re.sub(r"^[-*]\s+", "", s)
            for i, line in enumerate(lines):
                if line is None or core not in line:
                    continue
                if re.match(r"\s*[-*]\s", line):
                    lines[i] = None
                else:
                    lines[i] = re.sub(r"[ \t]{2,}", " ", line.replace(core, "")).strip()
                break
        text = "\n".join(line for line in lines if line is not None)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    out = dict(review)
    if "summary" in by_where:
        out["summary"] = scrub(out.get("summary") or "", by_where["summary"])
    out["comments"] = [
        c
        for c in review.get("comments") or []
        if not (isinstance(c, dict) and f"{c.get('path')}:{c.get('line')}" in by_where)
    ]
    return out
