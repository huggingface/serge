"""Sliding window over the browse transcript sent to the model each turn.

Why this exists: the agent loop is append-only, and the input-token cap
(``LLM_MAX_INPUT_TOKENS``) counts *every* call's whole prompt, so a session's
cost is the sum of a context that only grows — quadratic in turns, not linear.
Measured on prod task ``d9d4b022`` (transformers#48534, 2026-09-07, 56 turns,
2,094,215 cumulative input tokens):

* the fixed prefix was 17,526 chars (~5,100 tokens) — 14% of the budget;
* the other 86% was 62 accumulated tool results being re-sent;
* the last turn's prompt was 56,400 tokens, 37x the largest message in it;
* per-turn prompts grew 5,100 -> 56,400 at roughly 936 tokens a turn.

So the cap is not a context-window problem (the peak is a third of what the
model allows) and not a prefix problem — it is the transcript being re-billed
once per remaining turn. Windowing it turns that sum back into roughly linear:
with 20 results kept, that job's per-turn prompt would hold near ~21k instead
of climbing to 56k, and the turn ceiling under the same 2M cap rises from 56 to
roughly 95.

What this does NOT do is de-duplicate reads. It is the obvious next thought and
it is wrong here: on the same job, 3,101 lines were read across 33 ``read_file``
calls covering 2,802 distinct lines — 1.11x. The six reads of ``modular_blt.py``
were six *different windows*, not the same text six times. There is nothing to
reclaim by de-duplicating, and a de-duplicating cache would spend its
complexity on 9%.

Two constraints shape the implementation:

* **Every ``role: "tool"`` message must stay in place.** Providers reject a
  request where an assistant ``tool_calls`` entry has no matching tool reply,
  so dropping the message outright 400s the turn. Only ``content`` is replaced.
* **The stub has to read as "this happened and you can redo it".** A model that
  reads a bare ``[elided]`` can conclude the file does not exist and give up on
  the path; one that is told the call returned N characters and can be re-run
  will re-run it if it still needs the text.

Non-destructive by contract: the caller keeps its full ``messages`` list and
passes a windowed *copy* to the provider, so every turn re-windows from the
complete transcript. That keeps the function pure, keeps the journal's copy of
what was read intact, and means an elided message is never elided twice.
"""

from __future__ import annotations

from typing import Any

_ELIDED = (
    "[The earlier {name} call was made and returned {size:,} characters, but "
    "its output has been removed from this transcript to stay inside the "
    "input-token budget. Run the call again if you still need the output.]"
)


def elided_stub(name: str, size: int) -> str:
    """The placeholder left in place of a tool result's content."""
    return _ELIDED.format(name=name or "tool", size=size)


def elide_old_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int,
    min_chars: int = 0,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return ``messages`` with all but the newest ``keep_recent`` tool results
    replaced by :func:`elided_stub`, plus how many were replaced and how many
    characters that removed.

    ``keep_recent <= 0`` disables the window and returns the input list itself
    (not a copy) with zero counts — the default, so this is inert until an
    operator sets ``TOOL_RESULT_WINDOW``. ``min_chars`` leaves results at or
    below that size alone; a stub is not always shorter than the two-line grep
    output it would replace, and a result is never made *longer* than it was.
    """
    if keep_recent <= 0:
        return messages, 0, 0
    tool_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    if len(tool_positions) <= keep_recent:
        return messages, 0, 0

    windowed = list(messages)
    elided = 0
    saved = 0
    for index in tool_positions[:-keep_recent]:
        message = windowed[index]
        content = message.get("content")
        if not isinstance(content, str):
            continue
        stub = elided_stub(str(message.get("name") or "tool"), len(content))
        if len(content) <= max(len(stub), min_chars):
            continue
        windowed[index] = {**message, "content": stub}
        elided += 1
        saved += len(content) - len(stub)
    if not elided:
        return messages, 0, 0
    return windowed, elided, saved
