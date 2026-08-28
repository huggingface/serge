"""Prometheus exposition for finished serge jobs.

Why this exists: ``WEB_JOB_RETENTION`` is 25 jobs — roughly two days of traffic
— and everything about how a session actually went (how many turns it ran, how
much of the budget went on re-reading files it had already opened, and whether
the model finished or a guard cut it off) lived only in that job row. So no
change to the agent loop could be shown to have helped: the evidence for the
change was deleted before the change shipped.

This endpoint does not fix retention. It moves the *durable* copy to Prometheus,
which is already scraping this cluster and already keeps 90 days. The exposition
is deliberately a **rolling window** over whatever the store still holds: when a
job is pruned its series simply stops being exported and goes stale, while the
samples Prometheus already took stay queryable for the full retention. Reading
this endpoint directly tells you nothing about last week — that is what the range
query is for.

Shape follows the usual info-metric convention: one ``serge_job_info`` series
carries the descriptive labels, and the numeric series are keyed by ``job_id``
alone. Keeping the labels off the values means a job's numbers don't fork into a
new series when, say, its status moves from ``running`` to ``published``, and a
table panel joins them back with ``on(job_id) group_left(...)``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# (metric name, session key, help text). Everything here is a point-in-time
# reading of one finished job, so every one is a gauge — none of them are
# monotonic across scrapes the way a counter must be.
_JOB_GAUGES: tuple[tuple[str, str, str], ...] = (
    ("serge_job_turns", "turns", "LLM turns the job's agent loop(s) ran."),
    ("serge_job_tool_calls", "tool_calls", "Tool calls the job made."),
    (
        "serge_job_input_tokens",
        "prompt_tokens",
        "Cumulative input tokens billed to the job. The per-loop cap is "
        "LLM_MAX_INPUT_TOKENS; a job running several rounds can exceed it.",
    ),
    (
        "serge_job_output_tokens",
        "completion_tokens",
        "Cumulative output tokens billed to the job.",
    ),
    ("serge_job_llm_seconds", "seconds", "Wall time spent inside LLM calls."),
    (
        "serge_job_repeat_calls",
        "repeats",
        "Tool calls that were an exact re-run of an earlier call in the same "
        "session — what the tool-repeat guard counts.",
    ),
    (
        "serge_job_distinct_paths",
        "distinct_paths",
        "Distinct files/directories the job opened.",
    ),
    (
        "serge_job_path_revisits",
        "path_revisits",
        "Calls spent re-opening a path already opened in the same session, "
        "counted per path as visits-1. Not the same as repeat_calls: a re-read "
        "of a different line range of the same file is a revisit but not an "
        "exact repeat, and that is the shape that dominates.",
    ),
    (
        "serge_job_validation_retries",
        "validation_retries",
        "Times the patch/normalize gate rejected an answer and re-asked.",
    ),
    (
        "serge_job_truncation_retries",
        "truncation_retries",
        "Times a truncated or empty final answer was salvaged by re-asking.",
    ),
    (
        "serge_job_rounds",
        "rounds",
        "Agent loops the job ran: one per candidate group, plus one per GPU "
        "verify retry. 0 means the job never reached the LLM.",
    ),
)

_INFO_HELP = (
    "One series per finished job, always 1. Carries the labels the numeric "
    "series deliberately omit; join with `on(job_id) group_left(...)`."
)

_FINISHED_HELP = (
    "Unix time the job reached its terminal state. Lets a range query order "
    "jobs even though the series are gauges."
)


def _escape(value: str) -> str:
    """Escape a Prometheus label value (backslash, quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: Iterable[tuple[str, Any]]) -> str:
    rendered = ",".join(
        f'{name}="{_escape("" if value is None else str(value))}"'
        for name, value in pairs
    )
    return "{" + rendered + "}"


def _number(value: Any) -> Optional[float]:
    """Coerce a stored counter to a float, or None when it isn't one.

    Session records are JSON written by an older build as easily as this one, so
    a missing or junk value must skip its sample rather than break the scrape.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def render_job_metrics(
    sessions: list[dict[str, Any]],
    *,
    retention: Optional[int] = None,
    version: Optional[str] = None,
    commit: Optional[str] = None,
) -> str:
    """Render ``store.list_sessions()`` rows as Prometheus text exposition.

    ``retention`` is exported alongside so a dashboard can say how far back the
    live window reaches — a gap in the series means the job aged out of the
    store, not that serge stopped working.
    """
    lines: list[str] = []

    if version is not None or commit is not None:
        lines.append("# HELP serge_build_info The running serge build.")
        lines.append("# TYPE serge_build_info gauge")
        lines.append(
            "serge_build_info"
            + _labels((("version", version or ""), ("commit", commit or "")))
            + " 1"
        )
    if retention is not None:
        lines.append(
            "# HELP serge_job_retention Jobs the store keeps before pruning "
            "(WEB_JOB_RETENTION). The export above is a window this wide."
        )
        lines.append("# TYPE serge_job_retention gauge")
        lines.append(f"serge_job_retention {int(retention)}")

    lines.append(f"# HELP serge_job_info {_INFO_HELP}")
    lines.append("# TYPE serge_job_info gauge")
    for row in sessions:
        session = row.get("session") or {}
        owner = row.get("target_owner") or ""
        repo = row.get("target_repo") or ""
        pr_number = row.get("pr_number")
        lines.append(
            "serge_job_info"
            + _labels(
                (
                    ("job_id", row.get("id")),
                    ("kind", row.get("kind") or "review"),
                    ("repo", f"{owner}/{repo}" if owner or repo else ""),
                    ("number", row.get("target_number")),
                    ("status", row.get("status") or ""),
                    ("model", row.get("llm_model") or "none"),
                    # "answered" is the only value meaning the model decided it
                    # was done; every other value is a guard ending the session.
                    ("stop_reason", session.get("stop_reason") or "unknown"),
                    ("verify_verdict", row.get("verify_verdict") or "none"),
                    ("pr", "" if pr_number is None else pr_number),
                )
            )
            + " 1"
        )

    for name, key, help_text in _JOB_GAUGES:
        samples = []
        for row in sessions:
            value = _number((row.get("session") or {}).get(key))
            if value is None:
                continue
            samples.append(
                f"{name}" + _labels((("job_id", row.get("id")),)) + f" {_format(value)}"
            )
        if not samples:
            continue
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.extend(samples)

    finished = []
    for row in sessions:
        value = _number(row.get("updated_at"))
        if value is None:
            continue
        finished.append(
            "serge_job_finished_timestamp_seconds"
            + _labels((("job_id", row.get("id")),))
            + f" {_format(value)}"
        )
    if finished:
        lines.append(f"# HELP serge_job_finished_timestamp_seconds {_FINISHED_HELP}")
        lines.append("# TYPE serge_job_finished_timestamp_seconds gauge")
        lines.extend(finished)

    return "\n".join(lines) + "\n"
