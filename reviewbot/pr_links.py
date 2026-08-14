"""Evidence serge attaches to a task PR body: the CI error being fixed, the
dashboard links the dispatcher supplied for each failing test, and existing
issues/PRs that already discuss the same failure.

All of it is decoration. Every helper here degrades to ``""``/``[]`` instead of
raising, so a task dispatched without links or a rate-limited GitHub search never
costs us a PR that is otherwise ready to open.

The failure report serge receives (``TaskRequest.context``, rendered by
transformers-ci's ``integration_failure_triage``) lists each failing test as a
bullet, optionally followed by the CI traceback in a 2-space-indented fenced
block::

    - `tests/models/foo/test_modeling_foo.py::FooIntegrationTest::test_bar` [single-gpu] (output_mismatch, seen 5/7)
      ```
      E   AssertionError: Tensor-likes are not close!
      ```

``tasks._failure_blocks`` deliberately stops at that fence — it feeds the
GPU-verify target selection, which must stay keyed on the bullets alone — so the
traceback is parsed separately here and re-attached only when the PR body is
rendered.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_FENCE = "```"

# Bounds on the dispatcher-supplied ``test_links`` (see sanitize_test_links).
# Generous enough for the real callers, tight enough that a malformed payload
# cannot bloat a PR body.
MAX_LINKS_PER_TEST = 4
MAX_LINKED_TESTS = 60
MAX_LABEL_CHARS = 60
MAX_URL_CHARS = 2000

# Tail of the CI traceback kept in the PR body. The triage already caps each
# traceback (~3000 chars); this is the second belt so a pathological report
# cannot push the fix itself below the fold.
TRACEBACK_CHARS = 2500

# A serge PR is not a "related report" — it is this bot's own earlier attempt.
_SERGE_MARKERS = ("serge-task:", "produced automatically by serge")


def failure_tracebacks(context: str) -> dict[str, str]:
    """Map each failing test's node-id to the fenced CI traceback under it.

    Only the first fenced block after a bullet is taken; tests rendered without
    one (the triage gives full tracebacks to the first N failures of a group)
    are simply absent from the mapping.
    """
    out: dict[str, str] = {}
    node_id: Optional[str] = None
    collecting = False
    buf: list[str] = []
    for line in context.splitlines():
        stripped = line.strip()
        if not collecting and line.startswith("- `"):
            match = _BACKTICK_RE.search(line)
            node_id = match.group(1).strip() if match else None
            continue
        if stripped == _FENCE and line.startswith(" "):
            if collecting:
                if node_id and buf:
                    out.setdefault(node_id, "\n".join(buf).strip())
                buf, collecting = [], False
            elif node_id and node_id not in out:
                collecting = True
            continue
        if collecting:
            buf.append(line[2:] if line.startswith("  ") else line)
    return out


def trim_traceback(traceback: str, max_chars: int = TRACEBACK_CHARS) -> str:
    """Keep the *tail* — the assertion and the raising frame live at the end."""
    text = (traceback or "").strip()
    if len(text) <= max_chars:
        return text
    return "…(truncated)…\n" + text[-max_chars:].lstrip()


def error_section(context: str, node_ids: list[str]) -> str:
    """A collapsed ``<details>`` block holding the CI traceback of the failing
    tests, so a reviewer sees the actual error without leaving the PR."""
    tracebacks = failure_tracebacks(context)
    parts: list[str] = []
    for node_id in node_ids:
        traceback = tracebacks.get(node_id)
        if not traceback:
            continue
        parts.append(
            "<details>\n"
            f"<summary>CI traceback — <code>{node_id}</code></summary>\n\n"
            f"```\n{trim_traceback(traceback)}\n```\n\n"
            "</details>"
        )
    return "\n\n".join(parts)


def sanitize_test_links(raw: Any) -> dict[str, list[dict[str, str]]]:
    """Validate an inbound ``test_links`` mapping (node-id → list of links).

    The dispatcher that files the task knows where its failures are observable —
    which dashboard, which UID, which template variables — and serge does not
    want to. It sends the finished URLs; serge only renders them, so no
    deployment's monitoring stack leaks into this codebase.

    Anything malformed is dropped rather than raising: links are decoration, and
    a bad entry must not cost a PR. Only ``http(s)`` URLs survive, and labels are
    forced to a single short line — this text lands in an outward-facing PR body.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[dict[str, str]]] = {}
    for node_id, links in raw.items():
        if not isinstance(node_id, str) or not node_id.strip():
            continue
        if not isinstance(links, list):
            continue
        clean: list[dict[str, str]] = []
        for link in links[:MAX_LINKS_PER_TEST]:
            if not isinstance(link, dict):
                continue
            url = str(link.get("url") or "").strip()
            if not url.startswith(("http://", "https://")) or len(url) > MAX_URL_CHARS:
                continue
            if any(ch.isspace() for ch in url) or ")" in url:
                continue  # would break out of the markdown link
            label = " ".join(str(link.get("label") or "").split())[:MAX_LABEL_CHARS]
            clean.append({"label": label or "Dashboard", "url": url})
        if clean:
            out[node_id.strip()] = clean
        if len(out) >= MAX_LINKED_TESTS:
            break
    return out


def test_links_section(
    test_links: dict[str, list[dict[str, str]]], node_ids: list[str]
) -> str:
    """Render the dispatcher's links for the failing tests this PR claims to fix
    — the failure's own history (how long it has been red, how its duration
    moved) next to the fix.

    Only ``node_ids`` are rendered: a task can carry several candidate failure
    groups, and the PR fixes just the one :func:`tasks._select_failure_block`
    picked. Links for the other groups' tests would be noise.
    """
    if not test_links:
        return ""
    lines: list[str] = []
    for node_id in node_ids:
        for link in test_links.get(node_id) or []:
            test = node_id.rsplit("::", 1)[-1]
            lines.append(f"- [{test} — {link['label']}]({link['url']})")
    if not lines:
        return ""
    return "\n".join(["**Where to watch it:**", *lines])


def _test_functions(node_ids: list[str], limit: int = 3) -> list[str]:
    """Distinct test-function names from pytest node-ids, parametrization
    stripped (``test_x[bf16-cuda]`` and ``test_x[fp32-cpu]`` are one test)."""
    names: list[str] = []
    for node_id in node_ids:
        name = node_id.rsplit("::", 1)[-1].split("[", 1)[0].strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def related_search_query(owner: str, repo: str, node_ids: list[str], model: str) -> str:
    """The GitHub search query for prior reports of this failure.

    The failing test's function name is the discriminating token — it is
    distinctive enough to quote and match on its own, and it is what a human
    filing the same bug pastes into the issue. Only when no node-id could be
    parsed do we fall back to the model name, restricted to titles so the query
    does not return every issue that merely mentions the architecture.
    """
    functions = _test_functions(node_ids)
    if functions:
        terms = " OR ".join(f'"{name}"' for name in functions)
        if len(functions) > 1:
            terms = f"({terms})"
        return f"repo:{owner}/{repo} {terms}"
    if model:
        return f'repo:{owner}/{repo} "{model}" in:title'
    return ""


def _is_serge_item(item: dict[str, Any]) -> bool:
    login = ((item.get("user") or {}).get("login") or "").lower()
    if login.startswith("serge"):
        return True
    body = (item.get("body") or "").lower()
    return any(marker in body for marker in _SERGE_MARKERS)


def find_related_issues(
    gh: Any,
    owner: str,
    repo: str,
    *,
    node_ids: list[str],
    model: str = "",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Existing issues/PRs that mention the failing test, newest activity first.

    Best-effort: a failed or rate-limited search (the search API has its own
    30 req/min budget) is logged and reported as "no hits". serge's own task
    PRs are dropped — they are this bot's earlier attempts at the same group,
    already linked by the triage fingerprint.
    """
    query = related_search_query(owner, repo, node_ids, model)
    if not query:
        return []
    try:
        items = gh.search_issues(query, per_page=20)
    except Exception:  # noqa: BLE001 — decoration must never fail a PR
        log.warning("related-issue search failed for %r", query, exc_info=True)
        return []
    related: list[dict[str, Any]] = []
    for item in items:
        if _is_serge_item(item):
            continue
        related.append(
            {
                "number": item.get("number"),
                "title": (item.get("title") or "").strip(),
                "url": item.get("html_url") or "",
                "state": item.get("state") or "",
                "is_pr": item.get("pull_request") is not None,
                "updated_at": (item.get("updated_at") or "")[:10],
            }
        )
        if len(related) >= limit:
            break
    return related


def related_section(items: list[dict[str, Any]], node_ids: list[str]) -> str:
    """Render the related issues/PRs, flagged as the keyword match they are —
    serge does not verify that they share a root cause."""
    if not items:
        return ""
    functions = _test_functions(node_ids)
    matched = ", ".join(f"`{name}`" for name in functions) or "the failing test"
    lines = [
        "### Possibly related",
        "",
        f"Existing issues/PRs mentioning {matched} (keyword match — not verified "
        "to share a root cause):",
        "",
    ]
    for item in items:
        kind = "PR" if item["is_pr"] else "issue"
        title = item["title"]
        lines.append(
            f"- [#{item['number']}]({item['url']}) — {title} "
            f"({kind}, {item['state']}, updated {item['updated_at']})"
        )
    return "\n".join(lines)
