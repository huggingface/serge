"""Runs an optional repo-local script that influences the review.

The script lives in the target repo (typically at `.ai/context-script`),
must be executable, and is invoked with no arguments. We send a JSON
document on stdin describing the PR; the script's stdout can take two
shapes:

1. **Plain text** — stitched into the user prompt verbatim, inside a
   `REPO-PROVIDED CONTEXT` block.

2. **JSON object** with this shape (any field optional):

       {
         "context": "<text to inject into the prompt>",
         "skip_files": ["path/a.py", "path/b.py"]
       }

   `skip_files` lets the repo tell the reviewer which paths to drop
   from the diff entirely (e.g. auto-generated files in repos that
   regenerate them from a modular source). The reviewer mentions the
   skip list neutrally to the LLM but does not send those patches.

Trust model: the script is whatever sits at the configured path in the
checked-out tree — in Action mode, that's the default branch (Actions
checks out the event ref, which for `issue_comment` is the default
branch). Treat it at the same trust level as `.ai/review-rules.md`.

stdin schema:
    {
      "title": str,
      "body":  str,
      "files": [
        {"path": str,
         "status": "added"|"modified"|"removed"|"renamed"|"copied"|"changed",
         "additions": int,
         "deletions": int,
         "previous_path": str | null}
      ]
    }
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .sandbox import AUTO, SandboxUnavailable, wrap_command
from .tools import _helper_subprocess_env

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000


@dataclass
class ContextScriptResult:
    context: Optional[str] = None
    skip_files: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.context and not self.skip_files


def _shape_files(files: list[dict]) -> list[dict]:
    out: list[dict] = []
    for f in files:
        path = f.get("filename")
        if not path:
            continue
        out.append(
            {
                "path": path,
                "status": f.get("status") or "modified",
                "additions": int(f.get("additions") or 0),
                "deletions": int(f.get("deletions") or 0),
                "previous_path": f.get("previous_filename"),
            }
        )
    return out


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return (
        text[:MAX_OUTPUT_CHARS]
        + f"\n[... truncated, {len(text) - MAX_OUTPUT_CHARS} chars omitted ...]"
    )


def _parse_stdout(stdout: str) -> Optional[ContextScriptResult]:
    """Decide whether stdout is structured JSON or plain text, and shape it.

    JSON wins only if the entire stdout parses to a dict — otherwise we
    fall back to treating it as a plain context block, which keeps simple
    shell-script consumers working.
    """
    text = (stdout or "").strip()
    if not text:
        return None

    parsed: Optional[dict] = None
    if text.startswith("{") and text.endswith("}"):
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            parsed = candidate

    if parsed is None:
        return ContextScriptResult(context=_truncate(text), skip_files=[])

    ctx_raw = parsed.get("context")
    context = (
        _truncate(ctx_raw.strip())
        if isinstance(ctx_raw, str) and ctx_raw.strip()
        else None
    )

    skip_raw = parsed.get("skip_files") or []
    if not isinstance(skip_raw, list):
        log.warning("context script returned non-list skip_files; ignoring")
        skip_raw = []
    skip_files = [s for s in skip_raw if isinstance(s, str) and s]

    result = ContextScriptResult(context=context, skip_files=skip_files)
    return None if result.empty else result


def run_context_script(
    script_path: str,
    *,
    title: str,
    body: str,
    files: list[dict],
    timeout_seconds: int,
    cwd: Optional[str] = None,
    sandbox_mode: str = AUTO,
) -> Optional[ContextScriptResult]:
    """Run `script_path` and return its parsed result, or None if nothing useful.

    A return of None means "don't change anything" — for any of: not
    configured, no checkout to run against, file missing, not executable,
    timed out, crashed, non-zero exit, or empty output. We never raise to
    the caller; failures are logged and swallowed because a broken context
    hook should not break the review.

    The script is PR-influenced (it lives under .ai/ in the checkout), so
    it runs inside the bubblewrap sandbox — no network, no host secrets,
    only the checkout writable — and with a scrubbed environment.
    """
    if not script_path:
        return None
    # The context script only makes sense against a checkout; without one
    # there is nothing to sandbox or run.
    if not cwd:
        return None
    base = cwd
    full = (
        script_path if os.path.isabs(script_path) else os.path.join(base, script_path)
    )
    if not os.path.isfile(full):
        log.debug("context script %s not found, skipping", full)
        return None
    if not os.access(full, os.X_OK):
        log.warning("context script %s exists but is not executable; skipping", full)
        return None

    payload = json.dumps(
        {
            "title": title or "",
            "body": body or "",
            "files": _shape_files(files),
        }
    )

    try:
        argv = wrap_command(
            [full], workdir=base, write_root=base, mode=sandbox_mode
        )
    except SandboxUnavailable as exc:
        log.warning("context script not run (sandbox unavailable): %s", exc)
        return None

    try:
        proc = subprocess.run(
            argv,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=base,
            env=_helper_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        log.warning("context script %s timed out after %ds", full, timeout_seconds)
        return None
    except Exception:
        log.exception("context script %s failed to launch", full)
        return None

    if proc.stderr:
        log.info("context script stderr: %s", proc.stderr.strip()[:1000])
    if proc.returncode != 0:
        log.warning(
            "context script %s exited %d; ignoring output", full, proc.returncode
        )
        return None

    return _parse_stdout(proc.stdout)
