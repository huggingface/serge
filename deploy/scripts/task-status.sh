#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from typing import Any, Optional


TASK_URL_RE = re.compile(r"/tasks/([^/]+)/([^/]+)/([A-Za-z0-9_-]+)")


class CommandError(Exception):
    def __init__(self, return_code: int):
        self.return_code = return_code


def run_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise CommandError(exc.returncode) from exc


def parse_task_ref(ref: str) -> tuple[Optional[str], Optional[str], str]:
    match = TASK_URL_RE.search(ref)
    if match:
        owner, repo, task_id = match.groups()
        return owner, repo, task_id
    return None, None, ref.rstrip("/").split("/")[-1]


def fmt_time(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    local = dt.datetime.fromtimestamp(value).astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def task_query_code() -> str:
    return r"""
import json
import sqlite3
import sys

db_path, task_id, history_tail_s = sys.argv[1], sys.argv[2], sys.argv[3]
history_tail = int(history_tail_s)
c = sqlite3.connect(db_path)
c.row_factory = sqlite3.Row
row = c.execute("select * from jobs where id = ?", (task_id,)).fetchone()
if row is None:
    print(json.dumps({"found": False, "id": task_id}))
    raise SystemExit(0)

data = dict(row)
history_raw = data.pop("history_json", None)
try:
    history = json.loads(history_raw) if history_raw else []
except json.JSONDecodeError:
    history = []
if history_tail < 0:
    data["history"] = history
elif history_tail == 0:
    data["history"] = []
else:
    data["history"] = history[-history_tail:]
for key in ("task_spec_json", "result_json", "draft_json", "review_edits_json", "published_draft_json"):
    raw = data.get(key)
    if raw:
        try:
            data[key[:-5] if key.endswith("_json") else key] = json.loads(raw)
        except json.JSONDecodeError:
            pass
print(json.dumps({"found": True, "job": data}, ensure_ascii=False))
"""


def load_task(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    command = [
        "kubectl",
        "exec",
        "-n",
        args.namespace,
        f"deploy/{args.release}",
        "--",
        "python",
        "-c",
        task_query_code(),
        args.db_path,
        task_id,
        str(args.history_tail),
    ]
    return json.loads(run_text(command))


def print_human(
    data: dict[str, Any],
    *,
    owner: Optional[str],
    repo: Optional[str],
    full_error: bool,
    error_chars: int,
) -> int:
    if not data.get("found"):
        print(f"task not found: {data.get('id')}", file=sys.stderr)
        return 3

    job = data["job"]
    target = f"{job.get('target_owner')}/{job.get('target_repo')}"
    created = job.get("created_at")
    updated = job.get("updated_at")
    duration = "-"
    if isinstance(created, (int, float)) and isinstance(updated, (int, float)):
        duration = f"{updated - created:.1f}s"

    print(f"Task: {job.get('id')}")
    if owner and repo:
        print(f"URL: https://serge.huggingface.tech/tasks/{owner}/{repo}/{job.get('id')}")
    print(f"Target: {target}")
    print(f"Actor: {job.get('user')}")
    print(f"Status: {job.get('status')}")
    print(f"Model: {job.get('llm_provider') or '-'} {job.get('llm_model') or '-'}")
    print(f"Created: {fmt_time(created)}")
    print(f"Updated: {fmt_time(updated)}")
    print(f"Duration: {duration}")

    result = job.get("result")
    if result:
        print("\nResult:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    error = job.get("error")
    if error:
        error = str(error)
        print("\nError:")
        if not full_error and len(error) > error_chars:
            error = (
                error[:error_chars]
                + f"\n... [truncated; use --full-error to print all {len(error)} chars]"
            )
        print(error)

    history = job.get("history") or []
    if history:
        print("\nHistory tail:")
        for event in history:
            kind = event.get("kind", "?")
            text = str(event.get("text", ""))
            seq = event.get("seq", "-")
            ts = fmt_time(event.get("ts"))
            if len(text) > 500:
                text = text[:497] + "..."
            print(f"[{seq}] {ts} {kind}: {text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deploy/scripts/task-status.sh",
        description="Read Serge's persisted SQLite row for a /tasks run.",
    )
    parser.add_argument("task", help="Task id or full https://serge.../tasks/<owner>/<repo>/<id> URL")
    parser.add_argument("-n", "--namespace", default="serge", help="Kubernetes namespace")
    parser.add_argument("-r", "--release", default="serge", help="Deployment name")
    parser.add_argument("--context", dest="expected_context", help="Required kubectl context")
    parser.add_argument("--db-path", default="/var/lib/reviewbot/jobs.db", help="Path to jobs.db inside the Serge pod")
    parser.add_argument("--history-tail", type=int, default=25, help="Number of persisted history events to print")
    parser.add_argument("--error-chars", type=int, default=4000, help="Maximum stored error chars to print")
    parser.add_argument("--full-error", action="store_true", help="Print the full stored error")
    parser.add_argument("--json", action="store_true", help="Print raw JSON from the job row")
    args = parser.parse_args()

    if shutil.which("kubectl") is None:
        print("missing required command: kubectl", file=sys.stderr)
        return 1

    try:
        current_context = run_text(["kubectl", "config", "current-context"])
    except CommandError as exc:
        return exc.return_code
    if args.expected_context and current_context != args.expected_context:
        print(
            f"refusing to read task data from context '{current_context}' "
            f"(expected '{args.expected_context}')",
            file=sys.stderr,
        )
        return 1

    owner, repo, task_id = parse_task_ref(args.task)
    try:
        data = load_task(args, task_id)
    except CommandError as exc:
        return exc.return_code

    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0 if data.get("found") else 3
    return print_human(
        data,
        owner=owner,
        repo=repo,
        full_error=args.full_error,
        error_chars=args.error_chars,
    )


if __name__ == "__main__":
    sys.exit(main())
