#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import sys


class CommandError(Exception):
    def __init__(self, return_code):
        self.return_code = return_code


def run_text(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise CommandError(exc.returncode) from exc


def stream_logs(args, grep_pattern):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    pattern = re.compile(grep_pattern, re.IGNORECASE) if grep_pattern else None
    try:
        for line in proc.stdout:
            if pattern is None or pattern.search(line):
                print(line, end="")
    finally:
        if proc.stdout:
            proc.stdout.close()

    return proc.wait()


def latest_error_block(args, pod, pod_started, since):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None

    log_line_re = re.compile(r"^[0-9-]+ [0-9:,]+ (INFO|WARNING|ERROR|DEBUG) ")
    error_re = re.compile(r"^[0-9-]+ [0-9:,]+ ERROR ")
    last = []
    block = []

    def flush():
        nonlocal last, block
        if block:
            last = block
        block = []

    for line in proc.stdout:
        if error_re.match(line):
            flush()
            block = [line]
            continue

        if line.startswith("Traceback (most recent call last):"):
            block.append(line)
            continue

        if block and (log_line_re.match(line) or line.startswith("INFO:")):
            flush()

        if block:
            block.append(line)

    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        return return_code

    flush()
    if last:
        print("".join(last), end="")
        return 0

    print(f"no ERROR/Traceback block found for pod {pod} in the last {since}", file=sys.stderr)
    print(
        f"note: this only covers logs retained for the current pod, which started at {pod_started}",
        file=sys.stderr,
    )
    return 1


def main():
    parser = argparse.ArgumentParser(
        prog="deploy/scripts/logs.sh",
        description="Find the current running Serge pod and print recent logs.",
    )
    parser.add_argument("-n", "--namespace", default="serge", help="Kubernetes namespace")
    parser.add_argument("-r", "--release", default="serge", help="Helm release/app label")
    parser.add_argument("--since", default="2h", help="Log window, e.g. 30m, 2h")
    parser.add_argument("-f", "--follow", action="store_true", help="Follow logs")
    parser.add_argument("--grep", dest="grep_pattern", help="Filter logs with grep -Ei semantics")
    parser.add_argument(
        "--task-id",
        help=(
            "Filter to one task id. Useful for /tasks investigations; shows "
            "queue/finalizer/watcher lines and hides high-volume HTTP access logs."
        ),
    )
    parser.add_argument(
        "--task-http",
        action="store_true",
        help="With --task-id, include high-volume HTTP callback/status access logs.",
    )
    parser.add_argument("--last-error", action="store_true", help="Print the last ERROR/Traceback block")
    parser.add_argument("--context", dest="expected_context", help="Required kubectl context")
    args = parser.parse_args()

    if args.last_error and args.follow:
        print("--last-error cannot be combined with --follow", file=sys.stderr)
        return 2
    if args.task_http and not args.task_id:
        print("--task-http requires --task-id", file=sys.stderr)
        return 2
    if args.task_id and args.grep_pattern:
        print("--task-id cannot be combined with --grep", file=sys.stderr)
        return 2

    if shutil.which("kubectl") is None:
        print("missing required command: kubectl", file=sys.stderr)
        return 1

    try:
        current_context = run_text(["kubectl", "config", "current-context"])
    except CommandError as exc:
        return exc.return_code
    if args.expected_context and current_context != args.expected_context:
        print(
            f"refusing to read logs from context '{current_context}' "
            f"(expected '{args.expected_context}')",
            file=sys.stderr,
        )
        return 1

    try:
        pod_output = run_text(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                args.namespace,
                "-l",
                f"app={args.release}",
                "--field-selector=status.phase=Running",
                "--sort-by=.metadata.creationTimestamp",
                "-o",
                'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
            ]
        )
    except CommandError as exc:
        return exc.return_code
    pods = [line for line in pod_output.splitlines() if line]
    if not pods:
        print(
            f"no running pod found in namespace '{args.namespace}' with label app={args.release}",
            file=sys.stderr,
        )
        return 1

    pod = pods[-1]
    try:
        pod_started = run_text(
            [
                "kubectl",
                "get",
                "pod",
                pod,
                "-n",
                args.namespace,
                "-o",
                "jsonpath={.status.startTime}",
            ]
        )
    except CommandError as exc:
        return exc.return_code

    print(f"Context: {current_context}", file=sys.stderr)
    print(f"Namespace: {args.namespace}", file=sys.stderr)
    print(f"Pod: {pod}", file=sys.stderr)
    print(f"Pod started: {pod_started}", file=sys.stderr)
    print(f"Since: {args.since}", file=sys.stderr)

    log_args = ["kubectl", "logs", "-n", args.namespace, pod, f"--since={args.since}"]
    if args.follow:
        log_args.append("-f")

    if args.task_id:
        escaped = re.escape(args.task_id)
        if args.task_http:
            args.grep_pattern = (
                f"{escaped}|task .*{escaped}|/internal/tasks/{escaped}|"
                f"/tasks/[^ ]*/{escaped}"
            )
        else:
            args.grep_pattern = f"^[0-9].*(task .*{escaped}|{escaped}.*task)"

    if args.last_error:
        return latest_error_block(log_args, pod, pod_started, args.since)

    return stream_logs(log_args, args.grep_pattern)


if __name__ == "__main__":
    sys.exit(main())
