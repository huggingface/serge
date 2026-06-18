#!/usr/bin/env bash
set -euo pipefail

namespace="serge"
release="serge"
since="2h"
follow=0
grep_pattern=""
expected_context=""
last_error=0

usage() {
  cat <<'EOF'
Usage: deploy/scripts/logs.sh [options]

Options:
  -n, --namespace NAME       Kubernetes namespace (default: serge)
  -r, --release NAME         Helm release/app label (default: serge)
      --since DURATION       Log window, e.g. 30m, 2h (default: 2h)
  -f, --follow               Follow logs
      --grep PATTERN         Filter logs with grep -Ei
      --last-error           Print the last ERROR/Traceback block from recent logs
      --context NAME         Require this kubectl context before reading logs
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace)
      namespace="$2"
      shift 2
      ;;
    -r|--release)
      release="$2"
      shift 2
      ;;
    --since)
      since="$2"
      shift 2
      ;;
    -f|--follow)
      follow=1
      shift
      ;;
    --grep)
      grep_pattern="$2"
      shift 2
      ;;
    --last-error)
      last_error=1
      shift
      ;;
    --context)
      expected_context="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v kubectl >/dev/null 2>&1; then
  echo "missing required command: kubectl" >&2
  exit 1
fi

if [[ -n "${grep_pattern}" ]] && ! command -v grep >/dev/null 2>&1; then
  echo "missing required command: grep" >&2
  exit 1
fi

current_context="$(kubectl config current-context)"
if [[ -n "${expected_context}" && "${current_context}" != "${expected_context}" ]]; then
  echo "refusing to read logs from context '${current_context}' (expected '${expected_context}')" >&2
  exit 1
fi

pod="$(
  kubectl get pods -n "${namespace}" \
    -l "app=${release}" \
    --field-selector=status.phase=Running \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | tail -n 1
)"

if [[ -z "${pod}" ]]; then
  echo "no running pod found in namespace '${namespace}' with label app=${release}" >&2
  exit 1
fi

pod_started="$(kubectl get pod "${pod}" -n "${namespace}" -o jsonpath='{.status.startTime}')"

echo "Context: ${current_context}" >&2
echo "Namespace: ${namespace}" >&2
echo "Pod: ${pod}" >&2
echo "Pod started: ${pod_started}" >&2
echo "Since: ${since}" >&2

args=(logs -n "${namespace}" "${pod}" --since="${since}")
if [[ "${follow}" -eq 1 ]]; then
  args+=(-f)
fi

if [[ "${last_error}" -eq 1 ]]; then
  if [[ "${follow}" -eq 1 ]]; then
    echo "--last-error cannot be combined with --follow" >&2
    exit 2
  fi
  kubectl "${args[@]}" | awk '
    function flush() {
      if (in_block && block != "") {
        last = block
      }
      block = ""
      in_block = 0
    }

    /^[0-9-]+ [0-9:,]+ ERROR / {
      flush()
      in_block = 1
      block = $0 "\n"
      next
    }

    /^Traceback \(most recent call last\):/ {
      if (!in_block) {
        in_block = 1
        block = $0 "\n"
      } else {
        block = block $0 "\n"
      }
      next
    }

    in_block {
      if ($0 ~ /^[0-9-]+ [0-9:,]+ (INFO|WARNING|ERROR|DEBUG) / || $0 ~ /^INFO:/) {
        flush()
      }
    }

    in_block {
      block = block $0 "\n"
    }

    END {
      flush()
      if (last != "") {
        printf "%s", last
      } else {
        exit 1
      }
    }
  ' || {
    echo "no ERROR/Traceback block found for pod ${pod} in the last ${since}" >&2
    echo "note: this only covers logs retained for the current pod, which started at ${pod_started}" >&2
    exit 1
  }
  exit 0
fi

if [[ -n "${grep_pattern}" ]]; then
  kubectl "${args[@]}" | grep -Ei "${grep_pattern}"
else
  kubectl "${args[@]}"
fi
