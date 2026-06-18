#!/usr/bin/env bash
set -euo pipefail

namespace="serge"
release="serge"
since="2h"
follow=0
grep_pattern=""
expected_context=""

usage() {
  cat <<'EOF'
Usage: deploy/scripts/logs.sh [options]

Options:
  -n, --namespace NAME       Kubernetes namespace (default: serge)
  -r, --release NAME         Helm release/app label (default: serge)
      --since DURATION       Log window, e.g. 30m, 2h (default: 2h)
  -f, --follow               Follow logs
      --grep PATTERN         Filter logs with grep -Ei
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

echo "Context: ${current_context}" >&2
echo "Namespace: ${namespace}" >&2
echo "Pod: ${pod}" >&2
echo "Since: ${since}" >&2

args=(logs -n "${namespace}" "${pod}" --since="${since}")
if [[ "${follow}" -eq 1 ]]; then
  args+=(-f)
fi

if [[ -n "${grep_pattern}" ]]; then
  kubectl "${args[@]}" | grep -Ei "${grep_pattern}"
else
  kubectl "${args[@]}"
fi
