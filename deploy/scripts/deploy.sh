#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART_DIR="${ROOT_DIR}/deploy/helm"

release="serge"
namespace="serge"
values_file="${CHART_DIR}/env/prod.yaml"
secret_file=""
expected_context=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: deploy/scripts/deploy.sh [options]

Options:
  -n, --namespace NAME       Kubernetes namespace (default: serge)
  -r, --release NAME         Helm release name (default: serge)
  -f, --values FILE          Helm values file (default: deploy/helm/env/prod.yaml)
      --secret-file FILE     Apply a local Secret manifest before deploying
      --context NAME         Require this kubectl context before deploying
      --dry-run              Render manifests without changing the cluster
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
    -f|--values)
      values_file="$2"
      shift 2
      ;;
    --secret-file)
      secret_file="$2"
      shift 2
      ;;
    --context)
      expected_context="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
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

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd kubectl
require_cmd helm

if [[ ! -d "${CHART_DIR}" ]]; then
  echo "chart directory not found: ${CHART_DIR}" >&2
  exit 1
fi

if [[ ! -f "${values_file}" ]]; then
  echo "values file not found: ${values_file}" >&2
  exit 1
fi

current_context="$(kubectl config current-context)"
if [[ -n "${expected_context}" && "${current_context}" != "${expected_context}" ]]; then
  echo "refusing to deploy to context '${current_context}' (expected '${expected_context}')" >&2
  exit 1
fi

echo "Context: ${current_context}"
echo "Namespace: ${namespace}"
echo "Release: ${release}"
echo "Values: ${values_file}"

if [[ "${dry_run}" -eq 1 ]]; then
  helm template "${release}" "${CHART_DIR}" -n "${namespace}" -f "${values_file}"
  exit 0
fi

kubectl get namespace "${namespace}" >/dev/null 2>&1 || kubectl create namespace "${namespace}"

if [[ -n "${secret_file}" ]]; then
  if [[ ! -f "${secret_file}" ]]; then
    echo "secret file not found: ${secret_file}" >&2
    exit 1
  fi
  kubectl apply -n "${namespace}" -f "${secret_file}"
  kubectl annotate secret serge-secrets -n "${namespace}" kubectl.kubernetes.io/last-applied-configuration- >/dev/null 2>&1 || true
fi

helm upgrade --install "${release}" "${CHART_DIR}" \
  -n "${namespace}" \
  -f "${values_file}" \
  --wait \
  --timeout 10m

kubectl rollout status deployment/"${release}" -n "${namespace}" --timeout=10m
