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
from_head=0
image_wait_timeout=900

usage() {
  cat <<'EOF'
Usage: deploy/scripts/deploy.sh [options]

Options:
  -n, --namespace NAME       Kubernetes namespace (default: serge)
  -r, --release NAME         Helm release name (default: serge)
  -f, --values FILE          Helm values file (default: deploy/helm/env/prod.yaml)
      --secret-file FILE     Apply a local Secret manifest before deploying
      --context NAME         Require this kubectl context before deploying
      --from-head            Pin image.tag to HEAD's sha-<commit>, waiting for
                             CI to publish that image to GHCR first, then write
                             the tag into the values file before deploying
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
    --from-head)
      from_head=1
      shift
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

# --from-head: derive HEAD's image tag, wait for CI to publish it, then pin it
# in the values file. The CI tag format mirrors docker/metadata-action's
# `type=sha,prefix=sha-` (.github/workflows/docker.yml): the 7-char prefix of
# the full commit SHA. We cannot build from the working tree; the chart only
# deploys published GHCR images.
if [[ "${from_head}" -eq 1 ]]; then
  require_cmd git
  require_cmd gh

  full_sha="$(git rev-parse HEAD)"
  short_sha="${full_sha:0:7}"
  image_tag="sha-${short_sha}"
  echo "Resolving image for HEAD ${short_sha} (tag ${image_tag})"

  if [[ -n "$(git status --porcelain -- "${ROOT_DIR}/reviewbot" 2>/dev/null)" ]]; then
    echo "warning: uncommitted changes under reviewbot/ — they are NOT in ${image_tag}" >&2
  fi

  echo "Waiting up to ${image_wait_timeout}s for CI to publish ${image_tag} ..."
  deadline=$(( $(date +%s) + image_wait_timeout ))
  while true; do
    # The Docker build workflow; succeeds once the sha-<commit> image is pushed.
    conclusion="$(gh run list --workflow docker.yml --commit "${full_sha}" \
      --limit 1 --json conclusion --jq '.[0].conclusion' 2>/dev/null || true)"
    case "${conclusion}" in
      success)
        echo "CI image build succeeded for ${short_sha}."
        break
        ;;
      failure|cancelled|timed_out|action_required)
        echo "CI image build for ${short_sha} ended with '${conclusion}'; aborting." >&2
        exit 1
        ;;
      *)
        if [[ "$(date +%s)" -ge "${deadline}" ]]; then
          echo "timed out waiting for CI to publish ${image_tag}" >&2
          exit 1
        fi
        sleep 15
        ;;
    esac
  done

  # Pin the tag in the values file (portable in-place edit). Only the single
  # image.tag line is touched.
  tmp_values="$(mktemp)"
  sed -E "s|^([[:space:]]*tag:[[:space:]]*).*|\1${image_tag}|" "${values_file}" > "${tmp_values}"
  mv "${tmp_values}" "${values_file}"
  echo "Pinned image.tag: ${image_tag} in ${values_file}"
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

# Preflight: confirm we can actually reach the cluster. infra access keys
# expire, and `kubectl config current-context` is purely local — it succeeds
# even with stale creds. Surface the interactive login command (it cannot be
# scripted non-interactively by design) instead of a cryptic API error later.
if ! kubectl version --request-timeout=10s >/dev/null 2>&1; then
  cluster="${expected_context#infra:}"
  echo "cannot reach the cluster — infra credentials are likely expired." >&2
  echo "refresh them, then re-run this deploy:" >&2
  echo "  infra login infra-hq.internal.huggingface.tech" >&2
  if [[ -n "${cluster}" ]]; then
    echo "  infra use ${cluster}" >&2
  fi
  exit 1
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
