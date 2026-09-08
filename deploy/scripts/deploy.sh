#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHART_DIR="${ROOT_DIR}/deploy/helm"

release="serge"
namespace="serge"
values_files=()
secret_file=""
expected_context=""
dry_run=0
from_head=0
allow_removals=0
image_wait_timeout=900

usage() {
  cat <<'EOF'
Usage: deploy/scripts/deploy.sh [options]

Options:
  -n, --namespace NAME       Kubernetes namespace (default: serge)
  -r, --release NAME         Helm release name (default: serge)
  -f, --values FILE          Helm values file; repeat to layer overlays, later
                             files win. REQUIRED — the production values are not
                             in this repo, see deploy/helm/env/example.yaml
      --secret-file FILE     Apply a local Secret manifest before deploying
      --context NAME         Require this kubectl context before deploying
      --from-head            Pin image.tag to HEAD's sha-<commit>, waiting for
                             CI to publish that image to GHCR first, then write
                             the tag into the values file before deploying
      --dry-run              Render manifests without changing the cluster
      --allow-removals       Proceed even though these values DROP settings the
                             live release has (see the drift preflight below)
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
      values_files+=("$2")
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
    --allow-removals)
      allow_removals=1
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

# No default values file, on purpose. This chart's own env/ holds an EXAMPLE,
# never the deployed configuration: production values are tracked outside this
# repo (transformers-ci-playbooks, serge/env/prod.yaml) and are passed with -f.
# A default here is how a deploy silently ships example config — see the drift
# preflight below and deploy/scripts/values_drift.py.
if [[ "${#values_files[@]}" -eq 0 ]]; then
  echo "no values file given: pass -f <file> (production values live in the" >&2
  echo "transformers-ci-playbooks repo as serge/env/prod.yaml)" >&2
  exit 2
fi

for vf in "${values_files[@]}"; do
  if [[ ! -f "${vf}" ]]; then
    echo "values file not found: ${vf}" >&2
    exit 1
  fi
done

# Assemble helm's repeated -f flags once; later files override earlier ones.
helm_values_args=()
for vf in "${values_files[@]}"; do
  helm_values_args+=(-f "${vf}")
done

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

  # Pin the tag in the primary values file (portable in-place edit). All three
  # images are built from the same commit by docker.yml (serge, -task-runner,
  # -egress, each tagged sha-<commit>), so they must move together. We rewrite:
  #   * the top-level image.tag line (serge app), and
  #   * the inline `ghcr.io/...:sha-<commit>` refs for the task-runner + egress
  #     images (they carry a full image string, not an image.tag).
  # image.tag lives in the base values file (prod.yaml), so we edit the first
  # -f file; overlays layered after it do not carry serge's image refs.
  primary_values="${values_files[0]}"
  tmp_values="$(mktemp)"
  sed -E \
    -e "s|^([[:space:]]*tag:[[:space:]]*).*|\1${image_tag}|" \
    -e "s|(ghcr\.io/[^:[:space:]]+):sha-[0-9a-f]+|\1:${image_tag}|g" \
    "${primary_values}" > "${tmp_values}"
  mv "${tmp_values}" "${primary_values}"
  echo "Pinned all images to ${image_tag} in ${primary_values}"
fi

current_context="$(kubectl config current-context)"
if [[ -n "${expected_context}" && "${current_context}" != "${expected_context}" ]]; then
  echo "refusing to deploy to context '${current_context}' (expected '${expected_context}')" >&2
  exit 1
fi

echo "Context: ${current_context}"
echo "Namespace: ${namespace}"
echo "Release: ${release}"
echo "Values: ${values_files[*]}"

if [[ "${dry_run}" -eq 1 ]]; then
  helm template "${release}" "${CHART_DIR}" -n "${namespace}" "${helm_values_args[@]}"
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

# Preflight: refuse a values file that DROPS configuration the live release has.
#
# On 2026-09-04 serge was upgraded from inside this repo with the chart's own
# env/prod.yaml (then a stale copy of the real thing) instead of the tracked
# production values. helm said "Upgrade complete", the pod stayed 1/1 Running,
# and nothing else looked wrong -- but VERIFY_ON_GPU, VERIFY_REPRODUCE_FIRST,
# both *_COMMENT_BREVITY keys and the whole backups block were simply absent.
# For three nightlies serge opened fix PRs with no GPU verification (and no
# "not verified" note, because that footer only exists when a run does), one of
# which was merged; the SQLite backup CronJob was gone for four days.
#
# A values file that is *almost* right fails silently, so the check is on the
# shape of the change, not on a list of keys someone has to remember to update:
# an upgrade that removes settings has to say so with --allow-removals.
check_values_drift() {
  if ! helm status "${release}" -n "${namespace}" >/dev/null 2>&1; then
    return 0  # first install — there is nothing to drop yet
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "warning: python3 not found — skipping the values drift check" >&2
    return 0
  fi

  local live_json new_json dropped rc
  live_json="$(mktemp)"
  new_json="$(mktemp)"
  # shellcheck disable=SC2064  # expand the paths now, not at trap time
  trap "rm -f '${live_json}' '${new_json}'" RETURN

  helm get values "${release}" -n "${namespace}" -o json > "${live_json}"
  # `.config` of a client-side dry run is exactly the merged user-supplied
  # values this deploy would send — helm does the -f layering, so the check
  # needs no YAML parser of its own.
  helm upgrade --install "${release}" "${CHART_DIR}" \
    -n "${namespace}" \
    "${helm_values_args[@]}" \
    --dry-run=client -o json \
    | python3 -c 'import json,sys; json.dump(json.load(sys.stdin).get("config") or {}, sys.stdout)' \
    > "${new_json}"

  set +e
  dropped="$(python3 "${ROOT_DIR}/deploy/scripts/values_drift.py" "${live_json}" "${new_json}")"
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${rc}" -ne 3 ]]; then
    echo "values drift check failed (exit ${rc}); refusing to deploy blind" >&2
    exit 1
  fi

  echo >&2
  echo "These settings are live on release '${release}' and are NOT in ${values_files[*]}:" >&2
  echo "${dropped}" | sed 's/^/  - /' >&2
  echo >&2
  if [[ "${allow_removals}" -eq 1 ]]; then
    echo "--allow-removals given; proceeding." >&2
    return 0
  fi
  echo "Deploying would unset them. That is usually the wrong values file:" >&2
  echo "serge's production values are tracked in transformers-ci-playbooks as" >&2
  echo "serge/env/prod.yaml — deploy with -f ../env/prod.yaml from a checkout" >&2
  echo "inside that work root. If the removal is intended, re-run with" >&2
  echo "--allow-removals." >&2
  exit 1
}

check_values_drift

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
  "${helm_values_args[@]}" \
  --wait \
  --timeout 10m

kubectl rollout status deployment/"${release}" -n "${namespace}" --timeout=10m
