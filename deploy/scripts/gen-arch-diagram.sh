#!/usr/bin/env bash
# Regenerate the Kubernetes architecture diagram in the docs from the Helm
# chart, using KubeDiagrams (https://github.com/philippemerle/KubeDiagrams).
#
# The diagram is rendered from `helm template` output with the pod-per-task /
# pod-per-review backend enabled, so it shows every resource the chart creates
# in the full "kube" deployment flavor: the serge Deployment/Service/Ingress,
# the SQLite PVC + ConfigMap + ServiceAccount, the serge-egress allowlist proxy
# (Deployment/Service/ConfigMap/NetworkPolicy), the task-pod NetworkPolicy, and
# the task-runner Role/RoleBinding.
#
# Prerequisites:
#   pip install KubeDiagrams        # provides `kube-diagrams`
#   brew install graphviz helm      # `dot` + `helm`
#
# Usage:
#   deploy/scripts/gen-arch-diagram.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
chart="$here/deploy/helm"
out="$here/docs/assets/architecture-k8s.png"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

helm template serge "$chart" \
  --namespace serge \
  --set serviceAccount.create=true \
  --set existingSecret=serge-secrets \
  --set taskExecution.kubernetes.enabled=true \
  --set taskExecution.kubernetes.image=ghcr.io/huggingface/serge-task-runner:latest \
  --set taskExecution.kubernetes.reviewPods=true \
  --set taskExecution.kubernetes.egress.image=ghcr.io/huggingface/serge-egress:latest \
  --set ingress.enabled=true \
  --set ingress.host=serge.huggingface.tech \
  --set ingress.className=alb \
  > "$tmp/rendered.yaml"

kube-diagrams -o "$out" "$tmp/rendered.yaml"
echo "Wrote $out"
