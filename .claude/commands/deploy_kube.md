---
description: Deploy the current working tree to the serge Kubernetes stack via the Helm chart in deploy/
allowed-tools: Bash(deploy/scripts/deploy.sh:*), Bash(./deploy/scripts/deploy.sh:*), Bash(deploy/scripts/logs.sh:*), Bash(./deploy/scripts/logs.sh:*), Bash(infra use:*), Bash(infra login:*), Bash(kubectl config current-context:*), Bash(kubectl rollout status:*), Bash(kubectl get pods:*)
---

Update the running `serge` Kubernetes stack using the Helm chart in `deploy/`.

Deployment target:
- cluster / kube context: `infra:opensource-aws-use1-prod-54`
- namespace: `serge`, Helm release: `serge`
- ingress host: `serge.huggingface.tech` (internal ALB; VPN-only)
- production values: `deploy/helm/env/prod.yaml`
- optional normalize backend overlay: `deploy/helm/env/normalize.example.yaml`
  (gitignored — kept local only because it carries internal cluster/infra
  identifiers). Layer it after `prod.yaml` to enable the Kubernetes task
  sandbox (Backend B).

`-f` is repeatable and passes straight through to `helm`; later files override
earlier ones. `--from-head` pins `image.tag` in the **first** `-f` file
(`prod.yaml`, where serge's `image.tag` lives), so always pass `prod.yaml`
first and overlays after it.

`deploy/scripts/deploy.sh` checks the kube context, creates the namespace if
needed, optionally applies a local Secret manifest, then runs
`helm upgrade --install --wait` and waits for the rollout. The non-secret
runtime config lives in `deploy/helm/env/prod.yaml`; sensitive values come from
the pre-created `serge-secrets` Secret in the namespace.

Steps:

1. Confirm (or refresh) the kube context. If `kubectl config current-context`
   is not `infra:opensource-aws-use1-prod-54`, refresh credentials:

   ```bash
   infra use opensource-aws-use1-prod-54
   ```

   infra **access keys also expire** (a local context switch still "succeeds"
   with a stale key). The deploy script preflights cluster connectivity and, on
   failure, prints the exact recovery commands. That login is interactive and
   cannot be scripted — ask the user to run it in-session via `!`:

   ```bash
   infra login infra-hq.internal.huggingface.tech
   ```

2. To pick up new application code, bump `image.tag` in
   `deploy/helm/env/prod.yaml` to the `sha-<commit>` tag CI published for the
   merged commit (or `latest`). The chart deploys a published GHCR image —
   it does not build from the working tree.

   **Preferred:** pass `--from-head` to the deploy script (step 3) instead of
   editing the file by hand. It resolves `HEAD`'s `sha-<commit>` tag, waits for
   CI's Docker build to publish that image to GHCR, then writes the tag into the
   values file automatically. It aborts if the build failed and warns if there
   are uncommitted changes under `reviewbot/` (which won't be in the image).
   The commit must already be pushed so CI has built it.

3. Run the deploy, pinning the expected context so it refuses any other cluster:

   ```bash
   ./deploy/scripts/deploy.sh \
     --context infra:opensource-aws-use1-prod-54 \
     -n serge \
     -f deploy/helm/env/prod.yaml
   ```

   Add `--from-head` (see step 2) to auto-pin and wait for HEAD's CI image:

   ```bash
   ./deploy/scripts/deploy.sh \
     --context infra:opensource-aws-use1-prod-54 \
     -n serge \
     -f deploy/helm/env/prod.yaml \
     --from-head
   ```

   To also enable the Kubernetes normalize backend, layer the (local,
   gitignored) normalize overlay after `prod.yaml`:

   ```bash
   ./deploy/scripts/deploy.sh \
     --context infra:opensource-aws-use1-prod-54 \
     -n serge \
     -f deploy/helm/env/prod.yaml \
     -f deploy/helm/env/normalize.example.yaml \
     --from-head
   ```

   To preview the rendered manifests without touching the cluster, add
   `--dry-run`. To rotate secrets in the same run, also pass
   `--secret-file deploy/helm/serge-secrets.yaml` (never commit that file).

4. Report the rollout status the script prints and tail recent logs to confirm
   the new pod is serving:

   ```bash
   ./deploy/scripts/logs.sh \
     --context infra:opensource-aws-use1-prod-54 \
     -n serge --since 5m
   ```

   If the pod is unhealthy, use `--last-error` to surface the latest
   ERROR/Traceback block.

Do not commit or push anything — this command only deploys.
