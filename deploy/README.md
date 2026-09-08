# Deploying Serge

This directory packages Serge's web app (`reviewbot-web`) for Kubernetes.
The Helm chart is intentionally self-contained and values-driven so a team can
deploy it into its own cluster without changing application code.

## Contents

- `helm/` contains the Helm chart for the web app: Deployment, Service,
  ConfigMap, optional Ingress, optional ServiceAccount, and a PersistentVolumeClaim.
- `helm/env/example.yaml` is an EXAMPLE values file — it is what CI renders,
  and it is not deployed anywhere. **The production values are not in this
  repo**: they are tracked in `huggingface/transformers-ci-playbooks` as
  `serge/env/prod.yaml`, next to the runbooks, and passed with `-f`.
  `deploy.sh` has no default values file, and refuses an upgrade whose values
  drop settings the live release has (`scripts/values_drift.py`).
- `helm/serge-secrets.example.yaml` is a template for the sensitive runtime env.
  Copy it to `helm/serge-secrets.yaml`, fill it locally, and never commit it.
- `scripts/deploy.sh` checks the current Kubernetes context, creates the namespace
  when needed, optionally applies a local Secret file, and runs Helm.
- `scripts/logs.sh` finds the current running Serge pod and prints recent logs.

## Chart Behavior

Serge uses embedded SQLite for review/task history. The chart therefore runs a
single replica with a `Recreate` rollout strategy and mounts a PVC at
`persistence.mountPath` (`/var/lib/reviewbot` by default). `WEB_STORE_PATH` is
set to `<mountPath>/jobs.db`, so the database survives pod restarts.

The container runs as a non-root user, drops Linux capabilities, uses
`RuntimeDefault` seccomp, and sets `fsGroup` so the app user can write the
volume. Sensitive values are loaded from a pre-created Secret via
`existingSecret`; non-secret runtime config lives in `envVars`.

## Deploy

Create or update the Secret in the target namespace:

```bash
cp deploy/helm/serge-secrets.example.yaml deploy/helm/serge-secrets.yaml
$EDITOR deploy/helm/serge-secrets.yaml
deploy/scripts/deploy.sh -n serge --secret-file deploy/helm/serge-secrets.yaml
```

Deploy without applying a Secret file, assuming `serge-secrets` already exists:

```bash
deploy/scripts/deploy.sh -n serge -f ../env/prod.yaml
```

(`../env/prod.yaml` is the tracked production values file — run the script from
a serge checkout inside the playbooks work root, where `../env` resolves to it.)

Use `--context` when you want the script to refuse any other kube context:

```bash
deploy/scripts/deploy.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n serge \
  -f ../env/prod.yaml
```

Fetch recent logs:

```bash
deploy/scripts/logs.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n serge \
  --since 2h \
  --grep 'error|traceback|crashed|HTTPError'
```

Print only the latest error block:

```bash
deploy/scripts/logs.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n serge \
  --since 2h \
  --last-error
```

## Notes

- The production image is published to GHCR as `ghcr.io/huggingface/serge`.
- `HELPER_SANDBOX=require` needs nodes that allow unprivileged user namespaces.
  Set it to `auto` or `off` in `envVars` if the cluster cannot support that.
- Avoid `kubectl apply` for filled Secret manifests long-term: it can store
  plaintext Secret values in the `last-applied-configuration` annotation. The
  helper strips that annotation after applying a Secret file.
