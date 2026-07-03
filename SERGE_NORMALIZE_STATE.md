# Serge normalize — Backend B state (resume here)

Snapshot for resuming after a context clear. Companion to
`SERGE_NORMALIZE_NEXT_STEPS.md` (plan) and `SERGE_NORMALIZE_PLAN.md` (design).
Last updated 2026-07-01.

## Goal
Run each /tasks patch-validation normalizer in a **one-shot, locked-down
Kubernetes Job** (Backend B), so serge opens transformers PRs that are already
green on repo-consistency. Docker/bwrap backends stay first-class; k8s is opt-in.

## TL;DR of where we are
- **Serge code for Backend B is written, tested, and mostly committed.**
- **Storage (EFS) is provisioned and verified on the real cluster.**
- **A serge image was built** from the branch — but it **predates the last change**
  (configurable Job `nodeSelector`), which is **still uncommitted**.
- **Not yet done:** rebuild image with the node-selector delta → deploy to an
  isolated namespace → fire a real task. No live end-to-end run has happened.

---

## Environment facts (all verified)
- **Cluster / kube context:** `infra:opensource-aws-use1-prod-54` (AWS EKS, prod).
- **Namespace:** `serge` (live serge deployment runs here, image
  `ghcr.io/huggingface/serge:sha-2ce3810`, ingress `serge.huggingface.tech`).
  Secret `serge-secrets` holds GitHub App + LLM creds in this namespace.
- **Repo:** `huggingface/serge`, branch **`dockerized`**, HEAD **`8c3222c`**
  (already pushed to origin).
- **EFS StorageClass:** `serge-tasks-efs-sc` — `efs.csi.aws.com`, `efs-ap`,
  `Immediate`, `fileSystemId fs-0672399c64245075b`, **`uid=1000, gid=1000`**,
  `directoryPerms=700`.
- **Node scheduling:** pods target CASTAI template
  `scheduling.cast.ai/node-template=default-by-castai`.
- **Curated normalize command** (in `deploy/helm/env/normalize.example.yaml`):
  `make style` + `python utils/checkers.py <ALL checkers MINUS add_dates> --fix
  --keep-going`. `add_dates` is excluded (needs network → dies under deny-all
  egress; not a git issue).

## Verified on the real cluster
- **EFS RWX + uid 1000 + git works.** A pod mirroring serge's security context
  (`runAsUser/Group/fsGroup 1000`) bound the PVC, wrote files, and ran
  `git init/add/commit/status` → **GIT_OK**. Worktree owned `1000:1000`.
  - First attempt used SC `uid/gid=0` → files root-owned → `git` failed with
    "dubious ownership". Infra changed to `1000` → fixed. **Do not regress this.**
- **Chart:** `helm lint` clean; `kubectl apply --dry-run=server` accepted all
  new resources (Role, RoleBinding, NetworkPolicy, RWX PVC).
- **Image build** (`gh workflow run docker.yml --ref dockerized`) succeeded →
  `ghcr.io/huggingface/serge:sha-8c3222c`. **This image lacks the node-selector
  code** (see below). `latest` was NOT tagged (only tags on `main`), so prod is
  untouched.

---

## What's committed in `8c3222c`
- `reviewbot/clone_cache.py` — `acquire_ref(..., standalone=True)`: self-contained
  local clone (not a linked worktree), checked out on a branch named `main`, so
  in-sandbox `git` works with only the worktree mounted.
- `reviewbot/config.py` — `Config.needs_isolated_checkout` + `task_k8s_*` fields.
- `reviewbot/k8s_sandbox.py` — pure `build_job_manifest` + `run_job`
  (create→poll→logs→exit-code→delete). `kubernetes` client imported lazily.
- `reviewbot/normalize.py` / `tasks.py` — `_run_kubernetes` wired; k8s passed as
  plain config values so tasks/normalize don't hard-depend on the k8s module.
- `Dockerfile` — installs `.[web,kubernetes]` (was `.[web]`).
- `pyproject.toml` — `kubernetes` optional extra.
- Chart: `templates/normalize.yaml` (RWX PVC + Role + RoleBinding +
  deny-all-egress NetworkPolicy on label `serge.io/sandbox: normalize`),
  `config.yaml` env wiring, `deployment.yaml` worktree mount, `values.yaml`
  `normalize.kubernetes` block.
- Tests: `tests/test_k8s_sandbox.py`, `test_clone_cache_tasks.py`,
  `test_config.py` additions.

## ⚠️ UNCOMMITTED right now (the node-selector delta) — 7 files
Configurable Job `nodeSelector` so Job pods land on the CASTAI template
(serge's own `nodeSelector` already works via the chart; the **Job** needs this
code). Wired: `TASK_K8S_NODE_SELECTOR` ("k=v,k2=v2") → `parse_node_selector` →
`K8sSettings.node_selector` → `pod_spec.nodeSelector`.
- `reviewbot/k8s_sandbox.py` (K8sSettings.node_selector, parse_node_selector,
  pod_spec nodeSelector)
- `reviewbot/normalize.py` (k8s_node_selector param → _run_kubernetes)
- `reviewbot/tasks.py` (passes `k8s_node_selector=cfg.task_k8s_node_selector`)
- `reviewbot/config.py` (`task_k8s_node_selector` + env `TASK_K8S_NODE_SELECTOR`)
- `deploy/helm/values.yaml` (`normalize.kubernetes.nodeSelector: ""`)
- `deploy/helm/templates/config.yaml` (renders `TASK_K8S_NODE_SELECTOR`)
- `deploy/helm/env/normalize.example.yaml` (sets both selectors + `serge-tasks-efs-sc`)

All verified locally: `parse_node_selector` + manifest correct; chart renders
both the deployment `nodeSelector` map and the Job env string; `helm lint` clean;
**65 tests pass** (k8s + config + tasks + clone_cache subsets).

---

## Remaining steps to a live run
1. **Commit** the 7 node-selector files on `dockerized`; **push**; **rebuild**
   image: `gh workflow run docker.yml --ref dockerized` → new `sha-<commit>`.
2. **Deploy to an isolated namespace** (recommended `serge-normalize-test`, NOT
   the live `serge` release):
   ```bash
   helm install serge-nrm deploy/helm -n serge-normalize-test --create-namespace \
     -f deploy/helm/env/prod.yaml -f deploy/helm/env/normalize.example.yaml \
     --set image.tag=sha-<newcommit> \
     --set existingSecret=serge-secrets --set ingress.enabled=false
   ```
   NOTE: `serge-secrets` lives in the `serge` namespace — either deploy the test
   release **into `serge`** (reuse the secret directly, distinct release name) or
   **copy the secret** into `serge-normalize-test` first. Decide this.
3. **Fire a task** at the fork `tarekziade/transformers` and watch the Job:
   `kubectl -n <ns> get jobs,pods -w`. OPEN QUESTION: how to trigger — the
   `/tasks` API is GitHub-OIDC-gated; need either an OIDC token (from a
   transformers-ci `workflow_dispatch`) or a dev trigger path.
4. **Acceptance:** Job runs the normalizer on the EFS worktree → serge opens a
   clean PR; feedback loop works in-cluster.

## Gotchas / notes for the next session
- **rtk proxy filters `git status`** — plain `git status` under-reports; use
  `rtk proxy git status --short` for the true list.
- **Local test env lacks `flask`/`itsdangerous`** → `test_webapp_*` and
  `test_app.py` fail on import only (42 failures, all env). Run targeted suites
  with `rtk proxy env PYTHONPATH=. python -m pytest <files>`.
- **pytest via rtk** needs `rtk proxy env PYTHONPATH=. python -m pytest ...`
  (rtk strips PYTHONPATH otherwise, and there's no editable install).
- **EBS is RWO** on this cluster (only `ebs-gp2/gp3`); that's why EFS was needed.
  EBS-RWO + pinning the Job to serge's node is a valid no-EFS fallback (writes
  are sequential: serge writes → Job writes → serge reads), but EFS is cleaner
  and already done.
- **hf-mount is NOT suitable** for the worktree: object-store/xet FUSE can't do
  git's atomic `rename()`-over-existing (index/refs/config), so `git add` fails
  even single-writer. EFS (real NFS) is required.
- Job pod security (in `build_job_manifest`): non-root 1000, readOnlyRootFS +
  `/tmp` emptyDir, drop ALL caps, no-priv-esc, RuntimeDefault seccomp,
  `automountServiceAccountToken: false`, `backoffLimit 0`, activeDeadlineSeconds.
- Worktree PVC is mounted into the Job by **subPath** (only its own worktree,
  same absolute path as serge's `WEB_CLONE_CACHE_DIR`).
