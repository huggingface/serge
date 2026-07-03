# Serge normalize — next steps

Forward plan after **Phase 0** (in-loop normalize validation) landed and was
validated end-to-end. Order of work, as decided:

1. **Backend B — a Kubernetes Job per task.** Decided over the interim DinD
   sidecar (Backend A): proper isolation, no privileged container. Implement
   `normalize._run_kubernetes` (`reviewbot/k8s_sandbox.py`).
2. **Deploy it.**
3. **Build the transformers-ci workflow** that calls serge.

**Decisions locked in (2026-06-30):**
- **Backend B**, not A — go straight to the Kubernetes Job; skip the privileged
  DinD sidecar.
- **Image architecture** — runs on the proper arch in the cloud, so the arm64
  emulation problem does not apply to the deployed path.
- **git-freeness — RESOLVED.** See "Status" below: the sandbox checkout is now a
  self-contained clone, so in-sandbox `git` works and the full repo-consistency
  set runs. The remaining checker carve-out is `add_dates` (network, not git).

Background/design: [`SERGE_NORMALIZE_PLAN.md`](SERGE_NORMALIZE_PLAN.md) (esp.
§5 backend A/B) · user-facing: [`docs/tasks-normalize.md`](docs/tasks-normalize.md)
· task flow: [`TASKS_FLOW_PLAN.md`](TASKS_FLOW_PLAN.md).

---

## Status — what's done (Phase 0)

- In-loop normalize validation: the LLM patch is applied + the repo normalizer
  (`make style && make fix-repo`) is run **inside the agentic loop**; failures
  are fed back so the model corrects the patch before serge ever writes the PR.
  Lives in `tasks._validate_patch` + the `validate` hook on
  `reviewer._run_agentic_loop`; the gate fires before `publish_task`.
- Backends: `bwrap` / `docker` / `auto` implemented; **`kubernetes` is a stub**
  (`normalize._run_kubernetes` raises `NormalizeError`).
- Repo conventions wired into the task prompt (`REVIEW_RULES_PATH`, read from
  the worktree) + root-cause/no-`noqa` guidance + optional
  `TASK_NORMALIZE_GUIDANCE`.
- **Validated for real** against the fork `tarekziade/transformers` with the
  `huggingface/transformers-quality` image and `docker` backend:
  - happy path → clean PR ([#1]); feedback loop → PR ([#2]);
  - with `.ai/AGENTS.md` + guidance the model **removed a dead variable**
    instead of `# noqa`-ing it ([#3]) — guidance demonstrably changes behavior.
- **Mergeable + backward-compatible** with `main`: reviews untouched, no DB
  schema change, all new knobs default off, command-tasks (never on main)
  removed. 302 tests pass.

Learnings to carry forward:
- **git-freeness — RESOLVED.** The clone-cache worktree was a *linked* git
  worktree whose gitdir lives outside the bind mount, so in-sandbox `git`
  failed. Fixed in `clone_cache.acquire_ref(..., standalone=True)`: for tasks
  that normalize in a container backend it now produces a **self-contained
  local clone** (`git clone --local --no-hardlinks`, checked out on a branch
  named `main`) instead of a worktree, so its `.git` lives inside the mount and
  git works. Gated by `Config.needs_isolated_checkout` (docker/kubernetes/auto +
  a normalize command); reviews / bwrap / host runs keep the cheaper worktree.
  Investigation of `utils/checkers.py`: only `add_dates`, `modular_conversion`,
  and `docstrings` touch git. `modular_conversion`'s `git branch --show-current`
  is why the clone is checked out as `main` (forces the correct "check all"
  path); `docstrings` only needs GitPython importable (present in the image);
  **`add_dates` needs the *network* (`raw.githubusercontent.com` + paper API),
  so it cannot run under `--network none` regardless of git → must be excluded
  from the in-container checker list.**
- The image is `linux/amd64`; the deployed cluster runs the proper arch, so the
  arm64-emulation slowness only affected the local dev box.
- The sandbox runs `--network none`, so the image must already be present —
  **no pull at task time.**
- Cost: a correction round can be token-heavy (re-reads + reasoning). Tight repo
  conventions and trimmed feedback help more than mechanism changes.

[#1]: https://github.com/tarekziade/transformers/pull/1
[#2]: https://github.com/tarekziade/transformers/pull/2
[#3]: https://github.com/tarekziade/transformers/pull/3

---

## Step 1 — Backend B: a Kubernetes Job per task

**Goal:** the deployed serge runs the normalizer for each task in a one-shot,
locked-down Kubernetes Job — proper isolation (non-root, no privilege, deny-all
egress), no privileged DinD sidecar. Wiring point is ready:
`normalize.run_normalize` already dispatches to the `kubernetes` backend; only
`_run_kubernetes` (a stub today) needs implementing in `reviewbot/k8s_sandbox.py`.

**Why Backend B over A (DinD):** privileged DinD can escape to the node;
Backend B is the proper-isolation target and we have the time to do it right.
Docker stays first-class for OSS / non-k8s deployments (`docker`/`bwrap`
backends) — **Kubernetes must never be mandatory.**

**Prereq — DONE:** the self-contained checkout (`standalone=True`) means the Job
only needs the **worktree** on a shared volume; the gitdir lives inside it, so
no bare-repo mount and no git-in-sandbox breakage.

Sub-tasks:
- [x] Implement `k8s_sandbox.py`: build the Job manifest → create → poll to
      completion → read pod logs (tail) → delete. Mirrors the docker backend's
      contract: `(returncode, output_tail)`, raises `NormalizeError` on infra
      failure/timeout. **DONE** — pure `build_job_manifest` + `run_job`
      orchestration; `kubernetes` is an optional extra imported lazily (only
      when `TASK_SANDBOX_BACKEND=kubernetes`), so non-k8s deploys never need it.
      `tasks.py`/`normalize.py` pass k8s wiring as plain config values and stay
      decoupled from the module. Unit-tested (manifest + mocked orchestration).
- [x] **Worktree on an RWX volume** — DECIDED: shared RWX PVC. The chart adds a
      `ReadWriteMany` `<release>-worktrees` PVC mounted in serge at
      `WEB_CLONE_CACHE_DIR`; each Job mounts the same claim by **subPath**
      (only its own worktree, at the same absolute path). Requires an RWX
      storage class (`normalize.kubernetes.worktree.storageClass`).
- [x] **Pod security** — DONE in `build_job_manifest`: non-root serge uid/gid
      (+`fsGroup`), `readOnlyRootFilesystem` + `/tmp` emptyDir, drop ALL caps,
      no priv-esc, `RuntimeDefault` seccomp, `automountServiceAccountToken:
      false`, `backoffLimit 0`, `activeDeadlineSeconds`. Egress denied by a
      **deny-all NetworkPolicy** on the `serge.io/sandbox: normalize` label
      (chart `normalize.yaml`).
- [x] **RBAC** — DONE: chart adds a Role (jobs create/get/list/watch/delete;
      pods get/list/watch; pods/log get) + RoleBinding to serge's SA. Guarded:
      `normalize.kubernetes.enabled` requires `serviceAccount.create=true`.
- [ ] **Image:** `TASK_NORMALIZE_IMAGE` (`transformers-quality`) present for the
      node arch; no pull at task time inside the locked-down pod. *(operator:
      set `normalize.kubernetes.image`.)*
- [ ] **Curated checker command:** set `normalize.kubernetes.command`
      (→ `TASK_NORMALIZE_COMMAND`) to a checker list that **excludes
      `add_dates`** (network-bound, dies under deny-all egress). Deployment
      config, not serge code.
- [x] **Deployment env wiring** — DONE: the chart renders
      `TASK_SANDBOX_BACKEND=kubernetes`, `TASK_NORMALIZE_*`, `WEB_CLONE_CACHE_DIR`,
      `TASK_K8S_*`, and optional `REVIEW_RULES_PATH` from the `normalize.kubernetes`
      values block into the ConfigMap.

**Chart usage:** set `serviceAccount.create=true` and the
`normalize.kubernetes` block (`enabled`, `image`, `command`,
`worktree.storageClass`); everything else is wired. A ready-to-use overlay with
the curated transformers command lives at `deploy/helm/env/normalize.example.yaml`.
`helm lint` + `helm template` verified, and `kubectl apply --dry-run=server`
against the live prod cluster **accepted all four new resources** (Role,
RoleBinding, NetworkPolicy, RWX PVC) — nothing created. Resources gated off by
default.

**Acceptance:** trigger a task against the fork through the deployed serge →
normalize runs in a Job → a clean PR; the feedback loop works in-cluster.
*(Blocked on the two deploy prerequisites below — not yet exercised live.)*

---

## Step 2 — Deploy

### Deploy prerequisites (discovered 2026-06-30 against `opensource-aws-use1-prod-54`)

Two hard blockers must clear before a live run; both confirmed against the real
cluster.

**(a) EFS + an RWX StorageClass — DONE & VERIFIED.** Infra provisioned
`serge-tasks-efs-sc` (`efs.csi.aws.com`, `efs-ap`, `Immediate`, `uid/gid 1000`,
`directoryPerms 700`) on `opensource-aws-use1-prod-54`. Verified on-cluster with
a pod mirroring serge's security context (`runAsUser/Group/fsGroup 1000`): PVC
binds RWX, a uid-1000 process writes, and **`git init/add/commit/status`
succeed** (no dubious-ownership error — the worktree is owned `1000:1000`). The
`uid/gid 1000` alignment was essential: the first attempt used `uid/gid 0`,
which made worktree files root-owned and broke git for the non-root pods.

Original infra request (kept for reference):
EBS (the cluster's only storage today: `ebs-gp2`/`ebs-gp3`, all RWO) cannot back
the shared worktree volume — serge and the Job pod mount the *same* worktree at
once, which needs `ReadWriteMany`. The `efs.csi.aws.com` driver is installed but
there is **no EFS filesystem and no `efs-sc` StorageClass**. Infra request (sent)
covers:
- an **EFS filesystem** in the cluster VPC;
- **mount targets** in the worker-node subnets (one per AZ);
- a **security group** allowing inbound **TCP 2049** from the node SG;
- **IRSA** perms for the CSI controller (`CreateAccessPoint`/`DeleteAccessPoint`/
  `DescribeAccessPoints`/`DescribeFileSystems`/`DescribeMountTargets`) for
  dynamic provisioning;
- a **`StorageClass efs-sc`**: `provisioner efs.csi.aws.com`, `provisioningMode:
  efs-ap`, `fileSystemId: fs-…`, `directoryPerms "0775"`, `gidRangeStart/End:
  "1000"` (must align with serge's uid/gid 1000, or the normalizer can't write).

  Verify once provisioned (must reach `Bound`):
  ```bash
  kubectl -n <ns> apply -f - <<'EOF'
  apiVersion: v1
  kind: PersistentVolumeClaim
  metadata: {name: efs-rwx-test}
  spec: {accessModes: [ReadWriteMany], storageClassName: efs-sc, resources: {requests: {storage: 1Gi}}}
  EOF
  kubectl -n <ns> get pvc efs-rwx-test   # STATUS must be Bound
  kubectl -n <ns> delete pvc efs-rwx-test
  ```

**(b) A serge image built from this branch.** The live image
(`ghcr.io/huggingface/serge:sha-2ce3810`) predates this work, so the k8s backend
isn't in it. CI builds `sha-<commit>` on merge to `main`. The root `Dockerfile`
now installs `.[web,kubernetes]` (was `.[web]`) so the client is present — that
change must be in the built image.

### Sub-tasks
- [ ] Merge the branch to `main`; CI builds/pushes `sha-<commit>` (with the
      `[web,kubernetes]` Dockerfile change).
- [ ] Land prerequisite (a) — EFS `efs-sc` StorageClass (REQUESTED from infra).
- [ ] **Roll out to an isolated namespace first** (e.g. `serge-normalize-test`),
      pinned to the branch image, with `prod.yaml` + `normalize.example.yaml` —
      NOT the live `serge` release — and validate before touching prod.
- [ ] Update `deploy/helm/env/<env>.yaml` `image.tag` to the new sha
      (the branch currently carries a stale committed bump — fix on merge).
- [ ] Confirm the transformers provider_config has `task_write_enabled` and the
      normalize env is set for the deployment serving transformers.

**Acceptance:** the deployed serge opens a normalized PR (controlled real run or
the fork) that is **green on repo-consistency at first CI run** — the original
goal of the whole effort.

---

## Step 3 — transformers-ci workflow that uses serge

The caller: a workflow that computes a failure report and POSTs to serge
`/tasks` over GitHub Actions OIDC; serge returns a clean PR (normalization now
happens server-side, so the previously-built downstream normalize workflows are
**obsolete and already deleted**).

Sub-tasks:
- [ ] Confirm the serge endpoint URL + OIDC audience (`TASK_OIDC_AUDIENCE`) the
      workflow posts to.
- [ ] The integration-failure triage already lives in transformers-ci
      (`transformersci.agentic.integration_failure_triage` → calls serge);
      verify it needs **no change** now that normalize is server-side.
- [ ] Workflow declares `permissions: id-token: write`, mints OIDC, POSTs
      `{instruction, context:<failure report>, output: new_pr | existing_pr}`.
- [ ] **Test without touching main:** add/trigger on a branch via
      `workflow_dispatch` (Tier 2) — OIDC + PR-triggered workflows run from a
      branch, no merge required.

**Acceptance:** the nightly triage opens a serge PR that passes repo-consistency
on its first CI run — no red CI, no follow-up commit.

---

## Open items / decisions to revisit

- ~~**Backend A vs B**~~ — DECIDED: Backend B (k8s Job).
- ~~**Image architecture**~~ — DECIDED: cloud runs the proper arch; non-issue.
- ~~**`make fix-repo` git-freeness**~~ — RESOLVED: self-contained checkout
  (`standalone` clone). The curated `TASK_NORMALIZE_COMMAND` (all checkers MINUS
  network-bound `add_dates`) is provided in `deploy/helm/env/normalize.example.yaml`.
- ~~**RWX worktree mechanism**~~ — DECIDED: shared RWX PVC (Job mounts serge's
  worktree by subPath). Needs an EFS `efs-sc` StorageClass — see Step 2
  prerequisite (a), REQUESTED from infra.
- **Gate prompt guidance?** Conventions + root-cause guidance currently apply to
  tasks even when normalize is off; could gate behind "normalize configured" for
  zero behavioral change when off (decided: leave on unless we want strict
  parity).
- **Feedback polish:** strip the Docker amd64-emulation `WARNING` line from the
  normalizer output before it reaches the model; optionally trim feedback to the
  actionable lines to cut correction-round tokens.
- **Cleanup:** close fork test PRs #1/#2/#3 and their `serge/fix-*` branches.
- **Optional:** capture the model's reasoning stream in the transcript dump (only
  `reasoning_chars` is recorded today).
