# Plan: in-serge post-LLM normalization ("clean patches directly")

Status: **planning**. This supersedes the standalone command-task feature
currently committed on the `dockerized` branch (see "Current state"). Read
`TASKS_FLOW_PLAN.md` first for the task-flow background.

---

## 1. Goal

Make every Serge fix PR **conformant to the target repo's standards at
creation** — no red repo-consistency CI, no follow-up commit, no waiting.

Mechanism: in the existing LLM task flow, after the LLM patch is applied to the
worktree and before Serge commits, run the repo's own normalizer
(e.g. `make style && make fix-repo`) in a sandbox, re-stage, and commit the
combined diff as one clean patch.

This is an **opt-in, per-repo** capability. When a repo has no normalize
command configured, Serge behaves exactly as today (stays repo-agnostic).

### Explicit reversal to record
`TASKS_FLOW_PLAN.md` §2 says "serge does NOT run the repo's build / no per-repo
command config." This plan deliberately reverses that for the opt-in normalize
hook. Update §2 of that doc when implementing.

---

## 2. How we got here (decision trail)

1. Established `make fix-repo` needs only `.[quality]` (no torch); all `--fix`
   checkers are source/AST based (proven empirically in a clean venv).
2. Built a **standalone command-task** on the `dockerized` branch (a separate
   `/tasks` request type that runs an allowlisted command and PRs the diff).
3. Considered doing normalization **downstream** in the caller's CI
   (`push: serge/fix/**` → `make` → commit back). Built it as a
   caller+reusable pair. Decided this keeps Serge agnostic but yields briefly
   red PRs + an extra commit + GITHUB_TOKEN/branch-protection interplay.
4. Final direction: do it **inside Serge** as a post-LLM normalize hook so
   patches are clean at the source. Serge is deployed in **Kubernetes**, and we
   will add container-execution capability there.

The standalone command-task framing (#2) is the wrong product, but its
**infrastructure is reused** (docker sandbox backend, `stage_all`,
run-command-in-worktree, image/backend config).

---

## 3. Current state (what exists right now)

### `dockerized` branch in `/Users/tarek/Dev/serge` (commit `48d705a`, NOT pushed/merged)
Reusable infra (KEEP, possibly rename):
- `reviewbot/sandbox.py`: `build_docker_argv`, `wrap_task_command`,
  `docker_available`, `normalize_backend`, `DockerUnavailable`, backend
  constants (`BWRAP_BACKEND`/`DOCKER_BACKEND`/`AUTO_BACKEND`).
- `reviewbot/clone_cache.py`: `stage_all` (`git add -A` before
  `collect_changes`).
- `reviewbot/config.py`: `task_command_image`, `task_command_timeout`,
  `task_command_memory`, `task_sandbox_backend` (KEEP; rename the
  `task_command_*` ones to `task_normalize_*`).
- `reviewbot/tasks.py`: `_commit_changes` extracted from `publish_task` (KEEP —
  shared commit/PR tail).

Standalone command-task (REMOVE / repurpose):
- `reviewbot/tasks.py`: `command` field on `TaskRequest`, `_parse_command`,
  allowlist parsing in `build_task_request`, `run_command_task`,
  `publish_command_result`, `_decorate_command_body`.
- `reviewbot/config.py`: `task_command_enabled`, `task_command_allowlist`.
- `reviewbot/webapp.py`: command-task branch in `_run_task_worker`, allowlist
  gating in `submit_task`, `require_llm_key` param on
  `_resolve_task_worker_cfg`.
- `tests/test_tasks.py`: `CommandTaskParsingTests`, `CommandTaskRunTests`.
- `tests/test_sandbox.py`: docker tests (KEEP — still test the backend).
- `docker/Dockerfile.task-runner`, `docs/tasks-command.md` (rewrite as
  normalize docs).

### Untracked, downstream approach (DROP — superseded by in-serge)
- `/Users/tarek/Dev/transformers/.github/workflows/serge-fix-normalize-caller.yml`
- `/Users/tarek/Dev/transformers-ci/.github/workflows/serge-fix-normalize.yml`

Delete both — in-serge normalization makes them unnecessary.

### Already-correct, leave alone
- transformers `nightly-integration-failure-triage-caller.yml` (thin caller)
  → transformers-ci `integration-failure-triage.yml` (reusable) →
  `transformersci.agentic.integration_failure_triage:main` (console script).
  The triage flow already lives entirely in transformers-ci; no change.

---

## 4. Target design

### Task flow — normalize validation is **in the LLM loop** (decided)
The first cut ran normalize as a post-LLM, best-effort step in `publish_task`.
**Reversed:** if the normalizer rejects a patch, the LLM must *see the failure
and correct the patch* — so validation lives inside the agentic loop, not after
it.
```
prepare_task():                            # the agentic LLM loop
  _run_agentic_loop(..., validate=_validate, max_validation_retries=N):
    LLM emits a candidate patch
    _validate_patch():                     # verification gate
      reset_worktree(); apply_patch(); run_normalize(...)
        - patch won't apply     -> return git-apply error as feedback
        - normalizer exits != 0 -> return its output as feedback  (LLM revises)
        - passes                -> accept; worktree = applied + normalized
  # feedback re-enters the SAME conversation; up to N corrective re-prompts
publish_task():
  if plan.worktree_prepared: stage_all + collect_changes + _commit_changes
  else: apply_patch (fallback) + stage + commit
```
- `_run_agentic_loop` gained an opt-in `validate` callback (reviews unaffected).
- On success the worktree already holds applied+normalized files → one commit,
  no re-apply, no double-normalize.
- Applies to both `new_pr` and `existing_pr`.
- Config: `task_normalize_max_retries` (default 2 → 3 patch attempts).

### Execution seam
Add `reviewbot/normalize.py` (or extend `tasks.py`) with:
```
run_normalize(command: list[str], *, workdir, write_root, image, backend,
              timeout, memory) -> (returncode, output_tail)
```
One implementation per backend, selected by `task_sandbox_backend`:
- `bwrap` / `off`: reuse `sandbox.wrap_task_command` + `subprocess.run`
  (dev/test only — uses Serge's venv, no repo deps).
- `docker`: reuse `sandbox.build_docker_argv` (Backend A / DinD).
- `kubernetes`: NEW (Backend B) — see §5.

### Config (`reviewbot/config.py`)
Rename `task_command_*` → `task_normalize_*`:
- `task_normalize_command: Optional[list[str]]` — global default normalize
  argv (e.g. `["bash","-lc","make style && make fix-repo"]`). Optional.
- `task_normalize_image: Optional[str]` — image with the repo's toolchain.
- `task_normalize_timeout: int` (default 1800).
- `task_normalize_memory: Optional[str]`.
- `task_sandbox_backend: str` — `bwrap | docker | kubernetes | auto`.

Per-repo vs global (OPEN DECISION, §6.2): command+image likely differ per
repo, so prefer storing them on `provider_configs` (new columns
`normalize_command`, `normalize_image`) and falling back to the env defaults.
For a transformers-only / single-tenant deploy, env-only is enough to start.

### Removing the injection surface
The normalize command is **operator/repo config**, never request-supplied.
Delete the `command` field + allowlist from the `/tasks` request entirely.
This removes the command-injection concern that the allowlist existed to
contain.

---

## 5. OPEN DECISION — k8s execution backend (pick A or B before coding the backend)

Serge runs in Kubernetes. "Add Docker" forks here. The **application-layer
refactor (§4) is identical for both**; only the execution backend differs.

### Backend A — DinD sidecar
- Add a `docker:dind` sidecar to the Serge pod; Serge runs `docker run` (reuse
  `build_docker_argv`) against the pod-local daemon.
- Worktree (clone cache) must be on a volume shared between the serge and DinD
  containers so `-v <path>:<path>` resolves.
- Cost: sidecar needs `privileged: true` (privileged container can escape to
  the node). Weakest isolation; closest to committed code.

### Backend B — Kubernetes Job per task (RECOMMENDED)
- Serge (ServiceAccount scoped to a locked-down namespace) creates a one-shot
  Pod/Job from `task_normalize_image`, mounts the worktree from a shared **RWX**
  PVC (EFS/NFS on EKS), runs the command with `network: none`, non-root,
  no-privileged, dropped caps, read-only rootfs except workdir + /tmp.
- Untrusted `make` code runs in its OWN pod with its OWN security context;
  isolation from the kubelet, not from a privileged process Serge controls.
- Cost: new backend (`reviewbot/k8s_sandbox.py` using the `kubernetes` client),
  RBAC, and an RWX PVC for the clone cache. More work, properly isolated.
- New module sketch: create Job → poll to completion (timeout) → read pod logs
  → delete Job. Worktree path identical inside the pod (same PVC mount path).

Do NOT mount the host `docker.sock` (node-level root).

Recommendation: **Backend B.**

---

## 6. Other open decisions

1. **Normalize-failure policy.** ~~best-effort vs hard-fail~~ **DECIDED: feed
   the failure back into the LLM loop** (`task_normalize_max_retries`, default
   2). A non-zero normalizer exit becomes a corrective re-prompt so the model
   fixes the patch. Only *after* the retry budget is exhausted — or on an
   infrastructure failure (sandbox unavailable/timeout), which isn't the
   model's fault — do we fall back to best-effort (commit the raw applied
   patch; CI catches the rest). We never lose the LLM fix.
2. **Config location.** Per-repo (`provider_configs` columns) vs global env.
   Recommend per-repo for multi-tenant; env-only acceptable for single-tenant
   transformers.
3. **Keep the repo-agnostic example image + docs?** Rewrite
   `docs/tasks-command.md` → `docs/tasks-normalize.md` describing the hook,
   per-repo config, and backend setup, repo-agnostically (transformers as one
   example). Keep an example Dockerfile or point at
   `huggingface/transformers-quality`.
4. **`existing_pr` follow-ups** already flow through the same `publish_task`, so
   they get normalized automatically — confirm no special-casing needed.

---

## 7. Implementation steps (in order)

**Phase 0 — shared app-layer refactor (backend-agnostic, do first):**
1. Remove standalone command-task: `command` on `TaskRequest`,
   `_parse_command`, allowlist in `build_task_request`, `run_command_task`,
   `publish_command_result`, `_decorate_command_body`, the webapp routing
   branch + allowlist gating + `require_llm_key`. Drop
   `task_command_enabled/_allowlist`. Drop the two command-task test classes.
2. Rename `task_command_image/_timeout/_memory` → `task_normalize_*`; add
   `task_normalize_command`; add `kubernetes` to `task_sandbox_backend`.
3. Add `run_normalize(...)` seam with the `bwrap`/`off` + `docker`
   implementations (reuse existing sandbox helpers).
4. Insert the hook into `publish_task` between `apply_patch` and the
   `stage_all`/`collect_changes`/`_commit_changes` tail. Apply best-effort
   policy (§6.1). Skip when unconfigured.
5. Rewrite `docs/tasks-command.md` → `docs/tasks-normalize.md`.
6. Tests: a normalize-hook test (reuse the worktree-fixture pattern from the
   old `CommandTaskRunTests` with sandbox `off` and a fake `echo`/`make`
   command): verifies combined diff, best-effort on failure, skip-when-
   unconfigured. Keep `tests/test_sandbox.py` docker tests.
7. Delete the two downstream workflow files (§3).
8. `make test` (`.venv/bin/python -m pytest tests/`) + `ruff check`/`format`.

**Phase 1 — backend (after A/B decided):**
- If B: `reviewbot/k8s_sandbox.py` (kubernetes client: create Job, wait, logs,
  cleanup) + unit tests with a mocked client.

**Phase 2 — deployment (k8s, Backend B):**
- RWX PVC (EFS) for the clone cache, mounted in serge + task pods.
- ServiceAccount + Role/RoleBinding (create/get/delete jobs, get pods/log) in a
  dedicated task namespace.
- Task Pod template: image, securityContext (non-root, no-privileged, drop
  ALL), NetworkPolicy deny-all egress, resource limits.
- Helm chart edits under `deploy/helm`.

**Phase 3 — docs/record:**
- Update `TASKS_FLOW_PLAN.md` §2 to record the opt-in normalize reversal.

---

## 8. Test / validation

- Unit: normalize hook (combined diff, best-effort failure, unconfigured skip);
  backend selection; k8s backend with mocked client.
- Local E2E: sandbox `off` + a real `make`-like command against a temp git repo
  (pattern already in the old `CommandTaskRunTests`).
- Real E2E (after deploy): trigger a transformers nightly task; confirm the
  opened PR contains LLM fix + normalization in one commit and passes
  repo-consistency on first CI run.

---

## 9. Quick repo map
- serge app: `/Users/tarek/Dev/serge/reviewbot/{tasks,sandbox,clone_cache,config,webapp}.py`
- serge tests: `/Users/tarek/Dev/serge/tests/`
- serge deploy: `/Users/tarek/Dev/serge/deploy/helm`
- transformers-ci triage (reference pattern): `/Users/tarek/Dev/transformers-ci`
- `make fix-repo` analysis (why `.[quality]`, no torch): transformers
  `utils/checkers.py` + `docker/quality.dockerfile` / `docker/consistency.dockerfile`.
```
