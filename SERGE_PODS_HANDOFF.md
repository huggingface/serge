# Serge pods — handoff (2026-07-03)

Status of the two features so context can be cleared and resumed from here.

## The two features

**A. Shared git mirror cache** (`feat/shared-mirror-cache`)
Serge keeps warm bare git mirrors on an RWX-EFS PVC (`serge-mirror`); task/review
pods mount the per-repo bare read-only and use it as a fetch **seed** so their
GitHub fetch shrinks to a delta. In-process warmer thread (no k8s CronJob).
Gated by `mirror.enabled` (helm) — off = today's behaviour.

**B. Orchestrator pods** (`feat/orchestrator-pods`)
Serge = pure orchestrator; all LLM work in pods.
- Phase 1: **non-blocking** k8s launch (`create_task_job`/`poll_task_job`/
  `cleanup_task_job`) + in-process **Job watcher** + idempotent `_finalize_task`.
- Phase 2: **`/admin/pods`** page (auto-refresh 5s, kill), admin-gated.
- Phase 3: **UI reviews in pods** (`review_runner.py`, `acquire(standalone=)` with
  `.ai/` overlay, `store.encode_draft`, `RunnerSpec.request_type`, `review_execution`
  flag default `inprocess`).
- Phase 4: **webhook reviews in pods**; serge auto-publishes on the callback.
- Deploy toggle: helm `taskExecution.kubernetes.reviewPods=true` (+ optional
  `reviewImage`) → emits `REVIEW_EXECUTION=kubernetes`.
- Phase 5 (warm repo-pinned pool): **deferred**.

All phases implemented + unit-tested; full suite green except one unrelated
pre-existing `flask` import (test_app.py).

## Branches

- `feat/shared-mirror-cache` — mirror only (pushed).
- `feat/orchestrator-pods` — orchestrator only; **HEAD `d423e81`** includes the
  review-pods UX fixes below (pushed through `7e7da75`; `d423e81` local — PUSH IT).
- `deploy/mirror-plus-reviewpods` — **merge of both** (HEAD `35c960b`), used to
  build the combined prod image. Merge resolved conflicts in `webapp.py`
  (imports / `_launch_task_pod` dispatch / two startup hooks) + `config.yaml`, and
  threaded mirror params through the non-blocking `create_task_job`/`run_task_job`
  (auto-merge left them declared-but-unwired). **NOTE:** this branch's `35c960b`
  does NOT include the `d423e81` UX fixes. The full plan doc
  (`SERGE_ORCHESTRATOR_PODS_PLAN.md`) is tracked on THIS branch (got committed into
  the merge); the mirror plan (`SERGE_SHARED_MIRROR_PLAN.md`) is local-only
  (git-excluded, contains infra names).

## Prod state

- Cluster `opensource-aws-use1-prod-54`, ns `serge`, release `serge`.
- **Live: revision 22, image `sha-35c960b`** (combined branch) with
  `mirror.enabled=true` + `reviewPods=true`. Healthy.
- prod.yaml edits (image tags + mirror block + reviewPods) were made in the
  working tree for the deploy, **not committed**.
- CI gotcha: `docker.yml` builds images only on push to `main`. For a branch:
  `gh workflow run docker.yml --repo huggingface/serge --ref <branch>`
  (workflow_dispatch), then `deploy/scripts/deploy.sh --from-head`. Deploy uses
  `--from-head` which waits for the `serge` image + aborts if the build failed
  (safety net for the Recreate strategy = downtime-on-missing-image).

## Live validation findings

- **Mirror**: mechanism proven end-to-end in prod (warmer registers repo → fetches
  it (44.5s full mirror) → task pod mounts it read-only → seed path engages →
  checkout). BUT on the small test repo `transformers-test-ci`: cold checkout 8.1s
  vs warm/seeded 9.5s — mirror ~1.4s **slower** (tasks already depth-1; seed's
  alternate+`--dissociate` copy is net overhead with no network to save).
  **Conclusion: the mirror only pays off for LARGE repos (transformers itself);
  unproven for the real workload since the exerciser only hits the small test repo.
  Decision pending: validate on a large repo (opens a real PR — awkward), gate per
  repo-size, or drop it for tasks.**
- **reviewPods**: validated live — a UI review at serge.huggingface.tech spawned a
  review pod, checked out (1.1s standalone + .ai overlay), ran `prepare_review`,
  streamed events, serge reconstructed the draft (persisted, status=done).
  Cold-start ≈ **~2 min on a cold node** (node schedule ~44s + task-runner image
  pull ~1 min); checkout itself 1.1s. This is the Phase-5 (warm-pool) motivation.

## review-pods UX fixes (committed d423e81 on feat/orchestrator-pods — NOT yet in prod)

Found via the live review: the draft form didn't appear until manual refresh.
- **Bug fix**: `ingest_task_event` now pushes a `"done"` SSE event on the pod
  terminal callback, so the live page calls `loadDraft()` (it waited on `"done"`;
  the pod path set status+draft but never signalled the stream). Backend was fine.
- **"Launch pod" step**: added to `review.html`/`review.js` `STEP_ORDER` (before
  Clone; auto-completes for the in-process backend).
- **Console feedback**: emit "Runner pod … created; waiting to schedule + pull
  image (~1–2 min on a cold node)" during the pending gap.

## Outstanding / next steps

1. **Get the UX fixes (`d423e81`) into prod**: push `feat/orchestrator-pods`,
   re-merge into `deploy/mirror-plus-reviewpods`, workflow_dispatch build, redeploy.
2. **Mirror decision**: large-repo validation vs gate-by-size vs drop-for-tasks.
3. **Prod is on unmerged combined code (`35c960b`)**: eventually merge both branches
   to `main` (PRs/review) so prod tracks `main`, or roll back to `sha-da3caca`.
4. Phase 5 warm-pool (deferred) — justified by the ~2 min review cold-start.
5. Cosmetic: `/internal/tasks/{id}/events` → `/internal/jobs/{id}/events`.

See also the auto-memory `serge-orchestrator-pods-direction` +
`serge-must-run-without-kube` for the running record.
