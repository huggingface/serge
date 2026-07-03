# Serge as a Pure Orchestrator — All LLM Work in Pods

**Branch:** `feat/orchestrator-pods`
**Status:** Proposed
**Author:** (with Claude Code assistance)
**Related:** `feat/shared-mirror-cache` (SERGE_SHARED_MIRROR_PLAN.md) — synergistic; pod checkout gets fast once mirrors seed it.

## Goal

Move **all** LLM work out of the serge process. Serge becomes a pure orchestrator:
it authenticates, mints tokens, launches pods, relays their event streams to the UI,
and publishes results — but never runs an LLM loop or a repo checkout itself. Each
review and task runs in its own ephemeral pod. This removes the two hard scaling
ceilings (per-job memory of ~200–500 MB, and sync LLM threads starving the asyncio
loop via the GIL) and decouples concurrency from the serge host.

## Where we are today

- **Tasks** already pod-able: `launcher.launch_kubernetes` → `k8s_sandbox.run_task_job`
  builds a `batch/v1` Job; `task_runner.py` is the in-pod entrypoint; the pod streams
  events to `POST /internal/tasks/{id}/events`; per-job Secret carries spec + token +
  llm key + callback; ownerRef GC; egress NetworkPolicy on `serge.io/task-pod`.
- **UI reviews** (`webapp.py:2418`): **unbounded** `threading.Thread` per job →
  `_run_review_worker` (`webapp.py:1006`) → synchronous `prepare_review()`
  (`reviewer.py:1135`) **in-process**. No admission control.
- **Webhook reviews** (`webapp.py:1477`): `ThreadPoolExecutor(max_workers=2)`
  (`webapp.py:366`) → `_run_webhook_review_worker` (`webapp.py:486`), `auto_publish=True`
  + follow-ups, same in-process `prepare_review()`.

Reusable substrate already present:
- `launcher.build_spec()` (`launcher.py:79`) is **generic** — carries a `request` dict,
  token, llm config, callback. Not task-specific.
- Callback endpoint + per-job bearer auth + SSE relay (`_push_event` → `job.queue`).
- K8s RBAC grants `jobs` + `pods` **list/watch/delete** (`task-runner.yaml:19-41`).
- Per-job Secret + ownerRef GC + egress NetworkPolicy.

## The load-bearing change: non-blocking launch

Today `_launch_task_pod` (`webapp.py:1237`) **blocks a thread-pool thread** polling Job
status until the pod exits. If we pod reviews but keep this, we just relocate the
2-worker bottleneck — serge thread count still caps concurrent pods.

Launch must become **fire-and-forget**:
1. Handler mints token + builds spec + creates Secret+Job (a few fast API calls),
   returns immediately.
2. A **single background Job watcher** (k8s `watch` on
   `label_selector=app.kubernetes.io/managed-by=serge`) reconciles terminal state and
   catches crashed pods that never send a terminal callback.
3. Progress/results flow over the existing callback → SSE path. No serge thread is
   parked per running pod.

Once launch is non-blocking, concurrent pods are bounded by the cluster, not by
`TASK_MAX_WORKERS` / `WEBHOOK_MAX_WORKERS`.

## Review-specific design points

- **Human-in-the-loop stays clean.** The pod runs `prepare_review()` and streams the
  *draft* back; **publish stays in serge** (a GitHub API POST — not LLM work — so serge
  stays LLM-free). For webhook reviews (`auto_publish=True`), publish-from-serge on the
  terminal callback is cleaner than publishing inside the pod.
- **Follow-ups are LLM work** → each `run_followup` is its own webhook event → its own
  pod. No special-casing.
- **Untrusted checkout.** Reviews clone a fork PR head with the `.ai/` overlay via
  `acquire()` (not `acquire_ref`), currently a *linked* worktree. In a pod it must be
  `standalone=True` (self-contained gitdir), same reason as tasks. Shared-mirror seed
  (`feat/shared-mirror-cache`) shrinks this fetch to the PR delta.
- **Lighter image.** Review pods don't run `make fix-repo`, so they don't need the heavy
  `reviewbot-task-runner` toolchain image — a slim serge-base image starts faster.
- **Naming debt.** Generalize task-specific surface to job-neutral names:
  `/internal/tasks/{id}/events` → `/internal/jobs/{id}/events`, `serge.io/task-pod` →
  `serge.io/job-pod` (+ a `serge.io/job-type: review|task` label), `TASK_RUNNER_IMAGE`
  stays for tasks, add `REVIEW_RUNNER_IMAGE`.

## Cold-start caveat (deferred, see Phase 5)

Pod cold-start adds a few seconds a warm in-process thread doesn't pay — noticeable for
small/fast reviews. **Deferred mitigation: keep repo-specific warm pods around** — a
small pool of pre-started, repo-pinned pods (checkout already warm from the shared
mirror) that pick up the next review/task for that repo, avoiding cold start entirely.
Not needed for correctness; a latency optimization once the pod path is proven.

---

## Work breakdown

### Phase 1 — Non-blocking launch + Job watcher (task path first) ✅ DONE

- [x] Split the k8s launch: `k8s_sandbox.create_task_job()` (creates Job+Secret,
      returns immediately) + `poll_task_job()` / `collect_task_result()` /
      `cleanup_task_job()`. `run_task_job` refactored to compose them (blocking path
      kept for compat/tests). `launcher.create_kubernetes()` wraps create.
- [x] `webapp._launch_task_pod`: kubernetes path is now **non-blocking** — creates
      the Job, registers a `_PodTask`, returns (no thread parked). Docker stays
      blocking (local/dev; no watcher) so serge still runs **without kube**.
- [x] Single background **Job watcher** (`_start_pod_job_watcher`, asyncio startup
      hook, polls every 5s): for tracked k8s tasks, reconciles a Job that went
      terminal without a callback (after a 12s grace for an in-flight callback) →
      marks error, and reaps finished Jobs (TTL is the backstop). No-op for
      docker/inprocess (never touches the k8s client).
- [x] Idempotent `_finalize_task` (notify + persist, once) shared by the callback
      (happy path), the docker thread, and the watcher (crash path). Callback
      (`ingest_task_event`) now finalizes on the terminal event.
- [x] `_TASK_POOL` now only briefly holds the k8s create call (frees immediately);
      concurrency is bounded by the cluster, not `TASK_MAX_WORKERS`.
- [x] Tests: `test_k8s_sandbox` (create returns without polling/deleting; poll
      running→terminal; 404→failed; cleanup) + `test_webapp_tasks` (k8s launch is
      non-blocking + token live + tracked; create error finalizes; watcher
      reconciles a crashed pod; watcher reaps a finalized pod). Full suite green
      (352 tests; only the unrelated legacy-flask import fails in this env).

### Phase 2 — Admin "running pods" page (read-only ops value) ✅ DONE

- [x] `GET /admin/pods` (HTML, `static/admin_pods.html` + `admin_pods.js`) +
      `GET /admin/pods/data` (JSON), **admin-gated** (`_is_admin`, not just any user,
      since it exposes cross-user activity + kill).
- [x] `k8s_sandbox.list_task_pods()` lists live pods by
      `app.kubernetes.io/managed-by=serge,component=task`; the endpoint **joins** them
      to serge's tracked `_pod_tasks` on the auto-applied `job-name` label (no extra
      manifest labels needed) → repo, kind, user, serge status. Orphan pods (post
      serge-restart) still show with k8s data; just-launched tasks with no pod yet
      also show. Columns: started, kind, repo, user, job status, pod phase, node, pod.
- [x] **Kill** action → `POST /admin/pods/kill` → `cleanup_task_job`
      (admin + same-origin; RBAC already allows Job delete).
- [x] Graceful degradation: docker/inprocess list serge's tracked tasks; k8s errors
      surface as a banner instead of a 500.
- [x] **Auto-refresh** every 5s (client-side `setInterval`, matching the journal).
- [x] "Pods" nav link added to admin + journal headers.
- [x] Tests: admin-gating (403 for non-admin), k8s listing + join + orphan +
      just-launched, docker backend, kill. 28 `test_webapp_tasks` pass; page route
      renders (200, table). Full suite green (357; only legacy-flask import fails).

### Phase 3 — Review runner + pod path (UI reviews)

**Slice 3a — foundation (additive, zero behaviour change) ✅ DONE**
- [x] `review_execution` flag (inprocess|docker|kubernetes, default **inprocess** →
      reviews run without kube; podding opt-in + reversible) + `review_runner_image`
      config (falls back to task_runner_image). Validated + rejects bad values.
- [x] `store.encode_draft()` public (pairs with existing `decode_draft`) so a review
      pod can serialize its draft and serge reconstruct it.
- [x] `CloneCache.acquire(standalone=…)`: self-contained PR-head clone for a pod,
      with the **fork-`.ai/`-overlay-from-default-branch** security property
      preserved (`_overlay_base_ai_standalone`). Tests: self-containment (git works
      after cache removed) + overlay-uses-default-branch + fail-closed drop.

**Slice 3b — runner + launch ✅ DONE**
- [x] New `reviewbot/review_runner.py`: `acquire(standalone=True)` → `prepare_review()`
      → encode the draft (`store.encode_draft`) + token counts into the terminal
      callback. **No publish in the pod.** Reuses `task_runner`'s `CallbackEmitter`
      + `build_runner_config`. Tests: draft reported+round-trips, no-diff, crash.
- [x] `RunnerSpec` + `build_spec()` carry a `request_type` (review|task); the
      `task_runner.main` entrypoint dispatches to `review_runner.run` on it.
- [x] `review_runner_image` config (falls back to `task_runner_image`).

**Slice 3c — flip UI reviews (the switch) ✅ DONE**
- [x] `_run_review_worker` routes to `_launch_review_pod` when
      `review_execution != inprocess`; SSE queue + `_push_event` relay unchanged.
      `_launch_review_pod` mirrors `_launch_task_pod` (non-blocking k8s / blocking
      docker) and reuses the `_PodTask` registry + `_finalize_task` + Job watcher.
- [x] Callback ingest accepts the `review` kind and reconstructs `job.draft` via
      `decode_draft` + applies token counts; publish stays in serge (human approval).
- [x] `_finalize_task` skips the slack task-notifier for reviews (ReviewRequest has
      no notification fields).
- [x] Tests: review pod non-blocking launch, finalize-no-slack, ingest draft
      reconstruction. Full suite green (367; only legacy-flask import fails).
- Note: review pods reuse the task pod's Job manifest → same egress NetworkPolicy
  + RBAC, no new k8s manifests. **Deploy: enable with**
  `taskExecution.kubernetes.reviewPods=true` (+ optional `reviewImage`) — the chart
  then emits `REVIEW_EXECUTION=kubernetes` (+ `REVIEW_RUNNER_IMAGE`), reusing the
  task backend's egress proxy + RBAC + callback. helm-validated (off → 0 refs).
  Kept the callback path at `/internal/tasks/{id}/events` (rename to `/jobs/`
  deferred — cosmetic).

### Phase 4 — Webhook reviews to pods ✅ DONE

- [x] `_run_webhook_review_worker` routes full reviews through `_launch_review_pod`
      when `review_execution != inprocess` (else unchanged in-process path).
- [x] Pod streams the draft; **serge auto-publishes on the terminal callback**
      (`_webhook_publish_or_report`, off the event loop via `to_thread`) because
      `job.source == "webhook"` — mints a token, `publish_review`, sets
      `published_draft`/status. On error it posts the failure comment. Publish +
      error-comment stay in serge (GitHub writes, not LLM); the pod is pure LLM.
- [x] Webhook signature/verification path untouched.
- [x] Tests: webhook review auto-publishes on callback; webhook error posts the
      failure comment (and does not publish). Full suite green (369; only the
      unrelated legacy-flask import fails).
- Deferred: inline **follow-ups** stay in-process (small, distinct LLM+write
  path); `_WEBHOOK_REVIEW_POOL` kept (now frees fast for podded reviews — it still
  serves follow-ups + the inprocess fallback); a webhook review whose *launch*
  (not run) fails doesn't post a failure comment (rare config error; shows in UI).

### Phase 5 — (Deferred) Warm repo-pinned pod pool

- [ ] Small pool of pre-started pods pinned per active repo, checkout pre-warmed from
      the shared mirror, picking up the next review/task for that repo to skip cold
      start. Latency optimization only; design after Phases 1–4 are proven.

### Phase 6 — Verification

- [ ] Zero reviews run in serge threads (grep for `_run_review_worker` thread spawn gone;
      confirm no `prepare_review`/`prepare_task` call sites remain in `webapp.py`).
- [ ] Load test: N concurrent UI + webhook reviews → N pods, serge memory flat, event
      loop latency stable.
- [ ] Admin page shows the live pods with correct repo/PR/type/status; kill works.
- [ ] Crash path: killed pod reconciles to error via the watcher without parking a thread.

---

## Risks / decisions

- **Non-blocking launch is mandatory**, not optional — without it, podding reviews only
  relocates the thread-pool bottleneck.
- **SSE continuity across pod lifetime** — the UI stream must survive from launch through
  the pod's terminal callback; the existing `job.queue` + `_push_event` path covers this,
  but confirm the queue outlives a slow pod start.
- **Cold-start latency** for small reviews — accepted for now; Phase 5 mitigates.
- **Two images** (task toolchain vs slim review) add a build target; worth it for
  startup speed, but a single shared image is a valid v1 shortcut.
- **Naming migration** (`tasks` → `jobs` on the callback path + labels) needs an alias
  window so in-flight task pods keep reporting during rollout.
- **Publish-from-serge** keeps serge touching GitHub write APIs — that's fine (not LLM
  work) and preserves the human-approval gate for UI reviews.
