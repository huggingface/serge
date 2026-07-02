# Plan: one pod per task (whole LLM loop + normalize in-pod)

Status: **Phases 1–3 built (2026-07-02).** Phase 1 proven locally; Phase 2
(launcher wired into serge behind `TASK_EXECUTION`, docker backend + callback
ingest) and Phase 3 (kubernetes backend + `serge-egress` Helm infra) landed with
tests + `helm lint`. Phase 3 still needs live-cluster verification. Phase 4
(delete dead normalize-Job code) is next. Captured 2026-07-02.
**Supersedes** `SERGE_PERTASK_AGENT_POD_PROPOSAL.md`, whose recommendation ("do
not merge normalize into the agent pod, to preserve the deny-egress isolation")
is reversed by an explicit operator decision: the per-task pod may reach the
network, **but only an allowlist of trusted destinations** (git checkout, the
LLM service) — see "Network firewall" below. The normalizer no longer needs its
own isolated Job.

## Current state (2026-07-02)

**Done — Phase 1 (runner) + a docker launcher, verified locally:**

- `reviewbot/task_runner.py` — the in-pod entrypoint (`reviewbot-task-runner`
  console script). Reconstructs `Config` (`require_app=False`, no App key) +
  `TaskRequest` from a mounted `task.json`, does its own checkout, runs the
  existing `prepare_task`/`publish_task` candidate loop **verbatim**, and streams
  every event + a terminal outcome to serge over an HTTP callback
  (`CallbackEmitter`). Normalize runs in-process (`TASK_SANDBOX_BACKEND=off`).
- `docker/Dockerfile.task-runner` — combined, operator-overridable image
  (`ARG BASE_IMAGE`, serge layered on top; default base = transformers-quality).
- `reviewbot/launcher.py` — **docker backend** (`launch_docker` + `build_spec` +
  `DockerLaunchOptions`). This is the **"works without Kubernetes"** path: serge
  `docker run`s the runner (docker-in-docker via a mounted socket when serge is
  itself containerized). k8s Job backend is still TODO (Phase 3).
- Shared refactor: `reviewbot/errors.py` (`format_llm_error`,
  `format_github_http_error`, moved out of `webapp`); `format_pr_files_diff`
  moved to `tasks.py`. `webapp` imports both — no behavior change.
- New spec field `repo_remote_url` (GH Enterprise / mirror / local-e2e clone
  override), threaded into `acquire_ref`.
- Tests: `tests/test_task_runner.py` — in-process e2e (real local-git checkout +
  canned LLM + fake GitHub + a **real HTTP callback sink**) asserts a published
  PR + terminal `published`, and a `no_fix` path. Full suite: **332 passing**.
- Docker e2e (scratchpad `docker_e2e.py`): a **real container** ran the whole
  pipeline green — spec → in-container `git` checkout (mounted `file://` repo) →
  real LLM call to a host mock → publish → terminal `no_fix` → host callback,
  exit 0. No Kubernetes.

**Done — Phase 2 (launcher wired into serge, docker backend), 2026-07-02:**

- `webapp._launch_task_pod` — the out-of-process launcher. Mints the 1h GitHub
  installation token + a per-job callback token, builds the spec via
  `launcher.build_spec`, `docker run`s the runner (`wait=True`, block-in-pool),
  and reconciles to `error` if the runner dies without a terminal callback.
  Notifies Slack + persists the terminal snapshot like `_run_task_worker`.
- `webapp.submit_task` dispatch — `_run_task_worker` when
  `TASK_EXECUTION=inprocess` (default), else `_launch_task_pod`.
- `POST /internal/tasks/{id}/events` — callback ingest, authed by the per-job
  bearer token (in-memory `Job.callback_token`); events → `_push_event` (SSE
  unchanged), terminal → `job.status`/`task_result`/`error`.
- Config: `TASK_EXECUTION` (`inprocess`|`docker`|`kubernetes`, validated) +
  `TASK_RUNNER_IMAGE`, `TASK_CALLBACK_BASE_URL`, `TASK_RUNNER_TIMEOUT`,
  `TASK_RUNNER_NETWORK`/`_PROXY`/`_MEMORY`.
- Tests: `tests/test_webapp_tasks.py` — dispatch (inprocess vs docker),
  `_launch_task_pod` spec-build + reconcile + k8s-501, and the callback-ingest
  endpoint (auth rejects, event append, terminal published/error). Full
  suite: **343 passing**.

**Done — Phase 3 (k8s backend + Helm), built 2026-07-02 (pending cluster verify):**

- `k8s_sandbox.build_task_job_manifest` / `build_task_secret_manifest` /
  `run_task_job` — one Job pod runs the whole task; a per-job Secret (task.json)
  is mounted at `/etc/serge`, `ownerReferenced` to the Job for auto-GC; the pod
  clones into an ephemeral `emptyDir` (no PVC); egress proxy injected as
  `HTTPS_PROXY` with `NO_PROXY` for the callback. New `serge.io/task-pod` label.
- `launcher.launch_kubernetes` + `K8sLaunchOptions`; `_launch_task_pod`
  dispatches docker vs kubernetes; `K8sSandboxError` surfaces its safe message.
- Helm: `egress-proxy.yaml` (tinyproxy Deployment/Service/ConfigMap + its
  NetworkPolicy), `task-runner.yaml` (task RBAC — jobs/secrets/services — +
  allowlist-egress NetworkPolicy), `config.yaml`/`values.yaml` wiring under
  `taskExecution.kubernetes` (mutually exclusive with `normalize.kubernetes`).
  Validated with `helm template` + `helm lint`.
- Tests: task-Job/Secret builders, `run_task_job` orchestration, kubernetes
  dispatch. Full suite: **354 passing**.

**Not yet done:**

- **Phase 3 — live verify only.** On-cluster: `helm upgrade` with
  `taskExecution.kubernetes.enabled=true`, then confirm git clone + LLM succeed
  and a blocked host fails (`curl https://example.com` must fail) from inside a
  task pod. Also confirm the callback reaches serge and the per-job Secret is
  GC'd. Open items to validate on real infra: the tinyproxy image + config,
  and whether kube-dns egress should be tightened to ClusterIP injection (the
  plan's stricter no-DNS stance — currently a documented residual risk).
- **Phase 4** — delete dead normalize-Job paths + `task_k8s_worktree_*` config.
- A full-success **container** e2e (opening a real PR) needs either real creds or
  a `GITHUB_API_URL` override on `GitHubClient` (it hardcodes `api.github.com`) —
  see follow-ups.

## Why

Today serge runs the whole write-capable task **in-process** (a thread in the
single serge pod) and offloads **only** the normalizer to a one-shot
deny-egress Job on a shared RWX EFS PVC. That normalize Job is created on
**every validation attempt** (`task_normalize_max_retries + 1`, default 3, per
task) — each spawn pulls the large `transformers-quality` image, schedules a
pod, mounts EFS by `subPath`, polls status, collects logs, deletes. The shared
worktree forces the RWX EFS PVC, whose `git clone --local --no-hardlinks`
object-copy is the still-open 180s checkout timeout (status doc bug #5).

Two isolation mechanisms (serge thread + nested Job) and their glue — the
`jobs/status` RBAC subresource, subPath math, the deny-egress NetworkPolicy,
fail-open masking — is not scalable and is where bugs #1/#2/#3 lived.

## Target architecture

```
serge app pod   (thin orchestrator: /tasks, SQLite, SSE, launches + watches task pods)
  └─ per-task runner pod   [LLM egress + 1h GitHub token; runs the agent loop AND normalize in-process]
```

The runner pod runs the current `_run_task_worker` body end-to-end. Normalize
stops being a nested Job and becomes a **plain in-process subprocess**
(`TASK_SANDBOX_BACKEND=off`) — the pod is already the isolation boundary.

### What this removes / collapses

- The nested normalize Job entirely: 3 pod-spawns + image pulls per task → **one
  image pull per task**. Gone: `jobs/status` subresource RBAC, subPath mounting,
  `normalize.py::_run_kubernetes`. (The deny-all-egress NetworkPolicy is *replaced*,
  not removed — see "Network firewall.")
- The **RWX EFS PVC** and the `--local --no-hardlinks` copy: each pod clones the
  repo to its own ephemeral `emptyDir` over the network. Bug #5 disappears by
  construction; `needs_isolated_checkout` / the standalone-copy path is no longer
  needed for the task flow.
- **In-process LLM loops in serge**: a runaway/OOM task is contained in its own
  pod with its own resource limits, not a thread that can take down the single
  serge replica. SQLite stays single-writer in serge (pods report via HTTP, never
  touch the DB), which is what keeps it scalable.

Scope: the **write /tasks flow only**. The read-only webhook review flow
(`_WEBHOOK_REVIEW_POOL`) is orthogonal and unchanged.

## Confirmed decisions

1. **Combined, operator-overridable image.** A Dockerfile with
   `ARG BASE_IMAGE=huggingface/transformers-quality:latest`, `FROM ${BASE_IMAGE}`,
   then `pip install reviewbot` + a `reviewbot-task-runner` entrypoint. Operators
   who bring their own toolchain image just set `BASE_IMAGE` to it; serge is layered
   on top. Extends `docker/Dockerfile.task-runner`.
2. **HTTP callback for streaming + result.** serge mints a per-job bearer token;
   the pod POSTs each `emit` event and the final `TaskResult` to
   `POST /internal/tasks/{id}/events`. serge's Job/SSE model and the web UI are
   unchanged.
3. **serge mints the 1h installation token** and injects it into the pod. The pod
   never holds the long-lived GitHub App private key — only a scoped short-lived
   token (+ the LLM key).

## Network firewall (allowlist egress)

The pod runs arbitrary repo build code (`make fix-repo`) in the **same process
space that holds the LLM key + GitHub token**, so its egress must be locked to
trusted destinations only — otherwise compromised repo code could exfiltrate the
key. Required destinations: **git checkout (GitHub)** and the **LLM service
(`router.huggingface.co`)**. Image pull is *not* in scope — the node's
kubelet/containerd pulls the image before the pod netns exists, so a pod
NetworkPolicy never governs it.

**Cluster constraint:** the cluster runs the **AWS VPC CNI** (`aws-node`) with
NetworkPolicy enforcement on. VPC CNI NetworkPolicy is **IP-based only — no FQDN
egress** (that's a Cilium feature, and Cilium is not installed). GitHub and HF
sit behind rotating CDN IPs, so an `ipBlock` allowlist is unmaintainable.

**Mechanism: forced egress proxy.**
- A small forward proxy (squid/tinyproxy) runs as a **separate** namespace
  Deployment `serge-egress` (+ ClusterIP Service), with a domain allowlist:
  CONNECT permitted only to `.github.com`, `.githubusercontent.com`,
  `router.huggingface.co`; all else denied. The proxy performs DNS resolution.
- The task pod's NetworkPolicy **denies all egress except to the gateway
  ClusterIP on `:3128`** — including no DNS egress, closing the DNS-tunnel exfil
  channel. serge injects the gateway ClusterIP as `HTTPS_PROXY`/`HTTP_PROXY`; git
  gets `http.proxy`. The task pod thus only ever talks to one in-cluster IP.
- A **sidecar** proxy is rejected: containers share the pod netns and
  NetworkPolicy is per-pod, so a sidecar can't be granted egress without granting
  the arbitrary code the same egress. It must be a separate pod.

Arbitrary code in the pod can therefore reach only GitHub + the HF LLM (both
first-party), never an attacker-controlled host. **Residual risk (documented,
accepted):** a valid GitHub token could still tunnel data through requests to
`github.com`; fully closing that needs the old split-pod model (no secrets during
the arbitrary-code phase), which this design deliberately trades away.

## Change list

### Image
- `docker/Dockerfile.task-runner`: add `ARG BASE_IMAGE`, `FROM ${BASE_IMAGE}`,
  install `reviewbot` into the image, set CMD to the runner entrypoint.

### Code — split `_run_task_worker` into launcher (serge) + runner (pod)
- **Launcher** (`webapp.py`): `/tasks` submits `_launch_task_pod(job, cfg, req)`
  instead of `_run_task_worker`. It builds the pod spec, creates a per-job Secret
  holding `task.json` (spec + LLM key + 1h GitHub token + callback token,
  ownerReferenced to the Job for auto-GC), creates the Job, then watches it to
  terminal state. Reuse `k8s_sandbox`'s manifest/name/poll/log/delete helpers,
  repurposed from "normalize Job" to "task Job."
- **Runner** (new `reviewbot/task_runner.py` + `reviewbot-task-runner` console
  script): reads `/etc/serge/task.json`, reconstructs `Config` + `TaskRequest`,
  clones the repo to local scratch, then runs **today's `prepare_task` /
  `publish_task` / candidate loop verbatim**. `emit` is swapped for an HTTP POST to
  the callback; checkout is a local clone. Core task logic is untouched.
- **Callback ingest** (`webapp.py`): `POST /internal/tasks/{id}/events`, authed by
  the per-job token, feeds `_push_event` (SSE unchanged) and records the terminal
  `TaskResult` → `job.status`.
- **`normalize.py`**: drop `_run_kubernetes`; in-pod normalize uses `_run_subprocess`
  with backend `off`.
- **Reconcile**: a sweep for Jobs that finish/die without a terminal callback
  (pod OOM-killed, evicted) → mark the job `error`. `activeDeadlineSeconds` caps
  the pod; `ttlSecondsAfterFinished` reaps it.

### Helm
- Drop the worktrees RWX PVC and its serge mount (`deployment.yaml`).
- `serge-egress` proxy: Deployment + ClusterIP Service, squid/tinyproxy image with
  the domain-allowlist config (ConfigMap). Its own NetworkPolicy: egress to `:443`
  + DNS only.
- Replace the deny-all-egress NetworkPolicy (`normalize.yaml`) with the task-pod
  policy: egress allowed **only** to the `serge-egress` pods on `:3128`. serge
  templates/reads the gateway ClusterIP and injects it as the pod's proxy env.
- Keep/rename the Role granting serge Job/pod/log management; add `secrets:
  create,delete` (per-job Secret) and `services: get` (to read the gateway
  ClusterIP).
- Task pod: `automountServiceAccountToken: false` (creates no sub-Jobs), per-job
  Secret mounted at `/etc/serge/task.json`, proxy env pointing at the gateway.

### Config
- Retire `task_k8s_worktree_pvc` / `task_k8s_worktree_volume_root` and
  `needs_isolated_checkout` (task path). In-pod `TASK_SANDBOX_BACKEND=off`.
- Add launcher settings: task-runner image, secret-name prefix, pod
  resources/timeout; reuse the existing `task_k8s_namespace` / `_service_account`
  / `_node_selector`.

## Phasing

1. ✅ **DONE** — Runner entrypoint + combined image + docker launcher; proven
   locally in-process and in a real container (callback to a local sink).
2. ✅ **DONE** — Launcher wired into serge (`_launch_task_pod`) + temp-file spec
   (docker) + `POST /internal/tasks/{id}/events` callback ingest; feature-flagged
   (`TASK_EXECUTION` = `inprocess` | `docker` | `kubernetes`, in-process the
   default) so rollout is reversible. Per-job Secret (k8s) lands with Phase 3.
3. ✅ **BUILT** (pending cluster verify) — k8s Job backend (`run_task_job`) +
   Helm: `serge-egress` proxy + allowlist config, task RBAC + secret perms, the
   allowlist-egress NetworkPolicy, all under `taskExecution.kubernetes`. Still to
   do on a live cluster: verify git clone + LLM succeed and a blocked host fails
   (`curl https://example.com` must fail) from inside the pod; then remove the
   EFS PVC + old deny-all policy (folds into Phase 4).
4. ⏳ Delete the dead normalize-Job code paths and `task_k8s_worktree_*` config.

## Open questions
- Launcher watch: block a pool thread on the Job (mirrors today, simplest) vs. a
  k8s watch/event loop. Start with block-in-pool.
- Per-task concurrency cap: today `_TASK_POOL` bounds in-process tasks; the pod
  model needs a cap on concurrent Jobs (node capacity / cost) — a semaphore in the
  launcher or a queue.

## Design invariant: the pod checks out its own copy (nothing shared but the spec)

A runner pod has its own mount namespace — it **cannot see anything serge wrote
to serge's disk**, including a worktree serge might prepare. There are only two
ways to give a pod a checkout:

- **(a) share serge's worktree via a shared RWX mount** — the *old* normalize-Job
  approach: serge wrote the worktree to the EFS RWX PVC and the Job mounted it by
  `subPath`. This is the source of the EFS dependency and the slow object-copy.
- **(b) the pod does its own checkout** into its own ephemeral `emptyDir` scratch.

This design takes **(b)**: the runner calls `acquire_ref` *inside the pod*. The
**only** thing serge shares into the pod is the small `task.json` spec (a per-job
Secret in k8s / a bind-mounted temp file in docker) — never the repo. That is
exactly what removes the RWX EFS PVC. Verified in the local docker e2e: the
container mounted only the spec (+ a read-only `file://` origin) and did its own
`git clone` into `/tmp/serge-clones`.

## Follow-ups

- **Simplify the in-pod checkout (optional).** Because the pod checks out its own
  copy (invariant above), `CloneCache`'s shared bare-repo + `git worktree`
  machinery — which exists to share objects across many concurrent worktrees in
  one long-lived serge process — buys the runner nothing: one pod = one task = one
  clone. The runner already avoids linked worktrees (`standalone=True`) but still
  routes through `CloneCache` (bare-repo fetch → local clone → `checkout -B main`).
  Follow-up: let the runner do a **direct `git clone --depth 1 --single-branch`**
  into local scratch (no bare repo, no per-repo lock), keeping `CloneCache` only
  for the in-process review/webhook flow that still benefits from object sharing.
  *(Raised 2026-07-02.)*
- **`GITHUB_API_URL` override on `GitHubClient`.** It hardcodes
  `https://api.github.com` in every method. Honoring a base-URL env would (a)
  support GitHub Enterprise and (b) let a fully-mocked container e2e open a "PR"
  against a stub, closing the one gap in local end-to-end coverage.
