---
title: Architecture
nav_title: Architecture
---

This page explains how `serge` is put together — its components, the execution
backends that run the LLM work, and the two deployment flavors (**Docker** and
**Kubernetes**). For the step-by-step review lifecycle see
[How it works](how-it-works.md); for the trust model see
[Security](security.md) and [Security architecture](security-architecture.md).

## The big picture

serge is an **orchestrator**. Its long-lived process owns the durable state and
the credentials — the SQLite job store, the GitHub App private key, the OAuth /
session secrets, and LLM keys — and decides *when* a review or task runs and
*what* gets published. The actual LLM loop (fetch diff → prompt → tool calls →
draft) can either run **in that same process** or be pushed out into a
short-lived **runner pod**, depending on how you deploy. Either way, the model
only ever *proposes* output; serge validates it and publishes.

```text
                         ┌───────────────────────────────────────────┐
   GitHub  ── comment ──▶│  serge (orchestrator: reviewbot-web/-app)   │
   / webhook / OIDC      │  • trigger gate    • SQLite job store       │
                         │  • GitHub App auth • provider configs       │
                         │  • draft/publish   • per-job secrets        │
                         └───────────────┬─────────────────────────────┘
                                         │ run the LLM loop …
                        ┌────────────────┼─────────────────────┐
                        ▼                ▼                     ▼
                 in-process        docker runner         kubernetes pod
                 (same process)    (docker run)          (one Job / request)
```

## Entry points

serge ships three entry points around one shared core (`reviewbot/reviewer.py`):

| Command | Role | Where it runs |
| ------- | ---- | ------------- |
| `reviewbot-action` | [GitHub Action](github-action.md) runner; reads `GITHUB_EVENT_PATH` | GitHub's ephemeral runner |
| `reviewbot-app` | Flask [GitHub App webhook](github-app.md) server | Your server |
| `reviewbot-web` | FastAPI [staged web app](web-app.md) + `POST /webhook` + write-capable [`POST /tasks`](tasks-flow.md) | Your server |

The **Action** is self-contained: it runs on GitHub's disposable runner with a
job-scoped token, so the runner itself is the isolation boundary. The
**webhook** and **web app** are the persistent, hosted deployments — everything
below about execution backends and deployment flavors is about those.

## Core components

| Component | File | Responsibility |
| --------- | ---- | -------------- |
| Reviewer core | `reviewbot/reviewer.py` | Annotate diff, prompt the LLM, validate inline comments against real diff positions, publish or draft |
| Trigger gate | `reviewbot/triggers.py` | Decide whether an event should start a review (event type, author association, trigger phrase, PR state) |
| Tools + sandbox | `reviewbot/tools.py`, `sandbox.py` | Read-only repo browse tools and repo helper tools, each executing subprocess wrapped in bubblewrap |
| Clone cache | `reviewbot/clone_cache.py` | Shallow checkout of the PR head with an upstream `.ai/` overlay; standalone (self-contained) clones for pods |
| Store | `reviewbot/store.py` | Embedded SQLite: jobs, drafts, provider configs, cached installation tokens |
| GitHub auth | `reviewbot/github_auth.py` | Mint short-lived installation tokens from the App private key |
| Launchers | `reviewbot/launcher.py`, `k8s_sandbox.py` | Build the runner spec and start a docker container or a Kubernetes Job |
| Runners | `reviewbot/task_runner.py`, `review_runner.py` | The in-pod entry points that do the whole loop and stream results back |

## Execution backends

serge has **two independent backend selectors** — one for write-capable
[tasks](tasks-flow.md) and one for PR reviews. Both default to `inprocess`, so a
fresh install runs the entire loop inside serge's own thread pool with **no
Docker and no Kubernetes**.

| Setting | Env var | Values | Default |
| ------- | ------- | ------ | ------- |
| Task execution | `TASK_EXECUTION` | `inprocess` \| `docker` \| `kubernetes` | `inprocess` |
| Review execution | `REVIEW_EXECUTION` | `inprocess` \| `docker` \| `kubernetes` | `inprocess` |

- **`inprocess`** — the loop runs in a worker thread inside serge. Simplest;
  this is the legacy path and the default.
- **`docker`** — serge does `docker run` on a runner image per request, blocks
  on the worker thread, and finalizes inline.
- **`kubernetes`** — serge creates one short-lived Kubernetes **Job per
  request**, watches it out-of-band, and finalizes when the runner calls back.

The `docker` and `kubernetes` backends share the **same runner image, spec
format, and HTTP callback** — they differ only in how the runner process is
started. Inline PR-comment follow-ups always stay in-process regardless of the
setting.

> Kubernetes is never mandatory. The `kubernetes` client is imported lazily, the
> pod watcher no-ops when there are no pod tasks, and every k8s-only feature is
> gated — so a Docker or in-process deployment never needs the cluster libraries.

### Sandboxing vs. execution backend

Two other knobs control how *subprocesses within a review/task* are sandboxed —
they are separate from the execution backend above:

- **`HELPER_SANDBOX`** (`require` \| `auto` \| `off`) wraps the read-only helper
  tools, the `.ai/context-script`, and the pip-install hook in **bubblewrap**.
  Production sets `require`. See [Security architecture](security-architecture.md).
- **`TASK_SANDBOX_BACKEND`** (`bwrap` \| `docker` \| `kubernetes` \| `auto`)
  wraps the in-loop [normalize](tasks-normalize.md) command when tasks run
  in-process.

Inside a **runner pod** both are forced `off`: the ephemeral pod plus its
allowlist-egress firewall *is* the isolation boundary, so there is no nested
sandbox.

## Pods (the Kubernetes backend)

When `TASK_EXECUTION=kubernetes` (and, for reviews, `reviewPods=true` →
`REVIEW_EXECUTION=kubernetes`), serge becomes a pure orchestrator and all LLM
work happens in per-request pods.

### Lifecycle of a pod

```text
  request ─▶ serge mints installation token + callback token
             │
             ├─ build spec (request + GitHub token + LLM settings + callback)
             ├─ create Job (batch/v1, backoffLimit 0, ttlSecondsAfterFinished 300)
             ├─ create per-job Secret (task.json, ownerRef→Job) → returns immediately
             │
             ▼
   runner pod: read /etc/serge/task.json
             ├─ clone repo (standalone, self-contained) via the egress proxy
             ├─ run the LLM loop (tools, in-process normalize for tasks)
             ├─ POST each event  ──────────────▶  serge callback (SSE relay)
             └─ POST terminal result ──────────▶  serge: finalize + publish/draft
             │
   serge Job watcher (every 5s, k8s only): reconciles crashes, reaps the Job
```

1. **Launch (non-blocking).** serge mints a short-lived GitHub installation
   token and a per-job callback bearer token, serializes them plus the request
   and resolved LLM settings into a **spec**, creates the Job, then the Secret,
   and returns without parking a thread.
2. **Per-job Secret.** The spec is projected read-only into the pod at
   `/etc/serge/task.json`. The Secret carries an `ownerReference` back to the
   Job, so it is garbage-collected with the Job even on a crash.
3. **The runner pod** (`reviewbot-task-runner`) dispatches on `request_type`:
   `review` → `review_runner.run`, otherwise `task_runner.run`. It rebuilds a
   `Config` that has **no App private key and no web-auth env**, clones into its
   own `emptyDir` (no shared PVC), runs the loop, and POSTs events and a terminal
   payload back over the authenticated callback. A task runner *publishes* its PR
   itself; a review runner only ships an encoded **draft** — serge publishes on
   human approval (or auto-publishes for webhook reviews).
4. **Watcher + idempotent finalize.** An in-process asyncio watcher polls the
   Job status every 5s (only when there are Kubernetes pod tasks). Finalization
   is guarded so it happens **exactly once** whether it's triggered by the happy-
   path callback or by the watcher detecting a crashed/timed-out pod.
5. **Reaping.** serge deletes the Job (and its Secret) on completion;
   `ttlSecondsAfterFinished` is the backstop if a delete is missed.

### The Pods page

The web app exposes a **Pods** admin page (`/admin/pods`, admin-gated) that lists
running pods — joining live Kubernetes pods with serge's tracked tasks, so both
orphaned pods (after a serge restart) and just-launched ones appear — and offers
a **kill** action (`POST /admin/pods/kill`) as the manual stop for a runaway Job.

### Network isolation

A runner pod holds a live GitHub token and LLM key **while running arbitrary
repository build code** (e.g. `make style`). Its egress is therefore locked down
by two Kubernetes `NetworkPolicy` objects and a forward proxy:

- The **task-pod NetworkPolicy** denies all egress except to the `serge-egress`
  proxy, kube-dns, and serge's own callback Service.
- **`serge-egress`** is a small tinyproxy Deployment (built in-house) that
  `CONNECT`-allowlists only GitHub + the configured LLM hosts and denies
  everything else. serge injects it as the pod's `HTTPS_PROXY`/`HTTP_PROXY`;
  `NO_PROXY` keeps the in-cluster callback off the proxy. serge keeps the
  proxy's allowlist in sync with the configured LLM providers.

The residual risks (DNS egress, token tunneling through legitimate GitHub
requests) are documented in [Security](security.md#per-task-pod-network-firewall).

## Deployment flavors

serge runs in two shapes. Pick based on scale and isolation needs.

### Flavor 1 — Docker

A single container running `reviewbot-web` (`Dockerfile` at the repo root:
Python 3.11 + bubblewrap + the `[web]` extra, uvicorn on `$PORT`). The embedded
SQLite store lives on a mounted volume. This is the simplest hosted deployment —
the one that maps directly to the original EC2 host.

```text
  ┌──────────────────────── docker host / VM ────────────────────────┐
  │  serge container (reviewbot-web)                                  │
  │  • uvicorn :8080     • bubblewrap sandbox (HELPER_SANDBOX)        │
  │  • jobs.db on a mounted volume                                    │
  │                                                                   │
  │  reviews/tasks: TASK_EXECUTION=inprocess  (or =docker for the     │
  │  normalize/runner container on the same docker socket)            │
  └───────────────────────────────────────────────────────────────────┘
```

- **Backends:** `inprocess` by default. Optionally `docker` for tasks/reviews
  (docker-out-of-docker via a mounted socket) or `TASK_SANDBOX_BACKEND=docker`
  for the normalize step.
- **Isolation:** per-subprocess bubblewrap (`HELPER_SANDBOX=require`) is the
  boundary between fork code and host secrets.
- **State:** one container, one SQLite file on a volume. No orchestration.
- **Good for:** a single team, one or a few repositories, straightforward ops.

### Flavor 2 — Kubernetes (Helm)

The [Helm chart](https://github.com/huggingface/serge/tree/main/deploy) in
`deploy/helm` packages serge for a cluster. Because SQLite is a single writer,
the Deployment runs **one replica** with a `Recreate` strategy and an RWO PVC.
When the pod-per-task/-review backend is enabled, the chart also renders the
egress proxy, the network policies, and the RBAC serge needs to manage Jobs.

The diagram below is generated straight from the chart with
[KubeDiagrams](https://github.com/philippemerle/KubeDiagrams) (see
[Regenerating the diagram](#regenerating-the-diagram)):

![serge Kubernetes architecture]({{ "/assets/architecture-k8s.png" | relative_url }})

What the chart creates:

| Resource | Purpose |
| -------- | ------- |
| `Deployment serge` (1 replica, `Recreate`) | The orchestrator (`reviewbot-web`) |
| `Service serge` + `Ingress` | In-cluster + external access (ALB in prod) |
| `PVC serge-data` + `ConfigMap serge` | SQLite store + non-secret runtime env |
| `ServiceAccount` + `Role`/`RoleBinding serge-task-runner` | Lets serge create Jobs + per-job Secrets and read the egress Service |
| `Deployment/Service/ConfigMap serge-egress` | The allowlisting forward proxy for runner pods |
| `NetworkPolicy serge-egress` | Proxy may reach :443 + kube-dns only |
| `NetworkPolicy serge-task-pod` | Runner pods may reach only the proxy, kube-dns, and serge's callback |

Runner pods themselves are **not** in the chart — serge creates them at runtime,
one per request, as `batch/v1` Jobs.

- **Backends:** `taskExecution.kubernetes.enabled=true` turns on task pods;
  `reviewPods=true` also runs reviews in pods.
- **Isolation:** the ephemeral pod + the egress allowlist. Secrets never sit
  next to arbitrary fork code on the long-lived host.
- **Good for:** many repositories, heavier toolchains (the runner image carries
  the repo's build/lint stack), and horizontal isolation per request.

See [Deploying serge](https://github.com/huggingface/serge/tree/main/deploy) and
the production values in `deploy/helm/env/prod.yaml`.

#### Regenerating the diagram

The Kubernetes diagram is reproducible from the chart:

```bash
pip install KubeDiagrams        # provides `kube-diagrams`
brew install graphviz helm      # `dot` + `helm`

deploy/scripts/gen-arch-diagram.sh   # → docs/assets/architecture-k8s.png
```

The script runs `helm template` with the pod backend enabled and pipes the
manifests through `kube-diagrams`.

## Data and persistence

| Data | Where | Notes |
| ---- | ----- | ----- |
| Jobs, drafts, provider configs, cached tokens | SQLite (`WEB_STORE_PATH`) | Single writer → 1 replica; on a PVC/volume so it survives restarts |
| Clone cache | `WEB_CLONE_CACHE_DIR` | Shallow bare-clone cache; point at durable storage in production |
| Runner pod checkout | pod `emptyDir` | Self-contained standalone clone; evaporates with the pod |
| Secrets (App key, OAuth, LLM keys) | env / Kubernetes Secret | Never enter a sandbox or runner pod's git remote |

## Related pages

- [How it works](how-it-works.md) — the review lifecycle step by step.
- [Tasks flow](tasks-flow.md) — the write-capable `POST /tasks` path.
- [Configuration](configuration.md) — every env var, including the backend and
  pod settings.
- [Security](security.md) / [Security architecture](security-architecture.md) —
  the trust model and the sandbox/egress boundaries.
