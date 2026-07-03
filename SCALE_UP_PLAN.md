# Scale-up plan — `reviewbot-web`

**Target:** 20–50 concurrent reviews on a single self-managed EC2 host, with
the option to swap SQLite for Postgres via configuration. Single-node
availability is acceptable.

This document is the implementation plan, not a description of the
current state. Phases are ordered so each is independently mergeable and
the app keeps working end-to-end after every phase.

---

## 1. Why we need this

`reviewbot/webapp.py` is a single-process, single-uvicorn-worker FastAPI
app. It now serves both the staged review UI and the GitHub App webhook
surface at `POST /webhook`.

Each staged `POST /reviews` still spawns an unbounded daemon
`threading.Thread` that:

- Pulls a GitHub App installation token.
- Runs a **synchronous** shallow `git clone` via `subprocess.run`.
- Calls the LLM with synchronous `requests` (streaming), often for
  several minutes.
- Pushes events into an in-memory `history` list + `asyncio.Queue` per
  job.

Each GitHub App webhook request does cheap request-path work (HMAC
verification, JSON parsing, trigger gating), then submits immediate
review/follow-up work to `_WEBHOOK_REVIEW_POOL`, a bounded
`ThreadPoolExecutor` controlled by `WEBHOOK_MAX_WORKERS` (default 2).
That protects the FastAPI request path, but webhook reviews still share
the same process, LLM provider quota, GitHub API quota, CPU, memory, and
network as staged reviews.

State is held in:

- `_jobs: dict[str, Job]` guarded by `_jobs_lock` (in-memory, lost on
  restart for live jobs).
- `JobStore` — a single `sqlite3.Connection` with `check_same_thread=False`,
  WAL, and **one global `threading.Lock` serializing every read and write**
  (`reviewbot/store.py:98-110`).

The hard ceilings at ~50 concurrent reviews are, in order of severity:

1. **Memory**: each in-flight job holds a temp git checkout + history
   buffer + LLM context. ~200–500 MB/job. t3.medium (4 GB) OOMs well
   before 50.
2. **Unbounded thread spawn** in staged `submit_review` — no admission
   control, no queueing, no fairness. Webhook reviews already have a
   small bounded pool, but no persisted queue or global fairness with
   staged reviews.
3. **Single SQLite connection + global write lock** serializes all reads
   and writes (`store.py:_lock`).
4. **GIL pressure on the event loop**: the sync LLM streaming threads
   share the process with the asyncio loop that drives every SSE stream
   and every admin/journal request. Many noisy writers can stall the
   loop.

Phases 1–4 address these in order.

---

## 2. Target architecture (single node, ≤ 50 concurrent)

```
                ┌──────────────────────────────────────────┐
                │  uvicorn (1 worker, asyncio loop)        │
                │                                          │
   HTTP ──►     │  FastAPI                                 │
   SSE  ◄──     │   ├─ /reviews (admission + 429/202)      │
   Hooks ─►     │   ├─ /webhook (HMAC + admission + 202)    │
                │   ├─ /admin/*                            │
                │   ├─ /reviews/.../stream  (SSE)          │
                │   └─ /journal/*                          │
                │                                          │
                │  ┌────────────────────────────────────┐  │
                │  │  ReviewWorkerPool                  │  │
                │  │   ├─ staged jobs from /reviews      │  │
                │  │   ├─ immediate jobs from /webhook   │  │
                │  │   ├─ size = WEB_WORKER_POOL         │  │
                │  │   └─ bounded priority/fair queue    │  │
                │  └────────────────────────────────────┘  │
                │                                          │
                │  ┌────────────────────────────────────┐  │
                │  │  Store (SQLAlchemy Core)           │  │
                │  │   ├─ engine pool (size 10, overflow│  │
                │  │   │  20, pre-ping)                 │  │
                │  │   └─ URL: sqlite:///... | postgres │  │
                │  └────────────────────────────────────┘  │
                │                                          │
                │  ┌────────────────────────────────────┐  │
                │  │  CloneCache (shared bare repo +    │  │
                │  │  per-job worktree, capped tmpdir)  │  │
                │  └────────────────────────────────────┘  │
                └──────────────────────────────────────────┘
```

- Single uvicorn worker is preserved — live SSE state remains in-process,
  which is the simplest correct design for one node.
- The job-execution side is converted from "thread per staged call" plus
  "separate small webhook pool" to one bounded pool fed by a queue.
  Submissions beyond the queue cap are rejected (`429` for UI, `202`
  with a skipped/logged internal event or `503` for webhook, depending
  on the final GitHub retry strategy).
- The store grows a SQLAlchemy Core layer so the same code runs against
  SQLite (default) or Postgres (`postgresql+psycopg://…`).

---

## 3. Phase 0 — Dockerize

**Goal:** ship the app as a container image, with a `docker compose` file
that runs the full stack locally (web + optional Postgres) and on EC2.
This is a **packaging change, not a behavior change** — it lands first
because every later phase is easier to test and deploy in a consistent
environment.

### What changes

- **New file:** `Dockerfile` (multi-stage)
  - Stage `builder`: `python:3.12-slim`, installs build deps, builds a
    wheel of `reviewbot[web,postgres]` into a venv at `/opt/venv`.
  - Stage `runtime`: `python:3.12-slim`, copies `/opt/venv` from
    builder, installs **`git`** (required by `_clone_pr_head` /
    Phase 3's clone cache), `tini` as PID 1.
  - Runs as non-root user `app` (uid 10001).
  - `WORKDIR /app`, `EXPOSE 8080`.
  - `HEALTHCHECK CMD python -c "import urllib.request as u;
    u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"`
  - `ENTRYPOINT ["tini","--"]`, `CMD ["reviewbot-web"]`.
  - Resulting image target: **< 250 MB compressed**.
- **New file:** `.dockerignore`
  - Excludes `.venv/`, `aws/`, `.git/`, `tests/`, `*.db`, `*.pem`,
    `__pycache__/`, etc. Keeps the build context tiny.
- **New file:** `docker-compose.yml` (development + production base)
  - Service `web`:
    - `image: reviewbot:local` (built from `./Dockerfile`).
    - `env_file: .env` (gitignored; copied from
      `aws/reviewbot-web.env.example`).
    - Mounts a named volume `reviewbot-data` at `/var/lib/reviewbot` —
      holds the SQLite DB and the clone cache (Phase 3 will use this
      same volume).
    - Mounts the GitHub App private key as a read-only file:
      `./aws/serge.pem:/etc/reviewbot/github-app.pem:ro` with
      `GITHUB_PRIVATE_KEY_PATH=/etc/reviewbot/github-app.pem`.
    - `ports: ["8080:8080"]`.
    - `restart: unless-stopped`.
    - `tmpfs: /tmp` (small; clones live on the named volume, not /tmp).
  - **Optional sibling service `postgres`** behind a compose profile
    (`profiles: ["postgres"]`) so `docker compose up` defaults to SQLite
    and `docker compose --profile postgres up` brings up Postgres for
    testing the Postgres path. Postgres data on its own named volume.
- **New file:** `docker-compose.prod.yml` (override layered on top in
  production; minimal — just disables port publishing if running behind
  a reverse proxy on the host, sets `restart: always`, raises ulimits
  if needed).
- **Edit:** `aws/deploy.sh` and `aws/update.sh`
  - Replace the venv + systemd-running-`reviewbot-web` flow with:
    install Docker Engine via the AWS-recommended yum repo (Amazon
    Linux 2023: `dnf install -y docker`), enable + start `docker.service`,
    add `ec2-user` to the `docker` group.
  - `update.sh` becomes: `rsync` the repo to the host, `docker compose
    pull || docker compose build`, `docker compose up -d`,
    `docker compose ps` to verify, then `curl /healthz` against the
    local port.
  - The systemd unit is replaced by `docker compose` with
    `restart: unless-stopped`. (Optional: a wrapper systemd unit
    `reviewbot.service` that just runs `docker compose up` so the
    container starts on host boot — cleaner than enabling
    `docker.service` auto-start of compose stacks.)
- **Edit:** `aws/reviewbot-web.env.example` — point `WEB_STORE_PATH`
  (and the future `WEB_STORE_URL`) at `/var/lib/reviewbot/jobs.db`,
  i.e. the volume mountpoint. Keep `GITHUB_WEBHOOK_SECRET` documented;
  the single `reviewbot-web` container serves both UI and GitHub App
  webhooks at `/webhook`.
- **Optional CI:** `.github/workflows/docker.yml`
  - On push to `main`: `docker buildx build` + push to GHCR
    (`ghcr.io/<owner>/reviewbot:<sha>` and `:main`). `update.sh` can then
    `docker pull` instead of rebuilding on the host. Skip if not
    wanted — the host can build directly from a rsync'd checkout.

### Why a container, given we're staying on a single EC2 host

- **Dev/prod parity** — every later phase (Postgres swap, clone cache,
  bigger pool) is easier to validate locally when the runtime is the
  same.
- **Cleanup story** — `docker compose down -v` resets the entire stack
  including the DB. Right now, fresh deploys involve hand-curated
  systemd + venv state.
- **Sidecar Postgres for free** — Phase 1's Postgres path is just
  `docker compose --profile postgres up`. No host-level Postgres
  install needed for testing.
- **Restart semantics** — `restart: unless-stopped` + healthcheck does
  the job that the systemd unit does today, with one fewer concept.

### Things we explicitly do NOT do here

- **No multi-replica / Swarm / k8s.** Single host, single container,
  one uvicorn worker. The in-process `_jobs` dict survives because
  the container is the process boundary.
- **No image-pinning to a private registry** as a hard requirement.
  GHCR push is offered as optional. The default path builds on the
  host from the rsync'd checkout, which keeps `aws/deploy.sh`
  self-contained.
- **No nginx sidecar.** The ALB / EC2 SG already terminate networking
  in front of the app; uvicorn serves directly on `:8080`.

### Acceptance criteria

- `docker compose up -d` on a fresh checkout starts the app and
  `curl http://localhost:8080/healthz` returns `{"status":"ok"}` within
  10 seconds.
- `POST /webhook` is reachable through the same container and rejects
  bad signatures without touching worker capacity.
- `docker compose --profile postgres up -d` brings up both services and
  the web service connects to Postgres (verified once Phase 1 lands).
- Image size < 250 MB compressed.
- `git` is available inside the container (`docker compose exec web git
  --version` works).
- `aws/update.sh` against the EC2 host completes a deploy with no
  manual steps on the host beyond the first-time Docker install.
- `docker compose down && docker compose up -d` is a full restart with
  no data loss (SQLite DB on the named volume).

### Files touched

- `Dockerfile` (new)
- `.dockerignore` (new)
- `docker-compose.yml` (new)
- `docker-compose.prod.yml` (new)
- `aws/deploy.sh` (rewrite the runtime section; keep VPC/SG/EBS bits)
- `aws/update.sh` (rewrite to use `docker compose`)
- `aws/reviewbot-web.env.example` (path adjustments)
- `pyproject.toml` (no change here — the `[postgres]` extra is added in
  Phase 1)
- `.github/workflows/docker.yml` (optional)

---

## 4. Phase 1 — SQLAlchemy abstraction for the store

**Goal:** remove the single-connection global lock and make the backend
URL-selectable. No schema changes.

### What changes

- **New file:** `reviewbot/db.py`
  - `make_engine(url: str, *, echo: bool=False) -> Engine`
  - SQLite branch: `connect_args={"check_same_thread": False, "timeout": 30}`,
    `poolclass=StaticPool` only for `:memory:` tests; otherwise default
    `QueuePool` with `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`.
  - Apply `PRAGMA journal_mode=WAL`, `PRAGMA synchronous=NORMAL`,
    `PRAGMA busy_timeout=30000` via an `@event.listens_for(engine, "connect")`
    hook.
  - Postgres branch: `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`,
    `pool_recycle=1800`.
- **Rewrite:** `reviewbot/store.py`
  - Define tables with `sqlalchemy.MetaData` + `Table(...)` (Core, not
    ORM — keep the SQL explicit and the diff small). The schema is the
    full current shape in one place — no `_ensure_column` runtime DDL,
    no `ADD COLUMN` fallback.
  - Replace `self._conn` + `self._lock` with `self._engine`. Each method
    opens a `with engine.begin() as conn:` block.
  - On startup: `metadata.create_all(engine)` — idempotent, creates the
    tables on a fresh DB and no-ops if they already exist.
  - No migration tooling. Schema changes are made by dropping the DB
    (we don't preserve historical jobs).
  - Public method signatures stay identical so `webapp.py` is unchanged.
- **Config additions** (`reviewbot/config.py`):
  - `web_store_url: str` — defaults to `f"sqlite:///{web_store_path}"`
    so existing deployments keep working with no env change.
  - `web_db_pool_size: int = 10`
  - `web_db_max_overflow: int = 20`
- **Deps** (`pyproject.toml`): add `sqlalchemy>=2.0`. Postgres is
  optional: `psycopg[binary]>=3.1` as an extra (`reviewbot[postgres]`).

### Why SQLAlchemy Core (not ORM)

The current code is hand-written SQL with `json.dumps`/`json.loads` for
list columns. The ORM would force a model rewrite and an identity-map
that buys nothing here. Core gives us the engine + pool + dialect
abstraction without changing query shape.

### Deployment note: existing data is dropped

We do not preserve `jobs.db` history. Deploying Phase 1 means:

- The old SQLite file is removed (or ignored — the new schema is created
  fresh by `create_all`).
- Provider configs in the old DB must be re-entered via `/admin` after
  the deploy. **This is the one operator-visible action item.** Worth a
  pre-deploy step in `aws/update.sh` to dump existing `provider_configs`
  rows to a JSON file and re-import them after migration — purely as a
  convenience, not for correctness.

### Acceptance criteria

- `tests/` pass against SQLite (default) and against a local Postgres
  (`docker run -d -p 5432:5432 postgres:16`).
- No `_lock` left in `store.py`.
- Concurrent-write smoke test: 30 simultaneous `insert_job` + `save_terminal`
  complete with no `database is locked` errors on SQLite.

### Files touched

- `reviewbot/db.py` (new, ~50 lines)
- `reviewbot/store.py` (rewrite, same public API)
- `reviewbot/config.py` (+3 fields)
- `pyproject.toml`
- `aws/reviewbot-web.env.example` (document `WEB_STORE_URL`)
- `aws/update.sh` (optional: dump/restore `provider_configs` as a
  convenience step on the cutover deploy)

---

## 5. Phase 2 — Unified bounded worker pool + admission control

**Goal:** replace unbounded staged `threading.Thread` spawning and the
separate webhook-only executor with one bounded `ThreadPoolExecutor` +
queue. Submissions over the cap get explicit backpressure.

### What changes

- **New module:** `reviewbot/worker_pool.py`
  - `class ReviewWorkerPool:` wraps a `ThreadPoolExecutor(max_workers=N)`
    plus a `queue.Queue` of pending submissions.
  - `submit(item: ReviewWorkItem) -> SubmitResult` where `SubmitResult`
    is one of `{started, queued, rejected}`.
  - `ReviewWorkItem` supports both staged UI jobs and immediate webhook
    jobs. Staged jobs carry a `Job` and stream to SSE; webhook jobs
    carry the GitHub installation id + `ReviewRequest` and post directly.
  - Tracks per-job `Future`s in a dict so the pool can be drained on
    shutdown and so `/admin` can report depth.
  - Emits a `step:queued` event when a job is admitted to the queue but
    not yet started, so the SSE viewer shows "waiting for slot…".
- **Edit:** `reviewbot/webapp.py`
  - Replace the `threading.Thread(target=_run_review_worker,…).start()`
    block in `submit_review` with `pool.submit(job)`.
  - Replace `_WEBHOOK_REVIEW_POOL.submit(...)` in `/webhook` with the
    same pool. Webhook work keeps its immediate posting behavior; it
    just shares admission control with staged jobs.
  - On `SubmitResult.rejected` return HTTP 429 with a JSON body
    `{"error": "queue_full", "retry_after_seconds": N}` and a
    `Retry-After` header.
  - For webhook rejection, prefer a `503` so GitHub retries delivery
    later. If we decide duplicate webhook deliveries are worse than a
    dropped review, return `202` and log `queue_full`; make this a
    deliberate config choice (`WEBHOOK_QUEUE_FULL_STATUS=503|202`).
  - Add a startup hook to construct the pool from config; add a shutdown
    hook (`app.on_event("shutdown")`) that drains it with a timeout.
- **New endpoint:** `GET /admin/pool` returns `{active, queued, capacity,
  queue_cap}` for monitoring (cheap, no auth changes — admin page already
  gated by login).
- **Config additions:**
  - `web_worker_pool_size: int = 50`
  - `web_worker_queue_cap: int = 25` — total queued-but-not-started.
    Above this, submissions get 429.
  - `webhook_worker_share: int = 10` or a simpler priority policy:
    reserve at least N slots for staged jobs so a webhook burst cannot
    starve interactive users. Keep `WEBHOOK_MAX_WORKERS` as a temporary
    compatibility alias until it is removed.
  - `web_worker_job_timeout: int = 1800` — hard wall-clock cap per job;
    enforced inside `_run_review_worker` (a `threading.Event` set by a
    timer cancels the LLM stream cleanly).

### Backpressure shape

- `started`: 200 OK, same body as today.
- `queued`: 202 Accepted, same body — the UI polls or opens SSE which
  immediately shows the `queued` step.
- `rejected`: 429 with `Retry-After` (default 30s).
- Webhook `started`/`queued`: 202 Accepted, matching today's accepted
  async behavior.
- Webhook `rejected`: 503 + `Retry-After` by default so GitHub redelivers
  instead of silently dropping the mention.

### Acceptance criteria

- Smoke test: submit 100 staged reviews in a tight loop, observe at most
  `pool_size` worker threads, the next `queue_cap` jobs in `queued`
  state, the rest rejected with 429.
- Webhook burst test: deliver 100 signed `pull_request_review_comment`
  payloads, observe bounded worker count, accepted queued work up to the
  cap, and 503 + `Retry-After` after the cap.
- SSE for a queued job immediately shows `step: queued` then transitions
  to `step: clone` when a worker picks it up.
- `kill -TERM` on the uvicorn process waits up to `shutdown_grace`
  (default 30s) for in-flight jobs to finish before exit.

### Files touched

- `reviewbot/worker_pool.py` (new)
- `reviewbot/webapp.py` (`submit_review`, lifespan hooks, `/admin/pool`)
- `reviewbot/config.py` (+3 fields)
- `tests/test_worker_pool.py` (new)

---

## 6. Phase 3 — Memory & disk discipline

**Goal:** keep 50 concurrent jobs within ~12 GB RAM and a few GB of
disk. Today each job clones a full PR head into its own tmpdir.

> **Status:** the **clone cache** below landed early as low-hanging fruit
> (`reviewbot/clone_cache.py`, wired into `webapp.py`, config keys
> `WEB_CLONE_CACHE_DIR` / `WEB_CLONE_CACHE_TTL_SECONDS` / `WEB_CLONE_DEPTH`,
> tests in `tests/test_clone_cache.py`). The **history-buffer tightening**
> and the **dedicated EBS volume** below are still TODO.

### Clone cache  ✅ implemented

- **New module:** `reviewbot/clone_cache.py`
  - Maintains a per-`(owner, repo)` shared bare clone under
    `WEB_CLONE_CACHE_DIR` (default `/var/lib/reviewbot/clones`).
  - On clone request: `git fetch --depth 50 origin pull/<N>/head:pr-<N>-<job_id>`
    into the shared bare repo, then `git worktree add <tmpdir> pr-<N>-<job_id>`
    to give the worker a cheap isolated checkout.
  - Use a per-repo `threading.Lock` to serialize fetches into the same
    bare repo (git itself locks `.git/index.lock`, but a Python lock
    avoids fail-and-retry).
  - On worker completion: `git worktree remove --force <tmpdir>` and
    `git branch -D pr-<N>-<job_id>` — fast, no rewalk.
  - Periodic GC (every 1h via `asyncio.create_task` on startup): drop
    bare repos that haven't been touched in `clone_cache_ttl` (default
    7 days).

### Why this matters

Today's path does `git init` + `git fetch --depth 50` per job. Two
parallel reviews on the same repo do this twice. With 30 concurrent
reviews on `transformers`, that's 30 cold fetches of a large repo at
once — disk + outbound bandwidth disaster, plus disk-burst credit
starvation on EBS gp3. A shared bare repo turns this into one fetch +
30 cheap worktrees.

### Tmpdir cap

- Set `WEB_CLONE_CACHE_DIR` on a separately-sized EBS volume (`/var/lib/reviewbot`).
  Don't put clones in `/tmp` if `/tmp` is tmpfs — RAM-backed clones
  amplify the memory problem.
- Document in `aws/reviewbot-web.env.example` and create the volume in
  `aws/deploy.sh`.

### History buffer tightening

The current `_NOISY_HISTORY_CAP = 2000` (`webapp.py:473`) caps the
replay buffer. Two issues at scale:

- The cap is per kind aggregate (`token` + `reasoning` combined). On
  reasoning-heavy models (o-series, Kimi-K2 thinking), 2000 chunks is
  still meaningful memory per job (each chunk is a string ≥ a few
  hundred bytes).
- The `for i, e in enumerate(job.history): ... del job.history[i]` path
  is O(n) on a list eviction. Fine in isolation, ugly at 50 jobs ×
  thousands of chunks.

Changes:

- Replace the noisy-history list with a `collections.deque(maxlen=...)`
  to make eviction O(1).
- Add `WEB_NOISY_HISTORY_CAP` env knob (default still 2000).
- Cap structural events at 5000/job (currently unbounded — agentic loops
  with many tool calls can produce hundreds of structural events).

### Acceptance criteria

- 30 simultaneous reviews on the same medium-sized repo show one
  `fetch` in metrics, then 30 fast `worktree add` operations.
- RSS of the uvicorn process at steady state with 50 jobs running
  stays under 12 GB on a host with 16 GB.
- Disk usage of `WEB_CLONE_CACHE_DIR` capped by GC.

### Files touched

- `reviewbot/clone_cache.py` (new)
- `reviewbot/webapp.py` (`_run_review_worker` uses cache, replace
  `_clone_pr_head`, replace history list/lock with deque)
- `reviewbot/config.py` (+3 fields)
- `aws/deploy.sh` (separate EBS volume, mountpoint)
- `aws/reviewbot-web.env.example`
- `tests/test_clone_cache.py` (new)

---

## 7. Phase 4 — Async hygiene on the event loop

**Goal:** keep the event loop responsive when 50 workers are streaming.

### Move synchronous DB calls off the loop

Several endpoints currently call `_store.*` directly inside async handlers
(`submit_review`, `lookup_provider`, `admin_create_provider`, etc.). On
SQLite + connection pool the calls are fast, but on Postgres with
network latency they block the loop. Wrap with `asyncio.to_thread(...)`
or define a thin `AsyncStore` facade that `await`s a default executor.
`POST /webhook` should stay deliberately cheap on the loop: HMAC, JSON,
trigger gating, pool submission, response.

### Do not async-rewrite the worker

Tempting to swap `requests` for `httpx.AsyncClient`, but the agentic
loop in `reviewer.py` is deeply synchronous (tool dispatch, JSON
streaming parser, callback chain into `_push_event`). Async rewrite is
a multi-week project for marginal gain at 50 concurrent. Keep workers
sync; the GIL is fine because LLM streaming releases it on every socket
read.

### Lighten `_push_event`

`_push_event` runs in the worker thread on every token chunk. Today:

1. Acquires `job.history_lock`.
2. Appends to `job.history`.
3. Possibly evicts oldest noisy entry (O(n) scan).
4. Calls `job.loop.call_soon_threadsafe(job.queue.put_nowait, event)`.

At 50 jobs × tens of chunks/sec, the loop sees a constant trickle of
`call_soon_threadsafe` callbacks. Make this cheaper:

- Use `deque(maxlen=…)` (Phase 3) — drops the O(n) eviction.
- Batch token events: coalesce up to N (e.g. 8) token chunks or 50ms
  into a single SSE event before crossing the thread boundary. Keeps
  the UX (text still streams smoothly) but cuts cross-thread wakeups
  ~8×. Implement as a small per-job flush timer in the worker.

### Acceptance criteria

- `/healthz` p99 latency under load (50 active jobs) stays under 50 ms.
- `/admin/pool` p99 latency stays under 100 ms.
- No `asyncio` "Executing took ... seconds" warnings in logs at steady
  state.

### Files touched

- `reviewbot/webapp.py` (`_push_event`, async DB wrappers)
- `reviewbot/store.py` (optional `AsyncStore` shim, can also just use
  `asyncio.to_thread` at call sites — simpler and we have few of them).

---

## 8. Phase 5 — Host sizing, ops, observability

### Host

Move EC2 default from `t3.medium` (2 vCPU, 4 GB) to **`m5.xlarge`**
(4 vCPU, 16 GB) or **`m5.2xlarge`** (8 vCPU, 32 GB) if budget allows.
Use `gp3` EBS with a baseline ≥ 250 MB/s for the clone cache volume.

Update `aws/deploy.sh`:

- `INSTANCE_TYPE="${INSTANCE_TYPE:-m5.xlarge}"`
- Provision a second 50 GB gp3 volume mounted at `/var/lib/reviewbot`
  for clones + DB. Stops `git fetch` storms from competing with the OS
  disk.

### Metrics

Add a `GET /metrics` endpoint (Prometheus text format, no extra deps —
just hand-write it; ~30 lines). Export:

- `reviewbot_jobs_active` (gauge)
- `reviewbot_jobs_queued` (gauge)
- `reviewbot_webhook_deliveries_total{result}` (counter: accepted,
  skipped, rejected, bad_signature, bad_json)
- `reviewbot_jobs_total{status}` (counter)
- `reviewbot_job_duration_seconds` (histogram, status-tagged)
- `reviewbot_llm_tokens_total{kind=prompt|completion,provider}` (counter)
- `reviewbot_db_pool{state=in_use|idle}` (gauge)
- `reviewbot_clone_cache_repos` (gauge)

### Logs

- Switch to JSON structured logs (single env switch — keep the current
  format as a fallback). Existing `log.info(...)` calls keep working.
- Include `job_id`, `user`, `owner/repo`, `provider`, `model` as
  consistent fields.

### Alerts (whatever monitoring is in place)

- Job queue depth > 10 for > 5 min
- Pool saturation (active == capacity) > 10 min
- Webhook rejected/503 rate > 0 for > 5 min
- Disk usage on `/var/lib/reviewbot` > 80%
- Process RSS > 14 GB on a 16 GB box

### Acceptance criteria

- `m5.xlarge` deployment serves a sustained 50 concurrent jobs without
  OOM and with `/healthz` under 50 ms.
- `/metrics` scraped without errors.

---

## 9. Rollout order

Each phase is independently deployable. Order matters:

1. **Phase 0 (Dockerize)** — packaging only, no behavior change. Lands
   first because every subsequent phase is easier to test and roll out
   inside a consistent runtime. Bake for a day on production at current
   load before moving on.
2. **Phase 1 (SQLAlchemy)** — drops existing `jobs.db` data; re-add
   provider configs via `/admin` after deploy. With Docker already in
   place, the Postgres path can be smoke-tested in compose before the
   production swap.
3. **Phase 5a (host sizing)** — bump to `m5.xlarge` + add EBS volume
   *before* turning on the pool, so the pool has headroom. EBS volume
   becomes the bind target for the `reviewbot-data` docker volume.
4. **Phase 2 (worker pool)** — flips submission semantics and folds
   webhook work into the same admission-control path; bake with
   `web_worker_pool_size=10` first, `WEBHOOK_MAX_WORKERS=1` compatibility
   behavior, then raise once metrics show enough headroom.
5. **Phase 3 (clone cache + history deque)** — the biggest memory win,
   using the named volume Phase 0 already created.
6. **Phase 4 (event-loop hygiene)** — quality of service; nothing here
   blocks correctness, but it makes the page snappy under load.
7. **Phase 5b (metrics + alerts)** — fold in as each phase lands; tag
   each release with the new gauges/counters it adds.

---

## 10. Explicitly out of scope

These come later, when single-node isn't enough — they require a
separate plan and are noted only so we don't accidentally design
ourselves into a corner:

- **Multi-host / HA**: the `_jobs` dict and per-process SSE queues are
  intentionally kept. Going multi-host would require moving live job
  state to Redis (with pub/sub for SSE fan-out) and a shared queue
  (Redis-RQ / Celery / SQS). Defer until traffic warrants it.
- **Horizontal autoscaling**: same reason.
- **Job cancellation from UI**: the worker has no `cancel` story today.
  Worth adding when users have something to cancel (i.e., once they're
  visibly waiting in a queue).
- **Staged webhook drafts**: `/webhook` currently posts immediate reviews
  and follow-up replies. Creating draft jobs in the web UI from GitHub
  comments is a product change, not scale work. It can be layered on the
  unified pool later.

---

## 11. Acceptance for the whole effort

The plan is done when:

- A load test (provided as `tests/load/submit_many.py`) submits 50
  reviews in parallel against a real PR and:
  - 50 of them finish (or `WEB_WORKER_QUEUE_CAP` rejects the
    overflow cleanly with 429).
  - Peak RSS of the uvicorn process stays under 14 GB on an
    `m5.xlarge`.
  - `/healthz` p99 < 50 ms throughout.
  - No `database is locked` errors regardless of SQLite or Postgres
    backend.
- `WEB_STORE_URL=postgresql+psycopg://…` swaps the backend with no
  other code changes.
- `aws/deploy.sh` and `aws/update.sh` work unchanged for SQLite
  deployments and gain a documented `postgres` path.
- `reviewbot-web` is the only deployed app process needed for both the
  interactive UI and GitHub App `@askserge` comment webhooks.
