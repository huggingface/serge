# Security architecture: isolating PR/fork code from the review host

## Threat model

ai-reviewer reviews pull requests, including PRs from **forks opened by
untrusted authors**. The persistent review host (an AWS EC2 box running
Amazon Linux 2023 as `ec2-user` under systemd) holds high-value secrets:

- the GitHub App private key (`*.pem`) used to mint installation tokens,
- the OAuth client secret and web session secret,
- LLM API keys (in the systemd `EnvironmentFile`),
- `jobs.db`, the SQLite store that caches installation tokens.

**The threat we defend against is arbitrary code execution on the review
host originating from PR/fork content.** A fork author must not be able
to make the host run code that reads those secrets, reaches the network
on the host's behalf (e.g. the EC2 instance-metadata endpoint to steal
IAM credentials), or persists anything.

This is distinct from GitHub **Action** mode, which runs on GitHub's
*ephemeral, disposable* runner with a job-scoped token — there the runner
itself is the isolation boundary, and a compromise dies with the job.
The controls here are about the **persistent web/webhook host**.

## Where PR/fork code can execute

A review touches PR content in several places. Most are read-only and
safe; three can *execute* code:

| Surface | Reads or executes? | Source of truth |
|---|---|---|
| Diff + PR metadata (GitHub API) | data only | — |
| `read_file` / `grep` / `list_dir` browse tools | read-only | worktree (fork) |
| `fetch_url` builtin | network, host-allowlisted (`huggingface.co`) | — |
| `.ai/review-rules.md`, `.ai/review-tools.json` | data only, fetched via API from the **default branch** | upstream |
| **Helper-tool commands** | **execute** a subprocess `cwd`=worktree | upstream config, but command/scripts could resolve into the fork tree |
| **`.ai/context-script`** | **executes** | the worktree's copy (fork-editable) unless overlaid |
| Helper `install` hook (`pip install <pkg>`) | executes pip | upstream config, args hardened against VCS/URL installs — repo-owner trust, not PR-author |

The browse tools only *read* fork code — that is the point of a code
review and is acceptable. The danger is the three executing surfaces.

## Defense in depth

Three independent layers. Any one failing should not by itself hand a
fork author code execution against host secrets.

### Layer 1 — Configuration always comes from upstream

PR authors can edit `.ai/` in their branch. We must never trust that
copy. Two mechanisms already split by mode:

- **Review rules / helper-tool definitions** are fetched via the GitHub
  API from the base repo's **default branch** (`pr.base.repo.default_branch`),
  never from the PR head. (Pre-existing behavior.)
- **The worktree's `.ai/` directory is overlaid from the default branch.**
  In `CloneCache.acquire` (web mode), after checking out the PR head we:
  1. `git fetch --depth 1 <url> HEAD:_reviewbot_base` — the repo's
     default branch ("clone main"),
  2. `rm -rf <worktree>/.ai`,
  3. `git -C <worktree> checkout _reviewbot_base -- .ai` to materialize
     the upstream `.ai/` over the fork's (skipped if upstream has none).

  This is the literal "clone main, grab `.ai/`, then check out the
  fork/branch" flow. It is **fail-closed**: if the default-branch fetch
  fails, we delete the PR's `.ai/` rather than execute a copy we could
  not verify.

### Layer 2 — Helpers may only execute upstream code

`_resolve_helper_command` (in `tools.py`) accepts `command[0]` only if it
is either:

- a **bare PATH binary** (no `/` — e.g. `ruff`, `mypy`, a pip-installed
  helper), or
- a path that resolves **under `.ai/`** (now guaranteed upstream by
  Layer 1).

Any command resolving to a fork path **outside `.ai/`** (e.g.
`./scripts/lint.sh`) is rejected. Combined with Layer 1, every script a
helper can launch is upstream-controlled.

### Layer 3 — Every executing subprocess is sandboxed

Even an upstream-defined helper runs *against* the fork tree and may
import fork code (a linter discovering and importing PR files). So the
**helper-tool commands and the context-script** are wrapped in
**bubblewrap** (`bwrap`) — an unprivileged, daemonless user-namespace
sandbox, one invocation per subprocess.

The **`install` hook is deliberately *not* sandboxed by this layer.** It
runs `pip install <pkg>`, which inherently needs the network (PyPI) and
write access to the venv — both of which the offline, read-only-venv
profile below forbids. It is also not a fork-code surface: the package
name comes from the upstream default-branch config, and its arguments are
already hardened to reject VCS/URL/index installs (`_INSTALL_ARG_RE`,
`_INSTALL_DENY_FLAG_PREFIXES`). Its trust level is the repo owner's, not
the PR author's, so it stays outside the sandbox.

The sandbox profile:

- `--unshare-all --unshare-net` — **no network at all.** The main review
  loop, which talks to the LLM service, is a *separate, unsandboxed*
  process, so cutting helper network does not affect LLM connectivity.
  No network also means the EC2 instance-metadata endpoint
  (`169.254.169.254`) is unreachable.
- **Filesystem: deny by default.** Only `/usr` (+ usrmerge symlinks for
  `/bin`, `/lib`, `/lib64`, `/sbin`), the Python venv, and a small
  allowlist of `/etc` files (`passwd`, `group`, `nsswitch.conf`,
  `localtime`, TLS cert dirs) are bound **read-only**. The worktree is
  the only writable bind. `/home`, `/etc/reviewbot`, `jobs.db`, the
  app directory, and the `*.pem` keys are simply **not present** in the
  sandbox.
- `--tmpfs /tmp`, `HOME=/tmp`, `TMPDIR=/tmp` — scratch space that
  evaporates with the namespace.
- `--die-with-parent --new-session` — the sandbox cannot outlive the
  worker and is detached from the controlling terminal.
- Environment is independently scrubbed to an allowlist
  (`_HELPER_ENV_PASSTHROUGH`) so no secret reaches the child via env.

#### Sandbox availability policy — `HELPER_SANDBOX`

| Value | Behavior |
|---|---|
| `require` | If `bwrap` is missing or fails to launch, **refuse to run** the subprocess; the tool returns an error. **Set this in production.** |
| `auto` (default) | Sandbox if `bwrap` is on `PATH`, otherwise run unwrapped. For local dev (macOS has no bwrap) and ephemeral CI runners. |
| `off` | Never sandbox. Escape hatch only. |

Production (`aws/reviewbot-web.env`) sets `HELPER_SANDBOX=require`, and
`aws/deploy.sh` installs `bubblewrap`, so a misconfigured host fails
closed rather than silently running helpers unsandboxed.

## Why not just run the whole app in Docker?

Containerizing the *app* is defense-in-depth at the host edge, not a
substitute for the per-helper sandbox, because **the helper runs in the
same trust domain as the app**:

- A helper executing fork code inside the app container still sits next
  to the app's own secrets (GitHub key, LLM keys in env, `jobs.db`).
  Container-around-the-app does **not** isolate the helper from those.
- The boundary that matters is *per subprocess, inside the app* — which
  is what bubblewrap provides.
- Nesting bwrap inside Docker is awkward: Docker's default seccomp
  profile blocks the `clone`/`unshare` flags bwrap needs, so you would
  have to relax it (`--security-opt seccomp=unconfined`), weakening the
  container you just added.

If containers were to *be* the isolation primitive, the clean design is
**per-helper ephemeral containers** (rootless Podman: `--network none`,
`--read-only`, `--cap-drop=ALL`, worktree bind-mounted read-only)
*replacing* bubblewrap — heavier (image upkeep, per-call startup) but
equivalent in guarantee. We chose bubblewrap for its zero-daemon,
low-latency, per-call fit. App-in-Docker can still be layered on later
purely for host blast-radius containment without disturbing Layer 3.

## Residual risks / non-goals

- **Action mode** checkouts (`actions/checkout` into `GITHUB_WORKSPACE`)
  are not `.ai/`-overlaid by this code; they rely on the disposable
  runner's isolation. Layer 2's command restriction still applies; Layer
  3 is `auto` there (bwrap is usually absent on runners).
- A helper can still **read** fork code (by design) and emit it; helper
  output is already wrapped in untrusted-content markers for the model.
- Resource exhaustion (CPU/disk) is bounded by per-helper timeouts, not
  hard rlimits/quotas; the writable worktree bind could be filled within
  a single helper's timeout. Adding `setrlimit`/disk quotas is future
  work.
