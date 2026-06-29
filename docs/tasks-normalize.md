# Normalize validation (in-loop)

When serge produces a fix through the [tasks flow](tasks-flow.md), the LLM
proposes a logical patch — it does not run the target repo's build commands.
For repos with a consistency gate (regenerated files, formatting, lint), that
means the opened PR can fail repo-consistency CI even when the fix itself is
correct, forcing a follow-up commit.

**Normalize validation** closes that gap by making the repo's own normalizer
(e.g. `make style && make fix-repo`) part of the **LLM loop**. When the model
emits a patch, serge applies it to the worktree and runs the normalizer. If the
normalizer **rejects** the patch, that failure is fed straight back to the model
as a new turn in the same conversation, so it can *correct the patch* — exactly
the feedback a human contributor would get from CI, but before the PR is opened.
When the normalizer passes, the worktree already holds the applied + normalized
result and serge commits it as **one clean patch**. The PR is conformant the
moment it opens.

This is **opt-in, per deployment/repo**. When `TASK_NORMALIZE_COMMAND` is unset,
validation is skipped entirely and serge stays repo-agnostic, exactly as before.

## How it runs

```
prepare_task():                              ← the agentic LLM loop
  loop:
    LLM emits a candidate patch
    reset worktree → apply patch → run normalizer   ← the verification gate
      ├─ patch won't apply     → feed `git apply` error back → LLM revises
      ├─ normalizer exits != 0 → feed its output back        → LLM revises
      └─ normalizer passes     → accept; worktree = applied + normalized
publish_task():
  stage the prepared worktree → commit → open/update PR      ← Git Data API
```

The model gets up to `TASK_NORMALIZE_MAX_RETRIES` corrective re-prompts (default
2 → 3 patch attempts total). It applies to both `new_pr` and `existing_pr`
modes. A typical normalizer rejection the model learns to fix: *"you edited the
auto-generated `modeling_x.py` instead of its `modular_x.py` source"* — the
normalizer's own error text carries that signal.

### What happens when validation can't finish

The loop never costs you the fix:

- **Retries exhausted** (the model couldn't satisfy the normalizer): serge
  falls back to committing the model's last patch raw — the same PR you'd get
  today, which the repo's CI then flags. Better an imperfect PR than a lost fix.
- **Sandbox unavailable / timeout** (infrastructure, not the model's fault):
  the applied patch is accepted un-normalized rather than blaming the model.

### No injection surface

The normalize command is **operator/repo configuration, never request-supplied**.
The `/tasks` request cannot name a command — so there is nothing to allowlist
and no command-injection vector to contain. The OIDC `repository` claim
authorizes *which repo* serge acts on; the operator decides *what command* runs.

## Sandbox backends

The normalizer runs arbitrary repo build code, so it executes network-isolated,
with the worktree as the only writable path and no serge secrets in its
environment. `TASK_SANDBOX_BACKEND` selects how:

| Backend | Isolation | When to use |
| ------- | --------- | ----------- |
| `docker` | Throwaway container: `--network none`, read-only rootfs, `--cap-drop ALL`, `no-new-privileges`, pids cap, runs as serge's uid:gid. | **The portable default.** Works on any host with a Docker daemon — no Kubernetes required. |
| `kubernetes` | One-shot Job in a locked-down namespace (non-root, no-privileged, deny-all egress), worktree on a shared RWX volume. | k8s deployments that want pod-level isolation. *(Implemented in Phase 1.)* |
| `bwrap` | bubblewrap over serge's **own venv** (`--unshare-net`). | Dev/test only — viable just when the command needs no deps beyond serge's. |
| `auto` | docker when an image is set and the docker CLI is present, else bwrap. | Convenient default for mixed environments. |

Kubernetes is never mandatory: a classic Docker deployment is a first-class,
fully-isolated backend. Pick `docker` and you need nothing else.

## Setting up the Docker image

The `docker` backend runs the command in a throwaway container built from an
image with the target repo's toolchain baked in. serge runs it roughly as:

```
docker run --rm --init --network none --read-only \
  --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 512 --user <serge-uid>:<serge-gid> \
  --volume <worktree>:<worktree>:rw --workdir <worktree> \
  <TASK_NORMALIZE_IMAGE> bash -lc 'make style && make fix-repo'
```

1. **Write a Dockerfile** that installs the toolchain the normalizer needs —
   and nothing it doesn't (no torch / model deps for `make fix-repo`). See
   [`docker/Dockerfile.task-runner`](../docker/Dockerfile.task-runner) for a
   worked transformers example (just the `[quality]` extra).

2. **Build and tag it**, pinning versions so the normalizer produces the same
   output the repo's own CI would:
   ```
   docker build -f docker/Dockerfile.task-runner \
     --build-arg TRANSFORMERS_REF=main \
     -t serge/transformers-quality:latest .
   ```
   Build it wherever serge runs (it must be present in the local Docker
   daemon — serge never pulls at normalize time, and the container has no
   network). Rebuild when the repo bumps its pinned tool versions.

3. **Point serge at it** via `TASK_NORMALIZE_IMAGE`.

### Constraints on the image / command

- **Deps must be baked in.** The container has no network; `pip install` at run
  time will fail. Install everything at build time.
- **No `.pyc` writes to site-packages** — the rootfs is read-only. serge sets
  `PYTHONDONTWRITEBYTECODE=1`; informational.
- **The command must not need git history or remotes.** The checkout is a
  detached `git worktree` whose gitdir lives outside the bind mount, and there
  is no network, so commands that fetch/diff against `origin` won't work.
  File-based normalizers (`make fix-repo`, `ruff`, codegen) are fine — serge
  does the staging/diffing itself, on the host, after the command runs.

### The `bwrap` fallback

If you don't set an image (or set `TASK_SANDBOX_BACKEND=bwrap`), the command
runs under bubblewrap using **serge's own venv** — viable only when the command
needs no dependencies beyond what serge already has installed. `make fix-repo`
for transformers needs the `[quality]` toolchain, so use the `docker` backend
for it.

## Configuration

See [configuration](configuration.md#post-llm-normalize-hook) for the full env
var table. The minimum for transformers:

```
TASK_NORMALIZE_COMMAND=bash -lc 'make style && make fix-repo'
TASK_NORMALIZE_IMAGE=serge/transformers-quality:latest
TASK_SANDBOX_BACKEND=docker
```

The per-repo write opt-in (`task_write_enabled` on the repo's provider config)
is the same as for the LLM task flow — the hook commits through the same path.

## Security model

Same trust boundary as the LLM task flow, with stronger isolation for running
the repo's build code:

- **No secrets in the sandbox.** The command gets a scrubbed env; the docker
  container additionally has no network, a read-only rootfs, all capabilities
  dropped, `no-new-privileges`, and a pids cap.
- **serge owns the git write.** The command only mutates a throwaway worktree;
  serge commits via the Git Data API. The installation token never enters the
  container.
- **Operator-controlled command and image.** Both `TASK_NORMALIZE_COMMAND` and
  `TASK_NORMALIZE_IMAGE` are set by the operator, never by the caller.
- **The result is a PR a human reviews before merge.**
