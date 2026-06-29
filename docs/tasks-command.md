# Command tasks (deterministic, no LLM)

A **command task** is a `/tasks` request that runs a fixed, allowlisted
command — e.g. `make fix-repo` — against a checkout of the target repo and
opens a PR (or pushes a follow-up commit) with whatever the command changed.
No LLM is involved: **the command is the patch producer.** serge still does
the git write the same safe way as the LLM task flow — it reads the changed
files out of the worktree and commits them through the GitHub Git Data API,
so credentials never enter the sandbox.

Use it when the change is deterministic and the repo already encodes it as a
command (formatters, codegen, `make fix-repo`, `make style`, etc.). It's
cheaper and more reliable than asking an LLM to reproduce the diff.

## How it runs

```
POST /tasks (OIDC) { command: "make fix-repo", output: { mode: "new_pr" } }
        ▼
acquire_ref(base_ref)                       ← check out the branch
        ▼
run command in a network-isolated sandbox   ← docker (default) or bwrap
  (worktree writable, no secrets, no net)
        ▼
git add -A → collect_changes                ← serge stages + reads the diff
        ▼
create_blob → tree → commit → ref → PR      ← Git Data API, in serge's process
```

The command runs with **no network** and the **worktree as the only writable
path**. Its dependencies must therefore be present *before* it runs — that's
what the Docker image is for.

## Setting up the Docker image

The `docker` backend runs the command in a throwaway container built from a
per-repo image that has the target repo's toolchain baked in. serge runs it
roughly as:

```
docker run --rm --init --network none --read-only \
  --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 512 --user <serge-uid>:<serge-gid> \
  --volume <worktree>:<worktree>:rw --workdir <worktree> \
  <TASK_COMMAND_IMAGE> make fix-repo
```

1. **Write a Dockerfile** that installs the toolchain the command needs — and
   nothing it doesn't (no torch / model deps for `make fix-repo`). See
   [`docker/Dockerfile.task-runner`](../docker/Dockerfile.task-runner) for a
   worked transformers example (just the `[quality]` extra).

2. **Build and tag it**, pinning versions so the command produces the same
   output the repo's own CI would:
   ```
   docker build -f docker/Dockerfile.task-runner \
     --build-arg TRANSFORMERS_REF=main \
     -t serge/transformers-quality:latest .
   ```
   Build it wherever serge runs (it must be present in the local Docker
   daemon — serge never pulls at task time, and the task container has no
   network). Rebuild when the repo bumps its pinned tool versions.

3. **Point serge at it** via `TASK_COMMAND_IMAGE`.

### Constraints on the image / command

- **Deps must be baked in.** The container has no network; `pip install` at
  run time will fail. Install everything at build time.
- **No `.pyc` writes to site-packages** — the rootfs is read-only. serge sets
  `PYTHONDONTWRITEBYTECODE=1`; this is just informational.
- **The command must not need git history or remotes.** The checkout is a
  detached `git worktree` whose gitdir lives outside the bind mount, and there
  is no network, so commands that fetch/diff against `origin` won't work.
  File-based generators (`make fix-repo`, `ruff`, codegen) are fine — serge
  does the staging/diffing itself, on the host.
- **The command picks from a fixed menu.** Callers send a command *string*;
  serge runs it only if it exactly matches an entry in
  `TASK_COMMAND_ALLOWLIST`. Callers cannot add arguments or pass arbitrary
  shell.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `TASK_API_ENABLED` | off | Master switch for the whole `/tasks` flow. |
| `TASK_COMMAND_ENABLED` | off | Enable command tasks specifically. |
| `TASK_COMMAND_ALLOWLIST` | — | Comma-separated exact command strings, e.g. `make fix-repo,make style`. A request naming anything else is rejected (403). |
| `TASK_COMMAND_IMAGE` | — | Docker image (deps baked in) for the `docker` backend. |
| `TASK_SANDBOX_BACKEND` | `auto` | `docker` \| `bwrap` \| `auto`. `auto` = docker when an image is set and the docker CLI is present, else bwrap. |
| `TASK_COMMAND_TIMEOUT` | `1800` | Per-command timeout (seconds). |
| `TASK_COMMAND_MEMORY` | — | Optional docker `--memory` cap (e.g. `4g`). |

The per-repo write opt-in (`task_write_enabled` on the repo's provider config)
is still required — command tasks need `Contents: write` + `Pull Requests:
write` just like LLM tasks. They do **not** require a usable LLM API key.

### The `bwrap` fallback

If you don't set an image (or set `TASK_SANDBOX_BACKEND=bwrap`), the command
runs under bubblewrap using **serge's own venv** — viable only when the
command needs no dependencies beyond what serge already has installed.
`make fix-repo` for transformers does need the `[quality]` toolchain, so use
the `docker` backend for it.

## Request shape

```jsonc
POST /tasks
Authorization: Bearer <github-actions-oidc-jwt>
{
  "command": "make fix-repo",          // must be in TASK_COMMAND_ALLOWLIST
  "base_ref": "main",                  // new_pr: branch to start from
  "context": "optional notes shown in the PR body",
  "output": {
    "mode": "new_pr",                  // new_pr | existing_pr
    "pr_number": null,                 // required for existing_pr (serge-owned branch only)
    "title": "Apply make fix-repo",    // optional
    "branch_prefix": "serge/fix"       // new_pr only
  }
}
→ 202 { "id": "<job id>", "url": "/tasks/owner/name/<id>" }
```

`existing_pr` mode re-runs the command on an existing serge fix branch and
pushes a follow-up commit — the same branch-ownership guard and follow-up loop
cap as the LLM flow apply.

## Security model

Same trust boundary as the LLM task flow, with stronger isolation for running
arbitrary repo build code:

- **No secrets in the sandbox.** The command gets a scrubbed env; the docker
  container additionally has no network, a read-only rootfs, all capabilities
  dropped, `no-new-privileges`, and a pids cap.
- **serge owns the git write.** The command only mutates a throwaway worktree;
  serge commits via the Git Data API. The installation token never enters the
  container.
- **Operator-controlled image.** `TASK_COMMAND_IMAGE` is set by the operator,
  never by the caller. The OIDC `repository` claim authorizes *which repo*,
  not *what image* runs.
- **Allowlist.** Callers select from a fixed set of commands; they cannot
  inject arguments or arbitrary shell.
- **The result is a PR a human reviews before merge.**
