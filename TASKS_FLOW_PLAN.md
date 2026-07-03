# Serge Tasks Flow — Design & Implementation Plan

Status: **implemented.** User-facing docs live in
[`docs/tasks-flow.md`](docs/tasks-flow.md); this file is the design record.

## 1. Motivation

Serge today is a **read-only PR reviewer**: a GitHub webhook (an `@askserge`
mention) triggers an agentic LLM loop that browses the PR head with read-only
tools and publishes a *review* (comments) via the GitHub Reviews API. It never
writes code, never pushes branches, never opens PRs.

We want a second capability: an external agent — running inside a **GitHub
Actions** job — computes a report (e.g. failing tests) and asks serge to
**produce a change on the repo** (open a PR with a fix, or push a follow-up
commit to an existing serge-authored PR).

This is intentionally **generic**: the intake is "instruction + context →
contribution to a repo", not specifically "fix failing tests". Fixing failing
tests is just the first task kind.

## 2. Key decision: serge does NOT run tests

Test execution stays where it already happens — the **caller's GitHub Actions
runner / the repo's CI**. serge is a **stateless patch producer**:

```
report in  →  patch out (as a PR or a commit)
```

The feedback loop is closed by **GitHub CI**, not by serge:

```
CI workflow runs tests (in the runner)            ← test execution lives here
        │ tests fail
        ▼
POST /tasks (OIDC)  { instruction, context:<failure report>, output: new_pr }
        ▼
serge: checkout base_ref → agentic loop → patch → open PR  serge/fix-<id>
        ▼
CI runs on that PR (in the runner)                ← verification lives here
        │ still failing?
        ▼
POST /tasks  { output: existing_pr, pr_number:<fix PR>, context:<new report> }
        ▼
serge: checkout the fix PR head → amend → push another commit
        ▼
…repeat until green or the workflow gives up…
```

Consequences:
- **No sandbox test-run, no `verify.command`, no per-repo test-command config.**
  The "how does serge run this repo's suite" problem (the scaling + security
  headache) is gone.
- serge's per-task cost ≈ one review (checkout + LLM loop + git write), not a
  full CI run. This is what makes it scale.
- serge still uses the lightweight checkout + **read-only** browse tools
  (`read_file` / `grep` / `list_dir`) so the LLM can understand the code. It
  just never executes the test suite.

> **Narrow, opt-in reversal (in-loop normalize validation).** This "serge runs
> no per-repo command" stance is deliberately and narrowly reversed by
> [normalize validation](docs/tasks-normalize.md): when `TASK_NORMALIZE_COMMAND`
> is configured, serge runs the repo's own *normalizer* (e.g. `make style &&
> make fix-repo`) in a network-isolated sandbox **inside the LLM loop** — it
> applies each candidate patch, runs the normalizer, and feeds any failure back
> to the model so it corrects the patch, then commits the applied+normalized
> result. This is still **not a test run**: the normalizer is a formatter/codegen
> consistency gate (`make fix-repo`), not the repo's test suite — serge still
> never runs tests; CI still owns verification. It stays off by default, so the
> repo-agnostic posture above holds for any repo that doesn't opt in. After the
> retry budget is exhausted (or if the sandbox is unavailable) serge falls back
> to committing the raw patch, so a fix is never lost. See
> `SERGE_NORMALIZE_PLAN.md` for the design.

### The PR branch is the iteration state

On a follow-up, serge does `acquire_ref(<fix-branch>)`. That branch already
contains its previous attempt, so serge sees "what I tried last time" (the
branch diff) plus "what's still failing" (the new report) and amends. The PR
branch carries continuity — serge needs almost no persisted task state across
iterations. (`result_json = {pr_number, branch}` is stored on the job for the
journal / linking, but is not load-bearing.)

## 3. Authentication: GitHub Actions OIDC

Scope is **GitHub Actions only**. Auth is **Actions OIDC** — no shared secret.

- The calling workflow declares `permissions: id-token: write` and mints a
  short-lived, GitHub-signed JWT at job time.
- serge verifies the JWT against GitHub's JWKS, checking
  `iss` / `aud` / `exp` / signature.
- serge **authorizes the task on the token's `repository` claim** — it will
  only act on the repo named in the token. This gives per-repo scoping for free
  (a leaked token is useless within minutes and only ever scoped to one repo).

Why OIDC over a static `TASK_API_TOKEN`:
- The endpoint is **write-capable** (opens PRs). A short-lived, repo-scoped,
  GitHub-issued credential is the right trust level; a long-lived shared secret
  is not.
- Works on **GitHub-hosted and self-hosted** runners alike — the token is
  issued by GitHub's control plane, not the runner machine, so a self-hosted
  runner cannot forge the claims.

Config knobs (make the issuer configurable for GHES / self-hosted):
- `TASK_API_ENABLED` (default off)
- `TASK_OIDC_ISSUER` (default `https://token.actions.githubusercontent.com`)
- `TASK_OIDC_AUDIENCE` (the `aud` serge requires, e.g. `serge`)

serge verifies in its **main process** (which already reaches the GitHub API and
LLM providers), not the network-isolated sandbox, so `--unshare-net` does not
block JWKS fetches.

## 4. API surface

A machine-facing endpoint, distinct from the GitHub webhook (`/webhook`) and
from the human OAuth UI:

```
POST /tasks
Authorization: Bearer <github-actions-oidc-jwt>
{
  "repo": "owner/name",            # must match OIDC `repository` claim
  "base_ref": "main",             # branch the work starts from (new_pr mode)
  "instruction": "Fix the failing tests described below.",
  "context": "<failure report + logs>",   # UNTRUSTED — fed to the prompt
  "output": {
    "mode": "new_pr",             # new_pr | existing_pr
    "pr_number": null,            # required + serge-owned branch only for existing_pr
    "title": "...",               # optional; LLM proposes if omitted
    "branch_prefix": "serge/fix"  # new_pr only
  }
}
→ 202 { "id": "<job id>", "url": "/tasks/owner/name/<id>" }   # follow live via SSE
```

Reuses the existing SSE streaming + journal machinery (a task is a
`Job(kind="task")`).

## 5. Git-write mechanism (done safely)

This is the real inversion of serge's read-only posture, so the mechanism
matters:

- **The LLM only proposes a patch** (unified diff + PR title/body). serge
  applies, commits, and opens the PR itself — the model never touches push
  credentials. Same trust pattern as reviews (LLM proposes comments; serge
  publishes).
- **Commit via the GitHub Git Data API**, not `git push` with an embedded
  token. After the LLM patch is applied to the worktree, serge reads the
  changed files and uploads them through new `GitHubClient` methods:
  `create_blob` → `create_tree` → `create_commit` → `create_ref` →
  `create_pull_request`. **Credentials never enter the sandbox or the
  worktree's git remote**; the sandbox stays network-isolated exactly as today.
- For `existing_pr` mode, the commit target is the PR's head branch instead of
  a new branch.

## 6. Security model

- **New GitHub App permissions required:** `Contents: write` +
  `Pull Requests: write`. This is a privilege escalation over today's read-only
  App, so it is **opt-in per repo** (a flag on `provider_config`, off by
  default).
- **`context` / logs are untrusted** (prompt-injection vector, same class as a
  PR body today). Mitigations: the LLM can only emit a patch; serge owns the git
  write; the result is a PR a human reviews before merge. Mark `context`
  explicitly untrusted in the prompt.
- **Branch ownership guard:** `existing_pr` mode is valid **only for
  serge-owned fix branches** (`serge/fix-*`). serge must never push to an
  arbitrary head branch a caller names, otherwise the OIDC `repository` claim
  would authorize writing to *any* PR in the repo.
- **Loop cap:** the caller owns the loop, but serge enforces a low default cap
  on follow-ups per fix branch (e.g. count serge-authored commits on it) so a
  misconfigured workflow cannot burn tokens forever.

## 7. Reused vs. new components

Reused as-is:
- `Job` + SQLite store + SSE streaming + journal.
- `provider_configs` repo matching (`find_provider_config_for_repo`) to pick the
  LLM key for the task.
- `_run_agentic_loop` and the read-only browse tools + sandbox (for the LLM to
  read code — NOT to run tests).
- GitHub App installation-token auth (`installation_token`,
  `installation_id_for_repo`) for the write operations.

New:
- OIDC verification.
- Git Data API write methods + `create_pull_request`.
- `clone_cache.acquire_ref()` (today: PR head only).
- The `/tasks` endpoint + task worker + task prompts.

## 8. File-level change list

| File | Change |
|---|---|
| `reviewbot/github_client.py` | `create_blob`, `create_tree`, `create_commit`, `create_ref`, `create_pull_request` (Git Data + Pulls API). |
| `reviewbot/clone_cache.py` | `acquire_ref(owner, repo, ref)` — checkout an arbitrary branch (currently only `pull/N/head`). |
| `reviewbot/config.py` | `TASK_API_ENABLED`, `TASK_OIDC_ISSUER`, `TASK_OIDC_AUDIENCE`. |
| `reviewbot/oidc.py` *(new)* | JWKS fetch + cache; verify `iss`/`aud`/`exp`/signature; return claims. |
| `reviewbot/tasks.py` *(new)* | `TaskRequest`, `build_task_request()`, `prepare_task()` (loop → patch + PR meta), `publish_task()` (new PR or commit-to-existing serge branch). |
| `reviewbot/prompts.py` | Task system/user prompts: output = unified diff + PR title/body; mark `context` untrusted. |
| `reviewbot/store.py` | `kind` (`review`/`task`), `task_spec_json`, `result_json` columns on `jobs`; per-repo write opt-in flag on `provider_configs`. |
| `reviewbot/webapp.py` | `POST /tasks` route; OIDC auth; authorize on `repository` claim; `Job(kind="task")`; dedicated `_TASK_POOL`; reuse SSE + journal + `/tasks/...` view routes. |
| GitHub App config | Add Contents: write + Pull Requests: write; per-repo opt-in. |

## 9. Phased implementation

**Phase 1 — Git-write foundation** *(no LLM, no API — prove the write path)*
- `GitHubClient` Git Data + Pulls methods.
- `clone_cache.acquire_ref()`.
- New App permissions; per-repo opt-in flag.
- Throwaway script: hand-written patch on a real repo → branch → commit → PR.
  Validates end-to-end before any LLM is involved.

**Phase 2 — Task intake + OIDC**
- `config.py` task knobs; `oidc.py` verification helper.
- `webapp.py` `POST /tasks` route; authorize on `repository` claim;
  `Job(kind="task")`; `_TASK_POOL`.
- `store.py` columns; reuse `find_provider_config_for_repo` for the LLM key.

**Phase 3 — Agentic task loop**
- `tasks.py`: `prepare_task()` (loop → patch + PR meta), `publish_task()`
  (new PR or commit to existing serge fix branch per `output.mode`).
- `prompts.py`: task prompts, `context` untrusted.
- Output goes straight to a PR/commit; **CI validates, serge does not.**
- Enforce branch-ownership guard + follow-up loop cap.

## 10. Decided questions (record)

- **Output:** a new PR (first call) or a commit onto serge's existing fix branch
  (follow-up). Never a commit onto an arbitrary caller-named branch.
- **Verification:** done by the caller's CI, **not** by serge. No sandbox test
  runs.
- **Auth:** GitHub Actions OIDC, authorized on the `repository` claim. No static
  token.
- **Scope:** GitHub Actions callers only.
