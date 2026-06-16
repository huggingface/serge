---
title: Tasks flow (write-capable)
nav_title: Tasks
---

The **tasks flow** lets a GitHub Actions job ask `serge` to *produce a change*
on the repository — open a pull request with a fix, or push a follow-up commit
onto an existing serge-authored fix branch. It is the write-capable counterpart
to the read-only reviewer.

The canonical use case: CI runs a test suite, the tests fail, and the workflow
sends the failure report to `serge`, which opens a PR with a proposed fix. CI
then runs again on that PR. `serge` **never runs the test suite itself** — it is
a stateless patch producer; verification stays in the caller's CI.

```
CI runs tests  ──fail──▶  POST /tasks { instruction, context: <report> }
                              │
                              ▼
        serge: checkout base → agentic loop → patch → open PR  serge/fix-<id>
                              │
                              ▼
        CI runs on the PR  ──still failing?──▶  POST /tasks { output.mode: existing_pr, pr_number }
                              │
                              ▼
        serge: amend the fix branch with another commit
```

This endpoint is served by the [web app](web-app.md) deployment (`reviewbot-web`)
and is **off by default**. See [Security](security.md) and
[Security architecture](security-architecture.md) for the trust model.

## Enabling it

The tasks flow is a privilege escalation over the read-only reviewer: it
requires the backing GitHub App to hold **Contents: write** and
**Pull Requests: write**, and it is gated three ways.

1. **Deployment switch.** Set `TASK_API_ENABLED=1` on the web app.
2. **Per-repo opt-in.** In `/admin`, the repo's provider config must have
   *"Enable write-capable tasks"* checked. A config without it authorizes
   read-only reviews only.
3. **OIDC audience.** Callers must mint their OIDC token with the `aud` value
   serge expects (`TASK_OIDC_AUDIENCE`, default `serge`).

## Authentication: GitHub Actions OIDC

Authentication is **GitHub Actions OIDC** — there is no shared secret. The
calling workflow mints a short-lived, GitHub-signed JWT and serge verifies it
against GitHub's JWKS (`iss` / `aud` / `exp` / signature). serge authorizes the
task on the token's `repository` claim and will only ever act on that repo. A
leaked token is useless within minutes and is scoped to a single repository.

## Calling workflow

{% raw %}
```yaml
name: Auto-fix failing tests
on:
  workflow_run:
    workflows: [CI]
    types: [completed]

permissions:
  id-token: write   # required to mint the OIDC token

jobs:
  fix:
    if: github.event.workflow_run.conclusion == 'failure'
    runs-on: ubuntu-latest
    steps:
      - name: Request a fix from serge
        run: |
          TOKEN=$(curl -sSf \
            -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=serge" | jq -r .value)

          curl -sSf -X POST "$SERGE_URL/tasks" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" \
            -d @- <<JSON
          {
            "repo": "${{ github.repository }}",
            "base_ref": "${{ github.event.workflow_run.head_branch }}",
            "instruction": "Fix the failing tests described below.",
            "context": $(jq -Rs . < test-report.txt),
            "output": { "mode": "new_pr", "branch_prefix": "serge/fix" }
          }
          JSON
        env:
          SERGE_URL: https://serge.example.com
```
{% endraw %}

## API

```
POST /tasks
Authorization: Bearer <github-actions-oidc-jwt>
Content-Type: application/json
```

| Field | Required | Description |
| ----- | -------- | ----------- |
| `repo` | optional | `owner/name`. If present, must match the OIDC `repository` claim (the claim is authoritative). |
| `base_ref` | optional | Branch the work starts from in `new_pr` mode. Default `main`. |
| `instruction` | **required** | Trusted intent from the workflow, e.g. "Fix the failing tests below." |
| `context` | optional | The failure report / logs. **Untrusted** — treated as data, fed to the prompt, never as instructions. |
| `output.mode` | optional | `new_pr` (default) or `existing_pr`. |
| `output.pr_number` | required for `existing_pr` | The serge-authored fix PR to push onto. |
| `output.title` | optional | PR title. The LLM proposes one if omitted. |
| `output.branch_prefix` | optional | `new_pr` branch prefix. Must live in the `serge/` namespace. Default `serge/fix`. |

Response `202`:

```json
{ "id": "<job id>", "repo": "owner/name", "mode": "new_pr", "url": "/tasks/owner/name/<id>" }
```

Follow the run live at `url` (SSE console) in a browser, the same machinery the
review pages use.

## How serge writes safely

- **The LLM only proposes a patch** (a unified diff plus a PR title/body). serge
  applies it, commits, and opens the PR itself — the model never touches push
  credentials. Same trust pattern as reviews (the LLM proposes comments; serge
  publishes).
- **Commits go through the GitHub Git Data API** (`create_blob` → `create_tree`
  → `create_commit` → `create_ref` → `create_pull_request`), not `git push`. The
  installation token never enters the sandbox or the worktree's git remote, which
  stays network-isolated exactly as it is for reviews.
- **Branch-ownership guard.** `existing_pr` mode is valid *only* for serge-owned
  fix branches (`serge/*`). serge never pushes to an arbitrary head branch a
  caller names.
- **Follow-up loop cap.** serge counts its own commits on a fix branch and stops
  after `TASK_MAX_FOLLOWUPS` (default `5`) so a misconfigured workflow cannot
  burn tokens forever.
- **Untrusted context.** The `context`/logs are a prompt-injection vector (same
  class as a PR body). The prompt marks them untrusted, the model can only emit a
  patch, and the result is a PR a human reviews before merge.

## Configuration

| Env var | Default | Description |
| ------- | ------- | ----------- |
| `TASK_API_ENABLED` | `false` | Master switch for `POST /tasks`. |
| `TASK_OIDC_ISSUER` | `https://token.actions.githubusercontent.com` | OIDC issuer (override for GHES). |
| `TASK_OIDC_AUDIENCE` | `serge` | `aud` value serge requires on the token. |
| `TASK_LLM_MAX_TOKENS` | unset | Task-only completion-token cap. Unset means tasks use `LLM_MAX_TOKENS`; normal reviews are unchanged. |
| `TASK_LLM_MAX_INPUT_TOKENS` | unset | Task-only cumulative input-token cap. Unset means tasks use `LLM_MAX_INPUT_TOKENS`; normal reviews are unchanged. |
| `TASK_TOOL_MAX_ITERATIONS` | unset | Task-only tool-loop cap. Unset means tasks use `TOOL_MAX_ITERATIONS`; normal reviews are unchanged. |
| `TASK_MAX_FOLLOWUPS` | `5` | Max serge-authored commits per fix branch. Set `0` to disable the cap. |
| `TASK_MAX_WORKERS` | `2` | Concurrent task workers (separate pool from reviews). |
