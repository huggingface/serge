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
| `output.gpu` | optional | Target GPU flavor: `single-gpu` or `multi-gpu` (aliases `simple`/`single`, `multi`). Selects the GPU pod placement profile (`TASK_K8S_GPU_PROFILES`). Omit for a CPU task. |
| `output.tests` | optional | Array of pytest node-IDs (e.g. `tests/models/x/test_y.py::TestZ::test_w`) the patch **must** pass. With `TASK_TEST_COMMAND` set, serge runs them in the pod and opens a PR only if they pass — see the test gate below. |

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

## GPU test gate (run the target tests before opening a PR)

By default serge is a stateless patch producer and verification is left to the
caller's CI *after* the PR exists. For integration/GPU failures you can instead
have serge **prove the fix before opening a PR** (issue #20):

1. The caller passes the failing `output.tests` (pytest node-IDs) and the
   `output.gpu` flavor. Both are **trusted intent** (like `instruction`) — the
   node-IDs are validated (no leading `-`, no control chars) and passed as argv,
   never a shell string.
2. serge launches the task's runner pod **on the matching GPU node pool**
   (`TASK_K8S_GPU_PROFILES[output.gpu]` → nodeSelector + tolerations +
   `nvidia.com/gpu` reservation). A `gpu` with no configured profile is rejected.
3. Inside the pod, after the patch applies (and the normalizer passes), serge
   runs `TASK_TEST_COMMAND` + the node-IDs. A failure is fed back to the model to
   correct the patch (up to `TASK_TEST_MAX_RETRIES`). The model can also call the
   `run_tests` tool mid-loop to check a candidate patch.
4. **Fail-closed guarantee:** serge opens/appends a PR **only if those tests
   pass**. A test failure *or* the test infra being unavailable blocks the PR;
   the task ends `no_fix` with a "targeted tests did not pass" message.

The GPU test pod must reach the HF hub to download model weights — add the hub
hosts to the egress allowlist (`taskExecution.kubernetes.egress.allowDomains`,
e.g. `^huggingface\.co$` and `(^|\.)hf\.co$`). GPU tasks therefore run under a
slightly wider egress boundary than normal task pods (git + LLM + callback + HF
hub); still no arbitrary internet. Build the runner image against a CUDA
transformers base (see `docker/Dockerfile.task-runner`); the cluster needs the
NVIDIA device plugin and a GPU node pool matching the profiles.

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
| `TASK_TEST_COMMAND` | unset | Argv the request's `output.tests` node-IDs are appended to (e.g. `python -m pytest -q`). Unset = no test gate. Operator/repo config, never request-supplied. |
| `TASK_TEST_TIMEOUT` | `3600` | Wall-clock cap (seconds) on one test-gate run. |
| `TASK_TEST_MAX_RETRIES` | `2` | Corrective re-prompts when the tests reject the patch (the loop budget while the gate is active). |
| `TASK_K8S_GPU_PROFILES` | unset | JSON map: GPU flavor → `{ node_selector, tolerations, gpu_resource, gpu_count, memory }`. Selects per-task GPU pod placement (kubernetes backend). |
