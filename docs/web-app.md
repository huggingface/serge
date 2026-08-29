---
title: Staged Web App
nav_title: Staged review
---

The web app lets a signed-in reviewer start a review, watch the LLM stream its
draft, edit the summary and inline comments, discard noisy comments, and only
then publish the review to GitHub.

Reviews are published with the GitHub App identity. GitHub OAuth is used for
access control to the staging UI.

The web app deployment also hosts the optional, write-capable
[tasks flow](tasks-flow.md) (`POST /tasks`), which opens fix PRs from CI failure
reports. It is off unless `TASK_API_ENABLED` is set.

## Install

```bash
git clone https://github.com/huggingface/serge.git
cd serge
python -m venv .venv
source .venv/bin/activate
pip install -e '.[web]'
```

The web app reuses the GitHub App credentials from
[GitHub App webhook](github-app.md) and also needs a GitHub OAuth App.

```bash
export GITHUB_APP_ID=...
export GITHUB_PRIVATE_KEY_PATH=./private-key.pem
export GITHUB_OAUTH_CLIENT_ID=...
export GITHUB_OAUTH_CLIENT_SECRET=...
export GITHUB_OAUTH_CALLBACK_URL=http://localhost:8080/auth/callback
export WEB_SESSION_SECRET=$(openssl rand -hex 32)
export WEB_ALLOWED_USERS=octocat,hubot

reviewbot-web
```

Use `WEB_ALLOWED_ORG=org-a,org-b` instead of, or in addition to,
`WEB_ALLOWED_USERS`.

Set `DEV_NO_AUTH=1` only for local development.

## Provider Configs

The web app stores per-repository provider configs in SQLite. A provider config
chooses:

- provider: Hugging Face, OpenAI, Anthropic, or custom;
- API key;
- default model;
- repository pattern, either `owner/repo` or `owner/*`;
- users or orgs allowed to use the key.

Keys are write-only through the UI: they can be replaced, but not read back.
The most-specific matching config wins when a review is submitted.

## Review Flow

1. Open the New Review page.
2. Enter a PR URL or `owner/repo#123`.
3. Enter a trigger comment, for example `@askserge please review`.
4. Pick the provider and model. The model field is a dropdown of the models that
   provider serves: for Hugging Face, the tool-capable models on the
   [HF Inference Providers](https://router.huggingface.co) router; for the keyed
   providers, whatever their `/models` route advertises, listed server-side with
   a stored key you're authorized to use (the key never reaches the browser). It
   falls back to a free-text field when no list can be fetched.
5. Start the review and watch the stream.
6. Edit the summary and comments.
7. Publish or discard the draft.

The latest jobs are persisted in SQLite and can be reopened after a process
restart. Token and reasoning chunks are not replayed after completion to keep
stored history small.

## Webhook Surface

`reviewbot-web` also serves `POST /webhook`. In that mode, GitHub comment
events can kick off reviews that auto-publish to GitHub while still exposing
progress in the web UI.

## Storage and Cache

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `WEB_STORE_PATH` | `jobs.db` | SQLite path for jobs and provider configs |
| `WEB_JOB_RETENTION` | `25` | Number of recent jobs to retain |
| `WEB_CLONE_CACHE_DIR` | system temp dir | Shared bare clone cache |
| `WEB_CLONE_CACHE_TTL_SECONDS` | `604800` | Clone cache TTL |
| `WEB_CLONE_DEPTH` | `50` | Shallow fetch depth |

Point `WEB_CLONE_CACHE_DIR` at durable storage in production.

## Metrics

`GET /metrics` serves a Prometheus text exposition of the jobs the store still
holds. Unauthenticated, like `/healthz` — it is meant to be scraped in-cluster
over the pod port, and it carries no review content: job ids, the repo and PR
number, the model name, and counters.

Per finished job it exports turns, tool calls, input/output tokens, LLM seconds,
retries, and two numbers about how the budget was spent browsing:

| Metric | Meaning |
| ------ | ------- |
| `serge_job_repeat_calls` | Tool calls that re-ran an *earlier call verbatim* — what `TOOL_REPEAT_LIMIT` counts. |
| `serge_job_path_revisits` | Calls that re-opened a *path already opened*, counted per path as visits−1. A second `read_file` of the same file at a different line range is a revisit but not a verbatim repeat, and that is the shape that dominates in practice. |

Both have a nudge attached (`TOOL_REPEAT_LIMIT` / `TOOL_PATH_REVISIT_LIMIT`) and a separate cut-off budget (`TOOL_REPEAT_LIMIT` / `TOOL_PATH_TRIP_AFTER`), because they are different failures: one model is stuck on a single call, the other is browsing in circles.

The label that matters most is `stop_reason` on `serge_job_info`:

| `stop_reason` | The session ended because… |
| ------------- | -------------------------- |
| `answered` | the model decided it was done — the only value that means this |
| `input_token_cap` | `LLM_MAX_INPUT_TOKENS` was reached; the answer came from a tool-less final turn |
| `repeat_guard` | `TOOL_REPEAT_LIMIT` tripped — the model kept re-issuing one call verbatim |
| `path_revisit_guard` | `TOOL_PATH_TRIP_AFTER` tripped — the model kept re-opening files it had already opened |
| `blind_turn_cap` / `strict_tool_cap` / `absolute_ceiling` | a `TOOL_MAX_ITERATIONS` bound was reached |
| `chunk_input_token_cap` | a chunked review skipped chunks it could not afford |
| `no_llm_turns` | the job finished (or failed) without ever running the loop — e.g. reproduce-first classified the group ENVIRONMENT |

Every label on `serge_job_info` is immutable for the life of a job — `status` is
the session's outcome frozen when the loop ended, not the row's live status, so a
review a human later publishes does not fork into a second series and double its
row in a table.

Identity lives on `serge_job_info` alone (always `1`) and the numeric series are
keyed by `job_id` only, so a job's numbers don't fork into a new series when its
status changes. Join them back with `on(job_id) group_left(...)`:

```promql
max by (job_id) (serge_job_input_tokens)
  * on(job_id) group_left(repo, status, stop_reason) max by (job_id, repo, status, stop_reason) (serge_job_info)
```

The export is a **rolling window** over `WEB_JOB_RETENTION` jobs (exported as
`serge_job_retention`), not a history: when a job is pruned its series stops
being exported and goes stale, while the samples Prometheus already took stay
queryable for its full retention. So curling this endpoint tells you about the
last couple of days only — ask Prometheus for anything older.

The same record is returned in the `session` field of
`GET /tasks/{owner}/{repo}/{job_id}/status`, so a dispatcher polling its own task
can report why a group came back `no_fix` without a dashboard round trip.
