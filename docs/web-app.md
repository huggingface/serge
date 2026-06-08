---
title: Staged Web App
nav_title: Staged review
---

The web app lets a signed-in reviewer start a review, watch the LLM stream its
draft, edit the summary and inline comments, discard noisy comments, and only
then publish the review to GitHub.

Reviews are published with the GitHub App identity. GitHub OAuth is used for
access control to the staging UI.

## Install

```bash
git clone https://github.com/huggingface/ai-reviewer.git
cd ai-reviewer
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
4. Pick the provider and model.
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
