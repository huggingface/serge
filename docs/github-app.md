---
title: GitHub App Webhook
nav_title: Webhook
---

GitHub App mode runs a webhook server that listens for trigger comments, calls
the LLM, and posts reviews with a GitHub App installation token. It is useful
when you want one hosted reviewer to serve many repositories.

## Create the GitHub App

In GitHub, open **Settings -> Developer settings -> GitHub Apps -> New GitHub
App**.

Use these permissions:

| Permission | Access |
| ---------- | ------ |
| Pull requests | Read and write |
| Contents | Read |
| Issues | Read |
| Metadata | Read |

Subscribe to:

- Issue comment
- Pull request review comment

Set the webhook URL to:

```text
https://<your-host>/webhook
```

Generate a webhook secret, download the private key, note the App ID, and
install the App on the repositories you want reviewed.

## Run the Server

```bash
git clone https://github.com/huggingface/ai-reviewer.git
cd ai-reviewer
python -m venv .venv
source .venv/bin/activate
pip install .
cp .env.example .env
```

Fill in:

```bash
GITHUB_APP_ID=...
GITHUB_PRIVATE_KEY_PATH=./private-key.pem
GITHUB_WEBHOOK_SECRET=...
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=...
```

Run the webhook app:

```bash
reviewbot-app
```

Production deployments should run `reviewbot.app:app` behind a WSGI server
such as Gunicorn and expose it over HTTPS.

## Triggering

GitHub sends comment events to `POST /webhook`. A review starts only when:

- the event is `issue_comment` or `pull_request_review_comment`;
- the action is `created`;
- the comment contains the configured trigger, default `@askserge`;
- the author association is `MEMBER`, `OWNER`, or `COLLABORATOR`;
- the PR is open.

Inline review-comment mentions trigger the follow-up flow instead of a full PR
review.

## Webhook Concurrency

`WEBHOOK_MAX_WORKERS` controls how many review workers can run concurrently.
The default is `2`.
