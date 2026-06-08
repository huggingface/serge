---
title: Development
---

## Install Locally

```bash
git clone https://github.com/huggingface/ai-reviewer.git
cd ai-reviewer
python -m venv .venv
source .venv/bin/activate
pip install -e '.[web]'
```

## Entry Points

| Command | Purpose |
| ------- | ------- |
| `reviewbot-action` | GitHub Action runner. Reads `GITHUB_EVENT_PATH`. |
| `reviewbot-app` | Flask GitHub App webhook server. |
| `reviewbot-web` | FastAPI staged review web app. |

## Run Tests

```bash
pytest
```

The tests cover trigger gating, config parsing, LLM response handling, review
publishing, context scripts, helper tools, clone cache behavior, web app
webhooks, provider configs, and persistence.

## Useful Files

| Path | Purpose |
| ---- | ------- |
| `action.yml` | GitHub Action metadata and inputs. |
| `.env.example` | Server-mode environment template. |
| `reviewbot/action_runner.py` | Action entry point. |
| `reviewbot/app.py` | Flask webhook app. |
| `reviewbot/webapp.py` | FastAPI staged review app. |
| `reviewbot/reviewer.py` | Review preparation, validation, publishing, and follow-ups. |
| `reviewbot/tools.py` | Built-in and repo helper tools. |
| `reviewbot/context_script.py` | Context-script execution and parsing. |
| `reviewbot/store.py` | SQLite job and provider-config store. |

## Local Web App

For local UI development without OAuth:

```bash
export DEV_NO_AUTH=1
export GITHUB_APP_ID=...
export GITHUB_PRIVATE_KEY_PATH=./private-key.pem
reviewbot-web
```

Use `DEV_NO_AUTH=1` only on a local machine.
