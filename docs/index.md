---
title: ai-reviewer
---

`ai-reviewer` reviews pull requests with an OpenAI-compatible LLM and posts
inline comments through GitHub's Pull Request Reviews API. The default persona
is Serge, and the default trigger is `@askserge`.

The reviewer reads the PR diff, applies repository-specific rules, optionally
uses read-only repository context tools, validates every proposed inline
comment against the diff, and publishes only comments that point to real diff
positions.

## Choose a Mode

| Mode | Use it when | Where it runs |
| ---- | ----------- | ------------- |
| [GitHub Action](github-action.md) | You want the fastest setup and per-repo control | GitHub Actions |
| [GitHub App webhook](github-app.md) | You want a hosted reviewer across many repos | Your server |
| [Web app](web-app.md) | You want to edit or discard LLM output before it reaches a PR | Your server |

Start with [Getting started](getting-started.md), then use the guide for the
deployment mode that fits your repository.

## Core Features

- OpenAI-compatible chat completion endpoints.
- Inline PR comments validated against actual diff lines.
- Trigger comments and follow-up replies on inline review comments.
- Repository rules from `.ai/review-rules.md`.
- Optional repository context from `.ai/context-script`.
- Optional helper tools from `.ai/review-tools.json`.
- Human-in-the-loop staged reviews through the web app.

## Documentation

- [Getting started](getting-started.md)
- [GitHub Action](github-action.md)
- [GitHub App webhook](github-app.md)
- [Staged web app](web-app.md)
- [Configuration](configuration.md)
- [Repository customization](repository-customization.md)
- [LLM providers](llm-providers.md)
- [How it works](how-it-works.md)
- [Security](security.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)
