---
title: serge
layout: default
---

![Serge Code Reviewer]({{ "/assets/serge-hero.png" | relative_url }})

`serge` reviews pull requests with an OpenAI-compatible LLM and posts
inline comments through GitHub's Pull Request Reviews API. The default persona
is Serge, and the default trigger is `@askserge`.

The reviewer reads the PR diff, applies repository-specific rules, optionally
uses read-only repository context tools, validates every proposed inline
comment against the diff, and publishes only comments that point to real diff
positions.

serge also has a write-capable side. The [tasks flow](tasks-flow.md) lets CI
hand serge a failure report and get back a pull request with a proposed fix —
the same trust pattern as reviews (the LLM only proposes; serge applies and
publishes). It is off by default and gated behind GitHub Actions OIDC.

## What serge does

| Capability | What it produces | Guide |
| ---------- | ---------------- | ----- |
| **Review** | Inline PR comments validated against the diff | [How it works](how-it-works.md) |
| **Fix** (write-capable) | A fix PR or follow-up commit from a CI failure report | [Tasks flow](tasks-flow.md) |

## Choose a Mode

| Mode | Use it when | Where it runs |
| ---- | ----------- | ------------- |
| [GitHub Action](github-action.md) | You want per-repo control via a workflow file | GitHub Actions |
| [GitHub App webhook](github-app.md) | You want a hosted reviewer across many repos with no per-repo workflow | Your server |
| [Web app](web-app.md) | You want to edit or discard LLM output before it reaches a PR | Your server |

Start with [Getting started](getting-started.md), then use the guide for the
deployment mode that fits your repository. New to the UI? Take the
[web app tour](web-app-tour.md).

## Core Features

- OpenAI-compatible chat completion endpoints.
- Inline PR comments validated against actual diff lines.
- Trigger comments and follow-up replies on inline review comments.
- Repository rules from `.ai/review-rules.md`.
- Optional repository context from `.ai/context-script`.
- Optional helper tools from `.ai/review-tools.json`.
- Human-in-the-loop staged reviews through the web app.
- Write-capable [tasks flow](tasks-flow.md): CI sends a failure report, serge opens a fix PR (off by default, OIDC-gated).

## Documentation

- [Getting started](getting-started.md)
- [GitHub Action](github-action.md)
- [GitHub App webhook](github-app.md)
- [Staged web app](web-app.md)
- [Web app tour](web-app-tour.md)
- [Tasks flow (write-capable)](tasks-flow.md)
- [Configuration](configuration.md)
- [Repository customization](repository-customization.md)
- [LLM providers](llm-providers.md)
- [Architecture](architecture.md)
- [How it works](how-it-works.md)
- [Security](security.md)
- [Troubleshooting](troubleshooting.md)
- [Development](development.md)
