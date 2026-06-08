---
title: Getting Started
nav_title: Quick Start
---

The quickest way to run `ai-reviewer` is the GitHub Action. It needs an LLM API
key, a workflow file, and a trigger comment on an open pull request.

## Prerequisites

- A GitHub repository.
- A token for an OpenAI-compatible chat completion endpoint.
- Optional but recommended: `.ai/review-rules.md` in the target repository.

## Fastest Setup

1. Add a repository secret named `LLM_API_KEY`.
2. Add the workflow from the [GitHub Action guide](github-action.md).
3. Comment `@askserge please review` on an open PR.

The Action reacts only when the comment author is a `MEMBER`, `OWNER`, or
`COLLABORATOR`. Comments from outside contributors are ignored.

## Forked PRs

GitHub does not pass repository secrets to workflows triggered from forks, and
the default `GITHUB_TOKEN` is usually read-only on forked PRs. That means the
Action cannot safely receive `LLM_API_KEY` or post review comments on forked
PRs.

For fork-heavy repositories, use the [GitHub App webhook](github-app.md) or
[web app](web-app.md) mode.

## Next Steps

- Add repo-specific review policy with
  [repository customization](repository-customization.md).
- Tune models and endpoints with [LLM providers](llm-providers.md).
- Review the full [configuration reference](configuration.md).
