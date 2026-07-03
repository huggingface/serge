# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub App mode is documented as the zero-config default: installed repos need
  no workflow file or secret, with instructions for overriding gating via an
  explicit workflow and a note on the forked-PR limitation.

### Fixed

- The trigger gate now ignores comments authored by a bot, so the App never
  reacts to its own output (no self-trigger loops in App mode).

## [0.1.0] - 2026-06-17

First public release of `serge`, a GitHub-native AI code reviewer for any
OpenAI-compatible LLM.

### Added

#### Reviewing

- Pull request reviews that read the diff and post inline comments through
  GitHub's Pull Request Reviews API.
- Every proposed inline comment is validated against real diff positions before
  publishing; comments that don't map to a diff line are dropped.
- Trigger comments (`@askserge please review`) and follow-up replies on inline
  review comments.
- The model name used for a review is included in the review output.

#### Deployment modes

- **GitHub Action** — per-repo setup driven from GitHub Actions.
- **GitHub App webhook** — hosted reviewer across many repositories, with
  installation tokens read from the app database.
- **Web app** — human-in-the-loop staged reviews that can be edited or discarded
  before they reach a PR, with webhook calls surfaced in the app UI.

#### Repository customization

- Repository review rules from `.ai/review-rules.md`.
- Optional read-only repository context from `.ai/context-script`.
- Optional helper tools from `.ai/review-tools.json`.

#### LLM providers

- Support for any OpenAI-compatible chat completion endpoint.
- Hugging Face inference provider selection in the web UI.
- Configurable max tokens exposed in the UI.
- Optional context compression via [`headroom`](https://pypi.org/project/headroom-ai/),
  enabled at runtime with `HEADROOM_COMPRESS=1`.

#### Tasks flow (write-capable)

- Optional `/tasks` flow: CI posts a failure report and serge opens a fix PR.
- `existing_pr` mode appends follow-up commits to an existing fix branch instead
  of opening a duplicate PR, enabling a CI-retry loop.
- Loop-cap safety that counts serge-authored commits on a branch to prevent
  infinite retry loops.
- Original failure report included in the task PR body.

#### Security & operations

- Fork and PR code isolated from the review host.
- Explicit handling for reviews triggered on forked PRs.
- Dependabot weekly bumps for GitHub Actions.
- Public documentation site under `docs/`.

[0.1.0]: https://github.com/huggingface/serge/releases/tag/v0.1.0
