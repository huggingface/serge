---
title: GitHub Action
nav_title: GH Action
---

The GitHub Action runs inside the target repository's workflow. It is the
lowest-friction way to try `ai-reviewer` because it does not require a server
or GitHub App.

## Workflow

Create `.github/workflows/ai-review.yml`:

{% raw %}
```yaml
name: AI PR Review
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  review:
    if: |
      contains(github.event.comment.body, '@askserge') &&
      (github.event.comment.author_association == 'MEMBER' ||
       github.event.comment.author_association == 'OWNER' ||
       github.event.comment.author_association == 'COLLABORATOR')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: refs/pull/${{ github.event.issue.number || github.event.pull_request.number }}/head
        continue-on-error: true

      - uses: huggingface/ai-reviewer@main
        with:
          llm_api_key: ${{ secrets.LLM_API_KEY }}
          llm_api_base: ${{ secrets.LLM_API_BASE || 'https://api.openai.com/v1' }}
```
{% endraw %}

Pin `huggingface/ai-reviewer` to a tag or commit SHA when you need
reproducible behavior.

## Triggering Reviews

Comment on an open PR:

```text
@askserge please review
```

You can add instructions after the trigger:

```text
@askserge focus on the new auth flow and ignore generated files
```

For inline follow-up questions, reply to an existing PR review comment with the
trigger. The reviewer answers in the same review-comment thread.

## Repository Checkout and Tools

When `repo_checkout_path` is set, the reviewer can use read-only tools rooted
at the checkout:

- `read_file`
- `list_dir`
- `grep`
- `fetch_url` for `https://huggingface.co/*`

The Action defaults `repo_checkout_path` to `github.workspace`. Add an
`actions/checkout` step if you want the reviewer to browse files beyond the
diff. Set `repo_checkout_path: ''` to disable tools entirely.

## Inputs

The Action inputs mirror the environment variables in
[Configuration](configuration.md). The most common inputs are:

| Input | Default | Description |
| ----- | ------- | ----------- |
| `llm_api_key` | required | Bearer token for the LLM endpoint |
| `llm_api_base` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `llm_model` | auto-discovered | Model identifier |
| `mention_trigger` | `@askserge` | Phrase that triggers reviews |
| `review_rules_path` | `.ai/review-rules.md` | Rules file on the default branch |
| `context_script_path` | `.ai/context-script` | Optional context script |
| `helper_tools_path` | `.ai/review-tools.json` | Optional helper tool config |

See `action.yml` for the source-of-truth input list.

## Forked PR Limitation

The Action is not a good fit for forked PRs because secrets are not available
and the workflow token may not be able to write review comments. Use a
[GitHub App](github-app.md) or [web app](web-app.md) deployment for repositories
that receive many forked contributions.
