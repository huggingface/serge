# ai-reviewer

Reviews pull requests with any **OpenAI-compatible** LLM and posts **inline
comments** on the diff via GitHub's Pull Request Reviews API. The bot's
persona is **Serge**; you invoke it by mentioning `@askserge` in a PR comment.
The web app mode adds **phased reviews** — you validate and edit the LLM's
draft as a human before it's posted.

It runs in three modes off the same codebase. Pick one:

| Mode | Best for | Infra |
| ---- | -------- | ----- |
| **GitHub Action** | Trying it out, per-repo control on same-repo PRs | None (runs on Actions) |
| **GitHub App** | Many repos, auto-review on every mention | A hosted webhook server |
| **Web app** | Human-in-the-loop: edit a review before posting | A hosted server + OAuth |

## Setup

### Mode 1 — GitHub Action (quickest)

Add an LLM key as a repo secret (**Settings → Secrets and variables →
Actions**): `LLM_API_KEY`, and optionally `LLM_API_BASE`.

Drop this into `.github/workflows/ai-review.yml` in the repo you want reviewed:

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
    # Only react to @askserge from a member/owner/collaborator on an open PR.
    if: |
      contains(github.event.comment.body, '@askserge') &&
      (github.event.comment.author_association == 'MEMBER' ||
       github.event.comment.author_association == 'OWNER' ||
       github.event.comment.author_association == 'COLLABORATOR')
    runs-on: ubuntu-latest
    steps:
      - uses: huggingface/ai-reviewer@main   # pin to a tag or SHA for stability
        with:
          llm_api_key: ${{ secrets.LLM_API_KEY }}
          llm_api_base: ${{ secrets.LLM_API_BASE || 'https://api.openai.com/v1' }}
```

Then comment `@askserge please review` on any open PR. `issues: write` is
needed so the bot can react to your comment and post error messages.

**Forked PRs:** GitHub does not pass repository secrets to workflows triggered
from forks, and the `GITHUB_TOKEN` is usually read-only. That means the Action
cannot safely receive `LLM_API_KEY` or post review comments on forked PRs. For
fork-heavy repositories, use the GitHub App or Web app modes instead.

### Mode 2 — GitHub App webhooks

GitHub App webhooks are served by the same `reviewbot-web` FastAPI app used
for staged reviews. GitHub sends comment events to `POST /webhook`; the app
verifies the webhook, calls the LLM, and posts back to GitHub with the App
installation token. No separate `reviewbot-app` process is needed for the
hosted deployment.

1. **Create the App** (Settings → Developer settings → GitHub Apps → New):
   - Permissions: Pull requests **R/W**, Contents **R**, Issues **R**, Metadata **R**
   - Subscribe to events: **Issue comment**, **Pull request review comment**
   - Webhook URL `https://<your-host>/webhook`, set a webhook secret
   - Download the private key, note the **App ID**, install on your repos

2. **Configure and run:**

   ```bash
   git clone https://github.com/huggingface/ai-reviewer.git
   cd ai-reviewer
   python -m venv .venv && source .venv/bin/activate
   pip install .
   cp .env.example .env && $EDITOR .env   # fill in App + LLM credentials
   set -a; source .env; set +a

   pip install -e '.[web]'
   reviewbot-web                                  # http://localhost:8080
   ```

   Required env: `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH` (or
   `GITHUB_PRIVATE_KEY`), `GITHUB_WEBHOOK_SECRET`, `LLM_API_BASE`, `LLM_API_KEY`.

Expose the server over HTTPS and point the App's webhook at
`https://<your-host>/webhook`. For local testing, tunnel with
[smee.io](https://smee.io) or `cloudflared`.

### Mode 3 — Web app (review before posting)

A signed-in user starts a review from a form, watches the LLM stream live,
then edits the summary and per-comment text (or drops comments) before
publishing. Reviews are still posted under the GitHub App identity — OAuth is
only for access control. Per-repo LLM provider keys are stored in the app's
database, managed from an admin page.

```bash
pip install -e '.[web]'
# Reuses the Mode 2 GitHub App, plus a separate GitHub OAuth App:
export GITHUB_APP_ID=... GITHUB_PRIVATE_KEY_PATH=./private-key.pem
export GITHUB_OAUTH_CLIENT_ID=... GITHUB_OAUTH_CLIENT_SECRET=...
export GITHUB_OAUTH_CALLBACK_URL=http://localhost:8080/auth/callback
export WEB_SESSION_SECRET=$(openssl rand -hex 32)
export WEB_ALLOWED_USERS=octocat,hubot        # or WEB_ALLOWED_ORG=acme
reviewbot-web                                 # http://localhost:8080
```

Set `DEV_NO_AUTH=1` to skip OAuth for local clicking-around (never in
production). The repo also ships a single-VM EC2 bootstrap in
[`aws/`](aws/README.md).

## Configuration

All modes share the same settings — as **Action inputs** (Mode 1) or
**environment variables** (Modes 2/3). The common ones:

| Env var / input | Default | Notes |
| --------------- | ------- | ----- |
| `LLM_API_KEY` / `llm_api_key` | — | **Required.** Bearer token |
| `LLM_API_BASE` / `llm_api_base` | `https://api.openai.com/v1` | With or without trailing `/v1` |
| `LLM_MODEL` / `llm_model` | first id from `/models` | Auto-discovered if unset |
| `MENTION_TRIGGER` | `@askserge` | Phrase that triggers a review |
| `REVIEW_EVENT` | `COMMENT` | Fallback if the LLM omits one |
| `MAX_DIFF_CHARS` | `200000` | Diffs larger than this are truncated |
| `REVIEW_RULES_PATH` | `.ai/review-rules.md` | Repo rules, read from default branch |
| `HEADROOM_COMPRESS` | `false` | Compress context before each LLM call (see below) |

See [`action.yml`](action.yml) and [`.env.example`](.env.example) for the full
list (billing routing, streaming, reasoning effort, browse-tool limits, etc.).

### Context compression (optional)

Long agentic reviews accumulate token-heavy tool outputs (file reads, grep
dumps) and assistant turns. Set `HEADROOM_COMPRESS=true` to compress that
context with [headroom](https://github.com/chopratejas/headroom) before each
LLM call. It's an opt-in extra — install it with `pip install
'reviewbot[headroom]'` (the Action installs it automatically when the input is
on). If the package is missing or a compression call fails, messages are sent
uncompressed, so a review never breaks on it.

By default the annotated diff (a `user` message, whose line numbers the model
must cite) and the most recent turns are left intact; only tool outputs and
older turns shrink. Compression is model-aware — the resolved model id drives
token counting and the context limit, so it works for both OpenAI- and
Anthropic-family models over the OpenAI-compatible protocol.

| Env var / input | Default | Notes |
| --------------- | ------- | ----- |
| `HEADROOM_COMPRESS` | `false` | Master switch |
| `HEADROOM_TARGET_RATIO` | — | Keep-ratio for text compression (e.g. `0.5`) |
| `HEADROOM_COMPRESS_USER_MESSAGES` | `false` | Also compress the diff — off by default to keep cited lines intact |
| `HEADROOM_COMPRESS_SYSTEM_MESSAGES` | `true` | Compress system messages |
| `HEADROOM_PROTECT_RECENT` | `4` | Never compress the last N messages |
| `HEADROOM_MIN_TOKENS` | `250` | Skip messages shorter than this |
| `HEADROOM_KOMPRESS_MODEL` | — | Kompress model id, or `disabled` to skip ML compression |
| `HEADROOM_MODEL_LIMIT` | `200000` | Model context window used for sizing |

**LLM compatibility:** any service exposing
`POST {base}/chat/completions` with `response_format: json_object` works —
OpenAI, Hugging Face Router, Anthropic's OpenAI shim, vLLM, TGI, llama.cpp,
LM Studio. If the endpoint ignores JSON mode, the bot extracts JSON from the
response text as a fallback.

**Repo-specific rules:** drop a `.ai/review-rules.md` in the target repo; it's
read from the **default branch** (so PR authors can't rewrite the rules in
their own branch) and injected into the system prompt. Repos can also supply
an executable `.ai/context-script` and scoped helper CLIs via
`.ai/review-tools.json` — see [`action.yml`](action.yml) for details.

## How it works

```
PR comment "@askserge …"  ──►  trigger gate  ──►  build review  ──►  post review
  (Action event or App webhook)   (author +        (annotate diff,    (one PR review
                                   open PR)          call LLM,          with validated
                                                     validate output)   inline comments)
```

1. A comment containing the trigger, from a `MEMBER`/`OWNER`/`COLLABORATOR`,
   arrives via Actions event or webhook.
2. The bot fetches the PR and every changed file, annotating each addressable
   line so the LLM can reference it (`[R 42]` = new side line 42, `[L 11]` =
   old side). The valid `(side, line)` set is kept for validation.
3. The annotated diff + repo rules go to the LLM in JSON mode. The model
   returns a `summary`, an `event`, and a list of inline `comments`.
4. Each comment is validated against the recorded diff positions — anything
   pointing at a line not in the diff is dropped, so the model can't
   hallucinate locations.
5. One review is posted with all valid inline comments attached.

**Prompt safety:** PR content is treated as untrusted input. Injection
attempts ("ignore previous instructions", fake `SYSTEM` messages) are flagged
inline rather than obeyed, and any repo-defined tools run in a read-only
sandbox rooted at the checkout — no arbitrary shell access.
