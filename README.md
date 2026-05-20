# ai-reviewer

**Source:** [github.com/tarekziade/ai-reviewer](https://github.com/tarekziade/ai-reviewer)

Reviews pull requests with any **OpenAI-compatible chat-completion
service** and posts **inline comments** on the diff via GitHub's
Pull Request Reviews API.

Reimplements the
[`claude_review.yml`](https://github.com/huggingface/diffusers/blob/main/.github/workflows/claude_review.yml)
workflow from `huggingface/diffusers`, decoupled from the Anthropic
Claude Code action — so you can plug in OpenAI, a local vLLM, the
Hugging Face Router, Anthropic's OpenAI-compatible shim, LM Studio,
llama.cpp, or anything else that speaks `/v1/chat/completions`.

Runs in **two modes** off the same codebase:

1. **GitHub Action** (no server) — drop-in replacement for the original
   workflow. Triggers on `@serge` comments via `on: issue_comment`.
2. **GitHub App** (self-hosted) — long-lived webhook service. Install
   once, review PRs across many repos.

---

## Table of contents

- [How it works](#how-it-works)
- [Prompt safety](#prompt-safety)
- [Mode 1 — GitHub Action](#mode-1--github-action)
- [Mode 2 — GitHub App (self-hosted)](#mode-2--github-app-self-hosted)
- [Choosing between the two modes](#choosing-between-the-two-modes)
- [LLM endpoint compatibility](#llm-endpoint-compatibility)
- [Repo-specific review rules](#repo-specific-review-rules)
- [Project layout](#project-layout)
- [Limitations / possible extensions](#limitations--possible-extensions)

---

## How it works

```
 PR comment containing "@serge"
        │
        ▼
┌───────────────────────┐
│ Mode 1: Actions runner│   or   ┌──────────────────────┐
│  - reviewbot-action   │        │ Mode 2: reviewbot.app│
│  - $GITHUB_TOKEN      │        │  - HMAC verify       │
│  - $GITHUB_EVENT_PATH │        │  - App JWT → token   │
└───────────┬───────────┘        └──────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
             ┌────────────────────────────────────────────┐
             │ 1. Fetch PR, files, .ai/review-rules.md    │
             │ 2. Annotate each patch with [Rxxxx]/[Lxxxx]│
             │ 3. POST /chat/completions (JSON mode)      │
             │ 4. Validate (path, side, line) against diff│
             │ 5. POST /pulls/{n}/reviews + inline bodies │
             └────────────────────────────────────────────┘
```

1. A PR comment containing the trigger phrase (default `@serge`) from a
   `MEMBER`, `OWNER`, or `COLLABORATOR` arrives — either via webhook
   (App mode) or as an Actions event (Action mode).
2. The reviewer fetches PR metadata and every changed file with its
   patch.
3. Each patch is annotated so every addressable line gets a
   `[R   42]` (new file, RIGHT side) or `[L   11]` (old file, LEFT side)
   prefix. The set of valid `(side, line)` pairs is kept for validation.
4. The annotated diff plus repo-specific rules from
   `.ai/review-rules.md` on the default branch are sent to your LLM
   with JSON mode enabled. The model is required to reply with:
   ```json
   {
     "summary": "…",
     "event": "COMMENT" | "REQUEST_CHANGES" | "APPROVE",
     "comments": [
       {"path": "src/foo.py", "side": "RIGHT", "line": 42, "body": "…"}
     ]
   }
   ```
5. Every comment is validated against the recorded diff positions.
   Invalid ones are dropped and noted in the summary — the LLM cannot
   hallucinate a line that isn't in the diff.
6. A single review is posted with all inline comments attached and the
   chosen event (`COMMENT` by default).

---

## Prompt safety

The system prompt is ported from the diffusers workflow and keeps the
same anti-injection posture:

- PR content is declared **untrusted external input**.
- Overrides ("ignore previous instructions", "you are now…", fake
  `SYSTEM` messages) must be flagged inline with an
  `[INJECTION ATTEMPT]` prefix, not obeyed.
- Any filesystem browsing or repo-defined helper CLI runs through a
  narrow tool sandbox rooted at the checked-out repo. There is no
  arbitrary shell access.
- The final review still must be a single JSON object.

---

## Mode 1 — GitHub Action

Zero infra. Runs on the target repo's Actions runner, authenticated
with the job's `GITHUB_TOKEN`. No public server, no App registration.

### Setup

Drop this workflow into the repository you want reviewed as
`.github/workflows/ai-review.yml`. The action is referenced directly
from `tarekziade/ai-reviewer` — no vendoring, no submodules.

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
    # Only run for @serge mentions from trusted authors, on open PRs.
    if: |
      (github.event_name == 'issue_comment' &&
       github.event.issue.pull_request &&
       github.event.issue.state == 'open' &&
       contains(github.event.comment.body, '@serge') &&
       (github.event.comment.author_association == 'MEMBER' ||
        github.event.comment.author_association == 'OWNER' ||
        github.event.comment.author_association == 'COLLABORATOR')) ||
      (github.event_name == 'pull_request_review_comment' &&
       contains(github.event.comment.body, '@serge') &&
       (github.event.comment.author_association == 'MEMBER' ||
        github.event.comment.author_association == 'OWNER' ||
        github.event.comment.author_association == 'COLLABORATOR'))
    runs-on: ubuntu-latest
    steps:
      - uses: tarekziade/ai-reviewer@main
        with:
          llm_api_key: ${{ secrets.LLM_API_KEY }}
          llm_api_base: ${{ secrets.LLM_API_BASE || 'https://api.openai.com/v1' }}
```

Pin to a commit SHA (`tarekziade/ai-reviewer@<sha>`) or a tag once
you're happy with a version, rather than tracking `main`.

The workflow-level `if:` is redundant with the in-process trigger
gating but filters early so nothing spins up for irrelevant events.

`issues: write` is needed because the reviewer adds reactions to the
triggering comment and posts fallback/error comments on the PR when it
cannot produce a review.

Before the workflow can run you need to add the LLM credential as a
repository secret. If you do not want to hardcode the endpoint URL in
the workflow YAML, store that as a secret too:

- **Settings → Secrets and variables → Actions → New repository secret**
- Name: `LLM_API_KEY`
- Value: your OpenAI / Anthropic / HF Router / … bearer token
- Optional Name: `LLM_API_BASE`
- Optional Value: your endpoint base URL, for example
  `https://router.huggingface.co` or `https://router.huggingface.co/v1`

### Action inputs

| Input                   | Default                                       | Required |
| ----------------------- | --------------------------------------------- | -------- |
| `llm_api_key`           | —                                             | ✅       |
| `llm_api_base`          | `https://api.openai.com/v1`                   |          |
| `llm_model`             | first `id` from `{llm_api_base}/models`       |          |
| `mention_trigger`       | `@serge`                                      |          |
| `review_event`          | `COMMENT`                                     |          |
| `max_diff_chars`        | `200000`                                      |          |
| `review_rules_path`     | `.ai/review-rules.md`                         |          |
| `helper_tools_path`     | `.ai/review-tools.json`                       |          |
| `default_review_rules`  | generic Python correctness prompt             |          |
| `context_script_path`   | `.ai/context-script`                          |          |
| `context_script_timeout`| `30`                                          |          |
| `github_token`          | `${{ github.token }}`                         |          |
| `python_version`        | `3.12`                                        |          |

### Use

On any open PR, post a comment as a collaborator/member/owner:

```
@serge please review
```

Within a few seconds a full PR review lands with inline comments
anchored to the diff.

### Follow-up questions on inline comments

To ask a focused question about a specific line, leave an **inline
review comment** on that line containing the trigger phrase:

```
@serge could you help me understand this line of code?
```

The reviewer replies in the same comment thread (instead of posting a
new full-PR review). The reply is anchored to the line you commented
on and has access to the same browse tools as a full review, so it can
read surrounding code when needed.

Use this for "what does this do?", "why this approach?", or "is this
covered by tests?" — anything where a thread reply makes more sense
than another top-level review.

### Repo-supplied extra context

Drop an executable script at `.ai/context-script` (path configurable
via `context_script_path`) and the reviewer will run it on every
review, piping a JSON document on stdin:

```json
{
  "title": "PR title",
  "body":  "PR description",
  "files": [
    {"path": "foo.py", "status": "modified",
     "additions": 12, "deletions": 3, "previous_path": null}
  ]
}
```

The script's stdout can be either:

- **Plain text** — stitched into the user prompt inside a
  `REPO-PROVIDED CONTEXT` block. Use it to flag high-risk file
  combinations, point the reviewer at related code, or surface repo
  conventions.
- **A JSON object** — `{"context": str, "skip_files": [str]}`. Both
  fields are optional. `skip_files` lets the repo tell the reviewer to
  exclude specific paths from the diff entirely (e.g. auto-generated
  files in repos that regenerate them from a modular source); the
  reviewer mentions the skip list neutrally to the LLM but does not
  send those patches.

The script must be executable; if it's missing, not executable, exits
non-zero, times out (default 30 s), or emits malformed JSON, the
review proceeds without extra context or skip filtering. This repo
dogfoods the plain-text mode with a Python script at
`.ai/context-script`; the structured-output mode is used by
[`transformers`](https://github.com/huggingface/transformers) to skip
auto-generated model files.

The script runs in Action mode only (the App-mode webhook does not
check out the repo). It executes from the default branch (since
`actions/checkout@v4` on `issue_comment` events checks out the event
ref), so treat it at the same trust level as `.ai/review-rules.md`.

### Repo-supplied helper tools

If the reviewer has a local checkout of the PR head (`repo_checkout_path`
is set), a repo can also expose tightly scoped helper CLIs by adding a
JSON file at `.ai/review-tools.json` (path configurable via
`helper_tools_path`) on the **default branch**:

```json
{
  "helpers": [
    {
      "name": "mlinter",
      "description": "Run Transformers' model linter for new-model PRs.",
      "command": ["mlinter"],
      "install": ["pip", "install", "transformers-mlinter"],
      "allow_args": true,
      "max_args": 8,
      "timeout_seconds": 30
    }
  ]
}
```

Each helper becomes a tool the LLM can call by name. The config is read
from the target repo's default branch via the GitHub API, so a PR author
cannot smuggle in a new command by editing their branch. Commands run
without a shell, with stdout/stderr captured and truncated, and only
inside the checked-out repo root (or a configured subdirectory under it).

`install` is optional. When set, the reviewer runs the install command
once per worker process before the agent loop starts, so the helper
binary is on PATH inside the runner. Today only `pip` is supported —
it's executed as `python -m pip install …` against the reviewer's own
interpreter, so the package lands wherever `mlinter` will be looked up.
Successful installs are cached for the lifetime of the process;
failures aren't (we want a retry to refresh on the next review).
Install args are restricted to a conservative character set, so you
can't smuggle shell metacharacters or arbitrary indexes through this
field.

Use this for narrow repo-maintainer workflows like linters or metadata
inspectors. Keep descriptions concrete so the model knows when to call
the helper and what arguments are expected.

---

## Mode 2 — GitHub App (self-hosted)

Long-lived webhook service. One install, many repos, sub-second
latency, no Actions minutes consumed.

### 1. Create the GitHub App

In **Settings → Developer settings → GitHub Apps → New GitHub App**:

**Permissions** (repository):
- Pull requests: **Read & write**
- Contents: **Read**
- Issues: **Read**
- Metadata: **Read**

**Subscribe to events**:
- Issue comment
- Pull request review comment

**Webhook**:
- URL: `https://<your-host>/webhook`
- Secret: strong random string → `GITHUB_WEBHOOK_SECRET`

Download the private key (`.pem`), note the numeric **App ID**, and
install the App on the repositories you want reviewed.

### 2. Configure

```bash
git clone https://github.com/tarekziade/ai-reviewer.git
cd ai-reviewer
python -m venv .venv && source .venv/bin/activate
pip install .

cp .env.example .env
$EDITOR .env
set -a; source .env; set +a
```

Required variables:

| Variable                | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `GITHUB_APP_ID`         | Numeric ID of the GitHub App                   |
| `GITHUB_PRIVATE_KEY_PATH` or `GITHUB_PRIVATE_KEY` | PEM private key (path or inline) |
| `GITHUB_WEBHOOK_SECRET` | Shared secret set in the App's webhook config  |
| `LLM_API_BASE`          | Base URL of an OpenAI-compatible service (`...` or `.../v1`) |
| `LLM_API_KEY`           | Bearer token for that endpoint                 |

Optional variables (same defaults as the Action inputs above):
`LLM_MODEL`, `MENTION_TRIGGER`, `REVIEW_EVENT`, `MAX_DIFF_CHARS`,
`REVIEW_RULES_PATH`, `DEFAULT_REVIEW_RULES`, `PORT`, `LOG_LEVEL`.

### 3. Run

Install the package once:

```bash
pip install .
```

Development:

```bash
reviewbot-app
```

Production:

```bash
gunicorn -w 2 -b 0.0.0.0:8080 reviewbot.app:app
```

Expose the service publicly (HTTPS, stable URL) and point the App's
webhook at `https://<host>/webhook`. For local testing, use
[`smee.io`](https://smee.io) or `cloudflared tunnel run`.

### 4. Deployment targets

Any Python host works. Minimal configurations:

- **Cloud Run / Fly / Railway / Render**: deploy as a container with
  `gunicorn` as the entrypoint, expose port 8080.
- **Lambda + API Gateway**: wrap the Flask app with
  [`aws-wsgi`](https://pypi.org/project/aws-wsgi/) or Mangum-style
  shim; cold start is ~1–2s (fine for GitHub's 10s webhook timeout).
- **VM / systemd**: `gunicorn -w 2 -b 127.0.0.1:8080 reviewbot.app:app`
  behind nginx/caddy.

---

## Choosing between the two modes

| Need                                                | Action | App |
| --------------------------------------------------- | :----: | :-: |
| Zero infra                                          | ✅     |     |
| No Actions-minute usage                             |        | ✅  |
| Sub-second latency (no cold start, no setup-python) |        | ✅  |
| Serve many repos from one install                   |        | ✅  |
| Works on any repo without committing a workflow     |        | ✅  |
| Easiest to audit / pin by SHA per repo              | ✅     |     |
| Secrets stored in GitHub (not on your server)       | ✅     |     |

A common pattern: start with the Action, graduate to the App once
you're reviewing enough PRs that Actions minutes or queue time hurt.

---

## Mode 3 — Interactive web app

A small FastAPI app (`reviewbot-web`) lets a logged-in user kick off a
review from a form, watch the LLM stream live in a console, then tweak
the summary + per-comment text (or discard individual inline comments)
before publishing. The published review still goes out under the
GitHub App identity — OAuth is only used for access control.

### Setup

1. Install the web extras: `pip install -e '.[web]'`
2. Register a **GitHub OAuth App** (Settings → Developer settings →
   OAuth Apps → New OAuth App — this is *separate* from the GitHub App
   used to post reviews). Callback URL: `http://localhost:8080/auth/callback`
   for local testing.
3. Make sure the **GitHub App** from Mode 2 is installed on each repo
   you want to review.

### Run it locally

```bash
LLM_API_KEY=...                          # same as the other modes
GITHUB_APP_ID=...                        # same App as Mode 2
GITHUB_PRIVATE_KEY_PATH=./private-key.pem
GITHUB_OAUTH_CLIENT_ID=...               # the OAuth App's client id
GITHUB_OAUTH_CLIENT_SECRET=...
GITHUB_OAUTH_CALLBACK_URL=http://localhost:8080/auth/callback
WEB_SESSION_SECRET=$(openssl rand -hex 32)
WEB_ALLOWED_USERS=octocat,hubot          # comma-separated GitHub logins
# or WEB_ALLOWED_ORG=acme,other-org
reviewbot-web                            # listens on 0.0.0.0:8080
```

Then open `http://localhost:8080`, sign in with GitHub, fill in
`owner/repo#123` + the trigger comment, pick an LLM provider/model,
and watch the stream.

The web form has provider presets for Hugging Face Router
(`https://router.huggingface.co/v1`), OpenAI
(`https://api.openai.com/v1`), and Anthropic
(`https://api.anthropic.com`), plus a custom base URL field for any
OpenAI-compatible endpoint. The Anthropic preset defaults to
`claude-opus-4-6`. `LLM_API_BASE` and `LLM_MODEL` only set the default
selection shown in the form. HF can use the legacy `LLM_API_KEY`;
OpenAI, Anthropic, and Custom require `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `CUSTOM_LLM_API_KEY` respectively unless that
provider is the server's default `LLM_API_BASE`. If a user picks a
provider with no configured key, the form returns a clear error before
queueing a review.

For pure local "I just want to click around" testing, set
`DEV_NO_AUTH=1` to bypass OAuth (the OAuth env vars become optional).
Don't ship that flag to production.

### Single-VM AWS deployment

This repo also ships a small EC2 bootstrap in [`aws/`](aws/README.md)
for the interactive reviewboard app. It mirrors the dashboard
deployment pattern: launch a single Amazon Linux 2023 host, clone the
repo, install `.[web]`, copy an env file into
`/etc/reviewbot/reviewbot-web.env`, and enable a `systemd` unit that
starts the web app on port `8080`.

If your local launch uses `GITHUB_PRIVATE_KEY_PATH=/path/to/pem`,
`aws/deploy.sh` uploads that PEM and rewrites the deployed env file to
use `/etc/reviewbot/github-app.pem`. The generated service runs the
same `reviewbot-web` entrypoint you use locally.

Usage:

1. Copy `aws/reviewbot-web.env.example` to `aws/reviewbot-web.env`.
2. Fill in the GitHub App, OAuth, and LLM credentials.
3. Run `./aws/deploy.sh`.

The generated service is `reviewbot-web.service`, so after boot you can
inspect it with `systemctl status reviewbot-web` and
`journalctl -u reviewbot-web`.

### How it differs from Mode 2

| Aspect                              | Mode 2 (webhook) | Mode 3 (web) |
| ----------------------------------- | :--------------: | :----------: |
| Trigger surface                     | GitHub comment   | Web form     |
| Review posted automatically         | ✅              |              |
| Human can edit before publishing    |                  | ✅          |
| Multi-process / horizontal scaling  | ✅              |              |
| Needs an inbound webhook URL        | ✅              |              |

---

## LLM endpoint compatibility

Any server that accepts

```http
POST {normalized(LLM_API_BASE)}/chat/completions
Authorization: Bearer {LLM_API_KEY}

{"model": "...", "messages": [...], "response_format": {"type": "json_object"}}
```

where `normalized(LLM_API_BASE)` means `LLM_API_BASE` itself when it
already ends in `/v1`, otherwise `LLM_API_BASE + "/v1"`,

will work. Example bases:

- OpenAI: `https://api.openai.com` or `https://api.openai.com/v1`
- Hugging Face Router: `https://router.huggingface.co` or `https://router.huggingface.co/v1`
- Anthropic OpenAI shim: `https://api.anthropic.com` or `https://api.anthropic.com/v1`
- vLLM / TGI / llama.cpp / LM Studio: the service root or its `/v1` URL

If `llm_model` / `LLM_MODEL` is omitted, the reviewer calls
`{LLM_API_BASE}/v1/models` and uses the first returned `data[].id`
(or `{LLM_API_BASE}/models` when the base already ends in `/v1`). This
matches LoopSleuth's endpoint-discovery behavior.

If your endpoint ignores `response_format=json_object`, the reviewer
still recovers: it extracts fenced or bare JSON blocks from the
response body before giving up.

---

## Repo-specific review rules

The reviewer reads `REVIEW_RULES_PATH` (default `.ai/review-rules.md`)
from the **default branch** of the target repo and injects the contents
verbatim into the system prompt.

This matches the diffusers convention: reviewer guidance is configured
per-repo and pinned to the default branch, so fork contributors cannot
rewrite the rules as part of their PR.

Optional helper-tool config (`HELPER_TOOLS_PATH`, default
`.ai/review-tools.json`) follows the same trust model: it is also read
from the default branch, then used only to expose narrowly scoped helper
commands against the local PR checkout when one exists.

---

## Project layout

```
action.yml                       Composite Action manifest (Mode 1)
pyproject.toml                   Package metadata, deps, console scripts
.env.example                     Env template for webhook and web modes
.ai/review-rules.md              Repo-specific review guidance
.ai/context-script               Optional repo-supplied context hook
.ai/review-tools.json            Optional repo-supplied helper CLI config
aws/                             EC2 bootstrap scripts for reviewbot-web
reviewbot/
  action_runner.py   Action entry point — reads $GITHUB_EVENT_PATH
                     (console script: `reviewbot-action`)
  app.py             Flask webhook server (Mode 2)
                     (console script: `reviewbot-app`,
                      gunicorn target: `reviewbot.app:app`)
  webapp.py          FastAPI reviewboard UI (Mode 3)
                     (console script: `reviewbot-web`)
  triggers.py        Shared "should we review?" gating
  config.py          Env-driven config, App creds optional
  github_auth.py     App JWT + installation token (Mode 2 only)
  github_client.py   REST wrapper: PRs, files, contents, reviews
  llm_client.py      OpenAI-compatible chat-completions client
  patch.py           Unified diff parser + line annotator
  prompts.py         System / user prompt templates
  context_script.py  Runs `.ai/context-script`, captures stdout
  reviewer.py        Orchestration + JSON parsing + comment validation
tests/                           unittest-based test suite
```

---

## Limitations / possible extensions

- **No queue** in App mode: reviews run in a thread per webhook. Fine
  for small teams; swap for RQ/Celery/SQS if you need durability.
- **No rate limiting**: spamming `@serge` triggers one review per
  comment. Add a cooldown keyed on `(repo, pr, commenter)` if needed.
- **Single-shot LLM call**: no multi-turn repo exploration like
  `claude-code-action` would do. Add tool use if you want the model to
  grep the repo before commenting.
- **Binary / huge diffs** are skipped silently past `MAX_DIFF_CHARS`.
