---
title: Configuration
nav_title: Config
---

Configuration is passed as Action inputs in GitHub Action mode and as
environment variables in server modes.

## LLM

| Env var | Action input | Default | Description |
| ------- | ------------ | ------- | ----------- |
| `LLM_API_KEY` | `llm_api_key` | required | Bearer token for the LLM endpoint. In web mode, provider configs can supply per-repo keys. |
| `LLM_API_BASE` | `llm_api_base` | `https://api.openai.com/v1` | OpenAI-compatible API base. `LLM_BASE_URL` is also accepted as an env alias. |
| `LLM_MODEL` | `llm_model` | first model from `/models` | Model identifier. |
| `LLM_BILL_TO` | `llm_bill_to` | unset | Optional routing slug, used for Hugging Face Router requests. |
| `LLM_MAX_TOKENS` | `llm_max_tokens` | `4096` | Maximum completion tokens. |
| `LLM_STREAM` | `llm_stream` | env default `true`, Action default `false` | Consume streaming SSE responses. |
| `LLM_REASONING_EFFORT` | none | unset | Optional `reasoning_effort` value passed through to providers that support it. |
| `LLM_MAX_INPUT_TOKENS` | none | `2000000` | Hard cap on cumulative input tokens for a review. Set `0` to disable. |

## Review Behavior

| Env var | Action input | Default | Description |
| ------- | ------------ | ------- | ----------- |
| `MENTION_TRIGGER` | `mention_trigger` | `@askserge` | Phrase that triggers reviews. |
| `REVIEW_EVENT` | `review_event` | `COMMENT` | Fallback review event when the model omits one. |
| `MAX_DIFF_CHARS` | `max_diff_chars` | `200000` | Maximum diff characters sent to the LLM. |
| `REVIEW_RULES_PATH` | `review_rules_path` | `.ai/review-rules.md` | Rules file read from the target repo default branch. |
| `DEFAULT_REVIEW_RULES` | `default_review_rules` | general Python correctness and security rules | Fallback when no rules file exists. |
| `ALLOW_APPROVE` | none | `false` | Allows publishing `APPROVE` events in App/web mode. |
| `PERSONA_HEADER` | none | `🤗 **Serge** says:` | Prefix for failure comments and bot messages. |
| `STAGING` | `staging` | `false` | Marks a non-production deployment. Published reviews then carry a note that they were posted from staging. |

## Context Compression

Opt-in compression of token-heavy context (tool outputs, older turns) before
each LLM call, via the [`headroom-ai`](https://github.com/chopratejas/headroom)
package. Install the extra with `pip install '.[headroom]'` (the Action pulls
it in automatically when `headroom_compress` is on). It is a no-op if the
package is missing or a compression call fails, so a review never breaks on it.

| Env var | Action input | Default | Description |
| ------- | ------------ | ------- | ----------- |
| `HEADROOM_COMPRESS` | `headroom_compress` | `false` | Master switch. |
| `HEADROOM_TARGET_RATIO` | `headroom_target_ratio` | unset | Keep-ratio for text compression (e.g. `0.5`). Empty lets headroom decide. |
| `HEADROOM_COMPRESS_USER_MESSAGES` | `headroom_compress_user_messages` | `false` | Also compress user messages (the annotated diff). Off keeps cited lines intact. |
| `HEADROOM_COMPRESS_SYSTEM_MESSAGES` | `headroom_compress_system_messages` | `true` | Compress system messages. |
| `HEADROOM_PROTECT_RECENT` | `headroom_protect_recent` | `4` | Never compress the last N messages. |
| `HEADROOM_MIN_TOKENS` | `headroom_min_tokens` | `250` | Skip messages shorter than this many tokens. |
| `HEADROOM_KOMPRESS_MODEL` | `headroom_kompress_model` | unset | Kompress model id, or `disabled` to skip ML compression. |
| `HEADROOM_MODEL_LIMIT` | `headroom_model_limit` | `200000` | Model context window (tokens) used for sizing. |

## Repository Context and Tools

| Env var | Action input | Default | Description |
| ------- | ------------ | ------- | ----------- |
| `CONTEXT_SCRIPT_PATH` | `context_script_path` | `.ai/context-script` | Optional executable context script. |
| `CONTEXT_SCRIPT_TIMEOUT` | `context_script_timeout` | `30` | Seconds before the context script is ignored. |
| `HELPER_TOOLS_PATH` | `helper_tools_path` | `.ai/review-tools.json` | Optional helper tool config. |
| `REPO_CHECKOUT_PATH` | `repo_checkout_path` | Action: `github.workspace`; env: empty | Local checkout root for read-only tools. Empty disables tools. |
| `TOOL_MAX_ITERATIONS` | `tool_max_iterations` | env default `30`, Action default `8` | Maximum tool-calling rounds. Set `0` to disable the cap. |

## GitHub App

| Env var | Required for | Description |
| ------- | ------------ | ----------- |
| `GITHUB_APP_ID` | App/web publish | Numeric GitHub App ID. |
| `GITHUB_PRIVATE_KEY` | App/web publish | Inline PEM private key. Literal `\n` sequences are expanded. |
| `GITHUB_PRIVATE_KEY_PATH` | App/web publish | Path to the PEM private key. |
| `GITHUB_WEBHOOK_SECRET` | Webhook mode | Webhook signing secret. |
| `WEBHOOK_MAX_WORKERS` | `reviewbot-app` | Concurrent webhook review workers. Default `2`. |

## Web App

| Env var | Default | Description |
| ------- | ------- | ----------- |
| `GITHUB_OAUTH_CLIENT_ID` | required unless `DEV_NO_AUTH=1` | GitHub OAuth client ID. |
| `GITHUB_OAUTH_CLIENT_SECRET` | required unless `DEV_NO_AUTH=1` | GitHub OAuth client secret. |
| `GITHUB_OAUTH_CALLBACK_URL` | optional | Callback URL registered on the OAuth App. |
| `WEB_SESSION_SECRET` | required unless `DEV_NO_AUTH=1` | Secret for signed session cookies. |
| `WEB_ALLOWED_USERS` | unset | Comma-separated GitHub logins allowed into the UI. |
| `WEB_ALLOWED_ORG` | unset | Comma-separated GitHub orgs allowed into the UI. |
| `WEB_STORE_PATH` | `jobs.db` | SQLite path. |
| `WEB_JOB_RETENTION` | `25` | Number of recent jobs to retain. |
| `DEV_NO_AUTH` | `false` | Disables OAuth for local development only. |
| `WEB_INSECURE_COOKIES` | `false` | Drops the Secure flag from session cookies. |
| `WEB_CLONE_CACHE_DIR` | temp directory | Shared clone cache path. |
| `WEB_CLONE_CACHE_TTL_SECONDS` | `604800` | Clone cache TTL. |
| `WEB_CLONE_DEPTH` | `50` | Shallow fetch depth. |
| `WEB_GITHUB_APP_URL` | project default | Install/configure URL shown in the web help page. Set this to your GitHub App URL for public deployments. |

## Server

| Env var | Default | Description |
| ------- | ------- | ----------- |
| `PORT` | `8080` | Development server port. |
| `LOG_LEVEL` | `INFO` | Logging level. |
