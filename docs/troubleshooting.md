---
title: Troubleshooting
---

## The Trigger Was Ignored

Check that:

- the event is `issue_comment` or `pull_request_review_comment`;
- the comment was newly created;
- the comment contains `MENTION_TRIGGER`, default `@askserge`;
- the commenter is a `MEMBER`, `OWNER`, or `COLLABORATOR`;
- the issue comment is on an open PR, not a plain issue.

## `LLM_API_KEY` Is Missing

In Action and webhook modes, set `LLM_API_KEY` or pass `llm_api_key`.

In web mode, either configure a global `LLM_API_KEY` fallback or create a
matching provider config in the Admin page.

## No Provider Config Grants Access

In web mode, provider configs are matched by:

- provider;
- repository pattern, either `owner/repo` or `owner/*`;
- current user or current user's allowed orgs.

Create a matching config or choose the provider that already has one.

## The App Cannot Publish

Check that the GitHub App is installed on the target repository and has:

- Pull requests: read and write;
- Contents: read;
- Issues: read;
- Metadata: read.

Also verify `GITHUB_APP_ID` and the private key.

## Webhook Signature Failed

Make sure `GITHUB_WEBHOOK_SECRET` matches the secret configured on the GitHub
App. The server verifies `X-Hub-Signature-256`.

## Forked PR Review Failed in Actions

This is expected for many repositories. GitHub does not expose repository
secrets to forked PR workflows, and the token may be read-only. Use GitHub App
or web app mode.

## LLM Returned Bad JSON

Try:

- increasing `LLM_MAX_TOKENS`;
- choosing a model that follows JSON instructions reliably;
- enabling streaming if the provider or proxy times out on long requests;
- reducing `MAX_DIFF_CHARS`;
- adding clearer `.ai/review-rules.md` guidance.

## Context Script Did Nothing

Check that the script:

- exists at `CONTEXT_SCRIPT_PATH`;
- is executable;
- exits zero;
- finishes before `CONTEXT_SCRIPT_TIMEOUT`;
- prints either plain text or a JSON object with `context` and/or `skip_files`.

Failures are logged and ignored so a broken context hook does not block
reviews.

## Helper Tool Is Unavailable

Check `.ai/review-tools.json` for valid JSON, a valid helper name, a non-empty
command, and a command that exists inside the checkout or on `PATH`. If using
`install`, confirm the package spec passes the allowed installer restrictions.
