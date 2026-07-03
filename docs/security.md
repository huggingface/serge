---
title: Security
---

`serge` treats PR content as untrusted input.

## Prompt Injection

The reviewer prompt tells the model not to follow instructions embedded in PR
content, comments, docstrings, strings, or tool output. Suspicious instructions
inside a diff should be treated as code-review findings, not as instructions to
the reviewer.

## Diff Position Validation

The model cannot choose arbitrary GitHub comment locations. The reviewer tracks
valid `(path, side, line)` positions while annotating the diff. Any inline
comment that points outside those positions is dropped before publishing.

## Default-Branch Policy

Repository policy files are read from the target repository's default branch:

- `.ai/review-rules.md`
- `.ai/context-script`
- `.ai/review-tools.json`

This prevents a PR from changing its own review rules.

## Tool Sandboxing

Read-only tools are rooted at `REPO_CHECKOUT_PATH`. Paths are resolved with
real paths and rejected if they escape the checkout. Noisy or sensitive
directories such as `.git`, `node_modules`, virtualenvs, and build caches are
hidden.

`fetch_url` is restricted to `https://huggingface.co/*`.

## Helper Tools

Repo helper tools run without a shell and receive a stripped environment that
omits GitHub tokens, LLM keys, OAuth secrets, session secrets, and webhook
secrets.

Helper install hooks are limited to validated `pip` package installs. URL,
VCS, editable, custom-index, and target-directory installs are rejected.

## GitHub Action Forks

Do not rely on the Action for forked PRs. GitHub withholds secrets from forked
workflow runs, and the token is often read-only. Use GitHub App or web app mode
for fork-heavy repositories.

## Write-Capable Tasks

The [tasks flow](tasks-flow.md) (`POST /tasks`) can open PRs and push commits,
so it is off by default and gated on `TASK_API_ENABLED`, a per-repo opt-in flag,
and GitHub Actions OIDC (authorized on the token's `repository` claim — no shared
secret). The LLM only ever proposes a patch; serge applies and commits it through
the GitHub Git Data API, so push credentials never enter the sandbox. serge
writes only inside its own `serge/*` branch namespace, and a follow-up loop cap
bounds the number of commits per fix branch. Task `context`/logs are untrusted
input, handled like a PR body.

## Per-Task Pod Network Firewall

When tasks run in the pod-per-task model (`TASK_EXECUTION=kubernetes`), each task
runs in its own pod that holds a scoped GitHub token + the LLM key while executing
repo build code (e.g. `make style`). Its egress is therefore locked down:

- The task pod's `NetworkPolicy` **denies all egress** except to the in-cluster
  `serge-egress` forward proxy, kube-dns, and serge's own callback Service.
- `serge-egress` (a small tinyproxy Deployment, built in-house from the Debian
  package) permits `CONNECT` only to an allowlist — `*.github.com`,
  `*.githubusercontent.com`, and `router.huggingface.co` — and denies everything
  else. serge injects it as the pod's `HTTPS_PROXY`/`HTTP_PROXY`; `NO_PROXY` keeps
  the in-cluster callback off the proxy. Arbitrary repo code in the pod can thus
  reach only GitHub + the HF LLM, never an attacker-controlled host.

### Accepted residual risks

- **DNS egress.** Task pods may reach kube-dns (port 53) for name resolution. The
  stricter no-DNS stance — injecting the proxy's ClusterIP directly and denying
  all DNS to close the DNS-tunnel exfil channel — is **not** implemented; the DNS
  egress is an accepted residual risk. (Won't-fix unless the threat model changes.)
- **GitHub token tunnel.** Because the pod holds a valid GitHub token during the
  arbitrary-code phase, data could in principle be tunneled through legitimate
  requests to `github.com`. Fully closing this needs the old split-pod model (no
  secrets present while untrusted code runs), which this design deliberately
  trades away for simplicity and speed.

## Web App Sessions

Production web app deployments should use OAuth, a strong
`WEB_SESSION_SECRET`, HTTPS, and secure cookies. `DEV_NO_AUTH=1` and
`WEB_INSECURE_COOKIES=1` are local-development options only.
