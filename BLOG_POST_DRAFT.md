# Draft: Introducing serge: AI Code Review That Fits GitHub

Pull requests already have a workflow. Reviewers read the diff, ask for
changes, follow up in threads, and decide when a change is ready to merge. The
hard part for AI code review is not producing text about code. The hard part is
fitting into that workflow without creating another place to check, another
policy system to maintain, or another stream of comments that someone has to
clean up.

`serge` is a GitHub-native code reviewer built around that constraint. It
reviews pull requests with an OpenAI-compatible language model, follows
repository-owned review rules, and publishes comments through GitHub's normal
pull request review experience. The default reviewer persona is Serge. You
invoke it with a comment such as:

```text
@askserge please review
```

From there, Serge reads the pull request, applies the repository's review
policy, and optionally inspects additional read-only repository context. It then
returns a review that can be published directly or staged for a human to edit
first.

## Why We Built It This Way

Most teams do not need an AI reviewer that replaces code review. They need one
that catches issues early, helps maintainers keep up with pull request volume,
and adapts to the way the repository already works. That makes the important
questions practical:

- Can maintainers trigger it only when they want it?
- Can a repository define what kind of feedback is useful?
- Can the reviewer inspect enough context to avoid shallow comments?
- Can teams choose the model they trust, including open models?
- Can it work with GitHub permissions, forks, webhooks, and secrets?
- Can a human edit or discard the model's output before it reaches a pull
  request?

## Three Ways To Run It

There are three ways to run `serge`, depending on how much control and
automation you need: the GitHub Action for a quick single-repository setup, the
GitHub App webhook mode for organizations and fork-heavy projects, and the
staged web app for human-in-the-loop review.

The Action is the fastest path: add an LLM API key as a repository secret,
install the workflow, and comment on a pull request to start a review.

The GitHub App runs as a hosted service, receives GitHub comment events, and
publishes reviews with a GitHub App installation token. That avoids a common
forked pull request problem: GitHub Actions often cannot access repository
secrets or write review comments safely.

The staged web app is the human-in-the-loop path. A reviewer can start a
review, watch the model stream its work, edit or discard comments, and publish
only the parts that are useful. It also stores per-repository provider
configuration for different models, providers, and API keys.

## Repository Policy Lives In The Repository

AI review quality depends heavily on context. A general-purpose reviewer will
often spend time on the wrong things unless the repository tells it what matters.

`serge` lets repositories define their review policy in
`.ai/review-rules.md` on the default branch. A typical rules file might ask the
reviewer to focus on correctness, security, behavior changes, and missing tests
while ignoring generated files or style-only feedback. Loading those rules from
the default branch is intentional: a pull request should not be able to rewrite
the policy used to review itself.

Repositories can also provide an optional `.ai/context-script` for extra
context or files to skip, and read-only tools rooted in a local checkout. That
gives the reviewer bounded access to project context without exposing secrets
or arbitrary shell access. Built-in tools such as `read_file`, `list_dir`, and
`grep` can be exposed when bounded inspection beyond the diff is needed.

## A Review Flow That Matches GitHub

A full review starts with a GitHub comment event. The reviewer checks the
basics: event type, new comment, trigger phrase, trusted commenter, and open
pull request. Then it fetches the pull request, prepares the diff and repository
context, calls the model, validates the result, and publishes the review.

The same trigger also works in review-comment threads. If someone replies to an
existing inline review comment with `@askserge`, the reviewer answers that
specific question in the same thread instead of starting a full new review.

In web app mode, the model's output becomes a draft first. A human can edit the
summary, rewrite or discard comments, or drop the entire draft. Approval reviews
are blocked by default unless `ALLOW_APPROVE=1` is explicitly enabled.

## Bring Your Own Model Provider

`serge` talks to OpenAI-compatible chat completion endpoints. It works
with OpenAI, the Hugging Face Router, local vLLM/TGI/LM Studio endpoints, and
custom compatible providers.

That provider flexibility is intentional. Teams should be able to run the model
that fits their code, policy, cost, and deployment constraints, especially open
models they can evaluate and swap over time. The Hugging Face Router is a strong
fit: it exposes many open and commercial models through an OpenAI-compatible
API, keeping the integration simple while giving operators real model choice.

The basic configuration is intentionally small: set an API base, an API key,
and optionally a model.

If no model is configured, the reviewer can query the endpoint's `/models`
route and fall back to the first returned model. Streaming is supported so
hosted deployments can show output, tool calls, and progress while a review is
running.

## Security And Trust Boundaries

Pull request content is untrusted input. The prompt tells the model not to
follow instructions embedded in diffs, comments, strings, docstrings, or tool
output. If a pull request contains prompt injection text, the reviewer should
treat it as something to review, not as an instruction to obey.

Repository customization files are loaded from the default branch. Built-in
tools are read-only and confined to the checkout root. Helper tools run without
a shell and with a stripped environment that removes GitHub tokens, LLM keys,
OAuth secrets, session secrets, and webhook secrets.

The deployment modes also reflect GitHub's security model. The GitHub Action is
simple and fast, but it is not the right answer for many forked pull requests
because GitHub withholds secrets and often gives the workflow a read-only token.
For repositories that rely heavily on external contributors, the GitHub App or
staged web app modes are the safer operational choice.

## Built For Maintainers, Not Just Demos

`serge` is still a very young project, and it is still maturing. But it
has already proven useful in real open source review workflows, including
Hugging Face's `diffusers` and `transformers` projects, as well as
`serge` itself.

The webhook server can serve many repositories from one hosted GitHub App. The
web app includes GitHub OAuth, allowlists, persisted job history and provider
configuration, and a shared clone cache for large repositories. There is also a
path to scale beyond a demo: containerized deployment, a database layer that can
move from SQLite to Postgres, bounded worker admission control, and operational
metrics.

## Getting Started

For a quick trial, use the GitHub Action:

1. Add a repository secret named `LLM_API_KEY`.
2. Add the `serge` workflow.
3. Comment `@askserge please review` on an open pull request.

`serge` is an open source project, Apache-2.0 licensed, and
contributions are welcome. You can find the code at
[github.com/huggingface/serge](https://github.com/huggingface/serge)
and the documentation at
[huggingface.github.io/serge](https://huggingface.github.io/serge/).

The goal is straightforward: make AI review useful inside the code review
process maintainers already use, while keeping repository policy, security, and
final judgment under human control.
