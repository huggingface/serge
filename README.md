# ai-reviewer

`ai-reviewer` reviews GitHub pull requests with an OpenAI-compatible LLM and
posts validated inline comments on the diff. The default reviewer persona is
Serge, triggered by comments such as `@askserge please review`.

It can run as:

| Mode | Best for |
| ---- | -------- |
| GitHub Action | Quick setup and per-repo CI control |
| GitHub App webhook | Hosted automation across many repositories |
| Web app | Human-in-the-loop staged reviews before publishing |

## Quick Start

Add an LLM key as a repository secret named `LLM_API_KEY`, then install the
Action workflow from the [GitHub Action guide](docs/github-action.md). Comment
`@askserge please review` on an open PR to start a review.

For fork-heavy repositories or hosted deployments, use the
[GitHub App](docs/github-app.md) or [web app](docs/web-app.md) guides instead.

## Documentation

- [Getting started](docs/getting-started.md)
- [GitHub Action](docs/github-action.md)
- [GitHub App webhook](docs/github-app.md)
- [Staged web app](docs/web-app.md)
- [Configuration](docs/configuration.md)
- [Repository customization](docs/repository-customization.md)
- [LLM providers](docs/llm-providers.md)
- [How it works](docs/how-it-works.md)
- [Security](docs/security.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Development](docs/development.md)

## License

Apache-2.0. See [LICENSE](https://github.com/huggingface/ai-reviewer/blob/main/LICENSE).