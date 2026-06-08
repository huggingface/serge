---
title: LLM Providers
---

`ai-reviewer` talks to OpenAI-compatible chat completion endpoints. The
endpoint should support:

```text
POST {base}/chat/completions
GET  {base}/models
```

The reviewer asks for JSON output during full reviews. If a provider ignores
JSON mode, the reviewer attempts to extract JSON from the returned text.

## Common Bases

| Provider | Base URL |
| -------- | -------- |
| OpenAI | `https://api.openai.com/v1` |
| Hugging Face Router | `https://router.huggingface.co/v1` |
| Local vLLM/TGI/LM Studio | your local `/v1` endpoint |
| Custom | any compatible endpoint |

The web app has built-in provider choices for Hugging Face, OpenAI, Anthropic,
and custom endpoints. Custom provider configs must include an API base URL.

## Model Selection

Set `LLM_MODEL` or the Action input `llm_model` to choose a model explicitly.
If omitted, the reviewer asks the endpoint for `/models` and uses the first
returned model.

In the web app, model selection follows this order:

1. model entered on the review form;
2. provider config default model;
3. provider-specific static default, if any;
4. provider auto-discovery.

## Streaming

`LLM_STREAM=true` consumes streaming SSE responses. Streaming is useful for the
web app because the UI can show tokens, reasoning chunks, tools, and progress
live.

The Action defaults streaming off; server env defaults streaming on.

## Reasoning Models

For models that spend completion tokens on reasoning before emitting JSON,
increase `LLM_MAX_TOKENS`. If a provider supports a `reasoning_effort` field,
set `LLM_REASONING_EFFORT` to pass it through.

## Hugging Face Billing Header

`LLM_BILL_TO` sends `X-HF-Bill-To` for Hugging Face Router requests. Use it
when your Hugging Face token's Inference Providers permission is scoped to an
organization.
