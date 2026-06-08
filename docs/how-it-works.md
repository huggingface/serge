---
title: How It Works
---

## Full Review Flow

```text
trigger comment -> gate event -> fetch PR -> annotate diff -> call LLM -> validate -> publish
```

1. A comment event arrives from GitHub Actions or a GitHub App webhook.
2. The trigger gate checks the event type, action, comment author association,
   trigger phrase, and PR state.
3. The reviewer fetches PR metadata and changed files.
4. Each diff hunk is annotated with addressable line markers such as `[R 42]`
   for the new side and `[L 11]` for the old side.
5. Repository rules, optional context-script output, and optional tool specs
   are added to the prompt.
6. The LLM returns a review summary, review event, and inline comments.
7. Inline comments are validated against the real diff positions.
8. Invalid comments are dropped.
9. The review is published, or in web app mode stored as an editable draft.

## Follow-Up Flow

When someone mentions the trigger on a PR review comment, the reviewer answers
that specific inline question in the same thread. The prompt includes the
comment anchor, the diff hunk, the follow-up question, repo rules, and any
available tools.

Follow-up replies are plain Markdown, not full review JSON.

## Staged Reviews

In the web app, the LLM output becomes a draft. The reviewer can:

- edit the summary;
- change the review event;
- edit inline comment text;
- discard individual comments;
- publish or discard the whole draft.

`APPROVE` events are blocked unless `ALLOW_APPROVE=1`.

## Tool Loop

When a repository checkout is available, the model can request bounded tool
calls to inspect context beyond the diff. The loop stops when the model returns
a final review, when input-token caps are hit, or when tool-iteration limits
are reached.
