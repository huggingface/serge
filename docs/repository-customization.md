---
title: Repository Customization
---

Repositories can shape reviews with files on their default branch. Reading
customization from the default branch prevents PR authors from changing review
policy in the same PR being reviewed.

## Review Rules

Create `.ai/review-rules.md`:

```markdown
# Review rules

- Focus on correctness, security, and behavior changes.
- Flag missing tests for user-visible behavior.
- Ignore generated files unless they are the only source of truth.
- Do not comment on style-only issues unless they hide a bug.
```

The rules are injected into the reviewer prompt for full reviews and inline
follow-up replies.

## Context Script

An executable `.ai/context-script` can add repository-specific context. The
reviewer sends this JSON on stdin:

```json
{
  "title": "PR title",
  "body": "PR body",
  "files": [
    {
      "path": "src/example.py",
      "status": "modified",
      "additions": 12,
      "deletions": 3,
      "previous_path": null
    }
  ]
}
```

The script can print plain text, which is added to the prompt as repository
context.

It can also print a JSON object:

```json
{
  "context": "This repo generates files in src/generated from templates.",
  "skip_files": ["src/generated/example.py"]
}
```

`skip_files` removes matching paths from the review diff. Use it for generated
or mirrored files that should not receive inline comments.

If the script is missing, not executable, times out, exits non-zero, or prints
empty output, it is ignored.

## Helper Tools

`.ai/review-tools.json` can expose tightly scoped helper commands to the
reviewer when a local checkout is available.

```json
{
  "helpers": [
    {
      "name": "lint_file",
      "description": "Run the project linter on a file.",
      "command": ["python", "-m", "ruff", "check"],
      "cwd": ".",
      "allow_args": true,
      "max_args": 4,
      "timeout_seconds": 30
    }
  ]
}
```

Helper names must match `[A-Za-z][A-Za-z0-9_-]*` and cannot conflict with
built-in tools. Commands run without a shell, with a minimal environment that
omits secrets.

Helpers may declare an optional install hook:

```json
{
  "helpers": [
    {
      "name": "mlinter",
      "description": "Run the model linter.",
      "command": ["mlinter"],
      "allow_args": true,
      "max_args": 4,
      "install": ["pip", "install", "transformers-mlinter"]
    }
  ]
}
```

Install hooks are intentionally restricted. Only `pip` is supported, package
arguments are validated, and URL/VCS installs or custom indexes are rejected.

## Built-In Tools

When `REPO_CHECKOUT_PATH` is set, the reviewer can use:

| Tool | Purpose |
| ---- | ------- |
| `read_file` | Read a bounded slice of a file. |
| `list_dir` | List a directory inside the checkout. |
| `grep` | Search with `git grep -E`. |
| `fetch_url` | Fetch `https://huggingface.co/*` links for verification. |

Paths are resolved against the checkout root and rejected if they escape it.
