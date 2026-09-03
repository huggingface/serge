# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **One extra LLM pass that shortens serge's own prose** (`reviewbot/brevity.py`).
  Prompt discipline had already been tried: the task prompt's `LENGTH` block
  tells the model to comment only where the reason for a line is not evident
  from the line, and verbose models keep writing four lines of comment over a
  one-line assignment. So both flows now get a single dedicated call whose only
  job is to shorten:
  - **Tasks** (`TASK_COMMENT_BREVITY`, default on): every comment the patch
    added, in one call, **in the worktree just before the repo normalizer
    runs** — so the comments that reach a PR are the ones the repo's own
    formatter saw and the gate accepted.
  - **Reviews** (`REVIEW_COMMENT_BREVITY`, default on): the summary and each
    inline comment body, before the draft is stored, so the operator edits the
    text that will actually be posted.

  The pass can only ever shorten. The model is handed comment *text* and
  returns comment *text* — the marker, the indent and the wrapping are
  serge's, so a reply cannot become a code line. Comments are located with
  `tokenize` (running on the applied worktree is what makes that possible: a
  `#` inside a string literal is not a comment, and a diff line cannot tell
  you), and a rewritten file is kept only if its **code-token stream is
  byte-identical** and it still parses. Machine-read comments are never sent at
  all — lint suppressions, typing pragmas, formatter switches, licence headers,
  and transformers' own `Copied from`/`Ignore copy` (shortening one would fail
  `make repo-consistency`). In a review, a fenced ```suggestion block is masked
  out before the model sees it and must come back intact, an empty answer is
  treated as "keep" so brevity can never delete a finding, and any body that
  came back longer keeps its original. Every failure path — provider error,
  unparseable reply, guard tripping — leaves the original text, which is the
  behaviour that shipped before this existed. A normalize failure that follows
  a condensed patch says so in the feedback, because the pass moves the line
  numbers the normalizer reports. Its tokens are folded into the job's totals
  but not into `session`'s turn count: it is not a loop turn.

  **Measured live 2026-09-03** against Kimi at prod's caps, on real serge
  output rather than fixtures:

  - **Reviews are where the verbosity is.** Four published review bodies
    (three summaries + one inline comment, fetched back off the transformers
    PRs serge reviewed): **3,160 → 1,929 chars, −39%, in one call** of 1,177
    in / 443 out tokens, 2.6s. Every finding survived and the ```suggestion
    block came back byte-identical.
  - **Patch comments are a smaller prize than expected.** Of seven archived
    local task runs that produced a patch, only two contain a comment over the
    100-char floor at all; the worst (236 chars / 3 lines) condensed to 197
    (−17%), the phimoe one 134 → 126. A full `--brevity` replay of the phimoe
    group produced a patch with **no comments whatsoever** — the pass is
    cheap when there is nothing to do, and now says so.
  - The first live review run **dropped three verdict statements** — *"No
    correctness, security, or style issues were found"*, a *"correctly
    fixes"*, and the object of *"verified on a GPU runner that the new
    expectation matches current behaviour"*. Shortening a sign-off into
    silence is a different review, so `_REVIEW_SYSTEM_PROMPT` now keeps the
    verdict (a clean one included) and what a claim is *about*; the re-run
    restored all three at the same −39%.

- The tool-repeat guard is now **path-aware**. It keyed on the exact arguments,
  so fifty-three reads of one file at a different line range each time were
  fifty-three distinct signatures and it counted zero repeats — while that is
  the shape that dominates in practice (prod task `9d210794`: 137 of 153 calls
  re-opened a path it had already opened). A second counter tracks visits per
  opened path, with its own allowance (`TOOL_PATH_REVISIT_LIMIT`, default 3) and
  its own cut-off (`TOOL_PATH_TRIP_AFTER`, default 40). The correction names the
  file and the ranges already served — *"you have already read `modular_blt.py`
  6 times in this session (lines 1-200, 180-420, 400-650)"* — rather than only
  saying stop, because the first is something a model can act on. Still not a
  cache: every call executes for real, since a re-read after a patch must return
  the new content. A session cut off this way reports `stop_reason
  =path_revisit_guard`, distinct from `repeat_guard`.

- Per-job agent-loop metrics, exported for Prometheus at `GET /metrics`
  (unauthenticated, like `/healthz`; scraped in-cluster over the pod port).
  Every finished job now records how many turns and tool calls it ran, what it
  spent, how many of its tool calls re-opened a path it had already read, and —
  the one that reorders the rest — **which guard ended the session**, with
  `answered` meaning the model finished on its own terms and every other value
  meaning it did not. The exposition is a rolling window over the jobs the store
  still holds (`WEB_JOB_RETENTION`, 25); Prometheus is the durable side, so read
  history with a range query rather than by curling the endpoint. The same
  record is returned on `GET /tasks/{owner}/{repo}/{id}/status`, so a dispatcher
  can say *why* a group came back `no_fix`.

- GitHub App mode is documented as the zero-config default: installed repos need
  no workflow file or secret, with instructions for overriding gating via an
  explicit workflow and a note on the forked-PR limitation.

### Fixed

- A final answer that is **nothing but leaked tool-call markup** is now
  re-asked instead of parsed. `_needs_final_salvage` only recognised a
  *truncated* or *empty* reply, so markup-only content — non-empty, so it
  looked parseable — went straight to `_extract_json` and raised
  `LLM returned unparseable output` with **zero** recovery attempts. The
  review path had a backstop after parsing; the task path had none, which is
  what 3 of 10 task jobs in the 2026-08-24→26 window died of. The three
  unusable shapes are now one classifier (`_final_answer_defect`), so the
  loop, the log line and the recovery prompt agree on what was wrong, and the
  markup case gets a prompt that names the mistake.
- `_UnparseableLLMOutput` now carries `salvage_attempts`, so the error
  distinguishes "salvage never recognised this shape" (0) from "salvage ran
  and the model still could not produce JSON" (>0). Those are different bugs
  and they used to share one error string.

- The new-review form's per-provider model dropdown is visible again: it now
  lists a provider's models on page load (via `GET /llm-options/models`) instead
  of only after a PR reference resolved to a matching config, and Anthropic's
  `/v1/models` is called with the `anthropic-version` header it requires.
- The trigger gate now ignores comments authored by a bot, so the App never
  reacts to its own output (no self-trigger loops in App mode).
- `grep` no longer under-reports silently. It ran `git grep --max-count=10`, a
  cap that applies *per file* and left no trace in the output, so a
  single-file search returned ten matches that looked like the complete set —
  a rules file with 57 `description` lines came back as ten. The per-file cap
  is now derived from `max_results` (one above it, so a partial answer is
  distinguishable from a complete one), the output is streamed and stopped at
  the cap instead of captured whole and sliced, and a truncated result ends
  with an explicit note telling the model not to treat the count as final. The
  note used to be passed to the 8000-char truncator as a suffix, which dropped
  it whenever the output was under 8000 chars — i.e. on exactly the results
  that looked trustworthy. A single match line is also clipped at 300
  characters so one minified line cannot spend the whole output budget.
- `grep` runs Perl-compatible patterns where git supports PCRE2, falling back
  to POSIX ERE otherwise. Under `-E` a pattern like `TRF\d+` or
  `\bViolation\b` matched nothing and returned `no matches`, which reads as
  "not in this repo" — worse than a truncated answer, because it invents an
  absence. The result header now names the flag actually used, and a no-match
  answer under ERE says so when the pattern needed a feature ERE lacks.
- `grep` searches untracked files too, so a file created by an applied patch
  is visible to the search that follows it (`/tasks` greps after patching).
  Gitignored paths stay excluded, and matches under a denylisted directory
  (`node_modules`, `.venv`, …) are dropped rather than handed to a model that
  `read_file` would then refuse.

## [0.1.0] - 2026-06-17

First public release of `serge`, a GitHub-native AI code reviewer for any
OpenAI-compatible LLM.

### Added

#### Reviewing

- Pull request reviews that read the diff and post inline comments through
  GitHub's Pull Request Reviews API.
- Every proposed inline comment is validated against real diff positions before
  publishing; comments that don't map to a diff line are dropped.
- Trigger comments (`@askserge please review`) and follow-up replies on inline
  review comments.
- The model name used for a review is included in the review output.

#### Deployment modes

- **GitHub Action** — per-repo setup driven from GitHub Actions.
- **GitHub App webhook** — hosted reviewer across many repositories, with
  installation tokens read from the app database.
- **Web app** — human-in-the-loop staged reviews that can be edited or discarded
  before they reach a PR, with webhook calls surfaced in the app UI.

#### Repository customization

- Repository review rules from `.ai/review-rules.md`.
- Optional read-only repository context from `.ai/context-script`.
- Optional helper tools from `.ai/review-tools.json`.

#### LLM providers

- Support for any OpenAI-compatible chat completion endpoint.
- Hugging Face inference provider selection in the web UI.
- Configurable max tokens exposed in the UI.
- Optional context compression via [`headroom`](https://pypi.org/project/headroom-ai/),
  enabled at runtime with `HEADROOM_COMPRESS=1`.

#### Tasks flow (write-capable)

- Optional `/tasks` flow: CI posts a failure report and serge opens a fix PR.
- `existing_pr` mode appends follow-up commits to an existing fix branch instead
  of opening a duplicate PR, enabling a CI-retry loop.
- Loop-cap safety that counts serge-authored commits on a branch to prevent
  infinite retry loops.
- Original failure report included in the task PR body.

#### Security & operations

- Fork and PR code isolated from the review host.
- Explicit handling for reviews triggered on forked PRs.
- Dependabot weekly bumps for GitHub Actions.
- Public documentation site under `docs/`.

[0.1.0]: https://github.com/huggingface/serge/releases/tag/v0.1.0
