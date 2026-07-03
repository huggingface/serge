# Serge K8s normalize backend — deployment & verification status

_Last updated: 2026-07-01. Context: first live run of the Kubernetes normalize
gate against `huggingface/transformers`, driven by the nightly
integration-failure triage pipeline._

## TL;DR

The Kubernetes normalize gate now works **end-to-end in production**, verified on
a real PR. Getting there surfaced and fixed **three** distinct bugs plus a
checkout perf issue. Three fixes are merged + live; one fix and a test-mode
convenience are in open PRs awaiting merge.

**Proof:** PR **#46941** — the task exhausted its tool budget (force-final path),
the normalize gate ran (`make style` + checkers in a one-shot Job on the shared
EFS worktree), returned **clean**, and serge committed the normalized worktree and
pushed the PR. This exercised exactly the path that was previously bypassed.

## Deployment target

- Cluster / context: `infra:opensource-aws-use1-prod-54`
- Namespace / release: `serge` (prod, `serge.huggingface.tech`)
- Values: `deploy/helm/env/prod.yaml` + local (gitignored) overlay
  `deploy/helm/env/normalize.example.yaml`
- Current live image: `ghcr.io/huggingface/serge:sha-eab6420` (Helm rev 18)

## What was verified working

- Triage workflow → GitHub Actions OIDC → serge `/tasks` dispatch (fan-out of one
  task per failure group).
- serge stages the checkout on the **RWX EFS** worktree PVC (`serge-worktrees`,
  `serge-tasks-efs-sc`, mounted at `/var/lib/reviewbot-clones`).
- The normalize gate creates a one-shot **Job** on the CAST AI node pool, which
  pulls `huggingface/transformers-quality:latest`, runs the fix command as a
  non-root (uid 1000) pod under a deny-all-egress NetworkPolicy, on the shared
  EFS worktree; serge reads the exit code + logs and deletes the Job.
- On a clean normalizer result serge commits the **validated, normalized**
  worktree and opens/updates the PR (draft → ready).

## Bugs found & fixes

| # | Bug | Effect | Fix | PR | Status |
|---|-----|--------|-----|----|--------|
| 1 | `serge-normalize` Role missing the `batch/jobs/status` subresource | Every Job 403'd on the first status poll → gate silently fell back to un-normalized | Grant `jobs/status: get` in `deploy/helm/templates/normalize.yaml` | #32 | ✅ merged + live |
| 2 | Overlay pointed at private `ghcr.io/huggingface/transformers-quality` | `ErrImagePull` (anonymous pull denied) | Use the public Docker Hub image `huggingface/transformers-quality:latest` that transformers CI builds (`docker/quality.dockerfile`) | overlay (local) + #32 note | ✅ live |
| 3 | `_run_agentic_loop` skipped `validate()` on the force-final (budget-exhausted) path | The **root cause** — large-repo tasks always exhaust the tool budget, so the gate never ran; patches committed un-normalized | Route the forced final answer through the same `validate()` + retry gate | #32 | ✅ merged + live |
| 4 | LLM output truncation (`finish_reason=length`) hard-failed the task | Tasks lost all work. `Kimi-K2.7-Code` caps output at **16384** tokens; reasoning ate the budget before the JSON closed. Raising `LLM_MAX_TOKENS` is moot (already request 49152) | Retry a truncated final answer as a JSON-only, tools-off, low-reasoning re-ask (bounded) | #33 | ✅ merged + live |
| 5 | Standalone checkout `git clone --local --no-hardlinks` timed out at 180s | Intermittent `could not check out …` — physical object copy of transformers onto EFS is slow under load | `--depth 1` fetch for standalone checkouts + 600s clone timeout | **#34** | 🔲 open |

## Open PRs (merge order matters)

1. **serge #34** — standalone checkout EFS timeout fix. Merge → rebuild → redeploy.
2. **transformers-ci #7** — expose `max_groups` in the reusable triage workflow. **Merge first.**
3. **transformers #47012** — add `max_groups` `workflow_dispatch` input to the caller. Merge after #7.

## Single-task test mode (after #7 + #47012 merge)

Instead of fanning out ~10-20 PRs, dispatch exactly one Serge task:

```bash
gh workflow run nightly-integration-failure-triage-caller.yml \
  --repo huggingface/transformers -f dry_run=false -f max_groups=1
```

(The CLI already supports `--max-groups`; these PRs just expose it. Empty /
scheduled runs are unchanged.)

## Redeploy recipe

```bash
./deploy/scripts/deploy.sh \
  --context infra:opensource-aws-use1-prod-54 \
  -n serge \
  -f deploy/helm/env/prod.yaml \
  -f deploy/helm/env/normalize.example.yaml \
  --from-head
```

`-f` is repeatable (later files win); `--from-head` waits for CI to publish the
HEAD `sha-` image, pins it in `prod.yaml`, then rolls out. The normalize overlay
is gitignored (internal cluster identifiers) and kept local-only.

## Known-remaining issues / follow-ups

- **Mode B truncation:** task 46927 hit `finish_reason=tool_calls` after tool-budget
  exhaustion (forced-final still emitted tool calls, unparseable, hard error). The
  force-final path should also recover JSON-only. Not covered by #33.
- **Observability:** reasoning / `<think>` output is not persisted to
  `jobs.history_json` (0 `reasoning` events), so the web UI can't show *why* a task
  failed. Consider persisting a bounded reasoning tail.
- **`no_fix` → meta issue:** a `no_fix` outcome is recorded only in serge's task
  journal; it does **not** post back to the triage/meta issue. The nicely-formatted
  rationale ("…no safe fix… Relates to #NNNNN") sits unused in `result_json`. To
  close: plumb `issue_number` through the task spec (the CLI already knows it) and
  `post_issue_comment` on `no_change`.
- **Fail-open by design:** any sandbox error is treated as "normalizer unavailable"
  and the patch is accepted un-normalized. Intentional ("better an imperfect PR than
  a lost fix"), but it masked bugs #1/#2/#3 — consider louder alerting on
  `NormalizeError`.
- **Throughput:** every K8s task pays the full object-copy-to-EFS cost. A shared
  read-only base clone + per-task overlay (or a reflink/CoW FS) would remove it —
  larger change, see also `SERGE_PERTASK_AGENT_POD_PROPOSAL.md`.
- **Wiz admission (AUDIT):** `ghcr.io/huggingface/serge:sha-*` images don't pass the
  Wiz image-trust policy (AUDIT mode, non-blocking). Worth an infra follow-up.

## Run/observe cheatsheet

```bash
# watch normalize Job pods
kubectl -n serge get pods -l serge.io/sandbox=normalize -w

# recent serge logs
./deploy/scripts/logs.sh --context infra:opensource-aws-use1-prod-54 -n serge --since 15m

# a task's step trace (why it failed / did the gate run)
kubectl -n serge exec -i <serge-pod> -- python - <<'PY'
import sqlite3, os, json
db = os.environ.get("WEB_STORE_PATH", "/var/lib/reviewbot/jobs.db")
c = sqlite3.connect(db)
jid = "<task-id>"
_, status, hist = c.execute(
    "select id,status,history_json from jobs where id=?", (jid,)
).fetchone()
print("status=", status)
for e in json.loads(hist or "[]"):
    if e.get("kind") in ("step", "log", "error"):
        print(f"[{e['kind']}] {str(e.get('text'))[:140]}")
PY
```
