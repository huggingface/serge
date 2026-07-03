# Proposal: per-task agent pod (keep the normalizer isolated)

Status: design note / not scheduled. Captured 2026-07-01 during the first live
run of the Kubernetes normalize backend.

## Idea raised

Instead of the serge app pod running the LLM agent loop and offloading only the
normalizer to a separate Job pod, run *all* the per-task work (agent loop +
normalization) in one sandbox pod and have the serge app pod just orchestrate
and collect results.

## Why "everything in the normalize pod" doesn't work as-is

The normalize pod exists precisely to run **arbitrary repo build code**
(`make style`, checkers, libcst, importing the repo) under a hard isolation
contract:

- deny-all-egress NetworkPolicy (`serge.io/sandbox=normalize`)
- non-root, `automountServiceAccountToken: false`, no serge secrets

The agent loop needs the opposite: **network egress** (LLM API at
`router.huggingface.co`), the **LLM API key**, and the **GitHub App private key**
to open the PR. Merging them co-locates secrets + open egress with arbitrary
build-code execution — the exact trust-boundary collapse the isolation prevents.
Concretely, one pod can't be both deny-egress (for the normalizer) and have
egress (for the LLM). Note this is also why `add_dates` was dropped from the
normalize command (needs network, blocked by deny-egress).

("docker in the pod" also implies docker-in-docker → privileged/rootless daemon,
another escalation surface.)

## The version worth pursuing: promote the AGENT loop to a per-task pod

Keep the trust split; relocate only the agent:

```
serge app pod (thin orchestrator: DB, /tasks UI + SSE, dispatch)
  └─ per-task AGENT pod        [LLM egress + LLM key; does NOT run repo build code]
       └─ normalize sub-Job    [deny-egress, no secrets, runs make style/checkers]  ← unchanged
```

### Wins
- Per-task isolation: a runaway / OOM / crashing task is contained in its own
  pod instead of a thread inside serge; can't affect siblings.
- Clean per-task resource limits + node scheduling + accounting.
- Possibly drops the RWX EFS requirement for the agent worktree (local scratch);
  shared storage only needed for the normalize sub-step (or eliminated if the
  sub-Job mounts the agent pod's volume differently).

### Costs / open questions
- Nesting: an agent pod spawning a sub-Job needs Job-create RBAC (same shape as
  the `jobs`/`jobs/status` grant we just fixed).
- Per-task image pull / cold start (transformers-quality is large); need a
  combined agent-runtime image or bind-mounted serge code.
- Rebuild progress streaming: today `emit()` is in-process → SSE; a detached
  agent pod needs a side channel back to serge for live `/tasks` updates + state.
- PR-open credential placement: agent pod (has App key) vs. hand the validated
  patch back to serge to publish.

## Recommendation

Do **not** merge normalization into the agent pod (step backward on isolation).
Consider promoting the agent loop to a per-task pod if per-task isolation and
dropping the RWX-EFS dependency are worth the streaming/lifecycle refactor. It's
a meaningful change, orthogonal to the current backend working correctly.

## Related fixes from this run (already applied)
- Role missing `jobs/status` subresource → normalize Job 403'd instantly →
  silent un-normalized fallback. Fixed in `deploy/helm/templates/normalize.yaml`.
- Overlay pointed at private `ghcr.io/huggingface/transformers-quality`; correct
  image is the public Docker Hub `huggingface/transformers-quality:latest` that
  transformers CI builds (`docker/quality.dockerfile`).
