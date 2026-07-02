#!/usr/bin/env python3
"""Local driver for the full task pipeline (LLM patch -> Git Data API write),
authenticating as the GitHub App. Exercises everything the /tasks worker does
except the OIDC/HTTP front door, so it can be run from a VPN box without an
Actions OIDC token.

Usage:
    # new_pr (first call)
    python scripts/run_task_local.py owner/repo "instruction" "context" [base_ref]

    # existing_pr (follow-up call — simulates CI re-running after a failure)
    TASK_MODE=existing_pr TASK_PR_NUMBER=142 \\
      python scripts/run_task_local.py owner/repo "instruction" "context"

Reads LLM + App settings from aws/reviewbot-web.env (GITHUB_PRIVATE_KEY_PATH is
resolved relative to that file).
"""

import dataclasses
import os
import sys
import tempfile
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, "..", "aws", "reviewbot-web.env")


def _load_env():
    for line in open(ENV_FILE):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)
    # Resolve the PEM path relative to the env file (as deploy does).
    pk = os.environ.get("GITHUB_PRIVATE_KEY_PATH", "")
    if pk and not os.path.isabs(pk):
        os.environ["GITHUB_PRIVATE_KEY_PATH"] = os.path.normpath(
            os.path.join(HERE, "..", "aws", pk)
        )


def main() -> int:
    if len(sys.argv) < 4 or "/" not in sys.argv[1]:
        print(__doc__)
        return 2
    owner, repo = sys.argv[1].split("/", 1)
    instruction, context = sys.argv[2], sys.argv[3]
    base_ref = sys.argv[4] if len(sys.argv) > 4 else "main"

    _load_env()
    from reviewbot.clone_cache import CloneCache
    from reviewbot.config import Config
    from reviewbot.github_auth import installation_id_for_repo, installation_token
    from reviewbot.github_client import GitHubClient
    from reviewbot.tasks import (
        TaskError,
        TaskRequest,
        TaskResult,
        prepare_task,
        publish_task,
        resolve_existing_pr,
        task_candidate_requests,
    )

    cfg = Config.from_env(require_app=False, require_web=False)
    cfg = dataclasses.replace(
        cfg,
        llm_max_tokens=cfg.task_llm_max_tokens or cfg.llm_max_tokens,
        llm_max_input_tokens=cfg.task_llm_max_input_tokens
        or cfg.llm_max_input_tokens,
        tool_max_iterations=cfg.task_tool_max_iterations or cfg.tool_max_iterations,
        tool_max_iterations_strict=cfg.task_tool_max_iterations is not None,
    )
    app_id = os.environ["GITHUB_APP_ID"]
    pk = open(os.environ["GITHUB_PRIVATE_KEY_PATH"]).read()

    mode = os.environ.get("TASK_MODE", "new_pr")
    pr_number_env = os.environ.get("TASK_PR_NUMBER")

    iid = installation_id_for_repo(app_id, pk, owner, repo)
    token = installation_token(app_id, pk, iid)
    gh = GitHubClient(token)

    job_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        cache = CloneCache(os.path.join(tmp, "clones"))
        req = TaskRequest(
            owner=owner, repo=repo, base_ref=base_ref,
            instruction=instruction, context=context, mode=mode,
            pr_number=int(pr_number_env) if pr_number_env else None,
        )

        existing_diff = None
        if mode == "existing_pr":
            # Mirror the worker: resolve the serge fix branch (guard + loop
            # cap), check it out, and feed the prior attempt as context.
            head_branch = resolve_existing_pr(gh, req, cfg)
            print(f"[resolve] follow-up on {head_branch} (PR #{req.pr_number}), "
                  f"base {req.base_ref}")
            ref_to_checkout = head_branch
            files = gh.get_pr_files(owner, repo, req.pr_number)
            existing_diff = "\n".join(
                f"--- {f.get('filename')} ---\n{f.get('patch') or ''}" for f in files
            )
        else:
            ref_to_checkout = base_ref

        co = cache.acquire_ref(
            token,
            owner,
            repo,
            ref_to_checkout,
            job_id=job_id,
            depth=1,
            standalone=cfg.needs_isolated_checkout,
        )
        if co is None:
            print(f"could not checkout {owner}/{repo}@{ref_to_checkout}")
            return 1
        cfg = dataclasses.replace(cfg, repo_checkout_path=co.path)

        def emit(kind, text):
            print(f"[{kind}] {text}")

        candidate_reqs = task_candidate_requests(req)
        last_no_change = None
        result = None
        for index, candidate_req in enumerate(candidate_reqs, start=1):
            if len(candidate_reqs) > 1:
                emit(
                    "log",
                    f"Starting candidate {index}/{len(candidate_reqs)} in a fresh LLM cycle",
                )
            plan = prepare_task(
                cfg,
                candidate_req,
                checkout=co,
                clone_cache=cache,
                existing_diff=existing_diff,
                chunk_callback=emit,
            )
            print(f"\n--- proposed patch ({len(plan.patch)} chars) ---")
            print(plan.patch[:2000])
            print("--- end patch ---\n")
            try:
                attempt_result = publish_task(
                    cfg, gh, candidate_req, plan,
                    checkout=co, clone_cache=cache, job_id=job_id, emit=emit,
                )
            except TaskError as exc:
                if exc.status_code == 422 and index < len(candidate_reqs):
                    emit(
                        "log",
                        f"Candidate {index}/{len(candidate_reqs)} did not produce "
                        f"an applicable patch: {exc}. Moving to the next group.",
                    )
                    continue
                raise
            if attempt_result.no_change and index < len(candidate_reqs):
                last_no_change = attempt_result
                emit(
                    "log",
                    f"Candidate {index}/{len(candidate_reqs)} produced no fix. "
                    "Moving to the next group.",
                )
                continue
            result = attempt_result
            break
        if result is None:
            result = last_no_change or TaskResult(
                mode=req.mode,
                no_change=True,
                message="No candidate produced a safe fix.",
            )
        print("\nRESULT:", result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
