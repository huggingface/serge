#!/usr/bin/env python3
"""Throwaway end-to-end validation of the /tasks git-write path — no LLM, no
HTTP API. Hand-writes a patch against a real repo, applies it in a worktree,
and pushes the result through serge's Git Data API methods to open a PR.

This is the Phase-1 acceptance check from TASKS_FLOW_PLAN.md: prove
checkout → branch → commit → PR works before any LLM is wired in.

Usage:
    GITHUB_APP_ID=... GITHUB_PRIVATE_KEY_PATH=... \\
    python scripts/validate_task_write.py owner/repo [base_ref]

The App must be installed on the repo with Contents:write + Pull
Requests:write. Creates branch ``serge/validate-<n>`` and opens a draft-ish
PR titled "serge git-write validation". Delete the branch/PR afterwards.
"""

import os
import sys
import tempfile

from reviewbot.clone_cache import CloneCache
from reviewbot.github_auth import installation_id_for_repo, installation_token
from reviewbot.github_client import GitHubClient


def main() -> int:
    if len(sys.argv) < 2 or "/" not in sys.argv[1]:
        print(__doc__)
        return 2
    owner, repo = sys.argv[1].split("/", 1)
    base_ref = sys.argv[2] if len(sys.argv) > 2 else "main"

    app_id = os.environ["GITHUB_APP_ID"]
    pk_path = os.environ.get("GITHUB_PRIVATE_KEY_PATH")
    pk = os.environ.get("GITHUB_PRIVATE_KEY")
    if pk_path and not pk:
        with open(pk_path) as f:
            pk = f.read()
    assert pk, "set GITHUB_PRIVATE_KEY or GITHUB_PRIVATE_KEY_PATH"

    iid = installation_id_for_repo(app_id, pk, owner, repo)
    token = installation_token(app_id, pk, iid)
    gh = GitHubClient(token)

    with tempfile.TemporaryDirectory() as tmp:
        cache = CloneCache(os.path.join(tmp, "clones"))
        co = cache.acquire_ref(token, owner, repo, base_ref, job_id="validate", depth=1)
        assert co is not None, f"could not checkout {owner}/{repo}@{base_ref}"
        print(f"checked out {owner}/{repo}@{base_ref} -> {co.path}")

        marker = "serge-validation.txt"
        patch = (
            f"diff --git a/{marker} b/{marker}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{marker}\n"
            "@@ -0,0 +1 @@\n"
            "+serge git-write validation\n"
        )
        cache.apply_patch(co, patch)
        changes = cache.collect_changes(co)
        print(f"patch produced {len(changes)} change(s): {[c.path for c in changes]}")

        entries = []
        for ch in changes:
            blob = gh.create_blob(owner, repo, ch.content or b"")
            entries.append(
                {"path": ch.path, "mode": ch.mode, "type": "blob", "sha": blob}
            )
        parent = gh.get_ref_sha(owner, repo, f"heads/{base_ref}")
        base_tree = gh.get_commit_tree_sha(owner, repo, parent)
        tree = gh.create_tree(owner, repo, base_tree, entries)
        commit = gh.create_commit(
            owner,
            repo,
            message="serge git-write validation",
            tree_sha=tree,
            parents=[parent],
        )
        branch = f"serge/validate-{commit[:8]}"
        gh.create_ref(owner, repo, f"refs/heads/{branch}", commit)
        pr = gh.create_pull_request(
            owner,
            repo,
            title="serge git-write validation",
            head=branch,
            base=base_ref,
            body="Automated Phase-1 validation. Safe to close + delete branch.",
        )
        print(f"opened PR #{pr['number']}: {pr['html_url']}")
        print(f"branch: {branch}  commit: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
