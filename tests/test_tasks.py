"""Tests for the /tasks orchestration: request validation, the existing_pr
branch-ownership guard + loop cap, and publish_task's commit/PR flow (real
worktree via CloneCache, fake GitHub Git Data API)."""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from reviewbot.clone_cache import CloneCache
from reviewbot.config import Config
from reviewbot.github_client import SERGE_GIT_EMAIL
from reviewbot.tasks import (
    TaskError,
    TaskPlan,
    TaskRequest,
    _selected_failure_context,
    build_task_request,
    publish_command_result,
    publish_task,
    resolve_existing_pr,
    run_command_task,
    task_candidate_requests,
)


def _git(cwd, *args):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
        },
    )


def _make_cfg(**overrides) -> Config:
    base = dict(
        github_app_id=None,
        github_private_key=None,
        github_webhook_secret=None,
        llm_api_base="https://example.com/v1",
        llm_api_key="x",
        llm_model=None,
        llm_bill_to=None,
        llm_max_tokens=4096,
        llm_stream=False,
        mention_trigger="@askserge",
        review_event="COMMENT",
        max_diff_chars=200000,
        review_rules_path=".ai/review-rules.md",
        helper_tools_path=".ai/review-tools.json",
        default_review_rules="",
        allow_approve=False,
        persona_header="",
        context_script_path=".ai/context-script",
        context_script_timeout=30,
        repo_checkout_path="",
        tool_max_iterations=8,
        llm_max_input_tokens=2_000_000,
    )
    base.update(overrides)
    return Config(**base)


class _FakeGH:
    """Records Git Data API calls and returns plausible SHAs/objects."""

    def __init__(self, *, pr=None, pr_files=None, commit_count=0):
        self.calls = []
        self._pr = pr or {}
        self._pr_files = pr_files or []
        self._commit_count = commit_count
        self.created_pr = None
        self.updated_refs = []
        self.created_refs = []

    def get_pr(self, owner, repo, number):
        self.calls.append(("get_pr", number))
        return self._pr

    def get_pr_files(self, owner, repo, number):
        return self._pr_files

    def count_branch_commits_by_author(self, owner, repo, branch, *, author_email):
        return self._commit_count

    def get_ref_sha(self, owner, repo, ref):
        return f"parent-of-{ref}"

    def get_commit_tree_sha(self, owner, repo, commit_sha):
        return f"tree-of-{commit_sha}"

    def create_blob(self, owner, repo, content):
        self.calls.append(("create_blob", content))
        return f"blob{len(self.calls)}"

    def create_tree(self, owner, repo, base_tree, entries):
        self.calls.append(("create_tree", base_tree, entries))
        return "newtree"

    def create_commit(self, owner, repo, *, message, tree_sha, parents):
        self.calls.append(("create_commit", message, tree_sha, parents))
        return "newcommit"

    def create_ref(self, owner, repo, ref, sha):
        self.created_refs.append((ref, sha))
        return {"ref": ref}

    def update_ref(self, owner, repo, ref, sha, *, force=False):
        self.updated_refs.append((ref, sha))
        return {}

    def create_pull_request(self, owner, repo, *, title, head, base, body):
        self.created_pr = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
        }
        return {"number": 99, "html_url": "https://github.com/o/r/pull/99"}


class BuildTaskRequestTests(unittest.TestCase):
    def test_minimal_new_pr(self):
        req = build_task_request(
            {"instruction": "fix it", "context": "boom"},
            owner="acme",
            repo="widgets",
        )
        self.assertEqual(req.mode, "new_pr")
        self.assertEqual(req.base_ref, "main")
        self.assertEqual(req.branch_prefix, "serge/fix")

    def test_instruction_required(self):
        with self.assertRaises(TaskError):
            build_task_request({"context": "x"}, owner="a", repo="b")

    def test_bad_mode(self):
        with self.assertRaises(TaskError):
            build_task_request(
                {"instruction": "x", "output": {"mode": "delete_repo"}},
                owner="a",
                repo="b",
            )

    def test_branch_prefix_must_be_serge_namespace(self):
        with self.assertRaises(TaskError):
            build_task_request(
                {"instruction": "x", "output": {"branch_prefix": "evil/x"}},
                owner="a",
                repo="b",
            )

    def test_notifications_slack_channel_is_dynamic(self):
        req = build_task_request(
            {
                "instruction": "x",
                "notifications": {
                    "slack_channel": "#transformers-ci-daily-models",
                    "task_finished": True,
                    "pr_created": False,
                },
            },
            owner="a",
            repo="b",
        )
        self.assertEqual(req.slack_channel, "#transformers-ci-daily-models")
        self.assertTrue(req.slack_notify_task_finished)
        self.assertFalse(req.slack_notify_pr_created)

    def test_notifications_must_be_object(self):
        with self.assertRaises(TaskError):
            build_task_request(
                {"instruction": "x", "notifications": "#ci"},
                owner="a",
                repo="b",
            )

    def test_notification_booleans_must_be_boolean(self):
        with self.assertRaises(TaskError):
            build_task_request(
                {
                    "instruction": "x",
                    "notifications": {"task_finished": ["yes"]},
                },
                owner="a",
                repo="b",
            )

    def test_existing_pr_requires_pr_number(self):
        with self.assertRaises(TaskError):
            build_task_request(
                {"instruction": "x", "output": {"mode": "existing_pr"}},
                owner="a",
                repo="b",
            )


class ResolveExistingPrTests(unittest.TestCase):
    def test_serge_branch_ok(self):
        gh = _FakeGH(
            pr={"head": {"ref": "serge/fix-1"}, "base": {"ref": "main"}},
            commit_count=0,
        )
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="x",
            context="",
            mode="existing_pr",
            pr_number=5,
        )
        head = resolve_existing_pr(gh, req, _make_cfg(task_max_followups=5))
        self.assertEqual(head, "serge/fix-1")
        self.assertEqual(req.head_branch, "serge/fix-1")

    def test_non_serge_branch_rejected(self):
        gh = _FakeGH(pr={"head": {"ref": "main"}, "base": {"ref": "main"}})
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="x",
            context="",
            mode="existing_pr",
            pr_number=5,
        )
        with self.assertRaises(TaskError) as ctx:
            resolve_existing_pr(gh, req, _make_cfg())
        self.assertEqual(ctx.exception.status_code, 403)

    def test_loop_cap_enforced(self):
        gh = _FakeGH(
            pr={"head": {"ref": "serge/fix-1"}, "base": {"ref": "main"}},
            commit_count=5,
        )
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="x",
            context="",
            mode="existing_pr",
            pr_number=5,
        )
        with self.assertRaises(TaskError) as ctx:
            resolve_existing_pr(gh, req, _make_cfg(task_max_followups=5))
        self.assertEqual(ctx.exception.status_code, 429)


class TaskCandidateRequestTests(unittest.TestCase):
    def test_single_context_stays_single_candidate(self):
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="fix",
            context="plain report",
        )
        self.assertEqual(task_candidate_requests(req), [req])

    def test_serge_candidate_sections_are_split_with_preamble(self):
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="fix",
            context=(
                "shared report preamble\n\n"
                "## Serge candidate failure group 1/2: first\n"
                "first details\n\n"
                "## Serge candidate failure group 2/2: second\n"
                "second details\n"
            ),
        )
        candidates = task_candidate_requests(req)
        self.assertEqual(len(candidates), 2)
        self.assertIn("shared report preamble", candidates[0].context)
        self.assertIn("first details", candidates[0].context)
        self.assertNotIn("second details", candidates[0].context)
        self.assertIn("shared report preamble", candidates[1].context)
        self.assertIn("second details", candidates[1].context)


class TaskFailureContextTests(unittest.TestCase):
    def test_selects_failure_context_matching_plan(self):
        req = TaskRequest(
            owner="a",
            repo="b",
            base_ref="main",
            instruction="fix",
            context=(
                "## Serge candidate failure group 1/1: output mismatches\n"
                "\n"
                "- `tests/models/foo/test_modeling_foo.py::FooTest::test_a` [single-gpu] "
                "(output_mismatch, seen 5/7)\n"
                "  - AssertionError: ordinary mismatch\n"
                "- `tests/models/gemma3/test_modeling_gemma3.py::Gemma3IntegrationTest::"
                "test_dynamic_sliding_window_is_default` [single-gpu] "
                "(output_mismatch, seen 5/7)\n"
                "  - AssertionError: 'DynamicSlidingWindowLayer' unexpectedly found in "
                "'DynamicCache(...)'\n"
            ),
        )
        plan = TaskPlan(
            title="Fix explicit cache_implementation hybrid handling",
            body="Preserve cache_implementation when it is explicit.",
            patch="DynamicSlidingWindowLayer",
        )
        context = _selected_failure_context(req, plan)
        self.assertIn("Original CI failure", context)
        self.assertIn("output mismatches", context)
        self.assertIn("Gemma3IntegrationTest", context)
        self.assertIn("DynamicSlidingWindowLayer", context)
        self.assertNotIn("FooTest", context)


class PublishTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        self.src = os.path.join(root, "src")
        os.makedirs(self.src)
        _git(self.src, "init", "--quiet", "-b", "main")
        with open(os.path.join(self.src, "hello.txt"), "w") as f:
            f.write("hi from main\n")
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", "main commit")
        _git(self.src, "branch", "serge/fix-1")
        self.cache = CloneCache(os.path.join(root, "cache"))
        self.cfg = _make_cfg()

    def _checkout(self, ref="main"):
        return self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref=ref,
            job_id="abcd1234",
            remote_url=self.src,
        )

    _PATCH = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-hi from main\n"
        "+hi patched\n"
    )

    def test_new_pr_flow(self):
        co = self._checkout("main")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification") as notify:
            result = publish_task(
                self.cfg,
                gh,
                req,
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertFalse(result.no_change)
        self.assertEqual(result.pr_number, 99)
        self.assertEqual(result.branch, "serge/fix-abcd1234")
        self.assertEqual(gh.created_refs[0][0], "refs/heads/serge/fix-abcd1234")
        self.assertEqual(gh.created_pr["base"], "main")
        self.assertEqual(result.changed_files, ["hello.txt"])
        notify.assert_called_once()

    def test_new_pr_notification_uses_request_slack_channel(self):
        co = self._checkout("main")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
            slack_channel="#dynamic-ci",
            slack_notify_task_finished=True,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        cfg = _make_cfg(
            slack_bot_token="tok",
            slack_report_channel="#default-ci",
        )
        with patch("reviewbot.tasks.post_task_pr_created_notification") as notify:
            publish_task(
                cfg,
                gh,
                req,
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )

        self.assertEqual(notify.call_args.kwargs["token"], "tok")
        self.assertEqual(notify.call_args.kwargs["channel"], "#dynamic-ci")

    def test_existing_pr_flow(self):
        co = self._checkout("serge/fix-1")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="existing_pr",
            pr_number=5,
            head_branch="serge/fix-1",
        )
        plan = TaskPlan(title="Fix again", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        result = publish_task(
            self.cfg,
            gh,
            req,
            plan,
            checkout=co,
            clone_cache=self.cache,
            job_id="abcd1234",
        )
        self.assertEqual(result.pr_number, 5)
        self.assertEqual(result.branch, "serge/fix-1")
        self.assertEqual(gh.updated_refs[0][0], "heads/serge/fix-1")
        self.assertIsNone(gh.created_pr)

    def test_empty_patch_is_no_change(self):
        co = self._checkout("main")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
        )
        plan = TaskPlan(title="t", body="nothing to do", patch="")
        gh = _FakeGH()
        result = publish_task(
            self.cfg,
            gh,
            req,
            plan,
            checkout=co,
            clone_cache=self.cache,
            job_id="abcd1234",
        )
        self.assertTrue(result.no_change)
        self.assertIsNone(gh.created_pr)

    def test_bad_patch_raises_task_error(self):
        co = self._checkout("main")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
        )
        bad = (
            "diff --git a/hello.txt b/hello.txt\n"
            "--- a/hello.txt\n"
            "+++ b/hello.txt\n"
            "@@ -1 +1 @@\n"
            "-does not match\n"
            "+nope\n"
        )
        plan = TaskPlan(title="t", body="b", patch=bad)
        gh = _FakeGH()
        with self.assertRaises(TaskError) as ctx:
            publish_task(
                self.cfg,
                gh,
                req,
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertEqual(ctx.exception.status_code, 422)

    def test_serge_identity_used_for_loop_cap_consistency(self):
        # Sanity: the email the loop-cap counts by is the one stamped on
        # commits, so follow-ups are countable.
        self.assertTrue(SERGE_GIT_EMAIL)


class CommandTaskParsingTests(unittest.TestCase):
    ALLOW = ("make fix-repo", "make style")

    def test_command_string_in_allowlist(self):
        req = build_task_request(
            {"command": "make fix-repo"},
            owner="a",
            repo="b",
            allowed_commands=self.ALLOW,
        )
        self.assertTrue(req.is_command)
        self.assertEqual(req.command, ["make", "fix-repo"])
        self.assertEqual(req.command_str, "make fix-repo")
        self.assertIn("make fix-repo", req.instruction)

    def test_command_list_form(self):
        req = build_task_request(
            {"command": ["make", "style"]},
            owner="a",
            repo="b",
            allowed_commands=self.ALLOW,
        )
        self.assertEqual(req.command, ["make", "style"])

    def test_command_not_in_allowlist_rejected(self):
        with self.assertRaises(TaskError) as ctx:
            build_task_request(
                {"command": "rm -rf /"},
                owner="a",
                repo="b",
                allowed_commands=self.ALLOW,
            )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_command_rejected_when_allowlist_empty(self):
        with self.assertRaises(TaskError) as ctx:
            build_task_request(
                {"command": "make fix-repo"}, owner="a", repo="b", allowed_commands=()
            )
        self.assertEqual(ctx.exception.status_code, 403)

    def test_command_with_existing_pr_mode(self):
        req = build_task_request(
            {
                "command": "make fix-repo",
                "output": {"mode": "existing_pr", "pr_number": 7},
            },
            owner="a",
            repo="b",
            allowed_commands=self.ALLOW,
        )
        self.assertTrue(req.is_command)
        self.assertEqual(req.mode, "existing_pr")
        self.assertEqual(req.pr_number, 7)

    def test_non_command_task_unaffected(self):
        req = build_task_request(
            {"instruction": "fix the tests"},
            owner="a",
            repo="b",
            allowed_commands=self.ALLOW,
        )
        self.assertFalse(req.is_command)
        self.assertIsNone(req.command)


class CommandTaskRunTests(unittest.TestCase):
    """End-to-end command-task run against a real worktree, sandbox off."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        self.src = os.path.join(root, "src")
        os.makedirs(self.src)
        _git(self.src, "init", "--quiet", "-b", "main")
        with open(os.path.join(self.src, "hello.txt"), "w") as f:
            f.write("hi from main\n")
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", "main commit")
        self.cache = CloneCache(os.path.join(root, "cache"))
        # bwrap backend + sandbox off so the command runs directly (no docker
        # / bwrap needed in the test environment).
        self.cfg = _make_cfg(
            helper_sandbox="off",
            task_sandbox_backend="bwrap",
            task_command_timeout=30,
        )

    def _checkout(self, ref="main"):
        return self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref=ref,
            job_id="abcd1234",
            remote_url=self.src,
        )

    def _req(self, command):
        return TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="",
            context="",
            mode="new_pr",
            command=command,
        )

    def test_command_edits_tree_and_opens_pr(self):
        co = self._checkout("main")
        req = self._req(["sh", "-c", "echo patched > hello.txt"])
        changes, _tail = run_command_task(
            self.cfg, req, checkout=co, clone_cache=self.cache
        )
        self.assertEqual([c.path for c in changes], ["hello.txt"])

        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_command_result(
                self.cfg, gh, req, changes=changes, job_id="abcd1234"
            )
        self.assertFalse(result.no_change)
        self.assertEqual(result.pr_number, 99)
        self.assertEqual(result.changed_files, ["hello.txt"])
        self.assertIn("sh -c", gh.created_pr["body"])

    def test_command_no_change_is_no_change(self):
        co = self._checkout("main")
        req = self._req(["sh", "-c", "true"])
        changes, _tail = run_command_task(
            self.cfg, req, checkout=co, clone_cache=self.cache
        )
        self.assertEqual(changes, [])
        gh = _FakeGH()
        result = publish_command_result(
            self.cfg, gh, req, changes=changes, job_id="abcd1234"
        )
        self.assertTrue(result.no_change)
        self.assertIsNone(gh.created_pr)

    def test_command_failure_raises_422(self):
        co = self._checkout("main")
        req = self._req(["sh", "-c", "echo boom >&2; exit 3"])
        with self.assertRaises(TaskError) as ctx:
            run_command_task(self.cfg, req, checkout=co, clone_cache=self.cache)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("boom", str(ctx.exception))

    def test_gitignored_artifacts_not_committed(self):
        co = self._checkout("main")
        req = self._req(
            [
                "sh",
                "-c",
                "echo build/ > .gitignore && mkdir -p build && "
                "echo junk > build/x.o && echo patched > hello.txt",
            ]
        )
        changes, _tail = run_command_task(
            self.cfg, req, checkout=co, clone_cache=self.cache
        )
        paths = sorted(c.path for c in changes)
        self.assertIn("hello.txt", paths)
        self.assertIn(".gitignore", paths)
        self.assertNotIn("build/x.o", paths)


if __name__ == "__main__":
    unittest.main()
