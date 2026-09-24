"""Tests for the /tasks orchestration: request validation, the existing_pr
branch-ownership guard + loop cap, and publish_task's commit/PR flow (real
worktree via CloneCache, fake GitHub Git Data API)."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reviewbot import relore_tool
from reviewbot import tasks as tasks_module
from reviewbot.clone_cache import Checkout, CloneCache
from reviewbot.config import Config
from reviewbot.github_client import SERGE_GIT_EMAIL
from reviewbot.llm_client import ChatResult
from reviewbot.normalize import NormalizeError
from reviewbot.tasks import (
    MAX_REVIEWERS,
    NormalizeGateBroken,
    TaskError,
    TaskPlan,
    TaskRequest,
    _read_repo_conventions,
    _selected_failure_context,
    _task_validation_retry_messages,
    _validate_patch,
    build_task_request,
    check_task_preflight,
    prompt_prefix_summary,
    publish_task,
    resolve_existing_pr,
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


# Normalize commands for the gate tests. The distinction matters: a normalizer
# that fails only once the patch is in the tree is rejecting the PATCH (feed it
# back to the model), while one that fails on the pristine checkout is a BROKEN
# GATE (no patch can pass; raise NormalizeGateBroken). Both test suites' _PATCH
# rewrites hello.txt to "hi patched", so grepping for it tells them apart.
_REJECTS_THE_PATCH = (
    'grep -q "hi patched" hello.txt && { echo boom >&2; exit 3; }; exit 0'
)
_REJECTS_EVERYTHING = "echo boom >&2; exit 3"


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
        self.requested_reviewers = []

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

    def create_pull_request(self, owner, repo, *, title, head, base, body, draft=False):
        self.created_pr = {
            "title": title,
            "head": head,
            "base": base,
            "body": body,
            "draft": draft,
        }
        return {
            "number": 99,
            "html_url": "https://github.com/o/r/pull/99",
            "node_id": "PR_node_99",
        }

    def mark_pull_request_ready(self, node_id):
        self.marked_ready = node_id

    def request_reviewers(self, owner, repo, number, reviewers):
        self.requested_reviewers.append((number, list(reviewers)))
        return list(reviewers)


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

    def test_test_links_are_accepted_and_sanitized(self):
        req = build_task_request(
            {
                "instruction": "x",
                "test_links": {
                    "tests/a.py::T::test_x": [
                        {"label": "Dashboard", "url": "https://grafana/d/x?a=b"},
                        {"label": "bad", "url": "javascript:alert(1)"},
                    ]
                },
            },
            owner="a",
            repo="b",
        )
        self.assertEqual(
            req.test_links,
            {
                "tests/a.py::T::test_x": [
                    {"label": "Dashboard", "url": "https://grafana/d/x?a=b"}
                ]
            },
        )

    def test_test_links_are_optional_and_junk_is_not_fatal(self):
        # Links are decoration: a caller that sends none, or sends nonsense, still
        # gets its task run.
        self.assertEqual(
            build_task_request({"instruction": "x"}, owner="a", repo="b").test_links, {}
        )
        self.assertEqual(
            build_task_request(
                {"instruction": "x", "test_links": "not-a-map"}, owner="a", repo="b"
            ).test_links,
            {},
        )

    def test_reviewers_are_accepted_and_sanitized(self):
        req = build_task_request(
            {
                "instruction": "x",
                "reviewers": [
                    "@octocat",  # a leading @ is stripped
                    "octocat",  # dupe (case-insensitive) collapses
                    "OCTOCAT",
                    "dependabot[bot]",  # a bot cannot review, and 422s the call
                    "not a login",
                    "",
                    42,
                    "second-user",
                ],
            },
            owner="a",
            repo="b",
        )
        self.assertEqual(req.reviewers, ("octocat", "second-user"))

    def test_reviewers_are_optional_and_junk_is_not_fatal(self):
        # Same contract as test_links: a review request is a courtesy on top of
        # the fix, so a malformed field must never cost a PR.
        self.assertEqual(
            build_task_request({"instruction": "x"}, owner="a", repo="b").reviewers, ()
        )
        self.assertEqual(
            build_task_request(
                {"instruction": "x", "reviewers": "octocat"}, owner="a", repo="b"
            ).reviewers,
            (),
        )

    def test_reviewers_are_capped(self):
        req = build_task_request(
            {"instruction": "x", "reviewers": [f"user{i}" for i in range(25)]},
            owner="a",
            repo="b",
        )
        self.assertEqual(len(req.reviewers), MAX_REVIEWERS)

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

    def test_new_pr_requests_the_reviewers_the_dispatcher_named(self):
        # The triage blames a commit for the regression; its author is the one
        # person who knows what that change meant to do, so the fix PR should
        # land in their queue instead of waiting to be noticed.
        co = self._checkout("main")
        req = TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
            reviewers=("octocat",),
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            publish_task(
                self.cfg,
                gh,
                req,
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertEqual(gh.requested_reviewers, [(99, ["octocat"])])
        # After the draft->ready transition, so it adds to whatever the repo's
        # own reviewer-assignment workflow routes rather than racing it.
        self.assertEqual(gh.marked_ready, "PR_node_99")

    def test_no_reviewers_means_no_call(self):
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
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            publish_task(
                self.cfg,
                gh,
                req,
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertEqual(gh.requested_reviewers, [])

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

    def test_existing_pr_follow_up_does_not_re_request_reviewers(self):
        # The review was requested when the PR was opened. Re-requesting on every
        # follow-up push would reset a review the author may already have given.
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
            reviewers=("octocat",),
        )
        plan = TaskPlan(title="Fix again", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        publish_task(
            self.cfg,
            gh,
            req,
            plan,
            checkout=co,
            clone_cache=self.cache,
            job_id="abcd1234",
        )
        self.assertEqual(gh.requested_reviewers, [])

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


class PublishFallbackNormalizeTests(unittest.TestCase):
    """When validation's correction budget is exhausted, publish_task lands on
    the raw-apply fallback. With a normalizer configured it must re-run it so
    regenerated files (e.g. transformers' modeling_*.py) ride along, and refuse
    to open a PR the normalizer rejects — never commit a raw, un-normalized
    patch. Uses the bwrap backend + sandbox off so the command runs directly."""

    _PATCH = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-hi from main\n"
        "+hi patched\n"
    )

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

    def _checkout(self):
        return self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="abcd1234",
            remote_url=self.src,
        )

    def _req(self):
        return TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
        )

    def _cfg(self, **overrides):
        base = dict(helper_sandbox="off", task_sandbox_backend="bwrap")
        base.update(overrides)
        return _make_cfg(**base)

    def test_fallback_reruns_normalizer_and_ships_regenerated_files(self):
        # A raw patch (worktree_prepared=False) whose normalizer regenerates an
        # extra file: the commit must include BOTH the patched and generated
        # files, mirroring transformers' modular_*.py -> modeling_*.py flow.
        co = self._checkout()
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", "echo generated > extra.txt"],
            task_normalize_timeout=30,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertFalse(result.no_change)
        self.assertEqual(sorted(result.changed_files), ["extra.txt", "hello.txt"])
        self.assertIsNotNone(gh.created_pr)

    def test_unrelated_normalizer_output_is_left_out_of_the_commit(self):
        # The prod shape (task 433e8274): the base is stale, so the normalizer
        # rewrites files in directories the patch never touched. Those must not
        # land in the PR — one prod task patched 1 file and committed 32.
        co = self._checkout()
        cfg = self._cfg(
            task_normalize_command=[
                "sh",
                "-c",
                "mkdir -p models/other && echo drift > models/other/regen.py "
                "&& echo generated > extra.txt",
            ],
            task_normalize_timeout=30,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        events: list[tuple[str, str]] = []
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
                emit=lambda kind, text: events.append((kind, text)),
            )
        self.assertFalse(result.no_change)
        # hello.txt (patched) and extra.txt (root, alongside it) ship;
        # models/other/regen.py is drift from a stale base and does not.
        self.assertEqual(sorted(result.changed_files), ["extra.txt", "hello.txt"])
        logs = [text for kind, text in events if kind == "log"]
        self.assertTrue(
            any("Left 1 unrelated file(s) out of the commit" in line for line in logs),
            logs,
        )
        self.assertTrue(any("models/other/regen.py" in line for line in logs))

    def test_scoping_applies_to_the_prepared_worktree_path(self):
        """The path prod task 433e8274 actually took: validation accepted the
        patch in-loop (worktree_prepared=True), so publish_task commits the
        worktree as-is. The scope comes from plan.patch, which is set on both
        paths — without that this fix would no-op exactly where it is needed."""
        co = self._checkout()
        # Stand in for what the in-loop validation left behind: the patch applied
        # plus normalizer drift in an unrelated directory.
        self.cache.apply_patch(co, self._PATCH)
        os.makedirs(os.path.join(co.path, "models", "other"), exist_ok=True)
        with open(os.path.join(co.path, "models", "other", "regen.py"), "w") as f:
            f.write("drift\n")
        cfg = self._cfg(task_normalize_command=["true"], task_normalize_timeout=30)
        plan = TaskPlan(
            title="Fix hello",
            body="desc",
            patch=self._PATCH,
            worktree_prepared=True,
        )
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertEqual(result.changed_files, ["hello.txt"])

    def test_scoping_can_be_turned_off(self):
        co = self._checkout()
        cfg = self._cfg(
            task_normalize_command=[
                "sh",
                "-c",
                "mkdir -p models/other && echo drift > models/other/regen.py",
            ],
            task_normalize_timeout=30,
            task_scope_commit_to_patch=False,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertEqual(
            sorted(result.changed_files), ["hello.txt", "models/other/regen.py"]
        )

    def test_fallback_refuses_when_normalizer_rejects(self):
        # Normalizer exits non-zero on the raw patch -> no PR, worktree reset.
        # It must reject the PATCH specifically: one that fails on the pristine
        # checkout too is a broken gate, which raises instead (see
        # NormalizeBaselineTests).
        co = self._checkout()
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", _REJECTS_THE_PATCH],
            task_normalize_timeout=30,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        result = publish_task(
            cfg,
            gh,
            self._req(),
            plan,
            checkout=co,
            clone_cache=self.cache,
            job_id="abcd1234",
        )
        self.assertTrue(result.no_change)
        self.assertIsNone(gh.created_pr)
        self.assertIn("normalizer", result.message.lower())
        self.assertIn("exit 3", result.message)
        # Worktree restored to the pristine checkout.
        self.assertEqual(self.cache.collect_changes(co), [])

    def test_fallback_raises_when_the_gate_itself_is_broken(self):
        # The raw-apply fallback makes the same distinction: a normalizer that
        # rejects the pristine checkout too is an operator problem, and
        # reporting it as "the correction budget was exhausted" would hide it.
        co = self._checkout()
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", _REJECTS_EVERYTHING],
            task_normalize_timeout=30,
        )
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with self.assertRaises(NormalizeGateBroken):
            publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertIsNone(gh.created_pr)
        self.assertEqual(self.cache.collect_changes(co), [])

    def test_no_normalizer_configured_commits_raw_patch(self):
        # Without a normalizer the fallback still commits the raw patch as-is.
        co = self._checkout()
        cfg = self._cfg()  # no task_normalize_command
        plan = TaskPlan(title="Fix hello", body="desc", patch=self._PATCH)
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertFalse(result.no_change)
        self.assertEqual(result.changed_files, ["hello.txt"])


class _BrevityLLM:
    """Answers the brevity pass with a fixed comment mapping, and records the
    calls so a test can pin what the pass costs."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return ChatResult(
            content=json.dumps({"comments": self._mapping}),
            usage={"prompt_tokens": 90, "completion_tokens": 12},
        )


class CommentBrevityGateTests(unittest.TestCase):
    """The brevity pass inside the in-loop gate (reviewbot/brevity.py).

    The point of the placement is the ordering: the comments that reach a PR
    are the ones the repo's normalizer saw. These tests prove it by having the
    normalize command copy the file it is given, so what it copied is evidence
    of what it was handed."""

    _PATCH = (
        "diff --git a/mod.py b/mod.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/mod.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+import os\n"
        "+\n"
        "+# We keep the retry count at three because the upstream API rate limit\n"
        "+# window is sixty seconds, and three attempts is what fits inside it.\n"
        "+RETRIES = 3\n"
    )
    _SHORT = "Three retries fit in the API's 60s rate-limit window."

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        src = os.path.join(root, "src")
        os.makedirs(src)
        _git(src, "init", "--quiet", "-b", "main")
        with open(os.path.join(src, "hello.txt"), "w") as f:
            f.write("hi from main\n")
        _git(src, "add", "-A")
        _git(src, "commit", "--quiet", "-m", "main commit")
        self.cache = CloneCache(os.path.join(root, "cache"))
        self.co = self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="brev1234",
            remote_url=src,
        )

    def _cfg(self, **overrides):
        base = dict(
            helper_sandbox="off",
            task_sandbox_backend="bwrap",
            # Hand the normalizer's copy of the file back to the test.
            task_normalize_command=["sh", "-c", "cp mod.py seen.py"],
            task_normalize_timeout=30,
            comment_brevity_min_chars=40,
        )
        base.update(overrides)
        return _make_cfg(**base)

    def _wt(self, name):
        return os.path.join(self.co.path, name)

    def _read(self, name):
        with open(self._wt(name)) as fh:
            return fh.read()

    def _content(self):
        return json.dumps({"title": "t", "body": "b", "patch": self._PATCH})

    def test_the_normalizer_sees_the_condensed_comment(self):
        llm = _BrevityLLM({"c1": self._SHORT})
        feedback, prepared = _validate_patch(
            self._cfg(),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
            llm=llm,
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertEqual(len(llm.calls), 1)
        # In the worktree, which is what publish_task commits...
        self.assertIn(f"# {self._SHORT}\n", self._read("mod.py"))
        self.assertNotIn("rate limit\n", self._read("mod.py"))
        # ...and already there when the normalizer ran, which is the ordering
        # this pass depends on: a shortened comment is formatted and validated
        # like the code around it, never after the gate that would catch it.
        self.assertIn(f"# {self._SHORT}\n", self._read("seen.py"))
        # And it is the condensed file that publish_task would commit: the pass
        # edits the worktree, and staging picks that up (apply_patch's --index
        # staged the model's version, so this is the assertion that the later
        # edit is not left behind in the index).
        self.cache.stage_all(self.co)
        blob = {c.path: (c.content or b"") for c in self.cache.collect_changes(self.co)}
        self.assertIn(self._SHORT.encode(), blob["mod.py"])
        self.assertNotIn(b"rate limit", blob["mod.py"])

    def test_a_normalize_failure_says_the_comments_moved(self):
        # The normalizer reports line numbers in the worktree, which the pass
        # has just shifted. The model must not read that as its diff being
        # misapplied.
        llm = _BrevityLLM({"c1": self._SHORT})
        feedback, prepared = _validate_patch(
            self._cfg(
                task_normalize_command=["sh", "-c", "echo 'mod.py:3 boom' >&2; exit 3"]
            ),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
            llm=llm,
            baseline_state={"checked": True},
        )
        self.assertFalse(prepared)
        self.assertIn("boom", feedback)
        self.assertIn("comment TEXT in your patch was shortened", feedback)

    def test_no_such_note_when_nothing_was_condensed(self):
        feedback, _ = _validate_patch(
            self._cfg(
                task_normalize_command=["sh", "-c", "echo 'mod.py:3 boom' >&2; exit 3"]
            ),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
            baseline_state={"checked": True},
        )
        self.assertIn("boom", feedback)
        self.assertNotIn("shortened", feedback)

    def test_the_pass_is_skipped_when_it_is_turned_off(self):
        llm = _BrevityLLM({"c1": self._SHORT})
        feedback, prepared = _validate_patch(
            self._cfg(task_comment_brevity=False),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
            llm=llm,
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertEqual(llm.calls, [])
        self.assertIn("rate limit", self._read("mod.py"))

    def test_no_llm_means_no_pass_and_no_failure(self):
        # publish_task's fallback path has no client to spend; the patch ships
        # with the comments the model wrote, exactly as it did before.
        feedback, prepared = _validate_patch(
            self._cfg(),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertIn("rate limit", self._read("mod.py"))

    def test_a_provider_failure_does_not_fail_the_task(self):
        class _Broken:
            def complete(self, messages, **kwargs):
                raise RuntimeError("provider is down")

        feedback, prepared = _validate_patch(
            self._cfg(),
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(),
            emit=lambda *a: None,
            llm=_Broken(),
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertIn("rate limit", self._read("mod.py"))


class ValidatePatchTests(unittest.TestCase):
    """The in-loop verification gate (_validate_patch), against a real
    worktree. Runs with the bwrap backend + sandbox off so the normalize
    command runs directly (no docker / bwrap needed in the test environment)."""

    _PATCH = (
        "diff --git a/hello.txt b/hello.txt\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1 +1 @@\n"
        "-hi from main\n"
        "+hi patched\n"
    )

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
        self.co = self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="abcd1234",
            remote_url=self.src,
        )

    def _cfg(self, **overrides):
        base = dict(helper_sandbox="off", task_sandbox_backend="bwrap")
        base.update(overrides)
        return _make_cfg(**base)

    def _content(self, patch):
        return json.dumps({"title": "t", "body": "b", "patch": patch})

    def _validate(self, cfg, content):
        return _validate_patch(
            cfg,
            checkout=self.co,
            clone_cache=self.cache,
            content=content,
            emit=lambda *a: None,
        )

    def _wt(self, name):
        return os.path.join(self.co.path, name)

    def test_clean_normalize_prepares_combined_worktree(self):
        # Patch applies + normalizer creates an extra file + exits 0 ->
        # accepted (no feedback, prepared) with BOTH edits in the worktree.
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", "echo generated > extra.txt"],
            task_normalize_timeout=30,
        )
        feedback, prepared = self._validate(cfg, self._content(self._PATCH))
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        with open(self._wt("hello.txt")) as f:
            self.assertEqual(f.read(), "hi patched\n")
        self.assertTrue(os.path.exists(self._wt("extra.txt")))
        # The combined change is what publish_task would commit.
        self.cache.stage_all(self.co)
        changed = sorted(c.path for c in self.cache.collect_changes(self.co))
        self.assertEqual(changed, ["extra.txt", "hello.txt"])

    def test_normalizer_failure_feeds_back_and_resets(self):
        # Non-zero normalizer exit ON THE PATCH -> feedback to the model,
        # worktree reset clean (so the next attempt starts pristine).
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", _REJECTS_THE_PATCH],
            task_normalize_timeout=30,
        )
        events = []
        feedback, prepared = _validate_patch(
            cfg,
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(self._PATCH),
            emit=lambda kind, text: events.append((kind, text)),
        )
        self.assertFalse(prepared)
        self.assertIsNotNone(feedback)
        self.assertIn("normalizer", feedback.lower())
        self.assertIn("boom", feedback)
        normalize_errors = [text for kind, text in events if kind == "normalize_error"]
        self.assertEqual(len(normalize_errors), 1)
        self.assertIn("Normalizer failed (exit 3)", normalize_errors[0])
        self.assertIn("boom", normalize_errors[0])
        # Guides the model toward root-cause fixes over suppressions.
        self.assertIn("ROOT CAUSE", feedback)
        self.assertIn("noqa", feedback)
        # Worktree restored to the pristine checkout.
        with open(self._wt("hello.txt")) as f:
            self.assertEqual(f.read(), "hi from main\n")
        self.assertEqual(self.cache.collect_changes(self.co), [])

    def test_patch_apply_failure_is_persisted_for_humans(self):
        cfg = self._cfg(task_normalize_command=["true"], task_normalize_timeout=30)
        bad_patch = (
            "diff --git a/hello.txt b/hello.txt\n"
            "--- a/hello.txt\n"
            "+++ b/hello.txt\n"
            "@@ -1 +1 @@\n"
            "-this context does not match\n"
            "+hi patched\n"
        )
        events = []
        feedback, prepared = _validate_patch(
            cfg,
            checkout=self.co,
            clone_cache=self.cache,
            content=self._content(bad_patch),
            emit=lambda kind, text: events.append((kind, text)),
        )

        self.assertFalse(prepared)
        self.assertIn("git apply", feedback)
        apply_errors = [text for kind, text in events if kind == "patch_apply_error"]
        self.assertEqual(len(apply_errors), 1)
        self.assertIn("`git apply` rejected the proposed patch", apply_errors[0])
        self.assertIn("hello.txt", apply_errors[0])
        rejected_patches = [text for kind, text in events if kind == "rejected_patch"]
        self.assertEqual(rejected_patches, [bad_patch])

    def test_operator_guidance_is_appended_to_feedback(self):
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", _REJECTS_THE_PATCH],
            task_normalize_timeout=30,
            task_normalize_guidance="HOUSE RULE: never add new dependencies.",
        )
        feedback, prepared = self._validate(cfg, self._content(self._PATCH))
        self.assertFalse(prepared)
        self.assertIn("HOUSE RULE: never add new dependencies.", feedback)

    def test_task_validation_retry_prompt_is_compact_and_json_only(self):
        messages = _task_validation_retry_messages(
            repo_full_name="o/r",
            base_ref="main",
            instruction="fix the failing test",
            context="traceback",
            existing_diff="diff --git a/x b/x\n",
            rejected_content='{"patch": "bad"}',
            feedback="git apply failed",
            retry_number=2,
        )

        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("JSON object", messages[0]["content"])
        self.assertIn("no tool requests", messages[0]["content"])
        self.assertIn("fix the failing test", messages[1]["content"])
        self.assertIn("git apply failed", messages[1]["content"])
        self.assertIn('{"patch": "bad"}', messages[1]["content"])
        self.assertNotIn("tool_call_id", json.dumps(messages))

    def test_broken_gate_raises_instead_of_blaming_the_patch(self):
        # The prod shape (transformers#48037): the normalizer fails for a reason
        # that has nothing to do with the patch — a stale dependency in the
        # runner image — so it fails on the pristine checkout too. Feeding that
        # back would burn the whole correction budget and then report "the patch
        # does not pass the normalizer", so it must raise instead.
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", _REJECTS_EVERYTHING],
            task_normalize_timeout=30,
        )
        events = []
        with self.assertRaises(NormalizeGateBroken) as caught:
            _validate_patch(
                cfg,
                checkout=self.co,
                clone_cache=self.cache,
                content=self._content(self._PATCH),
                emit=lambda kind, text: events.append((kind, text)),
            )
        message = str(caught.exception)
        self.assertIn("unpatched checkout", message)
        self.assertIn("boom", message)
        # Not 422: that would make the runner skip to the next candidate group,
        # but a broken gate fails every group.
        self.assertEqual(caught.exception.status_code, 500)
        self.assertTrue(
            any(
                kind == "normalize_error" and "pristine checkout" in text
                for kind, text in events
            )
        )
        # Worktree still pristine for whatever runs next.
        self.assertEqual(self.cache.collect_changes(self.co), [])

    def test_baseline_runs_once_per_task(self):
        # The baseline costs a full normalizer run, so the answer is cached
        # across corrections rather than re-run for each one.
        marker = os.path.join(self._tmp.name, "runs")
        cfg = self._cfg(
            task_normalize_command=[
                "sh",
                "-c",
                f'echo x >> "{marker}"; echo boom >&2; exit 3',
            ],
            task_normalize_timeout=30,
        )
        state: dict = {}

        def _run():
            return _validate_patch(
                cfg,
                checkout=self.co,
                clone_cache=self.cache,
                content=self._content(self._PATCH),
                emit=lambda *a: None,
                baseline_state=state,
            )

        with self.assertRaises(NormalizeGateBroken):
            _run()
        with open(marker) as f:
            after_first = len(f.read().split())
        # Patch run + baseline run.
        self.assertEqual(after_first, 2)

        # A second correction attempt replays the cached verdict: it still
        # raises, but without paying for another baseline run.
        with self.assertRaises(NormalizeGateBroken):
            _run()
        with open(marker) as f:
            after_second = len(f.read().split())
        self.assertEqual(after_second, 3)

    def test_healthy_gate_pays_nothing_on_the_happy_path(self):
        # A clean normalizer never triggers the baseline check at all.
        marker = os.path.join(self._tmp.name, "runs")
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", f'echo x >> "{marker}"; exit 0'],
            task_normalize_timeout=30,
        )
        feedback, prepared = self._validate(cfg, self._content(self._PATCH))
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        with open(marker) as f:
            self.assertEqual(len(f.read().split()), 1)

    def test_unapplyable_patch_feeds_back(self):
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", "true"], task_normalize_timeout=30
        )
        bad = (
            "diff --git a/hello.txt b/hello.txt\n"
            "--- a/hello.txt\n"
            "+++ b/hello.txt\n"
            "@@ -1 +1 @@\n"
            "-does not match\n"
            "+nope\n"
        )
        feedback, prepared = self._validate(cfg, self._content(bad))
        self.assertFalse(prepared)
        self.assertIsNotNone(feedback)
        self.assertIn("apply", feedback.lower())

    def test_unavailable_sandbox_accepts_best_effort(self):
        # docker backend with no image -> NormalizeError; not the model's
        # fault, so the applied patch is accepted (prepared) un-normalized.
        cfg = self._cfg(
            task_sandbox_backend="docker",
            task_normalize_command=["make", "fix-repo"],
            task_normalize_timeout=30,
        )
        feedback, prepared = self._validate(cfg, self._content(self._PATCH))
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        with open(self._wt("hello.txt")) as f:
            self.assertEqual(f.read(), "hi patched\n")

    def test_empty_patch_accepted_not_prepared(self):
        cfg = self._cfg(
            task_normalize_command=["sh", "-c", "true"], task_normalize_timeout=30
        )
        feedback, prepared = self._validate(cfg, self._content(""))
        self.assertIsNone(feedback)
        self.assertFalse(prepared)


class AnchoredEditAnswerTests(unittest.TestCase):
    """An `edits` answer through the real gate (:mod:`reviewbot.anchored_edits`).

    The format stops at prepare_task: the gate applies the edits and has **git**
    write the diff, so `plan.patch` is git's and everything downstream —
    publish_task's apply path, `commit_scope`, `classify_patch`, the brevity
    pass's line numbers — keeps reading a unified diff with correct geometry.
    These tests pin that boundary, and that a bad anchor is rejected the same
    way a bad patch is (clean worktree, `rejection == "apply"`, tools kept)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        src = os.path.join(root, "src")
        os.makedirs(src)
        _git(src, "init", "--quiet", "-b", "main")
        with open(os.path.join(src, "hello.txt"), "w") as f:
            f.write("alpha\nbeta\ngamma\n")
        _git(src, "add", "-A")
        _git(src, "commit", "--quiet", "-m", "main commit")
        self.cache = CloneCache(os.path.join(root, "cache"))
        self.co = self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="edit1234",
            remote_url=src,
        )

    def _cfg(self, **overrides):
        base = dict(
            helper_sandbox="off",
            task_sandbox_backend="bwrap",
            task_normalize_command=["true"],
            task_normalize_timeout=30,
        )
        base.update(overrides)
        return _make_cfg(**base)

    def _content(self, edits, **extra):
        return json.dumps({"title": "t", "body": "b", "edits": edits, **extra})

    def _validate(self, content, cfg=None, report=None, events=None):
        return _validate_patch(
            cfg or self._cfg(),
            checkout=self.co,
            clone_cache=self.cache,
            content=content,
            emit=(lambda k, t: events.append((k, t)))
            if events is not None
            else (lambda *a: None),
            report=report,
        )

    def _read(self, name):
        with open(os.path.join(self.co.path, name)) as fh:
            return fh.read()

    def test_a_unique_anchor_is_applied_and_git_writes_the_diff(self):
        report: dict = {}
        feedback, prepared = self._validate(
            self._content([{"path": "hello.txt", "old": "beta", "new": "BETA"}]),
            report=report,
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertEqual(self._read("hello.txt"), "alpha\nBETA\ngamma\n")
        # git's diff, not the model's: correct headers and geometry, which is
        # what commit_scope.patch_paths and publish_task's apply path need.
        patch = report["patch"]
        self.assertIn("diff --git a/hello.txt b/hello.txt", patch)
        self.assertIn("@@ -1,3 +1,3 @@", patch)
        self.assertIn("-beta", patch)
        self.assertIn("+BETA", patch)
        # And it is the edited worktree publish_task would commit.
        self.cache.stage_all(self.co)
        blobs = {c.path: c.content for c in self.cache.collect_changes(self.co)}
        self.assertEqual(blobs["hello.txt"], b"alpha\nBETA\ngamma\n")

    def test_a_bad_anchor_is_an_apply_rejection_with_a_clean_worktree(self):
        report: dict = {}
        events: list = []
        feedback, prepared = self._validate(
            self._content(
                [
                    {"path": "hello.txt", "old": "beta", "new": "BETA"},
                    {"path": "hello.txt", "old": "not in the file", "new": "x"},
                ]
            ),
            report=report,
            events=events,
        )
        self.assertFalse(prepared)
        self.assertIn("does not occur", feedback)
        # "apply" is what keeps the correction turn's tools (see
        # test_patch_apply_feedback): the model has to re-read the file.
        self.assertEqual(report["rejection"], "apply")
        self.assertEqual(report["patch"], "")
        # All or nothing — the first edit was good and must NOT have landed.
        self.assertEqual(self._read("hello.txt"), "alpha\nbeta\ngamma\n")
        self.assertEqual(self.cache.collect_changes(self.co), [])
        # Persisted under the kinds the task page renders and the store keeps,
        # so a rejected edit set is as diagnosable as a rejected patch.
        kinds = [k for k, _ in events]
        self.assertIn("rejected_patch", kinds)
        self.assertIn("patch_apply_error", kinds)
        self.assertIn("edit 2", report["edits_error"])

    def test_an_ambiguous_anchor_is_refused_rather_than_guessed(self):
        with open(os.path.join(self.co.path, "hello.txt"), "w") as fh:
            fh.write("beta\nbeta\n")
        _git(self.co.path, "commit", "--quiet", "-am", "two betas")
        feedback, prepared = self._validate(
            self._content([{"path": "hello.txt", "old": "beta", "new": "x"}])
        )
        self.assertFalse(prepared)
        self.assertIn("occurs 2 times", feedback)

    def test_an_empty_edit_list_is_the_honest_decline(self):
        """Not a rejection: "the change I meant is not in the file" is the
        answer we asked for, and it must cost no correction budget."""
        report: dict = {}
        feedback, prepared = self._validate(self._content([]), report=report)
        self.assertIsNone(feedback)
        self.assertFalse(prepared)
        self.assertEqual(report["rejection"], "")

    def test_edits_win_over_a_patch_sent_alongside_them(self):
        events: list = []
        feedback, prepared = self._validate(
            self._content(
                [{"path": "hello.txt", "old": "gamma", "new": "GAMMA"}],
                patch=(
                    "diff --git a/hello.txt b/hello.txt\n--- a/hello.txt\n"
                    "+++ b/hello.txt\n@@ -1 +1 @@\n-alpha\n+ALPHA\n"
                ),
            ),
            events=events,
        )
        self.assertIsNone(feedback)
        self.assertTrue(prepared)
        self.assertEqual(self._read("hello.txt"), "alpha\nbeta\nGAMMA\n")
        self.assertTrue(any("ignoring the diff" in t for _, t in events))

    def test_a_path_outside_the_checkout_is_refused(self):
        outside = os.path.join(self._tmp.name, "outside.txt")
        with open(outside, "w") as fh:
            fh.write("secret\n")
        feedback, prepared = self._validate(
            self._content([{"path": "../../outside.txt", "old": "secret", "new": "x"}])
        )
        self.assertFalse(prepared)
        self.assertIn("repository-relative", feedback)
        with open(outside) as fh:
            self.assertEqual(fh.read(), "secret\n")

    def test_a_normalizer_rejection_still_resets_the_worktree(self):
        cfg = self._cfg(
            task_normalize_command=[
                "sh",
                "-c",
                "grep -q BETA hello.txt && { echo boom >&2; exit 3; }; exit 0",
            ]
        )
        report: dict = {}
        feedback, prepared = self._validate(
            self._content([{"path": "hello.txt", "old": "beta", "new": "BETA"}]),
            cfg=cfg,
            report=report,
        )
        self.assertFalse(prepared)
        self.assertIn("boom", feedback)
        # A normalizer rejection is NOT an apply rejection: it keeps the compact
        # correction prompt, and it carries its own reason.
        self.assertEqual(report["rejection"], "normalize")
        self.assertEqual(self._read("hello.txt"), "alpha\nbeta\ngamma\n")


class TaskPreflightTests(unittest.TestCase):
    """check_task_preflight — the cheap "can this gate be passed at all" probe
    that runs before any LLM work. Same real-worktree fixture as
    ValidatePatchTests, bwrap backend + sandbox off."""

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
        self.co = self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="abcd1234",
            remote_url=self.src,
        )

    def _cfg(self, **overrides):
        base = dict(helper_sandbox="off", task_sandbox_backend="bwrap")
        base.update(overrides)
        return _make_cfg(**base)

    def _run(self, cfg, events=None):
        return check_task_preflight(
            cfg,
            checkout=self.co,
            clone_cache=self.cache,
            emit=(lambda kind, text: events.append((kind, text)))
            if events is not None
            else (lambda *a: None),
        )

    def test_unset_command_runs_nothing(self):
        # Serge stays repo-agnostic: no probe configured, no probe run.
        self._run(self._cfg(task_preflight_command=None))

    def test_broken_environment_raises_before_any_llm_work(self):
        # The 2026-09-12 shape: the runner image drifted from the target repo's
        # main, so the unpatched checkout cannot pass the gate. Must raise here,
        # where it costs seconds, not after a full agent loop per candidate.
        marker = os.path.join(self._tmp.name, "runs")
        cfg = self._cfg(
            task_preflight_command=[
                "sh",
                "-c",
                f'echo x >> "{marker}"; echo "cannot import name httpx" >&2; exit 1',
            ],
            task_preflight_timeout=30,
        )
        events: list = []
        with self.assertRaises(NormalizeGateBroken) as caught:
            self._run(cfg, events)

        message = str(caught.exception)
        self.assertIn("cannot import name httpx", message)
        self.assertIn("no LLM work was started", message)
        self.assertIn("--no-deps", message)
        # Not 422: that makes the runner move to the next candidate, but a
        # broken environment breaks every one of them.
        self.assertEqual(caught.exception.status_code, 500)
        self.assertTrue(
            any(
                kind == "normalize_error" and "pristine checkout" in text
                for kind, text in events
            )
        )
        with open(marker) as f:
            self.assertEqual(len(f.read().split()), 1)

    def test_healthy_environment_passes_and_leaves_the_worktree_pristine(self):
        # A probe legitimately writes (an editable install, build artefacts);
        # everything downstream assumes an untouched base.
        cfg = self._cfg(
            task_preflight_command=[
                "sh",
                "-c",
                "echo scribble > hello.txt; echo built > artefact.txt; exit 0",
            ],
            task_preflight_timeout=30,
        )
        self._run(cfg)
        self.assertEqual(self.cache.collect_changes(self.co), [])
        with open(os.path.join(self.co.path, "hello.txt")) as f:
            self.assertEqual(f.read(), "hi from main\n")

    def test_a_failing_probe_also_leaves_the_worktree_pristine(self):
        cfg = self._cfg(
            task_preflight_command=[
                "sh",
                "-c",
                "echo scribble > hello.txt; exit 2",
            ],
            task_preflight_timeout=30,
        )
        with self.assertRaises(NormalizeGateBroken):
            self._run(cfg)
        self.assertEqual(self.cache.collect_changes(self.co), [])

    def test_unavailable_sandbox_fails_open(self):
        # The probe could not run at all. That says nothing about the gate, and
        # the real gate still runs later — so do not fail the task over it.
        cfg = self._cfg(
            task_preflight_command=["sh", "-c", "exit 0"], task_preflight_timeout=30
        )
        with patch(
            "reviewbot.tasks.run_normalize",
            side_effect=NormalizeError("normalize sandbox unavailable: no bwrap"),
        ):
            self._run(cfg)


class PublishPreparedTests(unittest.TestCase):
    """publish_task commits a worktree already prepared by validation,
    without re-applying the patch."""

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
        self.cfg = _make_cfg()

    def _checkout(self):
        return self.cache.acquire_ref(
            token="",
            owner="acme",
            repo="widget",
            ref="main",
            job_id="abcd1234",
            remote_url=self.src,
        )

    def _req(self):
        return TaskRequest(
            owner="acme",
            repo="widget",
            base_ref="main",
            instruction="fix",
            context="",
            mode="new_pr",
        )

    def test_prepared_worktree_is_committed_as_is(self):
        co = self._checkout()
        # Simulate what _validate_patch leaves behind: edits already in the
        # worktree (not staged). publish_task must NOT re-apply plan.patch.
        with open(os.path.join(co.path, "hello.txt"), "w") as f:
            f.write("hi patched\n")
        with open(os.path.join(co.path, "extra.txt"), "w") as f:
            f.write("generated\n")
        plan = TaskPlan(
            title="Fix hello",
            body="desc",
            patch="this patch text is never applied",
            worktree_prepared=True,
        )
        gh = _FakeGH()
        with patch("reviewbot.tasks.post_task_pr_created_notification"):
            result = publish_task(
                self.cfg,
                gh,
                self._req(),
                plan,
                checkout=co,
                clone_cache=self.cache,
                job_id="abcd1234",
            )
        self.assertFalse(result.no_change)
        self.assertEqual(sorted(result.changed_files), ["extra.txt", "hello.txt"])

    def test_prepared_but_clean_worktree_is_no_change(self):
        co = self._checkout()
        plan = TaskPlan(title="t", body="b", patch="x", worktree_prepared=True)
        gh = _FakeGH()
        result = publish_task(
            self.cfg,
            gh,
            self._req(),
            plan,
            checkout=co,
            clone_cache=self.cache,
            job_id="abcd1234",
        )
        self.assertTrue(result.no_change)
        self.assertIsNone(gh.created_pr)


class ReadRepoConventionsTests(unittest.TestCase):
    """_read_repo_conventions reads the repo's rules file straight from the
    task worktree, falling back to the deployment default."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.wt = self._tmp.name
        self.co = Checkout(path=self.wt, branch="main", bare="", owner="a", repo="b")

    def _cfg(self, **overrides):
        base = dict(
            review_rules_path=".ai/review-rules.md",
            default_review_rules="DEFAULT RULES",
        )
        base.update(overrides)
        return _make_cfg(**base)

    def test_reads_rules_file_from_worktree(self):
        os.makedirs(os.path.join(self.wt, ".ai"))
        with open(os.path.join(self.wt, ".ai", "review-rules.md"), "w") as f:
            f.write("Edit modular_*.py, never the generated file.\n")
        out = _read_repo_conventions(self._cfg(), self.co)
        self.assertEqual(out, "Edit modular_*.py, never the generated file.")

    def test_falls_back_to_default_when_absent(self):
        self.assertEqual(_read_repo_conventions(self._cfg(), self.co), "DEFAULT RULES")

    def test_configurable_path_can_point_at_agents_md(self):
        with open(os.path.join(self.wt, "AGENTS.md"), "w") as f:
            f.write("House conventions live here.\n")
        cfg = self._cfg(review_rules_path="AGENTS.md")
        self.assertEqual(
            _read_repo_conventions(cfg, self.co), "House conventions live here."
        )


if __name__ == "__main__":
    unittest.main()


class PromptPrefixSummaryTests(unittest.TestCase):
    """The prefix is resent on every turn, so its breakdown is what explains
    where an input-token budget went. Measured on a real 51-turn task: turn 1
    cost 25,335 tokens and the conversation then grew only ~510 a turn, so ~63%
    of the 2M cap was this prefix -- and which component dominates was reached
    by subtraction, which is what this line replaces."""

    def test_names_every_component(self) -> None:
        line = prompt_prefix_summary(
            system_prompt="s" * 4000,
            user_prompt="u" * 90000,
            conventions="c" * 3597,
            normalize_guidance="n" * 120,
            context="x" * 88000,
            instruction="i" * 1000,
            existing_diff="",
        )
        self.assertIn("Prompt prefix 94,000 chars", line)
        self.assertIn("system 4,000", line)
        self.assertIn("conventions 3,597", line)
        self.assertIn("normalize-guidance 120", line)
        self.assertIn("user 90,000", line)
        self.assertIn("context 88,000", line)
        self.assertIn("instruction 1,000", line)
        self.assertIn("existing-diff 0", line)

    def test_none_components_are_zero_not_a_crash(self) -> None:
        # normalize_guidance and existing_diff are Optional in the real call.
        line = prompt_prefix_summary(
            system_prompt="s",
            user_prompt="u",
            normalize_guidance=None,
            existing_diff=None,
        )
        self.assertIn("normalize-guidance 0", line)
        self.assertIn("existing-diff 0", line)
        self.assertIn("Prompt prefix 2 chars", line)

    def test_total_is_system_plus_user(self) -> None:
        line = prompt_prefix_summary(system_prompt="a" * 10, user_prompt="b" * 5)
        self.assertIn("Prompt prefix 15 chars", line)


class PriorArtStepTests(unittest.TestCase):
    """The pre-loop project-history lookup.

    Why it exists at all: across the 20 jobs in the production store that ran on
    a relore-indexed repository (2026-09-16..18), 6 called a history tool, and
    the earliest any of them did was the 14th tool call — median 22nd. The two
    tasks that ended in a published PR called none. The system prompt has said
    "BEFORE diagnosing" throughout. An instruction cannot buy that ordering,
    because by the time the model is picking tools it is already diagnosing.
    """

    NODE_ID = "tests/models/nemotron/test_modeling_nemotron.py::T::test_model_8b"

    def _req(self, **kw):
        base = dict(
            owner="huggingface",
            repo="transformers",
            base_ref="main",
            instruction="fix it",
            context="",
        )
        base.update(kw)
        return TaskRequest(**base)

    def test_node_ids_come_from_the_dispatcher_when_it_sent_them(self):
        req = self._req(test_links={self.NODE_ID: [{"label": "run", "url": "u"}]})
        self.assertEqual(tasks_module._failing_node_ids(req), [self.NODE_ID])

    def test_otherwise_they_are_parsed_out_of_the_failure_report(self):
        req = self._req(context=f"- `{self.NODE_ID}` [multi-gpu] (output_mismatch)")
        self.assertEqual(tasks_module._failing_node_ids(req), [self.NODE_ID])

    def test_no_relore_means_no_block_and_no_subprocess(self):
        # Every repo relore does not index, and every deployment without
        # RELORE_API. The task must be byte-for-byte what it was before.
        req = self._req(test_links={self.NODE_ID: []})
        self.assertEqual(
            tasks_module._history_notes(req, None, lambda *a: None), ("", "")
        )

    def test_a_task_with_no_identifiable_tests_skips_the_lookup(self):
        # A hand-dispatched task ("bump the pinned torch version") has no node
        # ids, so there is nothing to search for and no note to add.
        env = SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))
        with patch.object(tasks_module, "prior_art") as searched:
            note, _ = tasks_module._history_notes(self._req(), env, lambda *a: None)
        self.assertEqual(note, "")
        searched.assert_not_called()

    def test_hits_reach_the_note_the_job_log_and_a_step(self):
        env = SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))
        thread = relore_tool.PriorThread(
            number=37665,
            kind="pr",
            title="[tests] fix test_nemotron_8b_generation_sdpa",
            url="https://github.com/huggingface/transformers/pull/37665",
            author="someone",
            trust="reported",
            age="16mo",
            query="nemotron test_model_8b",
        )
        result = relore_tool.PriorArtResult([thread], ["nemotron test_x"], [], [])
        events = []
        with patch.object(tasks_module, "prior_art", return_value=result):
            note, _ = tasks_module._history_notes(
                self._req(test_links={self.NODE_ID: []}),
                env,
                lambda kind, text: events.append((kind, text)),
            )
        self.assertIn("#37665", note)
        # Its own step, so the task page lists it as something serge did
        # between the reproduce gate and the first turn — not a log line
        # buried in whichever phase happened to be open.
        self.assertIn(("step", "history"), events)
        self.assertIn(
            ("log", "Project history: #37665 already discuss these tests"), events
        )

    def test_the_task_page_says_when_the_quality_filter_dropped_something(self):
        # Otherwise a group whose only hits were rejected patches reads on the
        # page as a group with no history, which is the difference the filter
        # exists to make.
        env = SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))
        result = relore_tool.PriorArtResult([], ["nemotron test_x"], [], [], 2)
        events = []
        with patch.object(tasks_module, "prior_art", return_value=result):
            tasks_module._history_notes(
                self._req(test_links={self.NODE_ID: []}),
                env,
                lambda kind, text: events.append((kind, text)),
            )
        logs = [t for k, t in events if k == "log"]
        self.assertTrue(any("2 excluded" in t for t in logs), logs)

    def test_an_unanswered_query_is_logged_as_unanswered_not_as_empty(self):
        # "relore did not answer" and "there is nothing there" are different
        # facts, and the note tells the model to retry only the first.
        env = SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))
        result = relore_tool.PriorArtResult([], [], ["nemotron test_x"], [])
        events = []
        with patch.object(tasks_module, "prior_art", return_value=result):
            note, _ = tasks_module._history_notes(
                self._req(test_links={self.NODE_ID: []}),
                env,
                lambda kind, text: events.append((kind, text)),
            )
        logs = [t for k, t in events if k == "log"]
        self.assertTrue(any("did not answer" in t for t in logs), logs)
        self.assertFalse(any("no earlier thread matched" in t for t in logs), logs)
        self.assertIn("UNANSWERED", note)

    def test_a_relore_failure_costs_the_task_nothing(self):
        env = SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))
        events = []
        with patch.object(tasks_module, "prior_art", side_effect=OSError("down")):
            note, _ = tasks_module._history_notes(
                self._req(test_links={self.NODE_ID: []}),
                env,
                lambda kind, text: events.append((kind, text)),
            )
        self.assertEqual(note, "")
        # The step still opened, so the page shows the attempt rather than
        # silently skipping a stage that did run.
        self.assertIn(("step", "history"), events)


class CulpritThreadStepTests(unittest.TestCase):
    """The pre-loop lookup of the pull request CI's bisect blamed.

    Only regression clusters carry one. Before this, the triage prompt spent a
    paragraph asking the model to reconstruct what that PR was for by grepping
    what it left in the tree — and getting that wrong is the failure the cluster
    addendum is most afraid of: transformers #48535 re-guarded #47988's call
    with a condition the base class already applied, making it dead code, and
    OLMo started appending EOS to every prompt again.
    """

    CONTEXT = (
        "Attribution (from CI `git bisect`):\n"
        "- bad commit: ce5c8f5e4352\n"
        "- introduced by PR #47988 (https://github.com/huggingface/transformers/pull/47988)\n"
    )

    def _req(self, **kw):
        base = dict(
            owner="huggingface",
            repo="transformers",
            base_ref="main",
            instruction="fix it",
            context=self.CONTEXT,
        )
        base.update(kw)
        return TaskRequest(**base)

    def _env(self):
        return SimpleNamespace(relore=SimpleNamespace(repo="x", api="y"))

    def test_a_cluster_with_no_node_ids_still_gets_the_lookup(self):
        """The two lookups are independent.

        A group serge cannot pull node-ids out of used to return before the step
        was even opened; the culprit is knowable from the attribution line
        alone, so it must not be gated on the other lookup finding something.
        """
        events = []
        thread = relore_tool.CulpritThread(47988, page="<<<RELORE-UNTRUSTED>>>x")
        with patch.object(tasks_module, "culprit_thread", return_value=thread) as got:
            history, culprit = tasks_module._history_notes(
                self._req(),
                self._env(),
                lambda kind, text: events.append((kind, text)),
            )
        self.assertEqual(history, "")
        self.assertIn("#47988", culprit)
        got.assert_called_once()
        self.assertIn(("step", "history"), events)

    def test_one_step_covers_both_lookups(self):
        # Two `history` steps would render two rows for one thing serge does.
        events = []
        thread = relore_tool.CulpritThread(47988, page="<<<RELORE-UNTRUSTED>>>x")
        result = relore_tool.PriorArtResult([], ["nemotron test_x"], [], [])
        with (
            patch.object(tasks_module, "culprit_thread", return_value=thread),
            patch.object(tasks_module, "prior_art", return_value=result),
        ):
            tasks_module._history_notes(
                self._req(test_links={"tests/models/nemotron/t.py::T::test_x": []}),
                self._env(),
                lambda kind, text: events.append((kind, text)),
            )
        self.assertEqual([e for e in events if e[0] == "step"], [("step", "history")])

    def test_a_group_that_is_not_a_cluster_makes_no_call(self):
        # Most groups. No attribution line, so nothing to fetch and no block.
        with patch.object(tasks_module, "culprit_thread") as got:
            _history, culprit = tasks_module._history_notes(
                self._req(context="Failure group: whisper flakes."),
                self._env(),
                lambda *a: None,
            )
        self.assertEqual(culprit, "")
        got.assert_not_called()

    def test_no_relore_means_no_block_and_no_subprocess(self):
        self.assertEqual(
            tasks_module._history_notes(self._req(), None, lambda *a: None), ("", "")
        )

    def test_a_fetched_thread_is_reported_on_the_task_page(self):
        events = []
        thread = relore_tool.CulpritThread(47988, page="<<<RELORE-UNTRUSTED>>>x")
        with patch.object(tasks_module, "culprit_thread", return_value=thread):
            tasks_module._history_notes(
                self._req(), self._env(), lambda k, t: events.append((k, t))
            )
        logs = [t for k, t in events if k == "log"]
        self.assertTrue(any("culprit PR #47988" in t for t in logs), logs)

    def test_a_thread_relore_would_not_serve_is_logged_as_unanswered(self):
        # Same wording as the prior-art path, so task_report's `history` rules
        # score the step a warning rather than an ok.
        events = []
        thread = relore_tool.CulpritThread(47988, error="returned 404")
        with patch.object(tasks_module, "culprit_thread", return_value=thread):
            _history, culprit = tasks_module._history_notes(
                self._req(), self._env(), lambda k, t: events.append((k, t))
            )
        logs = [t for k, t in events if k == "log"]
        self.assertTrue(any("relore did not answer" in t for t in logs), logs)
        self.assertIn("UNREAD", culprit)

    def test_a_crash_costs_the_task_nothing(self):
        events = []
        with patch.object(tasks_module, "culprit_thread", side_effect=OSError("down")):
            _history, culprit = tasks_module._history_notes(
                self._req(), self._env(), lambda k, t: events.append((k, t))
            )
        self.assertEqual(culprit, "")
        self.assertIn(("step", "history"), events)


class PatchShapedForCheckersTests(unittest.TestCase):
    """Make the worktree answer "what did this patch change?" while the
    normalizer runs.

    transformers#49026: on 2026-09-22 the normalizer failed 7 of 10 ITF groups
    on `noisy_comments`. Nothing was wrong with the patches. serge's worktree
    has no `origin/main` (CloneCache detaches it) so the checker could not
    resolve the patch, fell back to scanning the whole tree, and
    `--fail-on-findings` blocked on 375 comments in files serge never opened.
    The shallow clone compounded it: `git blame` pins every line older than the
    graft to the boundary commit, so the checker's date and ownership filters —
    both of which read blame — silently became no-ops.
    """

    def _repo(self):
        import subprocess as sp
        import tempfile

        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, True)

        def run(*a):
            return sp.run(["git", "-C", path, *a], capture_output=True, text=True)

        run("init", "-q", "-b", "main")
        run("config", "user.email", "t@t")
        run("config", "user.name", "t")
        (Path(path) / "a.py").write_text("x = 1\n")
        run("add", "-A")
        run("commit", "-qm", "base")
        return path, run

    def _checkout(self, path):
        return Checkout(path=path, branch="b", bare="x", owner="o", repo="r")

    def test_the_patch_becomes_visible_as_a_commit_then_is_undone(self):
        path, run = self._repo()
        base = run("rev-parse", "HEAD").stdout.strip()
        (Path(path) / "a.py").write_text("x = 2\n")  # serge's patch, uncommitted

        with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
            inside_head = run("rev-parse", "HEAD").stdout.strip()
            # The whole point: `merge_base...HEAD` is now exactly the patch.
            diff = run("diff", "--name-only", f"{base}...HEAD").stdout.split()
            self.assertNotEqual(inside_head, base)
            self.assertEqual(diff, ["a.py"])
            self.assertEqual(
                run("rev-parse", "refs/remotes/origin/main").stdout.strip(), base
            )

        self.assertEqual(run("rev-parse", "HEAD").stdout.strip(), base)
        # Still uncommitted, and still the patched content.
        self.assertTrue(run("status", "--porcelain").stdout.strip())
        self.assertEqual((Path(path) / "a.py").read_text(), "x = 2\n")
        self.assertEqual(
            run("rev-parse", "--verify", "-q", "refs/remotes/origin/main").returncode, 1
        )

    def test_new_files_are_visible_too(self):
        # `add -A`, not a path list: a checker that cannot see a file serge
        # added reports it as unchanged, which is the silent half of #49026.
        path, run = self._repo()
        base = run("rev-parse", "HEAD").stdout.strip()
        (Path(path) / "new.py").write_text("y = 1\n")
        with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
            self.assertIn("new.py", run("diff", "--name-only", f"{base}...HEAD").stdout)
        self.assertTrue((Path(path) / "new.py").exists())

    def test_it_is_undone_when_the_normalizer_raises(self):
        path, run = self._repo()
        base = run("rev-parse", "HEAD").stdout.strip()
        (Path(path) / "a.py").write_text("x = 3\n")
        with self.assertRaises(RuntimeError):
            with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
                raise RuntimeError("normalizer blew up")
        self.assertEqual(run("rev-parse", "HEAD").stdout.strip(), base)
        self.assertEqual((Path(path) / "a.py").read_text(), "x = 3\n")

    def test_an_existing_base_ref_is_left_alone(self):
        # Only ever delete what we minted; clobbering a real origin/main would
        # outlive the normalizer.
        path, run = self._repo()
        base = run("rev-parse", "HEAD").stdout.strip()
        run("update-ref", "refs/remotes/origin/main", base)
        (Path(path) / "a.py").write_text("x = 4\n")
        with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
            pass
        self.assertEqual(
            run("rev-parse", "refs/remotes/origin/main").stdout.strip(), base
        )

    def test_a_clean_worktree_is_a_no_op(self):
        path, run = self._repo()
        base = run("rev-parse", "HEAD").stdout.strip()
        with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
            # Nothing to commit, so HEAD must not move.
            self.assertEqual(run("rev-parse", "HEAD").stdout.strip(), base)
        self.assertEqual(run("rev-parse", "HEAD").stdout.strip(), base)

    def test_somewhere_that_is_not_a_repo_still_runs_the_normalizer(self):
        # Fail-soft: un-shaped is today's behaviour, but not running the
        # normalizer at all would be worse.
        import tempfile

        path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, path, True)
        with tasks_module._patch_shaped_for_checkers(self._checkout(path)):
            pass

    def test_the_checker_env_asks_for_patch_scoping(self):
        # Without this the checker's `diff_only` is False and it scans the
        # whole tree however well-shaped the worktree is.
        self.assertEqual(tasks_module._CHECKER_ENV["CI_PULL_REQUEST"], "1")
