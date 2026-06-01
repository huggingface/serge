import os
import subprocess
import tempfile
import time
import unittest

from reviewbot.clone_cache import CloneCache


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


class CloneCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name

        # A source repo whose pull/1/head ref the cache will fetch.
        self.src = os.path.join(root, "src")
        os.makedirs(self.src)
        _git(self.src, "init", "--quiet", "-b", "main")
        with open(os.path.join(self.src, "hello.txt"), "w") as f:
            f.write("hi from the PR\n")
        # A symlink that must come out as a plain file (core.symlinks=false).
        os.symlink("hello.txt", os.path.join(self.src, "link.txt"))
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", "pr commit")
        _git(self.src, "update-ref", "refs/pull/1/head", "HEAD")

        self.cache = CloneCache(os.path.join(root, "cache"))

    def _acquire(self, number=1, job_id="job1"):
        return self.cache.acquire(
            token="",
            owner="acme",
            repo="widget",
            number=number,
            job_id=job_id,
            remote_url=self.src,
        )

    def _commit(self, path, content):
        full = os.path.join(self.src, path)
        os.makedirs(os.path.dirname(full) or self.src, exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", f"set {path}")

    def test_acquire_checks_out_pr_head(self):
        co = self._acquire()
        self.assertIsNotNone(co)
        with open(os.path.join(co.path, "hello.txt")) as f:
            self.assertEqual(f.read(), "hi from the PR\n")

    def test_symlink_written_as_plain_file(self):
        co = self._acquire()
        link = os.path.join(co.path, "link.txt")
        # core.symlinks=false: the symlink is materialized as a regular file
        # containing the target path, so helper tools can't escape the tree.
        self.assertFalse(os.path.islink(link))
        with open(link) as f:
            self.assertEqual(f.read(), "hello.txt")

    def test_bare_repo_shared_across_jobs(self):
        co1 = self._acquire(job_id="job1")
        co2 = self._acquire(job_id="job2")
        self.assertEqual(co1.bare, co2.bare)
        repos_dir = os.path.join(self.cache.root, "repos")
        self.assertEqual(len(os.listdir(repos_dir)), 1)  # one fetch, two worktrees
        self.assertNotEqual(co1.path, co2.path)
        self.assertTrue(os.path.isdir(co1.path))
        self.assertTrue(os.path.isdir(co2.path))

    def test_release_removes_worktree_and_branch(self):
        co = self._acquire()
        self.cache.release(co)
        self.assertFalse(os.path.exists(co.path))
        branches = subprocess.run(
            ["git", "-C", co.bare, "branch", "--list", co.branch],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(branches.stdout.strip(), "")

    def test_gc_removes_stale_repo(self):
        co = self._acquire()
        self.cache.release(co)
        # Nothing collected when the repo is fresh.
        self.assertEqual(self.cache.gc(max_age_seconds=3600), 0)
        self.assertTrue(os.path.isdir(co.bare))
        # Pretend the bare repo hasn't been touched in a week.
        self.assertEqual(
            self.cache.gc(max_age_seconds=3600, now=time.time() + 8 * 86400), 1
        )
        self.assertFalse(os.path.isdir(co.bare))

    def test_acquire_failure_returns_none(self):
        # No such pull ref → git fetch fails, acquire returns None cleanly.
        co = self.cache.acquire(
            token="",
            owner="acme",
            repo="widget",
            number=999,
            job_id="jobX",
            remote_url=self.src,
        )
        self.assertIsNone(co)

    def test_ai_overlay_uses_default_branch_not_pr_head(self):
        # main (the default branch) carries the trusted .ai/ config.
        self._commit(".ai/rules.md", "UPSTREAM RULES\n")
        # A PR commit tampers with .ai/ and ships a fork script.
        _git(self.src, "checkout", "--quiet", "-b", "attacker")
        self._commit(".ai/rules.md", "MALICIOUS RULES\n")
        self._commit("evil.sh", "#!/bin/sh\necho pwned\n")
        _git(self.src, "update-ref", "refs/pull/2/head", "HEAD")
        # Restore main as the remote's default branch (HEAD).
        _git(self.src, "checkout", "--quiet", "main")

        co = self._acquire(number=2, job_id="job-overlay")
        self.assertIsNotNone(co)
        # .ai/ is the default branch's copy, never the PR's.
        with open(os.path.join(co.path, ".ai", "rules.md")) as f:
            self.assertEqual(f.read(), "UPSTREAM RULES\n")
        # Non-.ai PR content is still the PR head's.
        self.assertTrue(os.path.isfile(os.path.join(co.path, "evil.sh")))

    def test_ai_overlay_drops_pr_ai_when_default_branch_has_none(self):
        # Default branch (main) has no .ai/ at all; the PR introduces one.
        _git(self.src, "checkout", "--quiet", "-b", "attacker2")
        self._commit(".ai/context-script", "#!/bin/sh\necho pwned\n")
        _git(self.src, "update-ref", "refs/pull/3/head", "HEAD")
        _git(self.src, "checkout", "--quiet", "main")

        co = self._acquire(number=3, job_id="job-noai")
        self.assertIsNotNone(co)
        # Fail-closed: an .ai/ the upstream default branch doesn't have is
        # dropped rather than trusted.
        self.assertFalse(os.path.exists(os.path.join(co.path, ".ai")))


if __name__ == "__main__":
    unittest.main()
