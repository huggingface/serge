"""Tests for the shared git mirror: CloneCache.update_mirror (git mechanics
against a local remote) and the MirrorWarmer scheduling/fail-soft logic."""

import os
import subprocess
import tempfile
import unittest

from reviewbot.clone_cache import CloneCache, mirror_bare_path
from reviewbot.mirror import MirrorWarmer


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


class UpdateMirrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = self._tmp.name
        self.src = os.path.join(root, "src")
        os.makedirs(self.src)
        _git(self.src, "init", "--quiet", "-b", "main")
        with open(os.path.join(self.src, "f.txt"), "w") as f:
            f.write("base\n")
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", "base")
        self.mirror_dir = os.path.join(root, "mirror")
        self.cache = CloneCache(self.mirror_dir)

    def _mirror_log(self):
        bare = mirror_bare_path(self.mirror_dir, "acme", "widget")
        out = subprocess.run(
            ["git", "-C", bare, "log", "--oneline", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout

    def test_creates_and_refreshes_mirror(self):
        bare = self.cache.update_mirror("", "acme", "widget", remote_url=self.src)
        self.assertEqual(bare, mirror_bare_path(self.mirror_dir, "acme", "widget"))
        self.assertTrue(os.path.isdir(os.path.join(bare, "objects")))
        self.assertIn("base", self._mirror_log())

        # A new upstream commit shows up after a refresh (fetch --prune).
        with open(os.path.join(self.src, "f.txt"), "a") as f:
            f.write("more\n")
        _git(self.src, "add", "-A")
        _git(self.src, "commit", "--quiet", "-m", "second")
        self.cache.update_mirror("", "acme", "widget", remote_url=self.src)
        log = self._mirror_log()
        self.assertIn("second", log)
        self.assertIn("base", log)


class _FakeCache:
    def __init__(self):
        self.calls = []
        self.raise_on = set()

    def update_mirror(self, token, owner, repo, **kw):
        self.calls.append((token, owner, repo))
        if (owner, repo) in self.raise_on:
            raise subprocess.CalledProcessError(1, ["git", "fetch"])
        return f"/mirror/{owner}__{repo}.git"


class MirrorWarmerTests(unittest.TestCase):
    def _warmer(self, provider):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        w = MirrorWarmer(tmp.name, provider, interval_seconds=1)
        fake = _FakeCache()
        w._cache = fake  # swap the real git backend for a recorder
        return w, fake

    def test_register_is_deduped(self):
        w, _ = self._warmer(lambda o, r: "tok")
        self.assertTrue(w.register("acme", "widget"))
        self.assertFalse(w.register("acme", "widget"))

    def test_refresh_calls_update_with_token(self):
        w, fake = self._warmer(lambda o, r: "tok")
        w.register("acme", "widget")
        self.assertTrue(w.refresh_one("acme", "widget"))
        self.assertEqual(fake.calls, [("tok", "acme", "widget")])

    def test_refresh_skips_when_no_token(self):
        # App not installed → provider returns None → skip, do not call git.
        w, fake = self._warmer(lambda o, r: None)
        self.assertFalse(w.refresh_one("acme", "widget"))
        self.assertEqual(fake.calls, [])

    def test_refresh_survives_git_failure(self):
        w, fake = self._warmer(lambda o, r: "tok")
        fake.raise_on.add(("acme", "widget"))
        # Must not raise; returns False.
        self.assertFalse(w.refresh_one("acme", "widget"))

    def test_refresh_survives_token_provider_error(self):
        def boom(o, r):
            raise RuntimeError("network")

        w, fake = self._warmer(boom)
        self.assertFalse(w.refresh_one("acme", "widget"))
        self.assertEqual(fake.calls, [])

    def test_refresh_all_iterates_tracked(self):
        w, fake = self._warmer(lambda o, r: "tok")
        w.register("acme", "widget")
        w.register("acme", "gadget")
        w.refresh_all()
        self.assertEqual(
            sorted(fake.calls), [("tok", "acme", "gadget"), ("tok", "acme", "widget")]
        )


if __name__ == "__main__":
    unittest.main()
