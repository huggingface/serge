import unittest
from unittest import mock

from reviewbot import sandbox
from reviewbot.sandbox import (
    AUTO,
    OFF,
    REQUIRE,
    SandboxUnavailable,
    build_bwrap_argv,
    normalize_mode,
    wrap_command,
)


class NormalizeModeTests(unittest.TestCase):
    def test_known_values_pass_through(self):
        self.assertEqual(normalize_mode("require"), REQUIRE)
        self.assertEqual(normalize_mode("AUTO"), AUTO)
        self.assertEqual(normalize_mode(" Off "), OFF)

    def test_unknown_and_blank_default_to_auto(self):
        self.assertEqual(normalize_mode(None), AUTO)
        self.assertEqual(normalize_mode(""), AUTO)
        self.assertEqual(normalize_mode("banana"), AUTO)


class BuildArgvTests(unittest.TestCase):
    def test_profile_isolates_network_and_secrets(self):
        # Pin the venv root so the assertions don't depend on where the
        # test happens to run (a venv under /home on CI would otherwise
        # trip the /home secrets check below).
        with mock.patch.object(sandbox, "_venv_root", return_value="/opt/venv"):
            argv = build_bwrap_argv(
                ["ruff", "check"], workdir="/wt/sub", write_root="/wt"
            )
        self.assertEqual(argv[0], "bwrap")
        # No network.
        self.assertIn("--unshare-net", argv)
        self.assertIn("--unshare-all", argv)
        # System binaries available read-only; worktree is the writable bind.
        self.assertIn("--ro-bind", argv)
        joined = " ".join(argv)
        self.assertIn("--ro-bind /usr /usr", joined)
        # The active venv is bound read-only so installed helpers resolve.
        self.assertIn("--ro-bind /opt/venv /opt/venv", joined)
        self.assertIn("--bind /wt /wt", joined)
        self.assertIn("--chdir /wt/sub", joined)
        # Host secrets are never bound in.
        self.assertNotIn("/home", joined)
        self.assertNotIn("/etc/reviewbot", joined)
        # The wrapped command follows the -- separator.
        sep = argv.index("--")
        self.assertEqual(argv[sep + 1 :], ["ruff", "check"])


class WrapCommandTests(unittest.TestCase):
    def test_off_returns_command_unchanged(self):
        cmd = ["echo", "hi"]
        self.assertEqual(
            wrap_command(cmd, workdir="/wt", write_root="/wt", mode=OFF), cmd
        )

    def test_require_raises_when_bwrap_missing(self):
        with mock.patch.object(sandbox.shutil, "which", return_value=None):
            with self.assertRaises(SandboxUnavailable):
                wrap_command(["echo"], workdir="/wt", write_root="/wt", mode=REQUIRE)

    def test_auto_runs_unwrapped_when_bwrap_missing(self):
        cmd = ["echo", "hi"]
        with mock.patch.object(sandbox.shutil, "which", return_value=None):
            self.assertEqual(
                wrap_command(cmd, workdir="/wt", write_root="/wt", mode=AUTO), cmd
            )

    def test_wraps_when_bwrap_available(self):
        with mock.patch.object(
            sandbox.shutil, "which", return_value="/usr/bin/bwrap"
        ):
            argv = wrap_command(
                ["echo", "hi"], workdir="/wt", write_root="/wt", mode=REQUIRE
            )
        self.assertEqual(argv[0], "bwrap")
        self.assertIn("--unshare-net", argv)
        self.assertEqual(argv[-2:], ["echo", "hi"])


if __name__ == "__main__":
    unittest.main()
