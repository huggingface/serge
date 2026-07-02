import unittest
from unittest import mock

from reviewbot import sandbox
from reviewbot.sandbox import (
    AUTO,
    AUTO_BACKEND,
    BWRAP_BACKEND,
    DOCKER_BACKEND,
    OFF,
    REQUIRE,
    DockerUnavailable,
    SandboxUnavailable,
    build_bwrap_argv,
    build_docker_argv,
    normalize_backend,
    normalize_mode,
    wrap_command,
    wrap_task_command,
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
        with mock.patch.object(sandbox.shutil, "which", return_value="/usr/bin/bwrap"):
            argv = wrap_command(
                ["echo", "hi"], workdir="/wt", write_root="/wt", mode=REQUIRE
            )
        self.assertEqual(argv[0], "bwrap")
        self.assertIn("--unshare-net", argv)
        self.assertEqual(argv[-2:], ["echo", "hi"])


class NormalizeBackendTests(unittest.TestCase):
    def test_known_values_pass_through(self):
        self.assertEqual(normalize_backend("docker"), DOCKER_BACKEND)
        self.assertEqual(normalize_backend(" BWRAP "), BWRAP_BACKEND)
        self.assertEqual(normalize_backend("auto"), AUTO_BACKEND)

    def test_unknown_and_blank_default_to_auto(self):
        self.assertEqual(normalize_backend(None), AUTO_BACKEND)
        self.assertEqual(normalize_backend(""), AUTO_BACKEND)
        self.assertEqual(normalize_backend("podman"), AUTO_BACKEND)
        # kubernetes is no longer a normalize backend (per-task-pod model).
        self.assertEqual(normalize_backend("kubernetes"), AUTO_BACKEND)


class BuildDockerArgvTests(unittest.TestCase):
    def test_isolation_flags(self):
        argv = build_docker_argv(
            ["make", "fix-repo"],
            image="serge/transformers-quality:latest",
            workdir="/wt",
            write_root="/wt",
            uid=1000,
            gid=1000,
            memory="4g",
        )
        self.assertEqual(argv[:3], ["docker", "run", "--rm"])
        joined = " ".join(argv)
        # No network by default; deps must be baked into the image.
        self.assertIn("--network none", joined)
        # Read-only rootfs + writable tmpfs + capability drop.
        self.assertIn("--read-only", argv)
        self.assertIn("--cap-drop", argv)
        self.assertIn("ALL", argv)
        self.assertIn("no-new-privileges", argv)
        self.assertIn("--memory 4g", joined)
        # Worktree bound rw at the same path; cwd is the worktree.
        self.assertIn("--volume /wt:/wt:rw", joined)
        self.assertIn("--workdir /wt", joined)
        # Runs as the host uid:gid so written files aren't root-owned.
        self.assertIn("--user 1000:1000", joined)
        # Image then command, in that order, at the end.
        self.assertEqual(
            argv[-3:], ["serge/transformers-quality:latest", "make", "fix-repo"]
        )

    def test_network_opt_in(self):
        argv = build_docker_argv(
            ["x"], image="img", workdir="/wt", write_root="/wt", network=True
        )
        self.assertIn("--network bridge", " ".join(argv))


class WrapTaskCommandTests(unittest.TestCase):
    def test_bwrap_backend_off_returns_command(self):
        cmd = ["make", "fix-repo"]
        self.assertEqual(
            wrap_task_command(
                cmd,
                workdir="/wt",
                write_root="/wt",
                backend=BWRAP_BACKEND,
                image=None,
                mode=OFF,
            ),
            cmd,
        )

    def test_docker_backend_without_image_raises(self):
        with self.assertRaises(DockerUnavailable):
            wrap_task_command(
                ["x"],
                workdir="/wt",
                write_root="/wt",
                backend=DOCKER_BACKEND,
                image=None,
                mode=OFF,
            )

    def test_docker_backend_without_cli_raises(self):
        with mock.patch.object(sandbox, "docker_available", return_value=False):
            with self.assertRaises(DockerUnavailable):
                wrap_task_command(
                    ["x"],
                    workdir="/wt",
                    write_root="/wt",
                    backend=DOCKER_BACKEND,
                    image="img",
                    mode=OFF,
                )

    def test_auto_picks_docker_when_image_and_cli_present(self):
        with mock.patch.object(sandbox, "docker_available", return_value=True):
            argv = wrap_task_command(
                ["make", "fix-repo"],
                workdir="/wt",
                write_root="/wt",
                backend=AUTO_BACKEND,
                image="img",
                mode=OFF,
            )
        self.assertEqual(argv[0], "docker")
        self.assertEqual(argv[-3:], ["img", "make", "fix-repo"])

    def test_auto_falls_back_to_bwrap_without_image(self):
        cmd = ["make", "fix-repo"]
        with mock.patch.object(sandbox.shutil, "which", return_value=None):
            # No image -> bwrap backend; bwrap missing + mode OFF -> unwrapped.
            argv = wrap_task_command(
                cmd,
                workdir="/wt",
                write_root="/wt",
                backend=AUTO_BACKEND,
                image=None,
                mode=OFF,
            )
        self.assertEqual(argv, cmd)


if __name__ == "__main__":
    unittest.main()
