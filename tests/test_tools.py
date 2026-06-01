import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from reviewbot import tools as tools_mod
from reviewbot.tools import (
    ALLOWED_FETCH_HOSTS,
    DENY_DIR_NAMES,
    MAX_FETCH_BODY_CHARS,
    MAX_READ_LINES,
    MAX_TOOL_OUTPUT_CHARS,
    HelperInstallResult,
    RepoHelperTool,
    ToolEnv,
    build_tool_specs,
    install_helper_tools,
    load_repo_helper_tools,
    run_tool,
)


class _TempRepo(unittest.TestCase):
    """Tests share a temp git repo under self.repo_root with a small
    fixture tree we know the contents of."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.repo_root = self.tmpdir

        # Fixture files
        os.makedirs(os.path.join(self.repo_root, "src", "pkg"))
        with open(os.path.join(self.repo_root, "src", "pkg", "main.py"), "w") as f:
            f.write(
                "def hello():\n    return 'world'\n\n\ndef goodbye():\n    return 'bye'\n"
            )
        with open(os.path.join(self.repo_root, "src", "pkg", "util.py"), "w") as f:
            f.write("CONSTANT = 1\n")
        with open(os.path.join(self.repo_root, "README.md"), "w") as f:
            f.write("# fixture repo\n")
        os.makedirs(os.path.join(self.repo_root, "tools"))
        with open(os.path.join(self.repo_root, "tools", "helper.sh"), "w") as f:
            f.write(
                "#!/bin/sh\n"
                "printf 'cwd=%s\\n' \"$(pwd)\"\n"
                "printf 'argc=%s\\n' \"$#\"\n"
                'for arg in "$@"; do\n'
                "  printf 'arg=%s\\n' \"$arg\"\n"
                "done\n"
            )
        os.chmod(os.path.join(self.repo_root, "tools", "helper.sh"), 0o755)
        # An identical helper under .ai/ — the only repo location a helper
        # command is allowed to resolve to (the clone cache forces .ai/ to
        # the upstream default branch).
        os.makedirs(os.path.join(self.repo_root, ".ai"))
        with open(os.path.join(self.repo_root, ".ai", "helper.sh"), "w") as f:
            f.write(
                "#!/bin/sh\n"
                "printf 'cwd=%s\\n' \"$(pwd)\"\n"
                "printf 'argc=%s\\n' \"$#\"\n"
                'for arg in "$@"; do\n'
                "  printf 'arg=%s\\n' \"$arg\"\n"
                "done\n"
            )
        os.chmod(os.path.join(self.repo_root, ".ai", "helper.sh"), 0o755)
        # Pretend there's a .git dir so the denylist has something to bite on
        os.makedirs(os.path.join(self.repo_root, ".git", "refs"))
        # Real git init so `git grep` works in the grep tests
        subprocess.run(["git", "init", "-q"], cwd=self.repo_root, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=a", "add", "."],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=a@b",
                "-c",
                "user.name=a",
                "commit",
                "-q",
                "-m",
                "init",
            ],
            cwd=self.repo_root,
            check=True,
        )

        self.env = ToolEnv(repo_root=self.repo_root)


class ResolvePathTests(_TempRepo):
    def test_read_file_returns_numbered_lines(self) -> None:
        out = run_tool(
            self.env,
            "read_file",
            {"path": "src/pkg/main.py", "start_line": 1, "end_line": 3},
        )
        self.assertIn("src/pkg/main.py (lines 1-3)", out)
        self.assertIn("     1\tdef hello():", out)
        self.assertIn("     2\t    return 'world'", out)

    def test_read_file_caps_at_max_read_lines(self) -> None:
        big = os.path.join(self.repo_root, "big.txt")
        with open(big, "w") as f:
            for i in range(1, MAX_READ_LINES + 50):
                f.write(f"line{i}\n")
        out = run_tool(
            self.env,
            "read_file",
            {"path": "big.txt", "start_line": 1, "end_line": MAX_READ_LINES + 100},
        )
        self.assertIn(f"lines 1-{MAX_READ_LINES}", out)

    def test_read_file_rejects_path_escape(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "../etc/passwd"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("escapes the repository root", out)

    def test_read_file_rejects_absolute_path(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "/etc/passwd"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("absolute or home-relative", out)

    def test_read_file_rejects_denylisted_dir(self) -> None:
        out = run_tool(self.env, "read_file", {"path": ".git/HEAD"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("denylisted", out)

    def test_read_file_errors_on_directory(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "src"})
        self.assertIn("error:", out)
        self.assertIn("not a file", out)

    def test_list_dir_lists_entries_and_hides_denylist(self) -> None:
        out = run_tool(self.env, "list_dir", {"path": "."})
        self.assertIn("README.md", out)
        self.assertIn("src/", out)
        for name in DENY_DIR_NAMES:
            self.assertNotIn(f"\n{name}\n", out)

    def test_list_dir_descends_into_subdirs(self) -> None:
        out = run_tool(self.env, "list_dir", {"path": "src/pkg"})
        self.assertIn("main.py", out)
        self.assertIn("util.py", out)

    def test_grep_finds_match(self) -> None:
        out = run_tool(self.env, "grep", {"pattern": "def hello", "path": "src"})
        self.assertIn("src/pkg/main.py:1:def hello", out)

    def test_grep_reports_no_matches(self) -> None:
        out = run_tool(
            self.env, "grep", {"pattern": "definitely_not_there", "path": "src"}
        )
        self.assertIn("no matches", out)

    def test_grep_rejects_non_ascii_pattern(self) -> None:
        out = run_tool(self.env, "grep", {"pattern": "héllo"})
        self.assertTrue(out.startswith("error:"), out)

    def test_unknown_tool_returns_error_message(self) -> None:
        out = run_tool(self.env, "delete_repo", {})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("unknown tool", out)

    def test_truncates_oversized_output(self) -> None:
        big = os.path.join(self.repo_root, "huge.txt")
        with open(big, "w") as f:
            f.write("x" * (MAX_TOOL_OUTPUT_CHARS * 2))
        out = run_tool(self.env, "read_file", {"path": "huge.txt"})
        self.assertIn("[... truncated", out)

    def test_build_tool_specs_appends_repo_helpers(self) -> None:
        env = ToolEnv(
            repo_root=self.repo_root,
            helper_tools={
                "fixture_helper": RepoHelperTool(
                    name="fixture_helper",
                    description="Run the fixture helper script.",
                    command=("./tools/helper.sh",),
                    allow_args=True,
                    max_args=2,
                )
            },
        )

        specs = build_tool_specs(env)

        helper_spec = next(
            spec for spec in specs if spec["function"]["name"] == "fixture_helper"
        )
        self.assertEqual(
            helper_spec["function"]["description"],
            "Run the fixture helper script.",
        )
        self.assertIn("args", helper_spec["function"]["parameters"]["properties"])

    def test_repo_helper_runs_dot_ai_command(self) -> None:
        env = ToolEnv(
            repo_root=self.repo_root,
            helper_tools={
                "fixture_helper": RepoHelperTool(
                    name="fixture_helper",
                    description="Run the fixture helper script.",
                    command=("./.ai/helper.sh",),
                    allow_args=True,
                    max_args=2,
                )
            },
        )

        out = run_tool(env, "fixture_helper", {"args": ["alpha", "beta"]})

        self.assertIn("exit_code: 0", out)
        self.assertIn("argc=2", out)
        self.assertIn("arg=alpha", out)
        self.assertIn("arg=beta", out)

    def test_repo_helper_rejects_command_outside_dot_ai(self) -> None:
        # A path-form command pointing into the fork tree (outside .ai/)
        # is refused — only PATH binaries or .ai/ scripts may run.
        env = ToolEnv(
            repo_root=self.repo_root,
            helper_tools={
                "fixture_helper": RepoHelperTool(
                    name="fixture_helper",
                    description="Run the fixture helper script.",
                    command=("./tools/helper.sh",),
                )
            },
        )

        out = run_tool(env, "fixture_helper", {})

        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("must be a PATH binary or live under .ai/", out)

    def test_repo_helper_rejects_args_when_not_allowed(self) -> None:
        env = ToolEnv(
            repo_root=self.repo_root,
            helper_tools={
                "fixture_helper": RepoHelperTool(
                    name="fixture_helper",
                    description="Run the fixture helper script.",
                    command=("./tools/helper.sh",),
                )
            },
        )

        out = run_tool(env, "fixture_helper", {"args": ["alpha"]})

        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("does not accept additional args", out)


class RepoHelperConfigTests(unittest.TestCase):
    def test_load_repo_helper_tools_parses_valid_config(self) -> None:
        helpers = load_repo_helper_tools(
            """
            {
              "helpers": [
                {
                  "name": "mlinter",
                  "description": "Run the model linter.",
                  "command": ["mlinter"],
                  "allow_args": true,
                  "max_args": 4,
                  "timeout_seconds": 30
                }
              ]
            }
            """
        )

        self.assertEqual(len(helpers), 1)
        self.assertEqual(helpers[0].name, "mlinter")
        self.assertEqual(helpers[0].command, ("mlinter",))
        self.assertTrue(helpers[0].allow_args)
        self.assertEqual(helpers[0].max_args, 4)

    def test_load_repo_helper_tools_rejects_builtin_name_conflict(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            load_repo_helper_tools(
                """
                {"helpers": [{"name": "grep", "description": "bad", "command": ["x"]}]}
                """
            )
        self.assertIn("conflicts with a built-in tool", str(ctx.exception))

    def test_load_repo_helper_tools_parses_install_spec(self) -> None:
        helpers = load_repo_helper_tools(
            """
            {
              "helpers": [
                {
                  "name": "mlinter",
                  "description": "Run the model linter.",
                  "command": ["mlinter"],
                  "install": ["pip", "install", "transformers-mlinter"]
                }
              ]
            }
            """
        )
        self.assertEqual(helpers[0].install, ("pip", "install", "transformers-mlinter"))

    def test_load_repo_helper_tools_rejects_unknown_installer(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            load_repo_helper_tools(
                """
                {"helpers": [{
                  "name": "mlinter",
                  "description": "x",
                  "command": ["mlinter"],
                  "install": ["curl", "https://evil.example/install.sh"]
                }]}
                """
            )
        self.assertIn("install command must start with", str(ctx.exception))

    def test_load_repo_helper_tools_rejects_empty_install(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            load_repo_helper_tools(
                """
                {"helpers": [{
                  "name": "mlinter",
                  "description": "x",
                  "command": ["mlinter"],
                  "install": []
                }]}
                """
            )
        self.assertIn("non-empty array", str(ctx.exception))

    def test_load_repo_helper_tools_rejects_install_arg_with_shell_meta(self) -> None:
        # `;` would be a problem if anyone ever shelled out — reject up front
        # so the config can't smuggle shell metacharacters into installer args.
        with self.assertRaises(ValueError) as ctx:
            load_repo_helper_tools(
                """
                {"helpers": [{
                  "name": "mlinter",
                  "description": "x",
                  "command": ["mlinter"],
                  "install": ["pip", "install", "foo; rm -rf /"]
                }]}
                """
            )
        self.assertIn("disallowed characters", str(ctx.exception))


class InstallHelperToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the process-local cache so each test is independent.
        tools_mod._INSTALL_CACHE.clear()

    def _helper(self) -> RepoHelperTool:
        return RepoHelperTool(
            name="mlinter",
            description="Model linter.",
            command=("mlinter",),
            install=("pip", "install", "transformers-mlinter"),
        )

    def test_skips_helpers_without_install(self) -> None:
        helper = RepoHelperTool(
            name="plain",
            description="x",
            command=("plain",),
        )
        with patch("reviewbot.tools.subprocess.run") as mock_run:
            results = install_helper_tools([helper])
        self.assertEqual(results, [])
        mock_run.assert_not_called()

    def test_runs_pip_via_python_module(self) -> None:
        helper = self._helper()
        mock_proc = Mock(returncode=0, stdout="ok", stderr="")
        with patch(
            "reviewbot.tools.subprocess.run", return_value=mock_proc
        ) as mock_run:
            results = install_helper_tools([helper])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok, results[0].message)
        cmd = mock_run.call_args.args[0]
        # Run via `python -m pip` so the install lands in this interpreter.
        self.assertEqual(cmd[1:], ["-m", "pip", "install", "transformers-mlinter"])

    def test_caches_successful_install(self) -> None:
        helper = self._helper()
        mock_proc = Mock(returncode=0, stdout="", stderr="")
        with patch(
            "reviewbot.tools.subprocess.run", return_value=mock_proc
        ) as mock_run:
            install_helper_tools([helper])
            results = install_helper_tools([helper])
        # Second call should hit the cache, not subprocess.
        self.assertEqual(mock_run.call_count, 1)
        self.assertTrue(results[0].ok)
        self.assertIn("cached", results[0].message)

    def test_failed_install_is_not_cached(self) -> None:
        helper = self._helper()
        mock_proc = Mock(returncode=1, stdout="", stderr="pkg not found")
        with patch(
            "reviewbot.tools.subprocess.run", return_value=mock_proc
        ) as mock_run:
            r1 = install_helper_tools([helper])
            r2 = install_helper_tools([helper])
        self.assertEqual(mock_run.call_count, 2)
        self.assertFalse(r1[0].ok)
        self.assertFalse(r2[0].ok)
        self.assertIn("pkg not found", r1[0].message)

    def test_install_timeout_is_reported(self) -> None:
        helper = self._helper()
        with patch(
            "reviewbot.tools.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=1),
        ):
            results = install_helper_tools([helper])
        self.assertFalse(results[0].ok)
        self.assertIn("timed out", results[0].message)

    def test_result_dataclass_shape(self) -> None:
        # Sanity: callers depend on .name / .ok / .message.
        r = HelperInstallResult(name="x", ok=True, message="m")
        self.assertEqual((r.name, r.ok, r.message), ("x", True, "m"))


class FetchUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ToolEnv(repo_root=tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.env.repo_root, ignore_errors=True)

    def test_fetch_returns_status_and_body_for_allowed_host(self) -> None:
        response = Mock(
            status_code=200,
            reason="OK",
            text="<html>Molmo2 paper</html>",
            content=b"<html>Molmo2 paper</html>",
            headers={"Content-Type": "text/html"},
            is_redirect=False,
        )
        with patch("reviewbot.tools.requests.get", return_value=response) as mock_get:
            out = run_tool(
                self.env,
                "fetch_url",
                {"url": "https://huggingface.co/papers/2403.09611"},
            )
        self.assertIn("HTTP 200 OK", out)
        self.assertIn("Content-Type: text/html", out)
        self.assertIn("Molmo2 paper", out)
        # Network call must be redirect-free so we can't be smuggled
        # off-host.
        self.assertFalse(mock_get.call_args.kwargs["allow_redirects"])

    def test_fetch_rejects_non_huggingface_host(self) -> None:
        with patch("reviewbot.tools.requests.get") as mock_get:
            out = run_tool(self.env, "fetch_url", {"url": "https://example.com/foo"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("not in the allowlist", out)
        mock_get.assert_not_called()

    def test_fetch_rejects_http_scheme(self) -> None:
        with patch("reviewbot.tools.requests.get") as mock_get:
            out = run_tool(self.env, "fetch_url", {"url": "http://huggingface.co/foo"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("only https", out)
        mock_get.assert_not_called()

    def test_fetch_reports_redirect_target(self) -> None:
        response = Mock(
            status_code=302,
            reason="Found",
            text="",
            content=b"",
            headers={"Location": "https://huggingface.co/papers/2403.09611"},
            is_redirect=True,
        )
        with patch("reviewbot.tools.requests.get", return_value=response):
            out = run_tool(
                self.env, "fetch_url", {"url": "https://huggingface.co/papers/foo"}
            )
        self.assertIn("HTTP 302", out)
        self.assertIn("Location: https://huggingface.co/papers/2403.09611", out)

    def test_fetch_truncates_long_body(self) -> None:
        big_body = "x" * (MAX_FETCH_BODY_CHARS * 3)
        response = Mock(
            status_code=200,
            reason="OK",
            text=big_body,
            content=big_body.encode(),
            headers={"Content-Type": "text/plain"},
            is_redirect=False,
        )
        with patch("reviewbot.tools.requests.get", return_value=response):
            out = run_tool(self.env, "fetch_url", {"url": "https://huggingface.co/foo"})
        self.assertIn("[... truncated", out)

    def test_fetch_skips_body_for_non_text(self) -> None:
        response = Mock(
            status_code=200,
            reason="OK",
            text="(binary)",
            content=b"\x00\x01\x02" * 100,
            headers={"Content-Type": "application/octet-stream"},
            is_redirect=False,
        )
        with patch("reviewbot.tools.requests.get", return_value=response):
            out = run_tool(self.env, "fetch_url", {"url": "https://huggingface.co/foo"})
        self.assertIn("non-text body", out)

    def test_fetch_handles_timeout(self) -> None:
        with patch(
            "reviewbot.tools.requests.get",
            side_effect=requests.Timeout("slow"),
        ):
            out = run_tool(self.env, "fetch_url", {"url": "https://huggingface.co/foo"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("timed out", out)

    def test_allowlist_contains_huggingface(self) -> None:
        # Sanity: the allowlist surface stays narrow on purpose.
        self.assertEqual(ALLOWED_FETCH_HOSTS, frozenset({"huggingface.co"}))


if __name__ == "__main__":
    unittest.main()
