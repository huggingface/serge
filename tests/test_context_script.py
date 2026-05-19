import os
import stat
import tempfile
import textwrap
import unittest

from reviewbot.context_script import ContextScriptResult, run_context_script


def _write_script(
    dir_: str, body: str, *, executable: bool = True, name: str = "ctx"
) -> str:
    path = os.path.join(dir_, name)
    with open(path, "w") as f:
        f.write(body)
    if executable:
        os.chmod(
            path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
    return path


class ContextScriptTests(unittest.TestCase):
    def test_returns_none_when_path_blank(self) -> None:
        self.assertIsNone(
            run_context_script("", title="t", body="b", files=[], timeout_seconds=5)
        )

    def test_returns_none_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(
                run_context_script(
                    "does-not-exist",
                    title="t",
                    body="b",
                    files=[],
                    timeout_seconds=5,
                    cwd=d,
                )
            )

    def test_returns_none_when_not_executable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho hi\n", executable=False)
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_plain_text_output_becomes_context(self) -> None:
        script = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            data = json.load(sys.stdin)
            print("title=" + data["title"])
            print("nfiles=" + str(len(data["files"])))
            print("first=" + data["files"][0]["path"])
            print("status=" + data["files"][0]["status"])
            """
        )
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, script)
            result = run_context_script(
                "ctx",
                title="My PR",
                body="desc",
                files=[
                    {
                        "filename": "a.py",
                        "status": "modified",
                        "additions": 3,
                        "deletions": 1,
                    },
                    {"filename": "b.py", "status": "added"},
                ],
                timeout_seconds=10,
                cwd=d,
            )
        assert result is not None
        self.assertIsInstance(result, ContextScriptResult)
        assert result.context is not None
        self.assertIn("title=My PR", result.context)
        self.assertIn("nfiles=2", result.context)
        self.assertIn("first=a.py", result.context)
        self.assertIn("status=modified", result.context)
        self.assertEqual(result.skip_files, [])

    def test_json_output_routes_context_and_skip_files(self) -> None:
        body = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            sys.stdin.read()
            print(json.dumps({
                "context": "skip the modeling files",
                "skip_files": ["a.py", "src/x/modeling_x.py"],
            }))
            """
        )
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, body)
            result = run_context_script(
                "ctx", title="t", body="b", files=[], timeout_seconds=10, cwd=d
            )
        assert result is not None
        self.assertEqual(result.context, "skip the modeling files")
        self.assertEqual(result.skip_files, ["a.py", "src/x/modeling_x.py"])

    def test_json_output_with_only_skip_files_has_no_context(self) -> None:
        body = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            sys.stdin.read()
            print(json.dumps({"skip_files": ["foo.py"]}))
            """
        )
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, body)
            result = run_context_script(
                "ctx", title="t", body="b", files=[], timeout_seconds=10, cwd=d
            )
        assert result is not None
        self.assertIsNone(result.context)
        self.assertEqual(result.skip_files, ["foo.py"])

    def test_invalid_json_falls_back_to_plain_text(self) -> None:
        # Looks like JSON but isn't valid → treated as plain text.
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho '{\"context\": broken'\n")
            result = run_context_script(
                "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
            )
        assert result is not None
        assert result.context is not None
        self.assertIn("broken", result.context)
        self.assertEqual(result.skip_files, [])

    def test_returns_none_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho oops\nexit 7\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_returns_none_on_empty_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\nexit 0\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_returns_none_when_json_yields_empty_result(self) -> None:
        # JSON parses but contains nothing useful → None.
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho '{}'\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_timeout_is_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\nsleep 10\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=1, cwd=d
                )
            )

    def test_truncates_oversized_plain_output(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\npython3 -c 'print(\"x\"*20000)'\n")
            result = run_context_script(
                "ctx", title="t", body="b", files=[], timeout_seconds=10, cwd=d
            )
        assert result is not None
        assert result.context is not None
        self.assertIn("truncated", result.context)
        self.assertLess(len(result.context), 9000)


if __name__ == "__main__":
    unittest.main()
