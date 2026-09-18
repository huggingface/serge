"""What the model is told when `git apply` refuses its patch.

Three prod tasks died this way on 2026-09-16/17 (`pvt_v2`, `qwen3_omni_moe`,
`zamba`; triage issues transformers#48881 and #48914). Each spent its whole
correction budget re-guessing: `rejected_patch: 3`, `patch_apply_error: 3`,
`validation_retries: 2`, then publish applied the same patch a fourth time and
reported that. 3.58M input tokens across the three, 0 PRs, 0 branches.

The cause was not a weak apply fallback — strict, `--recount`, `-C1`, `-C0`,
`--3way` and GNU `patch --fuzz=3` all reject all three patches. It was that the
correction turn could not see the file: tools were stripped, and the feedback
was two lines of `git apply` stderr naming a line number.
"""

import subprocess
import types

from reviewbot import tasks


# ── what the file actually holds ────────────────────────────────────────────


def _reader(files):
    return lambda path: files.get(path)


def test_windows_quote_the_real_file_around_each_hunk():
    patch = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -3,2 +3,2 @@\n-old\n+new\n"
    )
    out = tasks.patch_target_windows(
        patch, _reader({"f.py": [f"line{i}" for i in range(1, 21)]})
    )
    assert "f.py (lines 1-13):" in out
    # Numbered exactly as tools.read_file numbers its output, so the model is
    # not asked to reconcile two different renderings of the same file.
    assert "     3\tline3" in out


def test_windows_cover_every_hunk_of_a_multi_hunk_patch():
    patch = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -5,1 +5,1 @@\n-a\n+b\n"
        "@@ -60,1 +60,1 @@\n-c\n+d\n"
    )
    out = tasks.patch_target_windows(
        patch, _reader({"f.py": [f"l{i}" for i in range(1, 101)]})
    )
    assert out.count("f.py (lines") == 2
    assert "    60\tl60" in out


def test_windows_say_so_when_the_file_is_not_there():
    patch = "diff --git a/new.py b/new.py\n--- a/new.py\n+++ b/new.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
    out = tasks.patch_target_windows(patch, _reader({}))
    assert "not in the checkout" in out


def test_windows_say_so_when_the_hunk_is_past_the_end():
    """A hunk aimed past EOF is the loudest possible sign the model is patching
    a different version of the file than the one in the checkout."""
    patch = "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ -900,1 +900,1 @@\n-x\n+y\n"
    out = tasks.patch_target_windows(patch, _reader({"f.py": ["a", "b"]}))
    assert "past the end of the file" in out
    assert "2 lines" in out


def test_windows_are_empty_for_a_patch_with_no_hunks():
    assert tasks.patch_target_windows("not a diff", _reader({})) == ""


# ── git's own echo must not reach the model ─────────────────────────────────


_RAW = """Checking patch tests/models/zamba/test_modeling_zamba.py...
Hunk #1 succeeded at 468 (offset 1 line).
error: while searching for:
        self.assertEqual(
            output_sentences[1],
            "<s><s> Tell me a story about a time when you were in a difficult situation",
        )

error: patch failed: tests/models/zamba/test_modeling_zamba.py:488
error: tests/models/zamba/test_modeling_zamba.py: patch does not apply"""


def test_the_searched_for_block_is_stripped_for_the_model():
    """`git apply -v` quotes the context it looked for — the model's OWN wrong
    lines. Measured against the deployed model on the real zamba rejection: with
    that block in the prompt the true line appeared 3 times and the fabricated
    one once, and the model still reproduced the fabrication 3 times out of 3.
    """
    out = tasks.apply_error_for_model(_RAW)
    assert "<s><s>" not in out, "the model's fabrication was handed back to it"
    assert "while searching for" not in out


def test_the_location_survives_stripping():
    """It still has to say WHERE to look, just not what to look for."""
    out = tasks.apply_error_for_model(_RAW)
    assert "error: patch failed: tests/models/zamba/test_modeling_zamba.py:488" in out
    assert "patch does not apply" in out


def test_stripping_a_plain_error_changes_nothing():
    plain = "error: patch failed: f.py:12\nerror: f.py: patch does not apply"
    assert tasks.apply_error_for_model(plain) == plain


# ── the rejection kind decides how the correction turn is built ─────────────


def _cfg():
    return types.SimpleNamespace(
        task_normalize_command=["true"],
        task_normalize_timeout=60,
        task_normalize_max_retries=2,
        task_normalize_guidance=None,
        task_sandbox_backend="off",
        task_normalize_image=None,
        task_normalize_memory=None,
        helper_sandbox="off",
        task_scope_commit_to_patch=False,
    )


def test_an_apply_rejection_is_reported_as_such(monkeypatch, tmp_path):
    """`prepare_task` keys the correction turn off this: an apply rejection must
    keep the model's tools, because it is a disagreement about file contents."""
    (tmp_path / "f.py").write_text("real\n")

    class _CC:
        def reset_worktree(self, checkout):
            pass

        def apply_patch(self, checkout, patch):
            raise subprocess.CalledProcessError(
                1, ["git", "apply"], stderr=b"error: patch failed: f.py:1"
            )

    report: dict = {}
    feedback, prepared = tasks._validate_patch(
        _cfg(),
        checkout=types.SimpleNamespace(path=str(tmp_path)),
        clone_cache=_CC(),
        content='{"title": "t", "body": "b", "patch": "diff --git a/f.py b/f.py\\n'
        '--- a/f.py\\n+++ b/f.py\\n@@ -1,1 +1,1 @@\\n-wrong\\n+fixed\\n"}',
        emit=lambda *a: None,
        report=report,
    )
    assert report["rejection"] == "apply"
    assert prepared is False
    assert feedback and "ACTUALLY contain" in feedback
    assert "     1\treal" in feedback, "the real file content must be in the feedback"


def test_no_patch_to_validate_reports_no_rejection():
    report: dict = {}
    feedback, prepared = tasks._validate_patch(
        _cfg(),
        checkout=types.SimpleNamespace(path="/tmp"),
        clone_cache=types.SimpleNamespace(),
        content='{"title": "t", "body": "b", "patch": ""}',
        emit=lambda *a: None,
        report=report,
    )
    assert (feedback, prepared) == (None, False)
    assert report["rejection"] == ""


def test_apply_patch_surfaces_the_verbose_diagnostic(tmp_path):
    """The raised stderr must carry `git apply -v`'s search block: that is the
    operator's whole diagnosis of a rejected patch, and without it the stored
    job error says only "patch failed: <file>:<line>"."""
    import os

    from reviewbot.clone_cache import CloneCache

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "f.py").write_text("alpha\nbeta\ngamma\n")
    for args in (
        ["init", "-q", "."],
        ["add", "-A"],
        ["-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "b"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    cc = CloneCache(str(tmp_path / "cache"))
    bad = (
        "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n alpha\n-NOTHING LIKE THE FILE\n+x\n gamma\n"
    )
    try:
        cc.apply_patch(types.SimpleNamespace(path=str(repo)), bad)
        raise AssertionError("should not have applied")
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode()
    assert "while searching for" in err
    assert "NOTHING LIKE THE FILE" in err
    assert os.path.exists(repo / "f.py")
    assert (repo / "f.py").read_text() == "alpha\nbeta\ngamma\n", "tree untouched"
