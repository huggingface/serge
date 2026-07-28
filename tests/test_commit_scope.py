"""Tests for scoping the commit to the files the patch is responsible for.

Reference failure: prod task 433e8274 patched one file,
``tests/models/glm_ocr/test_modeling_glm_ocr.py``, and committed 32 — the other
31 were ``modeling_*``/``processing_*`` files the normalizer regenerated across
30 unrelated models because ``main`` was already stale.
"""

import unittest

from reviewbot.commit_scope import describe_dropped, patch_paths, scope_paths

PATCH = """diff --git a/tests/models/glm_ocr/test_modeling_glm_ocr.py b/tests/models/glm_ocr/test_modeling_glm_ocr.py
index 1111111..2222222 100644
--- a/tests/models/glm_ocr/test_modeling_glm_ocr.py
+++ b/tests/models/glm_ocr/test_modeling_glm_ocr.py
@@ -400,7 +400,7 @@ class GlmOcrIntegrationTest(unittest.TestCase):
-        expected = [1, 2, 3]
+        expected = [4, 5, 6]
"""

# The real 32, in the order serge logged them.
PROD_CHANGED = [
    "src/transformers/models/bamba/modeling_bamba.py",
    "src/transformers/models/colmodernvbert/processing_colmodernvbert.py",
    "src/transformers/models/colpali/processing_colpali.py",
    "src/transformers/models/colqwen2/processing_colqwen2.py",
    "src/transformers/models/deepseek_vl_hybrid/processing_deepseek_vl_hybrid.py",
    "src/transformers/models/exaone4_5/processing_exaone4_5.py",
    "src/transformers/models/gemma3n/modeling_gemma3n.py",
    "src/transformers/models/gemma4_unified/processing_gemma4_unified.py",
    "src/transformers/models/glm46v/image_processing_glm46v.py",
    "src/transformers/models/glm46v/processing_glm46v.py",
    "src/transformers/models/glm4v/processing_glm4v.py",
    "src/transformers/models/glm_image/image_processing_glm_image.py",
    "src/transformers/models/glmasr/processing_glmasr.py",
    "src/transformers/models/granite4_vision/processing_granite4_vision.py",
    "src/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py",
    "src/transformers/models/kimi_k25/processing_kimi_k25.py",
    "src/transformers/models/laguna/modeling_laguna.py",
    "src/transformers/models/mellum/modeling_mellum.py",
    "src/transformers/models/mimo_v2_flash/modeling_mimo_v2_flash.py",
    "src/transformers/models/modernbert/modeling_modernbert.py",
    "src/transformers/models/modernbert_decoder/modeling_modernbert_decoder.py",
    "src/transformers/models/nemotron_h/modeling_nemotron_h.py",
    "src/transformers/models/olmo3/modeling_olmo3.py",
    "src/transformers/models/pp_formulanet/processing_pp_formulanet.py",
    "src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py",
    "src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py",
    "src/transformers/models/qwen3_vl/processing_qwen3_vl.py",
    "src/transformers/models/t5gemma2/modeling_t5gemma2.py",
    "src/transformers/models/video_llama_3/processing_video_llama_3.py",
    "src/transformers/models/zamba2/modeling_zamba2.py",
    "src/transformers/models/zaya/modeling_zaya.py",
    "tests/models/glm_ocr/test_modeling_glm_ocr.py",
]


class PatchPathsTests(unittest.TestCase):
    def test_reads_git_headers(self):
        self.assertEqual(
            patch_paths(PATCH), {"tests/models/glm_ocr/test_modeling_glm_ocr.py"}
        )

    def test_reads_both_sides_of_a_rename(self):
        patch = "diff --git a/old/x.py b/new/x.py\n--- a/old/x.py\n+++ b/new/x.py\n"
        self.assertEqual(patch_paths(patch), {"old/x.py", "new/x.py"})

    def test_falls_back_to_plus_headers(self):
        patch = "--- a/x/y.py\n+++ b/x/y.py\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(patch_paths(patch), {"x/y.py"})

    def test_ignores_dev_null_for_deletions(self):
        patch = "--- a/x/y.py\n+++ /dev/null\n"
        self.assertEqual(patch_paths(patch), set())

    def test_empty_patch_has_no_paths(self):
        self.assertEqual(patch_paths(""), set())


class ScopePathsTests(unittest.TestCase):
    def test_prod_case_drops_all_31_unrelated_files(self):
        keep, dropped = scope_paths(PROD_CHANGED, PATCH)
        self.assertEqual(keep, ["tests/models/glm_ocr/test_modeling_glm_ocr.py"])
        self.assertEqual(len(dropped), 31)

    def test_keeps_regenerated_files_for_the_same_model(self):
        """The coupling that matters: a glm_ocr test patch legitimately
        regenerates src/transformers/models/glm_ocr/. Dropping that would
        reintroduce serge #58 (a PR with a modular file and no modeling file)."""
        changed = [
            "tests/models/glm_ocr/test_modeling_glm_ocr.py",
            "src/transformers/models/glm_ocr/modeling_glm_ocr.py",
            "src/transformers/models/glm_ocr/processing_glm_ocr.py",
            "src/transformers/models/bamba/modeling_bamba.py",
        ]
        keep, dropped = scope_paths(changed, PATCH)
        self.assertIn("src/transformers/models/glm_ocr/modeling_glm_ocr.py", keep)
        self.assertIn("src/transformers/models/glm_ocr/processing_glm_ocr.py", keep)
        self.assertEqual(dropped, ["src/transformers/models/bamba/modeling_bamba.py"])

    def test_keeps_files_in_the_same_directory(self):
        patch = (
            "diff --git a/src/pkg/a.py b/src/pkg/a.py\n--- a/src/pkg/a.py\n"
            "+++ b/src/pkg/a.py\n"
        )
        keep, dropped = scope_paths(
            ["src/pkg/a.py", "src/pkg/b.py", "src/other/c.py"], patch
        )
        self.assertEqual(keep, ["src/pkg/a.py", "src/pkg/b.py"])
        self.assertEqual(dropped, ["src/other/c.py"])

    def test_keeps_a_patched_directorys_subtree(self):
        patch = (
            "diff --git a/src/pkg/__init__.py b/src/pkg/__init__.py\n"
            "--- a/src/pkg/__init__.py\n+++ b/src/pkg/__init__.py\n"
        )
        keep, _ = scope_paths(["src/pkg/__init__.py", "src/pkg/sub/deep.py"], patch)
        self.assertIn("src/pkg/sub/deep.py", keep)

    def test_always_include_globs_survive(self):
        changed = [
            "tests/models/glm_ocr/test_modeling_glm_ocr.py",
            "src/transformers/utils/dummy_pt_objects.py",
            "src/transformers/models/bamba/modeling_bamba.py",
        ]
        keep, dropped = scope_paths(
            changed, PATCH, always_include=["src/transformers/utils/dummy_*.py"]
        )
        self.assertIn("src/transformers/utils/dummy_pt_objects.py", keep)
        self.assertEqual(dropped, ["src/transformers/models/bamba/modeling_bamba.py"])

    def test_a_patched_path_is_always_kept(self):
        """Even a patched file whose directory looks unrelated to the rest."""
        patch = "diff --git a/setup.py b/setup.py\n--- a/setup.py\n+++ b/setup.py\n"
        keep, dropped = scope_paths(["setup.py"], patch)
        self.assertEqual(keep, ["setup.py"])
        self.assertEqual(dropped, [])

    def test_empty_patch_keeps_everything(self):
        """With no patch we cannot tell drift from the fix; committing too much
        beats committing nothing."""
        keep, dropped = scope_paths(PROD_CHANGED, "")
        self.assertEqual(keep, PROD_CHANGED)
        self.assertEqual(dropped, [])

    def test_never_scopes_down_to_nothing(self):
        """If the rule rejected every change it is wrong about this repo —
        keep the change rather than silently publish an empty commit."""
        patch = "diff --git a/a/only.py b/a/only.py\n--- a/a/only.py\n+++ b/a/only.py\n"
        keep, dropped = scope_paths(["z/unrelated.py"], patch)
        self.assertEqual(keep, ["z/unrelated.py"])
        self.assertEqual(dropped, [])

    def test_a_root_patch_scopes_to_the_root(self):
        """Documenting a deliberate generosity: a patch to a root-level file
        adopts root-level normalizer output, because that is how a generated file
        sitting next to its source rides along (serge #58, whose test fixture is
        exactly hello.txt -> extra.txt). Nested files are still dropped, which is
        what the real drift looks like — generated files in the repos serge
        targets live in a model directory, never at the root."""
        patch = "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
        keep, dropped = scope_paths(["README.md", "setup.py", "src/pkg/a.py"], patch)
        self.assertEqual(keep, ["README.md", "setup.py"])
        self.assertEqual(dropped, ["src/pkg/a.py"])


class DescribeDroppedTests(unittest.TestCase):
    def test_lists_and_counts_the_remainder(self):
        out = describe_dropped([f"d/f{i}.py" for i in range(12)], limit=3)
        self.assertIn("d/f0.py", out)
        self.assertIn("+9 more", out)

    def test_short_list_has_no_suffix(self):
        out = describe_dropped(["a/b.py", "c/d.py"], limit=8)
        self.assertEqual(out, "a/b.py, c/d.py")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
