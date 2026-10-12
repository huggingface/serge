"""Tests for the expectation-shape rule (reviewbot/expectation_keys.py).

Each case is cut down from a real serge PR on huggingface/transformers; the
number in the test name is the PR."""

import textwrap
import unittest

from reviewbot.expectation_keys import expectation_key_problems


def _src(body: str) -> str:
    return textwrap.dedent(body).lstrip()


BASE = _src(
    """
    def test_logits(self):
        expectations = Expectations(
            {
                (None, None): [0.9989, 0.1953, -0.4382],
                ("xpu", 5): [0.9986, 0.1953, -0.4381],
            }
        )
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    """
)


def problems(old: str, new: str, eo: bool = True) -> list[str]:
    return expectation_key_problems("test_x.py", old, new, expectation_only=eo)


class AcceptedShapes(unittest.TestCase):
    def test_48838_adds_the_runner_key(self):
        new = BASE.replace(
            '("xpu", 5)',
            '("cuda", 8): [0.9987, 0.1953, -0.4381],\n                ("xpu", 5)',
        )
        self.assertEqual(problems(BASE, new), [])

    def test_49040_adds_a_major_minor_runner_key(self):
        new = BASE.replace(
            '("xpu", 5)',
            '("cuda", (8, 6)): [0.9987, 0.1953, -0.4381],\n                ("xpu", 5)',
        )
        self.assertEqual(problems(BASE, new), [])

    def test_48882_demotes_then_updates_the_runner_key(self):
        old = _src(
            """
            def test_bf16(self):
                e = Expectations({("cuda", 8): [-0.3145, -5.6562], ("xpu", 3): [-0.31, -5.65]})
            """
        )
        new = _src(
            """
            def test_bf16(self):
                e = Expectations({
                    (None, None): [-0.3145, -5.6562],
                    ("cuda", 8): [-0.3184, -5.6875],
                    ("xpu", 3): [-0.31, -5.65],
                })
            """
        )
        self.assertEqual(problems(old, new), [])

    def test_48198_demotes_cuda_none_to_generic(self):
        old = _src(
            """
            def test_t(self):
                e = Expectations({("cuda", None): [-6.7329, -16.9590]})
            """
        )
        new = _src(
            """
            def test_t(self):
                e = Expectations({(None, None): [-6.7329, -16.9590], ("cuda", 8): [-6.7667, -16.9915]})
            """
        )
        self.assertEqual(problems(old, new), [])

    def test_converting_a_plain_literal(self):
        old = _src(
            """
            from transformers.testing_utils import require_torch

            def test_t(self):
                expected_slice = torch.tensor([-0.2962, 0.4487])
            """
        )
        new = _src(
            """
            from transformers.testing_utils import Expectations, require_torch

            def test_t(self):
                expectations = Expectations({(None, None): [-0.2962, 0.4487], ("cuda", 8): [-0.2970, 0.4490]})
                expected_slice = torch.tensor(expectations.get_expectation())
            """
        )
        self.assertEqual(problems(old, new), [])

    def test_49507_input_length_is_not_a_hardware_value(self):
        old = "def test_t(self):\n    assert inputs.input_ids.shape == (2, 421)\n"
        new = "def test_t(self):\n    assert inputs.input_ids.shape == (2, 422)\n"
        self.assertEqual(problems(old, new), [])

    def test_an_output_count_is_still_held_to_the_rule(self):
        old = 'def test_t(self):\n    self.assertEqual(len(results["scores"]), 5)\n'
        new = 'def test_t(self):\n    self.assertEqual(len(results["scores"]), 3)\n'
        self.assertIn("outside any Expectations block", problems(old, new)[0])

    def test_new_text_wording_on_the_runner_key_is_fine(self):
        old = 'def test_t(self):\n    e = Expectations({(None, None): "a cat on a mat"})\n'
        new = 'def test_t(self):\n    e = Expectations({(None, None): "a cat on a mat", ("cuda", 8): "a cat on the mat"})\n'
        self.assertEqual(problems(old, new), [])

    def test_existing_stutter_is_not_blamed_on_the_patch(self):
        old = 'def test_t(self):\n    e = Expectations({("cuda", 8): "very_long very_long tail"})\n'
        new = 'def test_t(self):\n    e = Expectations({("cuda", 8): "very_long very_long tale"})\n'
        self.assertEqual(problems(old, new), [])

    def test_47697_test_code_fix_is_not_this_rules_business(self):
        old = 'def test_t(self):\n    config.rope_parameters["type"] = "default"\n'
        new = 'def test_t(self):\n    config.rope_parameters["rope_type"] = "default"\n'
        self.assertEqual(problems(old, new), [])


class RejectedShapes(unittest.TestCase):
    def test_47616_overwrites_the_generic_key(self):
        new = BASE.replace("(None, None): [0.9989", "(None, None): [0.9950")
        self.assertIn("(None, None)", problems(BASE, new)[0])

    def test_48580_overwrites_cuda_none(self):
        old = 'def test_t(self):\n    e = Expectations({("cuda", None): ["a Pystylus cat"]})\n'
        new = 'def test_t(self):\n    e = Expectations({("cuda", None): ["a snow leopard cat"]})\n'
        self.assertIn('("cuda", None)', problems(old, new)[0])

    def test_48955_rewrites_another_vendor(self):
        new = BASE.replace('("xpu", 5): [0.9986', '("xpu", 5): [0.9000')
        self.assertIn('"xpu"', problems(BASE, new)[0])

    def test_49407_adds_a_broad_key(self):
        new = BASE.replace(
            '("xpu", 5)',
            '("cuda", None): [1.0, 0.2, -0.4],\n                ("xpu", 5)',
        )
        out = problems(BASE, new)
        self.assertEqual(len(out), 1)
        self.assertIn('added a ("cuda", None) entry', out[0])

    def test_new_generic_value_that_is_not_a_demotion(self):
        old = 'def test_t(self):\n    e = Expectations({("xpu", 3): [1.0]})\n'
        new = 'def test_t(self):\n    e = Expectations({(None, None): [2.0], ("xpu", 3): [1.0]})\n'
        self.assertIn("(None, None)", problems(old, new)[0])

    def test_47750_runner_key_moving_far_past_drift(self):
        old = 'def test_t(self):\n    e = Expectations({(None, None): [-1.4192, -1.9158], ("cuda", 8): [-1.4194, -1.9161]})\n'
        new = 'def test_t(self):\n    e = Expectations({(None, None): [-1.4192, -1.9158], ("cuda", 8): [-1.0951, -1.9902]})\n'
        out = problems(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("regression", out[0])

    def test_49068_small_bf16_drift_is_fine(self):
        old = 'def test_t(self):\n    e = Expectations({("cuda", (8, 6)): [0.0608, -0.0933, 0.1348]})\n'
        new = 'def test_t(self):\n    e = Expectations({("cuda", (8, 6)): [0.0588, -0.0928, 0.1357]})\n'
        self.assertEqual(problems(old, new), [])

    def test_49143_loosened_tolerance_even_with_code_changes(self):
        new = BASE.replace("rtol=1e-4, atol=1e-4", "rtol=1e-2, atol=1e-2")
        out = problems(BASE, new, eo=False)
        self.assertEqual(len(out), 1)
        self.assertIn("atol 0.0001 -> 0.01", out[0])

    def test_47840_plain_literal_overwritten_in_place(self):
        old = (
            "def test_t(self):\n    expected_slice = torch.tensor([-0.2962, 0.4487])\n"
        )
        new = (
            "def test_t(self):\n    expected_slice = torch.tensor([-0.2782, 0.4466])\n"
        )
        self.assertIn("outside any Expectations block", problems(old, new)[0])

    def test_plain_literal_rule_only_for_expectation_only_patches(self):
        old = (
            "def test_t(self):\n    expected_slice = torch.tensor([-0.2962, 0.4487])\n"
        )
        new = (
            "def test_t(self):\n    expected_slice = torch.tensor([-0.2782, 0.4466])\n"
        )
        self.assertEqual(problems(old, new, eo=False), [])

    def test_49243_converted_literal_is_drift_checked(self):
        old = "def test_t(self):\n    expected_slice = torch.tensor([-1.4192, -1.9158, -0.9702])\n"
        new = (
            "def test_t(self):\n    expected_slice = torch.tensor(Expectations({(None, None): "
            '[-1.4192, -1.9158, -0.9702], ("cuda", 8): [-1.1939, -1.4547, -0.1076]})'
            ".get_expectation())\n"
        )
        out = problems(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("45%", out[0])

    def test_47698_a_different_count_is_not_drift(self):
        old = "def test_t(self):\n    e = Expectations({(None, None): [0.8732, 0.8563, 0.8554, 0.6080, 0.5895]})\n"
        new = (
            "def test_t(self):\n    e = Expectations({(None, None): [0.8732, 0.8563, 0.8554, 0.6080, 0.5895], "
            '("cuda", 8): [0.5987, 0.5883, 0.3454]})\n'
        )
        out = problems(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("3 numbers where it had 5", out[0])

    def test_47983_stuttering_generation(self):
        old = (
            'def test_t(self):\n    e = Expectations({(None, None): "def hello_world():\\n\\treturn 1", '
            '("cuda", 8): "def hello_world():\\n\\treturn 1"})\n'
        )
        new = old.replace(
            '("cuda", 8): "def hello_world():',
            '("cuda", 8): "def hello_world(): hello_world():',
        )
        out = problems(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("repeats `hello_world():`", out[0])

    def test_49244_control_character_in_text(self):
        old = 'def test_t(self):\n    e = Expectations({(None, None): "A blue bus on a road."})\n'
        new = (
            'def test_t(self):\n    e = Expectations({(None, None): "A blue bus on a road.", '
            '("cuda", 8): "\\x04 A bus on a road."})\n'
        )
        out = problems(old, new)
        self.assertEqual(len(out), 1)
        self.assertIn("control character", out[0])

    def test_unparseable_source_is_not_judged(self):
        self.assertEqual(problems("def broken(:\n", "def broken(:\n  x\n"), [])


if __name__ == "__main__":
    unittest.main()
