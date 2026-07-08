"""Per-task GPU placement resolution (issue #20)."""

import unittest

from reviewbot.launcher import (
    GpuPlacement,
    GpuPlacementError,
    resolve_gpu_placement,
)

_PROFILES = {
    "single-gpu": {
        "node_selector": "pool=gpu-1x",
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
        "gpu_resource": "nvidia.com/gpu",
        "gpu_count": 1,
        "memory": "64Gi",
    },
    "multi-gpu": {
        "node_selector": {"pool": "gpu-2x"},
        "gpu_count": 2,
    },
}


class ResolveGpuPlacementTests(unittest.TestCase):
    def test_no_gpu_uses_global_defaults(self):
        p = resolve_gpu_placement(
            None,
            _PROFILES,
            default_node_selector={"pool": "cpu"},
            default_memory="6Gi",
        )
        self.assertEqual(p, GpuPlacement(node_selector={"pool": "cpu"}, memory="6Gi"))
        self.assertIsNone(p.gpu_resource)

    def test_single_gpu_profile(self):
        p = resolve_gpu_placement(
            "single-gpu", _PROFILES, default_node_selector=None, default_memory="6Gi"
        )
        self.assertEqual(p.node_selector, {"pool": "gpu-1x"})
        self.assertEqual(p.gpu_resource, "nvidia.com/gpu")
        self.assertEqual(p.gpu_count, 1)
        self.assertEqual(p.memory, "64Gi")
        self.assertEqual(
            p.tolerations, [{"key": "nvidia.com/gpu", "operator": "Exists"}]
        )

    def test_multi_gpu_defaults_resource_and_dict_selector(self):
        # node_selector given as a dict; gpu_resource defaults; memory falls back.
        p = resolve_gpu_placement(
            "multi-gpu", _PROFILES, default_node_selector=None, default_memory="6Gi"
        )
        self.assertEqual(p.node_selector, {"pool": "gpu-2x"})
        self.assertEqual(p.gpu_resource, "nvidia.com/gpu")
        self.assertEqual(p.gpu_count, 2)
        self.assertEqual(p.memory, "6Gi")

    def test_unknown_flavor_raises(self):
        with self.assertRaises(GpuPlacementError):
            resolve_gpu_placement(
                "tpu", _PROFILES, default_node_selector=None, default_memory=None
            )

    def test_gpu_flavor_with_no_profiles_raises(self):
        with self.assertRaises(GpuPlacementError):
            resolve_gpu_placement(
                "single-gpu", None, default_node_selector=None, default_memory=None
            )

    def test_bad_gpu_count_raises(self):
        with self.assertRaises(GpuPlacementError):
            resolve_gpu_placement(
                "x",
                {"x": {"gpu_count": 0}},
                default_node_selector=None,
                default_memory=None,
            )


if __name__ == "__main__":
    unittest.main()
