"""CPU-only checks of the launch-duration audit of the reference profiles."""

import json
import unittest
from pathlib import Path

from microbench import launch_audit

ROOT = Path(__file__).resolve().parents[2]


class LaunchAuditTests(unittest.TestCase):
    def summary(self, name):
        return launch_audit.summary(json.loads((ROOT / "profiles" / name).read_text()))

    def test_gb10_short_launch_counts(self):
        rows = self.summary("gb10.json")
        self.assertEqual(rows["l2_closed_loop"]["under_1_ms"], 360)
        self.assertEqual(rows["dram_bw_by_bytes_in_flight"]["under_1_ms"], 38)
        self.assertEqual(rows["dram_bw_by_active_sms"]["under_1_ms"], 0)

    def test_mma_iterations_follow_the_benchmark(self):
        profile = json.loads((ROOT / "profiles" / "gb10.json").read_text())
        point = profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"][
            "mma.sync:f16:f16:f32:16x8x16"
        ][0]
        ms = launch_audit.durations(profile)[
            "mma_tflops_per_sm_by_warps[mma.sync:f16:f16:f32:16x8x16]"
        ][0]
        warps = point["mma_warps_per_sm"]
        expected = (
            warps
            * max(4000, 32000 // min(warps, 8))
            * 4
            * 4096
            / (point["tflops_per_sm"]["value"] * 1e9)
        )
        self.assertAlmostEqual(ms, expected)
