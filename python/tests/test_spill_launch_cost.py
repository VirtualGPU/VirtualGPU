"""CPU-only checks of how collect_profile takes the launch cost out of spill-latency."""

import unittest

from microbench import collect_profile as cp
from microbench import long_capture


def point(rc, steps, per_step_ns, cost_ns, frame=4096, mode="serial_chain"):
    total = steps * per_step_ns + cost_ns
    return {
        "dependency_mode": mode,
        "frame_bytes_per_thread": frame,
        "reload_count": rc,
        "steps": steps,
        "iter_time_ns_samples": [total / steps] * 3,
    }


class SpillLaunchCostTests(unittest.TestCase):
    def test_cost_cancels_between_different_lengths(self):
        cost = 126000.0
        raw = {
            "reload_latency_by_count": [point(0, 4096, 15.0, cost), point(32, 128, 900.0, cost)],
            "launch_cost_by_frame": [
                {
                    "dependency_mode": "serial_chain",
                    "frame_bytes_per_thread": 4096,
                    "launch_cost_ns_samples": [cost - 1000, cost, cost + 1000],
                }
            ],
        }
        zero, reloads = cp.spill_step_samples(raw)
        for value in zero:
            self.assertAlmostEqual(value, 15.0)
        for value in reloads:
            self.assertAlmostEqual(value, 900.0)

    def test_without_launch_cost_samples_are_kept(self):
        raw = {"reload_latency_by_count": [point(8, 512, 40.0, 5000.0)]}
        self.assertEqual(
            cp.spill_step_samples(raw), [raw["reload_latency_by_count"][0]["iter_time_ns_samples"]]
        )

    def test_cost_is_matched_by_frame_and_mode(self):
        raw = {
            "reload_latency_by_count": [
                point(0, 256, 10.0, 2560.0, frame=128, mode="batch_before_dep")
            ],
            "launch_cost_by_frame": [
                {
                    "dependency_mode": "serial_chain",
                    "frame_bytes_per_thread": 128,
                    "launch_cost_ns_samples": [99999.0],
                },
                {
                    "dependency_mode": "batch_before_dep",
                    "frame_bytes_per_thread": 128,
                    "launch_cost_ns_samples": [2560.0],
                },
            ],
        }
        self.assertAlmostEqual(cp.spill_step_samples(raw)[0][0], 10.0)

    def test_merged_rounds_use_round_medians(self):
        def round_raw(cost):
            return {
                "reload_latency_by_count": [
                    point(0, 4096, 15.0, cost),
                    point(32, 128, 900.0, cost),
                ],
                "launch_cost_by_frame": [
                    {
                        "dependency_mode": "serial_chain",
                        "frame_bytes_per_thread": 4096,
                        "steps": 40,
                        "launch_ns_samples": [40 * cost] * 3,
                        "launch_cost_ns_samples": [cost - 500, cost, cost + 500],
                    }
                ],
            }

        merged = long_capture.merge([round_raw(c) for c in (125000.0, 126000.0, 127000.0)])
        costs = merged["launch_cost_by_frame"][0]["launch_cost_ns_samples"]
        self.assertEqual(costs, [125000.0, 126000.0, 127000.0])
        zero, reloads = cp.spill_step_samples(merged)
        # Each round's point carries its own cost; the shared median cost leaves the
        # rounds off by (round cost - median cost) / steps.
        self.assertAlmostEqual(zero[1], 15.0)
        self.assertAlmostEqual(reloads[1], 900.0)
        self.assertAlmostEqual(reloads[0], 900.0 - 1000.0 / 128)


if __name__ == "__main__":
    unittest.main()
