"""CPU-only checks of the generic launch-schedule freeze and verification."""

import unittest

from microbench import long_launch


def point(ctas, steps, ns, trials=None):
    row = {
        "issue_mode": "cp_async",
        "tokens": 2,
        "active_ctas": ctas,
        "stage_bytes": 4096,
        "compute_ns": 0,
        "steps": steps,
        "launch_ns_samples": ns,
        "bw_gbps_samples": [1.0] * len(ns),
        "turnaround_ns_samples": [1.0] * len(ns),
    }
    if trials is not None:
        row["sizing_trials"] = trials
    return row


def output(*rows_per_field):
    return {
        "samples": 3,
        "dram_closed_loop": list(rows_per_field[0]),
        "l2_closed_loop": list(rows_per_field[1]),
    }


class LongLaunchTests(unittest.TestCase):
    def pilot(self):
        trials = [{"steps": 256, "launch_ns_samples": [300e3] * 3}]
        return output(
            [point(16, 1100, [1.3e6] * 3, trials)], [point(32, 4000, [1.26e6] * 3, trials)]
        )

    def test_freeze_writes_one_length_per_point_in_order(self):
        schedule, record = long_launch.freeze("closed-loop", self.pilot())
        self.assertEqual(schedule, "1100\n4000\n")
        self.assertEqual(
            [p["point"]["field"] for p in record["points"]], ["dram_closed_loop", "l2_closed_loop"]
        )
        self.assertNotIn("sizing_trials", record["points"][0]["point"])

    def test_freeze_rejects_a_short_pilot(self):
        pilot = output([point(16, 1024, [1.0e6] * 3)], [point(32, 4000, [1.3e6] * 3)])
        with self.assertRaisesRegex(ValueError, "pilot reached only"):
            long_launch.freeze("closed-loop", pilot)

    def test_verify_checks_points_lengths_and_launches(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        good = output([point(16, 1100, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)])
        self.assertEqual(long_launch.verify(record, good)["shortest_launch_ns"], 1.1e6)
        for bad, message in (
            (output([point(16, 1000, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)]), "froze"),
            (
                output([point(16, 1100, [0.9e6] * 15)], [point(32, 4000, [1.1e6] * 15)]),
                "a launch took",
            ),
            (output([point(64, 1100, [1.2e6] * 15)], [point(32, 4000, [1.1e6] * 15)]), "differs"),
            (output([point(16, 1100, [1.2e6] * 15)], []), "points captured"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                long_launch.verify(record, bad)
