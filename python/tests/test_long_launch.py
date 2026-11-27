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


def output(*rows_per_field, samples=15):
    return {
        "samples": samples,
        "dram_closed_loop": list(rows_per_field[0]),
        "l2_closed_loop": list(rows_per_field[1]),
    }


class LongLaunchTests(unittest.TestCase):
    def pilot(self):
        trials = [{"steps": 256, "launch_ns_samples": [300e3] * 3}]
        return output(
            [point(16, 1100, [1.3e6] * 3, trials)],
            [point(32, 4000, [1.26e6] * 3, trials)],
            samples=3,
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

    def test_verify_rejects_incomplete_or_invalid_samples(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        for samples in ([1.2e6] * 14, [1.2e6] * 14 + [float("nan")], [1.2e6] * 14 + [0.0]):
            bad = output([point(16, 1100, samples)], [point(32, 4000, [1.1e6] * 15)])
            with self.assertRaisesRegex(ValueError, "CUDA event samples"):
                long_launch.verify(record, bad)

    def test_freeze_rejects_an_empty_sample_list(self):
        pilot = output([point(16, 1100, [])], [point(32, 4000, [1.3e6] * 3)], samples=3)
        with self.assertRaisesRegex(ValueError, "no CUDA event samples"):
            long_launch.freeze("closed-loop", pilot)

    def test_identity_ignores_diagnostics_but_checks_them(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        row = point(16, 1100, [1.2e6] * 15)
        row["observed_sms"], row["target_sms"] = 16, 16
        good = output([row], [point(32, 4000, [1.1e6] * 15)])
        long_launch.verify(record, good)
        row["observed_sms"] = 15
        with self.assertRaisesRegex(ValueError, "SMs ran"):
            long_launch.verify(record, good)

    def test_schedule_file_must_match_the_freeze(self):
        schedule, record = long_launch.freeze("closed-loop", self.pilot())
        long_launch.check_schedule(record, schedule.encode())
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.check_schedule(record, b"1100\n4001\n")

    def test_unregistered_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no design fields"):
            long_launch.freeze("sync-loop", self.pilot())

    def test_mma_tables_and_fp32_fma_are_keyed_by_their_own_fields(self):
        def timed(steps):
            return {"steps": steps, "launch_ns_samples": [1.3e6] * 3}

        pilot = {
            "samples": 3,
            "mma": {
                "f16": [{"mma_warps_per_sm": 1, **timed(9000)}],
                "bf16": [{"mma_warps_per_sm": 1, **timed(9100)}],
            },
            "fp32_fma": {"warps_per_sm": 16, **timed(52000)},
            "fp32_fma_tflops_samples": [30.0] * 3,
        }
        schedule, record = long_launch.freeze("mma", pilot)
        self.assertEqual(schedule, "9000\n9100\n52000\n")
        self.assertEqual(
            [p["point"] for p in record["points"]],
            [
                {"field": "mma/f16", "mma_warps_per_sm": 1},
                {"field": "mma/bf16", "mma_warps_per_sm": 1},
                {"field": "fp32_fma", "warps_per_sm": 16},
            ],
        )

    def test_gmem_issue_points_are_keyed_by_grid_width_and_warps(self):
        row = {
            "active_sms": 1,
            "load_width_bytes": 4,
            "num_warps": 8,
            "steps": 9000,
            "launch_ns_samples": [1.3e6] * 3,
        }
        _, record = long_launch.freeze(
            "gmem-issue", {"samples": 3, "gmem_load_issue_per_sm": [row]}
        )
        self.assertEqual(
            record["points"][0]["point"],
            {
                "field": "gmem_load_issue_per_sm",
                "active_sms": 1,
                "load_width_bytes": 4,
                "num_warps": 8,
            },
        )

    def test_read_sweeps_bind_their_work(self):
        def row(steps, transfer):
            return {
                "issue_mode": "cp_async",
                "bytes_in_flight": 4096,
                "active_ctas": 1,
                "bytes_per_cta": 4096,
                "steps": steps,
                "transfer_bytes": transfer,
                "source_buffer_bytes": 1 << 30,
                "launch_ns_samples": [1.3e6] * 3,
            }

        _, record = long_launch.freeze(
            "mlp", {"samples": 3, "dram_bw_by_bytes_in_flight": [row(4096, 1 << 24)]}
        )
        self.assertEqual(record["points"][0]["transfer_bytes"], 1 << 24)
        capture = {"samples": 3, "dram_bw_by_bytes_in_flight": [row(4096, 1 << 25)]}
        with self.assertRaisesRegex(ValueError, "work differs"):
            long_launch.verify(record, capture)

    def test_unregistered_field_is_refused(self):
        pilot = {"samples": 3, "other": [{"steps": 1, "launch_ns_samples": [1.3e6] * 3}]}
        with self.assertRaisesRegex(ValueError, "other: no design fields"):
            long_launch.freeze("closed-loop", pilot)

    def test_l2_merge_points_are_keyed_by_sharing_and_stagger(self):
        def merge(stagger, steps):
            return {
                "active_ctas": 48,
                "sharing": 4,
                "bytes_per_cta": 8192,
                "stagger_stages": stagger,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
                "turnaround_ns_samples": [900.0] * 3,
            }

        pilot = {"samples": 3, "l2_merge_by_sharing": [merge(0, 2000), merge(16, 2100)]}
        schedule, record = long_launch.freeze("l2-merge", pilot)
        self.assertEqual(schedule, "2000\n2100\n")
        self.assertEqual([p["point"]["stagger_stages"] for p in record["points"]], [0, 16])
        swapped = {"samples": 3, "l2_merge_by_sharing": [merge(16, 2000), merge(0, 2100)]}
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(record, swapped)
