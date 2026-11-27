"""CPU-only checks of attaching long-launch records to a profile."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from microbench import profile_launch as pl

ROOT = Path(__file__).resolve().parents[2]


def rounds_around(value):
    return [value * f for f in (0.98, 1.0, 1.01, 0.99, 1.02)]


def launch(steps=1000):
    return {
        "length": steps,
        "min_ns": 1.2e6,
        "median_ns": 1.3e6,
        "sm_mhz_before": [2500.0] * 5,
        "sm_mhz_after": [2520.0] * 5,
    }


class ProfileLaunchTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/gb10.json").read_text())

    def test_closed_loop_values_get_launch_and_round_range(self):
        entry = self.profile["measured"]["memory"]["dram_closed_loop"][3]
        point = {
            k: entry[k]
            for k in ("issue_mode", "tokens", "active_ctas", "stage_bytes", "compute_ns")
        }
        bw, turn = entry["delivered_bw_gbps"]["value"], entry["mean_stage_turnaround_ns"]["value"]
        point |= {
            "steps": 1000,
            "launch_ns_samples": [1.3e6] * 5,
            "bw_gbps_samples": rounds_around(bw),
            "turnaround_ns_samples": rounds_around(turn),
            "launch": launch(),
        }
        pl.attach(self.profile, "closed-loop", {"samples": 5, "dram_closed_loop": [point]})
        value = entry["delivered_bw_gbps"]
        self.assertEqual(value["launch"]["length"], 1000)
        self.assertEqual(value["round_median_range"], [bw * 0.98, bw * 1.02])
        self.assertNotIn("per_cycle", value)
        self.assertIn("launch", entry["mean_stage_turnaround_ns"])

    def test_compute_rates_get_per_cycle_values(self):
        table = self.profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"]
        op = next(iter(table))
        entry = table[op][0]
        rate = entry["tflops_per_sm"]["value"]
        point = {
            "mma_warps_per_sm": entry["mma_warps_per_sm"],
            "steps": 9000,
            "launch_ns_samples": [1.3e6] * 5,
            "tflops_per_sm_samples": [rate] * 5,
            "launch": launch(9000),
        }
        fp32 = self.profile["measured"]["compute"]["fp32_fma_tflops"]["value"]
        merged = {
            "mma": {op: [point]},
            "fp32_fma": {
                "warps_per_sm": 16,
                "steps": 9e4,
                "launch_ns_samples": [1.3e6] * 5,
                "launch": launch(),
            },
            "fp32_fma_tflops_samples": [fp32] * 5,
        }
        pl.attach(self.profile, "mma", merged)
        self.assertAlmostEqual(entry["tflops_per_sm"]["per_cycle"]["value"], rate / 2510.0)
        self.assertEqual(entry["tflops_per_sm"]["per_cycle"]["unit"], "TFLOP/s per MHz")
        self.assertAlmostEqual(
            self.profile["measured"]["compute"]["fp32_fma_tflops"]["per_cycle"]["value"],
            fp32 / 2510.0,
        )

    def test_write_points_are_scalars_and_capacity_sweep_has_no_value(self):
        write = self.profile["measured"]["memory"]["dram_bw_write_gbps"]["value"]
        point = {
            "target_sms": 48,
            "steps": 256,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [write] * 5,
            "launch": launch(256),
        }
        pl.attach(self.profile, "dram-bw", {"write_full": [point]})
        self.assertEqual(
            self.profile["measured"]["memory"]["dram_bw_write_gbps"]["launch"]["length"], 256
        )
        sweep = {
            "working_set_bytes": 1 << 24,
            "steps": 9,
            "launch_ns_samples": [1.3e6] * 5,
            "launch": launch(9),
        }
        pl.attach(self.profile, "l2", {"capacity_sweep": [sweep]})

    @patch.dict(
        pl.long_launch.DESIGN_FIELDS["l2"], {"read_curve_cp_async": ("target_sms",)}
    )  # registered with the l2 schedule on its own branch
    def test_l2_read_scalar_follows_whichever_curve_supplied_it(self):
        memory = self.profile["measured"]["memory"]
        memory.setdefault(
            "l2_bw_by_active_sms_cp_async", copy.deepcopy(memory["l2_bw_by_active_sms"])
        )
        mlp = next(e for e in memory["l2_bw_by_bytes_in_flight"])
        sources = (
            ("l2", "read_curve", memory["l2_bw_by_active_sms"][-1], "target_sms"),
            ("l2", "read_curve_cp_async", memory["l2_bw_by_active_sms_cp_async"][-1], "target_sms"),
            ("mlp", "l2_bw_by_bytes_in_flight", mlp, None),
        )
        for mode, field, entry, sms_key in sources:
            memory["l2_bw_read_gbps"] = entry["bw_gbps"]
            entry["bw_gbps"].pop("launch", None)
            ident = (
                {sms_key: entry["active_sms"]}
                if sms_key
                else {
                    k: entry[k]
                    for k in ("issue_mode", "bytes_in_flight", "active_ctas", "bytes_per_cta")
                }
            )
            point = ident | {
                "steps": 77,
                "launch_ns_samples": [1.3e6] * 5,
                "gbps_samples": [entry["bw_gbps"]["value"]] * 5,
                "launch": launch(77),
            }
            pl.attach(self.profile, mode, {field: [point]})
            self.assertEqual(memory["l2_bw_read_gbps"]["launch"]["length"], 77, field)

    def test_l2_write_view_must_equal_the_point(self):
        write = self.profile["measured"]["memory"]["l2_bw_write_gbps"]["value"]
        point = {
            "working_set_bytes": 1 << 23,
            "steps": 64,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [write] * 5,
            "launch": launch(64),
        }
        pl.attach(self.profile, "l2", {"write_full": point, "write_full_gbps_samples": [write] * 5})
        self.assertIn("launch", self.profile["measured"]["memory"]["l2_bw_write_gbps"])
        with self.assertRaisesRegex(ValueError, "differs"):
            pl.attach(
                self.profile,
                "l2",
                {"write_full": point, "write_full_gbps_samples": [write] * 4 + [1.0]},
            )

    def test_overridden_points_are_annotated_from_their_last_source(self):
        entry = self.profile["measured"]["pipeline"]["sync_stage_loop"][0]
        names = (
            "mem_level",
            "transfer_mode",
            "ctas_per_sm",
            "num_warps",
            "stage_bytes",
            "mma_chain_len",
        )
        values = {
            m: entry[m]["value"]
            for m in ("delivered_bw_gbps", "mean_stage_turnaround_ns", "achieved_mma_tflops_per_sm")
        }

        def point(scale, steps):
            return {k: entry[k] for k in names} | {
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 5,
                "bw_gbps_samples": [values["delivered_bw_gbps"] * scale] * 5,
                "turnaround_ns_samples": [values["mean_stage_turnaround_ns"] * scale] * 5,
                "mma_tflops_per_sm_samples": [values["achieved_mma_tflops_per_sm"] * scale] * 5,
                "launch": launch(steps),
            }

        later = "sync-loop-later"
        with (
            patch.dict(
                pl.long_launch.DESIGN_FIELDS, {later: pl.long_launch.DESIGN_FIELDS["sync-loop"]}
            ),
            patch.dict(
                pl.PROFILE_PATH, {(later, "sync_stage_loop"): ("pipeline", "sync_stage_loop")}
            ),
            patch.dict(pl.METRIC, {later: pl.STAGE_LOOP}),
            patch.object(pl, "OVERRIDE_ORDER", ("sync-loop", later)),
        ):
            merged = {
                "sync-loop": {"sync_stage_loop": [point(1.2, 128)]},
                later: {"sync_stage_loop": [point(1.0, 700)]},
            }
            pl.attach_all(self.profile, merged)
        self.assertEqual(entry["delivered_bw_gbps"]["launch"]["length"], 700)

    def test_lists_that_are_not_values_are_left_out(self):
        entry = self.profile["measured"]["compute"]["mma_chain_latency"][0]
        point = {
            k: entry[k]
            for k in ("variant", "num_warps", "tile_bytes", "ldmatrix_x4_per_warp", "mma_chain_len")
        }
        point |= {
            "steps": 900,
            "launch_ns_samples": [1.3e6] * 5,
            "stage_time_ns_samples": [entry["stage_time_ns"]["value"]] * 5,
            "window_ns_samples": [5.0] * 5,
            "launch": launch(900),
        }
        with self.assertRaisesRegex(ValueError, "NOT_A_VALUE"):
            pl.attach(copy.deepcopy(self.profile), "mma-chain", {"mma_chain_latency": [point]})
        with patch.object(pl, "NOT_A_VALUE", {("mma-chain", "window_ns_samples")}):
            pl.attach(self.profile, "mma-chain", {"mma_chain_latency": [point]})
        self.assertIn("launch", entry["stage_time_ns"])

    def test_exception_flags_the_entry_and_covers_the_rounds(self):
        entry = self.profile["measured"]["memory"]["dram_bw_by_active_sms"][0]
        bw = entry["bw_gbps"]["value"]
        point = {
            "target_sms": entry["active_sms"],
            "steps": 81,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [bw * 0.7, bw, bw, bw * 1.01, bw],
            "launch": launch(81),
            "exceptions": ["low_in_flight_unstable"],
        }
        pl.attach(self.profile, "dram-bw", {"read_curve": [point]})
        self.assertTrue(entry["low_in_flight_unstable"])
        self.assertAlmostEqual(entry["bw_gbps"]["rel_uncertainty"], 0.3, places=3)

    def test_mismatches_are_errors(self):
        entry = self.profile["measured"]["memory"]["dram_bw_by_active_sms"][0]
        point = {
            "target_sms": 999,
            "steps": 81,
            "launch_ns_samples": [1.3e6] * 5,
            "gbps_samples": [1.0] * 5,
        }
        with self.assertRaisesRegex(ValueError, "0 profile entries"):
            pl.attach(copy.deepcopy(self.profile), "dram-bw", {"read_curve": [point]})
        point["target_sms"] = entry["active_sms"]
        point["launch"] = launch(81)
        with self.assertRaisesRegex(ValueError, "not the median"):
            pl.attach(self.profile, "dram-bw", {"read_curve": [point]})
