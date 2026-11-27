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
    def test_l2_capacity_freezes_passes_and_working_set(self):
        row = {"working_set_bytes": 24 * 1024**2, "steps": 32, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "capacity_sweep": [row]}
        schedule, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(schedule, "32\n")
        long_launch.verify(freeze, raw)
        row["steps"] = 31
        with self.assertRaisesRegex(ValueError, "froze"):
            long_launch.verify(freeze, raw)

    def test_l2_write_legacy_samples_do_not_duplicate_the_scheduled_point(self):
        bandwidth = [100.0] * 3
        raw = {
            "samples": 3,
            "write_full": {
                "working_set_bytes": 6 * 1024**2,
                "steps": 128,
                "launch_ns_samples": [1.3e6] * 3,
                "gbps_samples": bandwidth,
            },
            "write_full_gbps_samples": bandwidth,
        }
        schedule, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(schedule, "128\n")
        long_launch.verify(freeze, raw)
        raw["write_full"]["working_set_bytes"] *= 2
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(freeze, raw)

    def test_l2_read_instructions_are_separate_frozen_curves(self):
        row = {"target_sms": 48, "steps": 2048, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "read_curve": [row], "read_curve_cp_async": [dict(row)]}
        _, freeze = long_launch.freeze("l2", raw)
        self.assertEqual(
            [entry["point"] for entry in freeze["points"]],
            [
                {"field": "read_curve", "target_sms": 48},
                {"field": "read_curve_cp_async", "target_sms": 48},
            ],
        )
        long_launch.verify(freeze, raw)
        raw["read_curve_cp_async"][0]["target_sms"] = 24
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(freeze, raw)

    def test_dram_write_has_its_own_frozen_point(self):
        row = {"target_sms": 48, "steps": 384, "launch_ns_samples": [1.3e6] * 3}
        raw = {"samples": 3, "read_curve": [row], "write_full": [dict(row, steps=512)]}
        schedule, freeze = long_launch.freeze("dram-bw", raw)
        self.assertEqual(schedule, "384\n512\n")
        self.assertEqual(freeze["points"][1]["point"], {"field": "write_full", "target_sms": 48})
        long_launch.verify(freeze, raw)
        with self.assertRaisesRegex(ValueError, "points captured"):
            long_launch.verify(freeze, {"samples": 3, "read_curve": [row]})

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

    def test_freeze_sizes_in_kernel_probes_by_their_window(self):
        row = point(16, 1100, [1.3e6] * 3)
        row["window_ns_samples"] = [1.27e6, 0.8e6, 1.27e6]
        pilot = output([row], [point(32, 4000, [1.3e6] * 3)], samples=3)
        with self.assertRaisesRegex(ValueError, "pilot reached only 800000"):
            long_launch.freeze("closed-loop", pilot)
        row["window_ns_samples"][1] = 1.26e6
        long_launch.freeze("closed-loop", pilot)

    def test_verify_checks_in_kernel_windows(self):
        _, record = long_launch.freeze("closed-loop", self.pilot())
        row = point(16, 1100, [1.2e6] * 15)
        row["window_ns_samples"] = [1.19e6] * 15
        good = output([row], [point(32, 4000, [1.1e6] * 15)])
        long_launch.verify(record, good)
        row["window_ns_samples"][3] = 0.99e6
        with self.assertRaisesRegex(ValueError, "in-kernel window"):
            long_launch.verify(record, good)

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

    def test_closed_loop_mma_freezes_its_mma_iterations_ahead_of_each_length(self):
        def row(iters, steps, samples=3):
            return {
                "ctas_per_sm": 1,
                "active_ctas": 48,
                "tokens": 1,
                "num_warps": 4,
                "stage_bytes": 8192,
                "ldmatrix_bytes_per_stage": 0,
                "mma_iters_per_stage": iters,
                "solo_compute_ns": iters * 27.278,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * samples,
            }

        pilot = {"samples": 3, "dram_closed_loop_mma": [row(0, 1000), row(37, 400)]}
        schedule, record = long_launch.freeze("closed-loop-mma", pilot)
        self.assertEqual(schedule, "0\n1000\n37\n400\n")
        long_launch.check_schedule(record, schedule.encode())
        capture = {"samples": 15, "dram_closed_loop_mma": [row(0, 1000, 15), row(37, 400, 15)]}
        capture["dram_closed_loop_mma"][1]["solo_compute_ns"] = (
            37 * 27.9
        )  # recalibrated, not a design field
        long_launch.verify(record, capture)
        capture["dram_closed_loop_mma"][1]["mma_iters_per_stage"] = 36
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(record, capture)
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.check_schedule(record, b"0\n1000\n36\n400\n")

    def test_frozen_parameters_are_design_fields(self):
        for mode, fields in long_launch.FROZEN_PARAMETERS.items():
            for field, names in fields.items():
                self.assertTrue(
                    set(names) <= set(long_launch.DESIGN_FIELDS[mode][field]), (mode, field)
                )

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

    def test_sync_loop_multicta_is_keyed_by_level_mode_ctas_bytes_and_chain(self):
        def stage(level, cps, steps):
            return {
                "mem_level": level,
                "transfer_mode": "cp_async_s1",
                "ctas_per_sm": cps,
                "active_ctas": 48 * cps,
                "num_warps": 4,
                "stage_bytes": 4096,
                "mma_chain_len": 8,
                "solo_compute_ns": 120.0,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
            }

        occupancy = {
            "transfer_mode": "cp_async_s1",
            "stage_bytes": 4096,
            "max_active_ctas_per_sm": 8,
        }
        pilot = {
            "samples": 3,
            "occupancy_check": [occupancy],
            "sync_stage_loop": [stage("dram", 1, 900), stage("l2", 6, 5000)],
        }
        schedule, record = long_launch.freeze("sync-loop-multicta", pilot)
        self.assertEqual(schedule, "900\n5000\n")
        self.assertEqual(
            record["points"][1]["point"],
            {
                "field": "sync_stage_loop",
                "mem_level": "l2",
                "transfer_mode": "cp_async_s1",
                "ctas_per_sm": 6,
                "num_warps": 4,
                "stage_bytes": 4096,
                "mma_chain_len": 8,
            },
        )

    def test_sync_loop_ablation_is_keyed_by_variant(self):
        def stage(variant, steps):
            return {
                "variant": variant,
                "ctas_per_sm": 1,
                "active_ctas": 48,
                "num_warps": 8,
                "stage_bytes": 16384,
                "mma_chain_len": 64,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
            }

        pilot = {
            "samples": 3,
            "sync_stage_loop_ablation": [stage("cp_async_s1", 700), stage("ld_st_sync", 760)],
        }
        schedule, record = long_launch.freeze("sync-loop-ablation", pilot)
        self.assertEqual(schedule, "700\n760\n")
        self.assertEqual(
            [p["point"]["variant"] for p in record["points"]], ["cp_async_s1", "ld_st_sync"]
        )
        swapped = {
            "samples": 3,
            "sync_stage_loop_ablation": [stage("ld_st_sync", 700), stage("cp_async_s1", 760)],
        }
        with self.assertRaisesRegex(ValueError, "differs from the freeze"):
            long_launch.verify(record, swapped)

    def test_sync_loop_reconcile_freezes_the_pilot_length_not_its_own_probe(self):
        row = {
            "mem_level": "dram",
            "transfer_mode": "ld_st_sync",
            "ctas_per_sm": 2,
            "active_ctas": 96,
            "num_warps": 8,
            "stage_bytes": 24576,
            "mma_chain_len": 32,
            "solo_compute_ns": 300.0,
            "steps": 1600,
            "launch_ns_samples": [1.3e6] * 3,
        }
        schedule, record = long_launch.freeze(
            "sync-loop-reconcile", {"samples": 3, "sync_stage_loop": [row]}
        )
        self.assertEqual(schedule, "1600\n")
        self.assertNotIn("steps", record["points"][0]["point"])
        self.assertEqual(record["points"][0]["point"]["stage_bytes"], 24576)

    def test_mma_softmax_chain_freezes_calibrations_before_the_chain(self):
        def chain(cps, m_half, k_sf, steps):
            return {
                "ctas_per_sm": cps,
                "num_warps": 4,
                "mma_per_half": m_half,
                "softmax_ops": k_sf,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
                "iter_time_ns_samples": [300.0] * 3,
            }

        solo = chain(1, 8, 0, 4000)
        pilot = {
            "samples": 3,
            "solo_calibration": [solo],
            "mma_softmax_chain": [
                chain(2, 8, 64, 2500) | {"solo_mma_ns": 200.0, "solo_softmax_ns": 90.0}
            ],
        }
        schedule, record = long_launch.freeze("mma-softmax-chain", pilot)
        self.assertEqual(schedule, "4000\n2500\n")
        self.assertEqual(
            [p["point"] for p in record["points"]],
            [
                {
                    "field": "solo_calibration",
                    "ctas_per_sm": 1,
                    "num_warps": 4,
                    "mma_per_half": 8,
                    "softmax_ops": 0,
                },
                {
                    "field": "mma_softmax_chain",
                    "ctas_per_sm": 2,
                    "num_warps": 4,
                    "mma_per_half": 8,
                    "softmax_ops": 64,
                },
            ],
        )

    def test_serial_fill_is_keyed_by_transfer_read_warps_bytes_and_grid(self):
        row = {
            "transfer_mode": "cp_async",
            "read_mode": "fill_and_ldmatrix",
            "num_warps": 8,
            "stage_bytes": 16384,
            "active_ctas": 48,
            "steps": 9000,
            "launch_ns_samples": [1.3e6] * 3,
            "stage_time_ns_samples": [144.0] * 3,
        }
        schedule, record = long_launch.freeze(
            "serial-fill", {"samples": 3, "serial_stage_fill": [row]}
        )
        self.assertEqual(schedule, "9000\n")
        self.assertEqual(
            record["points"][0]["point"],
            {
                "field": "serial_stage_fill",
                "transfer_mode": "cp_async",
                "read_mode": "fill_and_ldmatrix",
                "num_warps": 8,
                "stage_bytes": 16384,
                "active_ctas": 48,
            },
        )

    def test_stage_overhead_points_carry_in_kernel_windows(self):
        def probe(steps, window, **fields):
            return {
                **fields,
                "steps": steps,
                "launch_ns_samples": [1.3e6] * 3,
                "window_ns_samples": [window] * 3,
                "overhead_ns_samples": [window / steps] * 3,
            }

        def output(window):
            return {
                "samples": 3,
                "cycles_per_ns": 2.4,
                "stage_sync_overhead_ns": [
                    probe(6000, window, issue_mode="cp_async", num_warps=4, tokens=2)
                ],
                "fixed_sync_ns": [probe(30000, window, num_warps=8, tokens=1)],
                "min_async_roundtrip": probe(2000, window, num_warps=4),
                "min_async_roundtrip_ns_samples": [window / 2000] * 3,
            }

        schedule, record = long_launch.freeze("stage-overhead", output(1.29e6))
        self.assertEqual(schedule, "6000\n30000\n2000\n")
        self.assertEqual(
            [p["point"]["field"] for p in record["points"]],
            ["stage_sync_overhead_ns", "fixed_sync_ns", "min_async_roundtrip"],
        )
        long_launch.verify(record, output(1.2e6))
        with self.assertRaisesRegex(ValueError, "in-kernel window"):
            long_launch.verify(record, output(0.98e6))

    def test_sfu_shfl_curves_and_latency_are_keyed_by_their_own_fields(self):
        def timed(steps):
            return {"steps": steps, "launch_ns_samples": [1.3e6] * 3}

        curve = [{"warps_per_sm": w, **timed(10000 * w)} for w in (1, 2)]
        pilot = {
            "samples": 3,
            "sfu_issue_rate_per_sm_gips": curve,
            "sfu_rcp_issue_rate_per_sm_gips": curve,
            "shfl_issue_rate_per_sm_gips": curve,
            "shfl_idx_issue_rate_per_sm_gips": curve,
            "shfl_latency": {"warps": 1, **timed(70000)},
            "shfl_latency_ns_samples": [20.0] * 3,
        }
        schedule, record = long_launch.freeze("sfu-shfl", pilot)
        self.assertEqual(schedule.split(), ["10000", "20000"] * 4 + ["70000"])
        self.assertEqual(record["points"][-1]["point"], {"field": "shfl_latency", "warps": 1})
        first = record["points"][0]["point"]
        self.assertEqual(first, {"field": "sfu_issue_rate_per_sm_gips", "warps_per_sm": 1})

    def test_cp_async_and_smem_points_record_their_lengths(self):
        pilot = {
            "samples": 3,
            "sm_count": 48,
            "cp_async": {"threads_per_sm": 256, "steps": 16384, "launch_ns_samples": [1.5e6] * 3},
            "cp_async_bw_per_sm_gbps_samples": [44.0] * 3,
            "smem": {"threads_per_sm": 512, "steps": 16384, "launch_ns_samples": [1.9e6] * 3},
            "smem_bw_per_sm_gbps_samples": [300.0] * 3,
        }
        schedule, record = long_launch.freeze("cp-async", pilot)
        self.assertEqual(schedule, "16384\n16384\n")
        self.assertEqual([p["point"]["field"] for p in record["points"]], ["cp_async", "smem"])

    def test_launch_batches_are_keyed_by_their_grids(self):
        def timed(steps):
            return {"steps": steps, "launch_ns_samples": [1.3e6] * 3}

        pilot = {
            "samples": 3,
            "graph_launch": {"ctas": 1, **timed(2600)},
            "device_launch_overhead_us_samples": [0.5] * 3,
            "per_cta": {
                "ctas": 1,
                "large_ctas": 4096,
                **timed(2600),
                "large_graph_ns_samples": [2e6] * 3,
            },
            "per_cta_launch_latency_ns_samples": [0.1] * 3,
            "stream_launch": {"ctas": 1, **timed(1900)},
            "host_launch_overhead_us_samples": [0.7] * 3,
        }
        schedule, record = long_launch.freeze("launch", pilot)
        self.assertEqual(schedule, "2600\n2600\n1900\n")
        self.assertEqual(
            record["points"][1]["point"], {"field": "per_cta", "ctas": 1, "large_ctas": 4096}
        )

    def test_latency_probes_freeze_chase_steps_and_check_their_windows(self):
        def probe(name, steps, window):
            return {
                "probe": name,
                "steps": steps,
                "launch_ns_samples": [1.4e6] * 3,
                "window_ns_samples": [window] * 3,
            }

        keys = (
            "smem_latency",
            "l2_latency",
            "l2_latency_loaded",
            "dram_latency",
            "dram_latency_loaded",
        )
        pilot = {"samples": 3, **{k: probe(k, 8192, 1.3e6) for k in keys}}
        schedule, record = long_launch.freeze("latency", pilot)
        self.assertEqual(schedule.split(), ["8192"] * 5)
        self.assertEqual(
            record["points"][3]["point"], {"field": "dram_latency", "probe": "dram_latency"}
        )
        short = {"samples": 3, **{k: probe(k, 8192, 1.3e6) for k in keys}}
        short["dram_latency"]["window_ns_samples"] = [0.9e6] * 3
        with self.assertRaises(ValueError):
            long_launch.verify(record, short)

    def test_latency_curve_levels_are_keyed_by_their_background(self):
        def level(blocks, threads, steps):
            row = {"bg_blocks": blocks, "bg_threads": threads, "steps": steps}
            row.update({"launch_ns_samples": [1.4e6] * 3, "window_ns_samples": [1.3e6] * 3})
            return row

        pilot = {
            "samples": 3,
            "dram_latency_by_bw": [level(0, 0, 4096), level(1, 64, 4096)],
            "l2_latency_by_bw": [level(0, 0, 9000), level(47, 256, 8192)],
        }
        schedule, record = long_launch.freeze("latency-curve", pilot)
        self.assertEqual(schedule.split(), ["4096", "4096", "9000", "8192"])
        self.assertEqual(
            record["points"][3]["point"],
            {"field": "l2_latency_by_bw", "bg_blocks": 47, "bg_threads": 256},
        )

    def test_l1_carveout_points_carry_their_carveout(self):
        def point(carveout, ws, steps):
            row = {"smem_carveout_bytes": carveout, "ws_bytes": ws, "steps": steps}
            row.update({"launch_ns_samples": [1.4e6] * 3, "window_ns_samples": [1.3e6] * 3})
            return row

        pilot = {
            "samples": 3,
            "l1_carveout_sweep": [
                {"smem_carveout_bytes": 0, "points": [point(0, 8192, 50000)]},
                {"smem_carveout_bytes": 16384, "points": [point(16384, 8192, 51000)]},
            ],
        }
        schedule, record = long_launch.freeze("l1-carveout", pilot)
        self.assertEqual(schedule, "50000\n51000\n")
        self.assertEqual(
            record["points"][1]["point"],
            {"field": "l1_carveout_sweep/points", "smem_carveout_bytes": 16384, "ws_bytes": 8192},
        )

    def test_l2_rescan_freezes_groups_per_working_set(self):
        def point(ws, groups):
            row = {"working_set_bytes": ws, "ratio_to_l2": ws / 2**24, "steps": groups}
            row.update({"launch_ns_samples": [9e6] * 3, "window_ns_samples": [1.3e6] * 3})
            return row

        pilot = {"samples": 3, "points": [point(2**23, 40), point(2**25, 6)]}
        schedule, record = long_launch.freeze("l2-rescan", pilot)
        self.assertEqual(schedule, "40\n6\n")
        self.assertEqual(
            record["points"][1]["point"], {"field": "points", "working_set_bytes": 2**25}
        )
