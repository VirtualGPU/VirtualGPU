"""CPU-only tests for the repeated MLP reduction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from microbench import host_quiet
from microbench import repeat_mlp as mlp

ROOT = Path(__file__).resolve().parents[2]


class RepeatMlpTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / "profiles/gb10.json"
        self.profile = json.loads(self.base.read_bytes())

    def raw(self, mode, value):
        memory = self.profile["measured"]["memory"]
        return {
            field: [{**row, "gbps_samples": [value] * mlp.SAMPLES} for row in memory[field]]
            for field in mlp.MODES[mode]
        }

    def test_geometry_and_full_sample_count_are_required(self):
        raw = self.raw("mlp", 20.0)
        mlp.point_maps(self.profile, "mlp", raw)
        point = raw[mlp.CONTIGUOUS[0]][0]
        point["gbps_samples"].pop()
        with self.assertRaisesRegex(ValueError, "finite positive samples"):
            mlp.point_maps(self.profile, "mlp", raw)
        point["bytes_in_flight"] += 1
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            mlp.point_maps(self.profile, "mlp", raw)

    def test_gate_requires_temperature_clock_and_event_stability(self):
        row = {key: "Not Active" for key in mlp.THERMAL_EVENT_FIELDS}
        row.update({"temperature.gpu": "48", "clocks.current.sm": "2500"})
        trace = [{**row, "elapsed_seconds": second} for second in range(60)]
        windows = mlp.thermal_windows(trace, 60)
        self.assertEqual(mlp.thermal_stability(windows)["samples_per_window"], [30, 30])
        for override in (
            {"temperature.gpu": "49"},
            {"clocks.current.sm": "2530"},
            {mlp.THERMAL_EVENT_FIELDS[-1]: "Active"},
            {mlp.THERMAL_EVENT_FIELDS[-1]: "[N/A]"},
        ):
            with self.subTest(override=override):
                recent = [{**sample, **override} for sample in windows[1]]
                self.assertIsNone(mlp.thermal_stability((windows[0], recent)))

    def test_default_warmup_requires_both_observation_windows(self):
        row = {key: "Not Active" for key in mlp.THERMAL_EVENT_FIELDS}
        row.update({"temperature.gpu": "48", "clocks.current.sm": "2500"})
        with (
            patch.object(mlp.time, "monotonic", side_effect=[0, 0, 15, 15, 45, 60]),
            patch.object(mlp, "host_snapshot", return_value={"compute_apps": ""}),
            patch.object(mlp, "command") as warm_launch,
            patch.object(mlp, "telemetry", return_value=row),
        ):
            trace, gate = mlp.warm_device("smi", "binary", 0, timeout_report=Path("unused"))
        self.assertEqual(warm_launch.call_count, 2)
        self.assertEqual(len(trace), 2)
        self.assertEqual(gate["elapsed_seconds"], 60)
        self.assertEqual(gate["samples_per_window"], [1, 1])

    def test_capture_rules_one_and_two_count_only_selected_geometry(self):
        steady = [100.0] * mlp.SAMPLES
        points = {"read": {(number,): list(steady) for number in range(20)}}
        points["read"][(0,)] = [100.0] * 14 + [160.0]
        receipt = mlp.dispersion_for_points(points)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["sample_lists"], 20)
        self.assertEqual(receipt["rules"]["spike"]["share"], 0.05)
        for number in range(1, 4):
            points["read"][(number,)] = [100.0] * 14 + [160.0]
        self.assertEqual(mlp.dispersion_for_points(points)["status"], "failed")
        wide = {"read": {(number,): list(steady) for number in range(20)}}
        for number in range(2):
            wide["read"][(number,)] = [80.0, 90.0, 100.0, 110.0, 120.0] * 3
        self.assertEqual(mlp.dispersion_for_points(wide)["status"], "failed")
        self.assertEqual(mlp.dispersion_for_points({})["status"], "failed")
        short = mlp.dispersion_for_points({"read": {(0,): [1.0]}})
        self.assertEqual((short["status"], short["invalid_lists"]), ("failed", 1))

    def test_capture_rule_three_compares_repeats_point_by_point(self):
        base = {"read": {(number,): [100.0] * mlp.SAMPLES for number in range(20)}}
        moved = {"read": {key: list(v) for key, v in base["read"].items()}}
        moved["read"][(3,)] = [111.0] * mlp.SAMPLES
        self.assertEqual(mlp.reproducibility_for_runs([base, moved])["status"], "passed")
        moved["read"][(4,)] = [89.0] * mlp.SAMPLES
        self.assertEqual(mlp.reproducibility_for_runs([base, moved])["status"], "failed")

    def test_failed_dispersion_archives_raw_and_exits(self):
        record = {"dispersion_check": {"status": "failed"}, "raw": {"read_curve": [1, 2]}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "void.json"
            argv = [
                "repeat_mlp",
                "capture",
                "profile",
                "binary",
                "mlp",
                "1",
                str(output),
                "--revision",
                "fixture",
                "--source-sha256",
                "fixture",
                "--nvcc",
                "nvcc",
            ]
            with patch("sys.argv", argv), patch.object(mlp, "capture", return_value=record):
                with self.assertRaisesRegex(SystemExit, "capture void"):
                    mlp.main()
            self.assertEqual(json.loads(output.read_bytes()), record)

    def test_host_not_quiet_archives_raw_and_exits(self):
        record = {
            "dispersion_check": {"status": "passed"},
            "host_quiet_check": {"status": "failed", "offenders": {"7": {"comm": "cc1"}}},
            "raw": {"read_curve": [1, 2]},
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "void.json"
            argv = [
                "repeat_mlp",
                "capture",
                "profile",
                "binary",
                "mlp",
                "1",
                str(output),
                "--revision",
                "fixture",
                "--source-sha256",
                "fixture",
                "--nvcc",
                "nvcc",
            ]
            with patch("sys.argv", argv), patch.object(mlp, "capture", return_value=record):
                with self.assertRaisesRegex(SystemExit, "Host CPU was not quiet"):
                    mlp.main()
            self.assertEqual(json.loads(output.read_bytes()), record)

    def test_reduce_uses_independent_medians_and_existing_ceiling(self):
        snapshot = {key: "Not Active" for key in mlp.TELEMETRY}
        snapshot.update({"clocks.current.sm": "2000", "clocks.current.memory": "[N/A]"})
        paths = []
        source = {
            "source_revision": "fixture",
            "source_bundle_sha256": "fixture",
            "base_profile_sha256": mlp.digest(self.base),
            "binary_sha256": "fixture",
            "nvcc": "fixture",
            "target": "sm_121",
            "host": "fixture",
            "initial_gpu": {"uuid": "fixture", "driver_version": "fixture"},
            "clock_policy": "record_only",
            "telemetry": [snapshot],
            "command": ["fixture"],
            "started_utc": "2026-11-26T10:00:00+00:00",
            "finished_utc": "2026-11-26T10:01:00+00:00",
        }

        def record(mode, number, raw):
            receipt = mlp.dispersion_for_points(mlp.point_maps(self.profile, mode, raw))
            return {
                **source,
                "mode": mode,
                "repeat": number,
                "raw": raw,
                "dispersion_check": receipt,
            }

        with tempfile.TemporaryDirectory() as temp:
            spread = []
            for mode in mlp.MLP_MODES:
                for number, value in enumerate((20.0, 23.0, 22.0, 90.0, 1.0), start=1):
                    path = Path(temp) / f"spread-{mode}-{number}.json"
                    mlp.write_json(path, record(mode, number, self.raw(mode, value)))
                    spread.append(path)
            with self.assertRaisesRegex(ValueError, "rule 3"):
                mlp.reduce_runs(self.base, spread, "fixture")
            for mode in mlp.MLP_MODES:
                for number, value in enumerate((20.0, 20.4, 20.2, 19.8, 20.6), start=1):
                    path = Path(temp) / f"{mode}-{number}.json"
                    mlp.write_json(path, record(mode, number, self.raw(mode, value)))
                    paths.append(path)
            updated, audit = mlp.reduce_runs(self.base, paths, "fixture")
            # An integrated GPU (GB10) needs a recorded host quiet check that recomputes.
            quiet = [{"total": 100 * i, "busy": 3 * i, "ncpu": 1, "procs": {}} for i in range(3)]
            first = json.loads(paths[0].read_bytes())
            gated = {
                **first,
                "device_query": {"integrated": True},
                "host_samples": quiet,
                "host_quiet_roots": [1],
                "host_quiet_check": host_quiet.check(quiet, [1]),
            }
            paths[0].write_text(json.dumps(gated))
            mlp.reduce_runs(self.base, paths, "fixture")
            paths[0].write_text(json.dumps({**gated, "host_quiet_check": {"status": "passed"}}))
            with self.assertRaisesRegex(ValueError, "host quiet"):
                mlp.reduce_runs(self.base, paths, "fixture")
            paths[0].write_text(json.dumps(first))
            with self.assertRaisesRegex(ValueError, "Expected repeats"):
                mlp.reduce_runs(self.base, paths[:-1], "fixture")
            session = {
                "session_id": "fixture-session",
                "thermal_stability": {"elapsed_seconds": 60},
            }
            for path in paths:
                data = json.loads(path.read_bytes())
                path.write_text(json.dumps({**data, **session}))
            for mode, (field,) in mlp.ACTIVE_MODES.items():
                raw = {
                    "read_curve": [
                        {"target_sms": row["active_sms"], "gbps_samples": [100.0] * mlp.SAMPLES}
                        for row in self.profile["measured"]["memory"][field]
                    ]
                }
                for number in range(1, mlp.REPEATS + 1):
                    path = Path(temp) / f"{mode}-{number}.json"
                    mlp.write_json(path, {**record(mode, number, raw), **session})
                    paths.append(path)
            combined, combined_audit = mlp.reduce_runs(self.base, paths, "fixture")
            self.assertEqual(combined_audit["session_id"], "fixture-session")
            for level in ("dram", "l2"):
                self.assertEqual(
                    combined["measured"]["memory"][f"{level}_bw_read_gbps"]["value"], 100.0
                )
                self.assertEqual(
                    combined_audit["ceilings"][f"{level}_bw_read_gbps"]["source"],
                    f"{level}_bw_by_active_sms",
                )
            latest = json.loads(path.read_bytes())
            path.write_text(json.dumps({**latest, "session_id": "mismatched"}))
            with self.assertRaisesRegex(ValueError, "require one session"):
                mlp.reduce_runs(self.base, paths, "fixture")
        self.assertEqual(audit["previous_provenance"], self.profile["provenance"])
        self.assertEqual(
            updated["provenance"]["clock_policy"], self.profile["provenance"]["clock_policy"]
        )
        self.assertEqual(audit["measurement_telemetry"]["active_sm_clock_range_mhz"], [2000, 2000])
        memory = updated["measured"]["memory"]
        for field in (*mlp.CONTIGUOUS, mlp.STRIDED):
            self.assertEqual(memory[field][0]["bw_gbps"]["value"], 20.2)
            self.assertEqual(memory[field][0]["bw_gbps"]["rel_uncertainty"], 0.15)
            point = audit["fields"][field]["points"][0]
            self.assertEqual(point["repeat_range_gbps"], [19.8, 20.6])
            self.assertEqual(point["pooled_median_gbps"], 20.2)
        self.assertEqual(
            memory["dram_bw_read_gbps"], memory["dram_bw_by_active_sms"][-1]["bw_gbps"]
        )
        self.assertIsNone(audit["fields"][mlp.CONTIGUOUS[0]]["active_clock_range_mhz"])


if __name__ == "__main__":
    unittest.main()
