"""CPU-only checks of the five-round long-launch capture."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from microbench import long_capture as lc
from microbench import long_launch


def test_l2_capacity_sweep_near_capacity_exempts_only_gb10_rule_three():
    mib = 1024**2
    raw = {
        "capacity_sweep": [
            {"working_set_bytes": size * mib, "launch_ns_samples": [1.3e6]}
            for size in (17, 18, 36, 48, 49)
        ],
        "write_full": {"working_set_bytes": 24 * mib, "launch_ns_samples": [1.3e6]},
    }
    for device, rule in (("NVIDIA GB10", 1), ("NVIDIA GB10", 2), ("NVIDIA GeForce RTX 4070", 3)):
        assert lc.judged("l2", raw, device, rule, 24 * mib) == raw
    judged = lc.judged("l2", raw, "NVIDIA GB10", 3, 24 * mib)
    assert [point["working_set_bytes"] for point in judged["capacity_sweep"]] == [
        17 * mib,
        49 * mib,
    ]
    assert judged["write_full"] == raw["write_full"]
    name = ["l2_near_capacity_unstable"]
    assert [
        point.get("exceptions")
        for point in lc.mark_exceptions("l2", raw, "NVIDIA GB10", 24 * mib)["capacity_sweep"]
    ] == [None, name, name, name, None]


def test_l2_rescan_keeps_the_bistable_band():
    mib = 1024**2
    raw = {
        "points": [
            {"working_set_bytes": size * mib, "launch_ns_samples": [1.3e6]} for size in (17, 30, 31)
        ]
    }
    marked = lc.mark_exceptions("l2-rescan", raw, "NVIDIA GB10", 24 * mib)["points"]
    assert [p.get("exceptions") for p in marked] == [None, ["bistable_band"], None]


STEADY = [1.3e6] * lc.SAMPLES
SPIKY = [1.3e6] * 13 + [2.0e6, 2.0e6]


def chain_raw(samples_by_point, steps=1000):
    return {
        "samples": len(samples_by_point[0]),
        "mma_chain_latency": [
            {
                "variant": "mma_only_independent",
                "num_warps": 4,
                "tile_bytes": 0,
                "ldmatrix_x4_per_warp": 0,
                "mma_chain_len": m,
                "steps": steps,
                "launch_ns_samples": list(samples),
                "stage_time_ns_samples": [x / steps for x in samples],
                "sm_mhz_before": 2500.0,
                "sm_mhz_after": 2490.0,
            }
            for m, samples in zip((1, 2, 4), samples_by_point)
        ],
    }


class Fixture:
    """A run directory frozen for two modes that both emit mma-chain output."""

    def __init__(self, tmp, plan):
        self.dir = Path(tmp)
        self.plan = plan  # (round, mode) -> samples of the first two points
        self.calls = []
        schedule, record = long_launch.freeze("mma-chain", chain_raw([[1.3e6] * 3] * 3))
        modes = {}
        for mode in ("a", "b"):
            (self.dir / f"{mode}.schedule").write_text(schedule, newline="\n")
            (self.dir / f"{mode}.freeze.json").write_text(json.dumps({**record, "mode": mode}))
            modes[mode] = {"schedule": f"{mode}.schedule", "record": f"{mode}.freeze.json"}
        self.manifest = {
            "binary": "vgpu_bench",
            "samples": lc.SAMPLES,
            "device": {"name": "test", "l2_size_bytes": 0},
            "modes": modes,
        }
        self.gates = 0
        self.round = 0

    def run(self, cmd):
        mode = cmd[cmd.index("--mode") + 1]
        self.calls.append(mode)
        first = self.plan.get((self.round, mode), STEADY)
        if first == "crash":
            raise RuntimeError("process could not run")
        return {
            "stdout": json.dumps(chain_raw([first, first, STEADY])),
            "host_check": {"status": "passed"},
        }

    def gate(self):
        self.gates += 1
        return {"thermal": "passed"}


def capture(fx, state=None):
    state = state or {"modes": ["a", "b"], "rounds": []}

    def run(cmd):
        fx.round = state["rounds"][-1]["round"]
        return fx.run(cmd)

    return lc.capture(fx.dir, fx.manifest, state, run, fx.gate)


class LongCaptureTests(unittest.TestCase):
    def setUp(self):
        # mode names a and b map onto the mma-chain design fields
        self.addCleanup(lc.long_launch.DESIGN_FIELDS.pop, "a")
        self.addCleanup(lc.long_launch.DESIGN_FIELDS.pop, "b")
        lc.long_launch.DESIGN_FIELDS["a"] = lc.long_launch.DESIGN_FIELDS["mma-chain"]
        lc.long_launch.DESIGN_FIELDS["b"] = lc.long_launch.DESIGN_FIELDS["mma-chain"]

    def test_five_clean_rounds_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {})
            state = capture(fx)
            self.assertEqual((state["status"], len(state["rounds"]), fx.gates), ("passed", 5, 5))
            self.assertEqual(fx.calls, ["a", "b"] * 5)

    def test_void_round_adds_a_round_for_that_mode_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {(2, "b"): SPIKY, (3, "b"): SPIKY})
            state = capture(fx)
            self.assertEqual(state["status"], "passed")
            self.assertEqual([r["modes"] for r in state["rounds"][5:]], [["b"], ["b"]])
            self.assertEqual(lc.valid_rounds(state, "a"), [1, 2, 3, 4, 5])
            self.assertEqual(lc.valid_rounds(state, "b"), [1, 4, 5, 6, 7])
            void = state["rounds"][1]["results"]["b"]
            self.assertEqual((void["status"], void["reason"]), ("void", "rules 1 and 2"))

    def test_short_launch_after_timing_voids_the_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {(1, "a"): [0.9e6] * lc.SAMPLES})
            state = capture(fx)
            void = state["rounds"][0]["results"]["a"]
            self.assertEqual(void["status"], "void")
            self.assertIn("a launch took", void["reason"])
            self.assertEqual((state["status"], len(state["rounds"])), ("passed", 6))

    def test_more_than_two_extra_rounds_fail_the_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {(n, "b"): SPIKY for n in (1, 2, 3)})
            state = capture(fx)
            self.assertEqual((state["status"], len(state["rounds"])), ("failed", 7))
            self.assertIn("['b']", state["reason"])

    def test_interruption_resumes_without_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {(3, "b"): "crash"})
            state = {"modes": ["a", "b"], "rounds": []}
            with self.assertRaises(RuntimeError):
                capture(fx, state)
            saved = json.loads((fx.dir / "capture-state.json").read_text())
            self.assertNotIn("b", saved["rounds"][2]["results"])
            fx.plan = {}
            state = capture(fx, saved)
            self.assertEqual((state["status"], len(state["rounds"])), ("passed", 5))
            self.assertEqual(len(state["rounds"][2]["gates"]), 2)

    def test_schedule_mismatch_stops_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp, {})
            (fx.dir / "a.schedule").write_text("1\n2\n3\n", newline="\n")
            with self.assertRaisesRegex(ValueError, "differs from the freeze"):
                capture(fx)
            self.assertEqual(fx.calls, [])

    def test_exceptions_leave_rules_one_and_three_only(self):
        wide = [1.0e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6] * 3

        def curve(first, second):
            return {
                "samples": 15,
                "read_curve": [
                    {"target_sms": sms, "steps": 81, "launch_ns_samples": list(s)}
                    for sms, s in zip((1, 2, 4, 12, 24, 48), (first, second) + (STEADY,) * 4)
                ],
            }

        raw = curve(wide, wide)
        self.assertEqual(lc.rules_one_two("dram-bw", raw, "test")["status"], "failed")
        self.assertEqual(lc.rules_one_two("dram-bw", raw, "NVIDIA GB10")["status"], "passed")
        self.assertEqual(lc.rules_one_two("l2", raw, "NVIDIA GB10")["status"], "failed")
        spiky = curve(SPIKY, SPIKY)
        self.assertEqual(lc.rules_one_two("dram-bw", spiky, "NVIDIA GB10")["status"], "failed")
        moved = curve([1.6e6] * 15, [1.6e6] * 15)
        steady = curve(STEADY, STEADY)
        self.assertEqual(lc.rule_three("dram-bw", [steady, moved], "test")["status"], "failed")
        self.assertEqual(
            lc.rule_three("dram-bw", [steady, moved], "NVIDIA GB10")["status"], "passed"
        )

    def test_merge_takes_round_medians_and_launch(self):
        rounds = []
        for n in range(5):
            raw = chain_raw([[1.3e6 + n] * 15, STEADY, STEADY])
            raw["cycles_per_ns"] = 2.5 + n / 100
            raw["mma_chain_latency"][0]["sm_mhz_before"] = 2500.0 + n
            rounds.append(raw)
        merged = lc.merge(rounds)
        point = merged["mma_chain_latency"][0]
        self.assertEqual(point["launch_ns_samples"], [1.3e6 + n for n in range(5)])
        self.assertEqual(point["launch"]["min_ns"], 1.3e6)
        self.assertEqual(point["launch"]["median_ns"], 1.3e6 + 2)
        self.assertEqual(point["launch"]["sm_mhz_before"], [2500.0 + n for n in range(5)])
        self.assertEqual((point["mma_chain_len"], merged["cycles_per_ns"]), (1, 2.52))
        rounds[1]["mma_chain_latency"][0]["variant"] = "other"
        with self.assertRaisesRegex(ValueError, "variant differs"):
            lc.merge(rounds)

    def test_merged_points_name_their_exceptions(self):
        curve = {
            "samples": 5,
            "read_curve": [
                {
                    "target_sms": sms,
                    "steps": 81,
                    "launch_ns_samples": [1.3e6, 1.9e6, 1.4e6, 1.3e6, 1.35e6],
                }
                for sms in (1, 4)
            ],
        }
        marked = lc.mark_exceptions("dram-bw", copy.deepcopy(curve), "NVIDIA GB10")
        first, second = marked["read_curve"]
        self.assertEqual(first["exceptions"], ["low_in_flight_unstable"])
        self.assertEqual(first["round_median_range_ns"], [1.3e6, 1.9e6])
        self.assertNotIn("exceptions", second)
        self.assertNotIn(
            "exceptions", lc.mark_exceptions("l2", curve, "NVIDIA GB10")["read_curve"][0]
        )

    def test_capacity_sweep_band_leaves_rule_three_only(self):
        l2 = 24 << 20

        def sweep(values):
            return {
                "capacity_sweep": [
                    {"working_set_bytes": ws, "steps": 9, "launch_ns_samples": [v] * 15}
                    for ws, v in zip((12 << 20, 24 << 20, 60 << 20), values)
                ]
            }

        base, moved = sweep([1.3e6] * 3), sweep([1.3e6, 1.6e6, 1.3e6])
        self.assertEqual(lc.rule_three("l2", [base, moved], "test", l2)["points"], 3)
        judged = lc.rule_three("l2", [base, moved], "NVIDIA GB10", l2)
        self.assertEqual((judged["points"], judged["failing"]), (2, 0))
