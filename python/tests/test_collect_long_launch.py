"""CPU-only checks of the 0.23.0 profile entry of collect_profile."""

import json
import tempfile
import unittest
from pathlib import Path

from microbench import collect_profile as cp


class LongLaunchProfileTests(unittest.TestCase):
    def run_dir(self, state):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        (path / "capture-state.json").write_text(json.dumps(state))
        return path

    def test_refuses_a_capture_that_did_not_pass(self):
        path = self.run_dir({"status": "failed", "modes": list(cp.BUILD_MODES)})
        with self.assertRaisesRegex(SystemExit, "did not pass"):
            cp.long_launch_profile(Path("vgpu_bench"), 0, path)

    def test_refuses_to_mix_protocols(self):
        path = self.run_dir({"status": "passed", "modes": ["mma", "gmem-issue"]})
        with self.assertRaisesRegex(SystemExit, "mixes no protocols") as ctx:
            cp.long_launch_profile(Path("vgpu_bench"), 0, path)
        self.assertIn("'dram-bw'", str(ctx.exception))
        self.assertNotIn("'mma'", str(ctx.exception))


def test_near_capacity_knee_covers_each_rounds_knee_and_flags_a_covered_bracket():
    from microbench import collect_profile as cp

    mib = 1 << 20
    l2 = 24 * mib

    def point(size, rounds, exempt=False):
        p = {"working_set_bytes": size * mib, "gbps_samples": rounds}
        p["median_gbps"] = sorted(rounds)[len(rounds) // 2]
        if exempt:
            p["exceptions"] = ["l2_near_capacity_unstable"]
        return p

    # The knee falls between 16 and 20 MiB, below the cap, at a different place each round.
    sweep = [
        point(2, [2000] * 3),
        point(6, [2000] * 3),
        point(16, [1900, 1750, 1600]),
        point(20, [1500, 1500, 1500], exempt=True),
        point(48, [300] * 3, exempt=True),
        point(60, [290] * 3),
    ]
    knee, unc = cp.estimate_l2_knee(sweep, l2)
    value = {"value": round(knee), "rel_uncertainty": unc}
    rounds = [
        cp.estimate_l2_knee(
            [
                {"working_set_bytes": p["working_set_bytes"], "median_gbps": p["gbps_samples"][r]}
                for p in sweep
            ],
            l2,
        )[0]
        for r in range(3)
    ]
    cp.near_capacity_knee(value, sweep, l2)
    spread = max(abs(k - round(knee)) / round(knee) for k in rounds)
    assert value["rel_uncertainty"] == round(min(0.5, max(unc, spread)), 4)
    assert value["l2_near_capacity_unstable"] is True

    plain = {"value": round(knee), "rel_uncertainty": unc}
    cp.near_capacity_knee(
        plain, [{k: v for k, v in p.items() if k != "exceptions"} for p in sweep], l2
    )
    assert plain == {"value": round(knee), "rel_uncertainty": unc}
