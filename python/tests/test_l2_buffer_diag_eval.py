"""The re-registered readings of the GB10 L2 buffer experiment on synthetic runs."""

import unittest

from microbench import l2_buffer_diag_eval as ev

MIB = 1 << 20
REFS = ((20, 1960.0), (28, 1900.0), (48, 300.0), (60, 290.0))


def point(size, bw, spread=0.0, **ids):
    samples = [bw * (1 + spread * ((i % 3) - 1)) for i in range(15)]
    return ids | {
        "working_set_bytes": size * MIB,
        "gbps_samples": samples,
        "launch_ns_samples": [1e6] * 15,
    }


def run(levels, control=1160.0, random_bw=600.0, spread=0.0):
    steady = []
    for fill in ("constant", "random"):
        for visit in (1, 2):
            for b, level in enumerate(levels, 1):
                ids = {"fill": fill, "visit": visit, "buffer": b}
                steady += [point(size, bw, **ids) for size, bw in REFS]
                bw36 = level if fill == "constant" else random_bw
                steady.append(point(36, bw36, spread, **ids))
    once = [
        {
            "fill": "constant",
            "buffer": 1,
            "pass1_gbps_samples": [240.0],
            "pass2_gbps_samples": [250.0],
        }
    ]
    return {"control": [point(36, control)], "steady": steady, "one_pass": once}


LEVELS = (1470.0, 1160.0, 885.0)


class ReadingTests(unittest.TestCase):
    def test_placement(self):
        spread = [1470.0, 1160.0, 885.0, 1470.0, 1160.0, 885.0]
        runs = [run(spread, control=c) for c in LEVELS + LEVELS[:2]]
        r = ev.readings(runs)
        self.assertEqual(r["reading"], "placement")
        self.assertTrue(r["compression"])

    def test_session_state(self):
        runs = [run([c] * 6, control=c) for c in LEVELS + LEVELS[:2]]
        self.assertEqual(ev.readings(runs)["reading"], "session state")

    def test_not_reproduced_by_the_control(self):
        spread = [1470.0, 1160.0, 885.0, 1470.0, 1160.0, 885.0]
        r = ev.readings([run(spread, control=1160.0, random_bw=1160.0)] * 5)
        self.assertEqual(r["reading"], "not reproduced")
        self.assertFalse(r["compression"])

    def test_unsteady_buffers_do_not_keep_their_level(self):
        spread = [1470.0, 1160.0, 885.0, 1470.0, 1160.0, 885.0]
        runs = [run(spread, control=c, spread=0.2) for c in LEVELS + LEVELS[:2]]
        r = ev.readings(runs)
        self.assertEqual(r["buffers_keeping_level"], 0)
        self.assertEqual(r["unsteady_share"]["constant"], 1.0)
        self.assertEqual(r["reading"], "neither supported")

    def test_gate_judges_only_the_reference_lists(self):
        record = {"raw": run([1160.0] * 6, spread=0.3), "host_quiet_check": {"status": "passed"}}
        self.assertTrue(ev.gate(record)[0])
        record["raw"]["steady"][0]["gbps_samples"][0] *= 3  # a 20 MiB reference list
        for p in record["raw"]["steady"]:
            if p["working_set_bytes"] // MIB == 20:
                p["gbps_samples"] = [1.0, 2.0] * 7 + [1.0]
        self.assertFalse(ev.gate(record)[0])
