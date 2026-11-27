"""The registered readings of the GB10 carveout probe on synthetic runs."""

import unittest

from microbench import carveout_probe_eval as ev

KIB = 1024
WS = [16, 32, 64] + list(range(96, 161, 4)) + [192, 224, 256, 320]


def chase(cliff_kib):
    """Latency 30 ns below the cliff, 300 ns above, a 4 KiB linear ramp across it."""
    out = []
    for kb in WS:
        frac = min(1.0, max(0.0, (kb - cliff_kib + 2) / 4))
        out.append(30 + 270 * frac)
    return out


def run(cliffs, m_by_warp=None):
    pts = []
    for name, kib in cliffs.items():
        for kb, lat in zip(WS, chase(kib)):
            pts.append({"config": name, "ws_bytes": kb * KIB, "latency_ns_samples": [lat] * 15})
    m_by_warp = m_by_warp or {w: (w - 6) / 4 for w in range(6, 11)}
    spill = []
    for context in range(1, 6):
        for w, m in m_by_warp.items():
            t128, t4096 = 100.0, 500.0
            for frame, t in ((128, t128), (4096, t4096), (512, t128 + m * (t4096 - t128))):
                spill.append(
                    {
                        "context": context,
                        "warps": w,
                        "frame_bytes_per_thread": frame,
                        "step_ns_samples": [t] * 15,
                    }
                )
    return {"l1_cliff": pts, "spill_transition": spill}


class CarveoutReadings(unittest.TestCase):
    def test_cliff_is_the_midpoint_crossing(self):
        self.assertAlmostEqual(
            ev.cliff([p for p in run({"R0": 128})["l1_cliff"]]), 128 * KIB, delta=KIB
        )

    def test_labelled_by_request(self):
        r = ev.readings([run({"R0": 125, "R8": 116, "R16": 104, "S": 125})] * 3)
        self.assertEqual(r["step1"], "labelled by request")
        self.assertEqual(r["step2"], "consistent")  # m crosses 0.5 at 8 warps: 128 KiB, vs 125
        late = {w: (w - 8) / 4 for w in range(6, 11)}  # crosses at 10 warps: 160 KiB
        r = ev.readings([run({"R0": 125, "R8": 116, "R16": 104, "S": 125}, late)] * 3)
        self.assertEqual(r["step2"], "inconsistent")
        r = ev.readings(
            [
                run(
                    {"R0": 125, "R8": 116, "R16": 104, "S": 125},
                    {w: (w - 5.8) / 4 for w in range(6, 11)},
                )
            ]
            * 3
        )
        self.assertLess(abs(r["T"] - 125 * KIB), 8 * KIB)
        self.assertEqual(r["step2"], "consistent")

    def test_labelled_by_carveout(self):
        r = ev.readings([run({"R0": 125, "R8": 116, "R16": 104, "S": 104})] * 3)
        self.assertEqual(r["step1"], "labelled by carveout")

    def test_neither_and_reference(self):
        # S at R8 with R8 far from R0 is the carveout reading; between all three is neither.
        cfg = {"R0": 125, "R8": 116, "R16": 104}
        self.assertEqual(ev.readings([run(cfg | {"S": 116})] * 3)["step1"], "labelled by carveout")
        self.assertEqual(ev.readings([run(cfg | {"S": 110})] * 3)["step1"], "neither")
        self.assertEqual(
            ev.readings([run({"R0": 140, "R8": 116, "R16": 104, "S": 140})] * 3)["step1"],
            "reference not reproduced",
        )

    def test_contexts_without_a_crossing(self):
        r = ev.readings(
            [run({"R0": 125, "R8": 116, "R16": 104, "S": 125}, {w: 0.1 for w in range(6, 11)})] * 3
        )
        self.assertEqual(r["step2"], "no transition in the swept range")
        self.assertTrue(all(x["side"] == "below" for x in r["contexts"]))
