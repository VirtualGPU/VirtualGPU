"""The registered readings of the GB10 L2 buffer experiment on synthetic runs."""

import unittest

from microbench import l2_buffer_diag_eval as ev

MIB = 1 << 20


def run(levels, random_hit_bw=600.0):
    steady = []
    for fill in ("constant", "random"):
        for visit in (1, 2):
            for b, level in enumerate(levels, 1):
                for size, bw in ((20, 1960.0), (28, 1900.0), (48, 300.0), (60, 290.0)):
                    steady.append(point(fill, visit, b, size, bw))
                steady.append(
                    point(fill, visit, b, 36, level if fill == "constant" else random_hit_bw)
                )
    once = [
        {
            "fill": "constant",
            "buffer": 1,
            "pass1_gbps_samples": [240.0],
            "pass2_gbps_samples": [250.0],
        }
    ]
    return {"steady": steady, "one_pass": once}


def point(fill, visit, buffer, size, bw):
    return {
        "fill": fill,
        "visit": visit,
        "buffer": buffer,
        "working_set_bytes": size * MIB,
        "gbps_samples": [bw] * 15,
    }


class ReadingTests(unittest.TestCase):
    def test_placement(self):
        spread = [1470.0, 1160.0, 885.0, 1470.0, 1160.0, 885.0]
        r = ev.readings([run(spread)] * 3)
        self.assertEqual(r["reading"], "placement")
        self.assertTrue(r["compression"])

    def test_session_state(self):
        r = ev.readings([run([1470.0] * 6), run([1160.0] * 6), run([885.0] * 6)])
        self.assertEqual(r["reading"], "session state")

    def test_not_reproduced_and_no_compression(self):
        r = ev.readings([run([1160.0] * 6, random_hit_bw=1160.0)] * 3)
        self.assertEqual(r["reading"], "not reproduced")
        self.assertFalse(r["compression"])

    def test_neither(self):
        mild = [1160.0, 1100.0, 1220.0, 1160.0, 1100.0, 1220.0]
        r = ev.readings([run(mild), run(mild), run([885.0] * 6)])
        self.assertEqual(r["reading"], "neither supported")
