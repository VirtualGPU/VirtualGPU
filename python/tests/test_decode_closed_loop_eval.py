"""The registered readings of the GB10 strided closed-loop test on synthetic runs."""

import unittest

from microbench import decode_closed_loop_eval as ev
from microbench import long_launch


def run(gap_by_segment):
    """Points whose bandwidth at 4 buffers is (1 + gap) times that at 1 buffer."""
    pts = []
    for ctas in (96, 128):
        for stage in (4096, 8192):
            for segment in (0, 128, 256):
                gap = gap_by_segment[segment]
                if callable(gap):
                    gap = gap(ctas, stage)
                for tokens, bw in ((1, 200.0), (2, 200.0 * (1 + gap / 2)), (4, 200.0 * (1 + gap))):
                    pts.append(
                        {
                            "access": "strided" if segment else "contiguous",
                            "segment_bytes": segment,
                            "stride_bytes": 16384 if segment else 0,
                            "active_ctas": ctas,
                            "stage_bytes": stage,
                            "tokens": tokens,
                            "gbps_samples": [bw] * 15,
                        }
                    )
    return {"decode_closed_loop": pts}


class Readings(unittest.TestCase):
    def reading(self, gaps):
        return ev.readings([run(gaps)] * 3)["reading"]

    def test_control_not_reproduced(self):
        self.assertEqual(self.reading({0: 0.03, 128: 0.10, 256: 0.10}), "control not reproduced")

    def test_supported_needs_three_pairs_of_one_segment(self):
        def three(ctas, stage):
            return 0.0 if (ctas, stage) == (96, 4096) else 0.06

        self.assertEqual(self.reading({0: 0.01, 128: three, 256: 0.0}), "H-gran supported")

        def two(ctas, stage):
            return 0.06 if ctas == 96 else 0.0

        self.assertEqual(self.reading({0: 0.01, 128: two, 256: two}), "inconclusive")

    def test_rejected(self):
        self.assertEqual(self.reading({0: 0.0, 128: 0.019, 256: -0.019}), "H-gran rejected")

    def test_inconclusive_between(self):
        self.assertEqual(self.reading({0: 0.0, 128: 0.03, 256: 0.0}), "inconclusive")

    def test_median_over_processes(self):
        runs = [run({0: 0.0, 128: 0.10, 256: 0.10})] * 2 + [run({0: 0.0, 128: 0.0, 256: 0.0})]
        self.assertEqual(ev.readings(runs)["reading"], "H-gran supported")


class Lengths(unittest.TestCase):
    def test_freeze_needs_twice_the_pilot_floor(self):
        pilot = run({0: 0.0, 128: 0.0, 256: 0.0})
        for p in pilot["decode_closed_loop"]:
            p.pop("gbps_samples")
            p["steps"] = 1000
            p["launch_ns_samples"] = [2.4e6] * 3
        with self.assertRaises(ValueError):
            long_launch.freeze("decode-closed-loop", pilot)
        for p in pilot["decode_closed_loop"]:
            p["launch_ns_samples"] = [2.6e6] * 3
        schedule, _ = long_launch.freeze("decode-closed-loop", pilot)
        self.assertEqual(len(schedule.split()), 36)


if __name__ == "__main__":
    unittest.main()
