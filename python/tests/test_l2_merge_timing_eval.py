"""The registered readings of the l2-merge episode timing experiment on synthetic runs."""

import unittest

from microbench import l2_merge_timing_eval as ev


def block(steps, per_sample_ms, period_ms=None, period_samples=None, n=200):
    host = [i * per_sample_ms for i in range(n)]
    every = period_samples or round(period_ms / per_sample_ms)
    launch = [1.3e6 if i % every else 1.8e6 for i in range(n)]
    return {"steps": steps, "launch_ns_samples": launch, "host_start_ms": host}


def run(**kw):
    return {"blocks": [block(1464, 2.4, **kw), block(732, 1.8, **kw)] * 2}


class TimingReadingTests(unittest.TestCase):
    def test_fixed_wall_clock(self):
        self.assertEqual(
            ev.readings([run(period_ms=21.6)] * 3)["reading"], "fixed wall-clock period"
        )

    def test_fixed_sample_count(self):
        self.assertEqual(ev.readings([run(period_samples=8)] * 3)["reading"], "fixed sample count")

    def test_not_separable_and_too_few(self):
        same = {"blocks": [block(1464, 2.0, period_samples=8), block(732, 1.9, period_samples=8)]}
        self.assertEqual(ev.readings([same] * 3)["reading"], "not separable")
        rare = {
            "blocks": [block(1464, 2.4, period_samples=150), block(732, 1.8, period_samples=150)]
        }
        self.assertEqual(ev.readings([rare] * 3)["reading"], "not enough episodes")
