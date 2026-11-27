"""CPU-only tests for the host CPU quiet check of GB10 captures."""

import json
import os
import sys
import time
import unittest
from unittest.mock import patch

from microbench import host_quiet


def series(foreign_per_second, busy_share=0.03, seconds=4, parent=1):
    """One snapshot per second; pid 50 uses `foreign_per_second` jiffies per second."""
    cpus = os.cpu_count()
    out = []
    for i in range(seconds):
        out.append(
            {
                "total": 100 * cpus * i,
                "ncpu": cpus,
                "busy": round(busy_share * 100 * cpus * i),
                "procs": {
                    10: ("capture", 0, 1),
                    11: ("bench", 0, 10),
                    50: ("cc1", foreign_per_second * i, parent),
                },
            }
        )
    return out


class HostQuietTests(unittest.TestCase):
    def test_quiet_host_passes(self):
        self.assertEqual(host_quiet.check(series(0), [10])["status"], "passed")

    def test_sustained_foreign_process_fails(self):
        result = host_quiet.check(series(60), [10])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["offenders"]["50"]["comm"], "cc1")

    def test_busy_share_above_the_limit_fails(self):
        self.assertEqual(host_quiet.check(series(0, busy_share=0.2), [10])["status"], "failed")

    def test_descendants_of_the_capture_are_its_own(self):
        self.assertEqual(host_quiet.check(series(60, parent=11), [10])["status"], "passed")


class JsonRoundTripTests(unittest.TestCase):
    def test_saved_samples_judge_the_same(self):
        for samples in (series(0), series(60), series(60, parent=11)):
            saved = json.loads(json.dumps(samples))
            self.assertEqual(host_quiet.check(saved, [10]), host_quiet.check(samples, [10]))

    def test_check_uses_the_sampled_cpu_count(self):
        samples = series(3)  # 3 jiffies per second out of 100 per CPU
        for s in samples:
            s["ncpu"] = 1  # on a 1-CPU host this is 3% of a CPU, under the limit
        self.assertEqual(host_quiet.check(samples, [10])["status"], "passed")

    def test_reused_pid_is_not_compared(self):
        samples = series(60)
        samples[2]["procs"][50] = ("other", 0, 1)
        samples[3]["procs"][50] = ("other", 60, 1)
        self.assertEqual(host_quiet.check(samples, [10])["status"], "passed")


class SlimTests(unittest.TestCase):
    def test_slim_keeps_the_verdict_bit_for_bit(self):
        for samples in (series(0), series(60), series(60, parent=11), series(0, busy_share=0.2)):
            for s in samples:
                s["procs"].update({100 + i: (f"idle{i}", 7, 1) for i in range(50)})
            slimmed = host_quiet.slim(samples, [10])
            self.assertEqual(host_quiet.check(slimmed, [10]), host_quiet.check(samples, [10]))
            self.assertEqual(
                json.loads(json.dumps(host_quiet.check(slimmed, [10]))),
                host_quiet.check(json.loads(json.dumps(slimmed)), [10]),
            )
            self.assertLess(len(slimmed[0]["procs"]), len(samples[0]["procs"]))

    def test_slim_on_real_snapshots(self):
        samples = []
        for _ in range(3):
            samples.append(host_quiet.sample())
            time.sleep(0.2)
        slimmed = host_quiet.slim(samples, [os.getpid()])
        self.assertEqual(
            host_quiet.check(slimmed, [os.getpid()]), host_quiet.check(samples, [os.getpid()])
        )
        self.assertIn(os.getpid(), slimmed[-1]["procs"])


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class WaitForQuietHostTests(unittest.TestCase):
    def test_returns_once_the_window_is_quiet(self):
        samples = iter(series(0, seconds=20))
        with patch.object(host_quiet, "sample", lambda: next(samples)):
            gate = host_quiet.wait_for_quiet_host([10], seconds=5, clock=FakeClock())
        self.assertEqual(gate["check"]["status"], "passed")
        self.assertEqual(gate["waited_seconds"], 5.0)

    def test_times_out_while_a_foreign_process_runs(self):
        samples = iter(series(60, seconds=100))
        with patch.object(host_quiet, "sample", lambda: next(samples)):
            with self.assertRaises(RuntimeError):
                host_quiet.wait_for_quiet_host([10], seconds=5, timeout=30, clock=FakeClock())


class RunTracedTests(unittest.TestCase):
    def test_returns_stdout_trace_and_check(self):
        run = host_quiet.run_traced(
            [sys.executable, "-c", "import time; time.sleep(0.05); print('ok')"]
        )
        self.assertEqual(run["stdout"], "ok\n")
        self.assertGreaterEqual(len(run["cpu_10ms"]), 1)
        self.assertLess(run["started_monotonic_ns"], run["finished_monotonic_ns"])
        self.assertIn(run["host_check"]["status"], ("passed", "failed"))
        self.assertGreaterEqual(run["host_check"]["samples"], 1)
