"""CPU-only tests of the post-hoc attention L2 counter collection."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from microbench import attention_l2_counters as l2


def action(values):
    metrics = {
        name: SimpleNamespace(name=lambda n=name: n, value=lambda v=v: v)
        for name, v in values.items()
    }
    return SimpleNamespace(name=lambda: "ablated_attention", metric_by_name=metrics.get)


def launch_values(hit, miss, grid=512, block=256):
    return {l2.HIT: hit, l2.MISS: miss, "launch__grid_size": grid, "launch__block_size": block}


class AttentionL2CounterTests(unittest.TestCase):
    def test_permission_failure_does_not_publish(self):
        denied = subprocess.CompletedProcess(["ncu"], 0, "", "==ERROR== ERR_NVGPUCTRPERM")

        def measure(*_):
            with patch.object(l2.spill.subprocess, "run", return_value=denied):
                return l2.spill.run(["ncu"], {})

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(l2, "POST_HOC", Path(folder) / "post-hoc"),
            patch.object(l2.shutil, "which", return_value="/tmp/ncu"),
            patch.object(l2, "measure", side_effect=measure),
            patch.object(l2.sys, "argv", ["attention_l2_counters"]),
            patch.object(l2.sys, "stderr") as stderr,
        ):
            with self.assertRaises(SystemExit) as error:
                l2.main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn("ERR_NVGPUCTRPERM", stderr.write.call_args.args[0])
            self.assertFalse(l2.POST_HOC.exists())

    def test_row_reports_hit_fraction_and_checks_geometry(self):
        launch = {"grid": [16, 32, 1], "num_warps": 8}
        row = l2.row("m128n32-w8-s2", action(launch_values(750, 250)), launch)
        self.assertEqual(row["read_hit_fraction"], 0.75)
        self.assertEqual((row["read_sectors_hit"], row["read_sectors_miss"]), (750, 250))
        with self.assertRaisesRegex(ValueError, "geometry"):
            l2.row("m128n32-w8-s2", action(launch_values(1, 1, block=128)), launch)
        with self.assertRaisesRegex(ValueError, "no L2 read sectors"):
            l2.row("m128n32-w8-s2", action(launch_values(0, 0)), launch)

    def test_configs_are_ablation_configs(self):
        from microbench.attention_ablation import CONFIGS

        self.assertLessEqual(set(l2.CONFIGS), {c["config_id"] for c in CONFIGS})


if __name__ == "__main__":
    unittest.main()
