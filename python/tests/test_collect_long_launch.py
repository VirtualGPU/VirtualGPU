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
