"""CPU-only checks of the MLP point-order runner."""

import copy
import hashlib
import unittest

from microbench import mlp_order


def manifest():
    points = [
        {"field": "dram_bw_by_bytes_in_flight", "key": ["cp_async", 4096, 1, 4096], "steps": 4096},
        {
            "field": "l2_bw_by_bytes_in_flight",
            "key": ["ld_global_v4", 65536, 8, 8192],
            "steps": 900,
        },
    ]
    steps = "".join(f"{p['steps']}\n" for p in points)
    return {
        "steps_file": "x.steps",
        "steps_sha256": hashlib.sha256(steps.encode()).hexdigest(),
        "points": points,
    }


class MlpOrderTests(unittest.TestCase):
    def test_order_file_lists_points_in_order(self):
        self.assertEqual(
            mlp_order.order_lines(manifest()),
            "dram_bw_by_bytes_in_flight cp_async 4096 1 4096 4096\n"
            "l2_bw_by_bytes_in_flight ld_global_v4 65536 8 8192 900\n",
        )

    def test_steps_must_match_the_digest(self):
        order = manifest()
        order["points"][1]["steps"] = 901
        with self.assertRaisesRegex(ValueError, "digest"):
            mlp_order.order_lines(order)

    def test_result_must_follow_the_manifest(self):
        order = manifest()
        ran = [
            {
                "field": p["field"],
                "issue_mode": p["key"][0],
                "bytes_in_flight": p["key"][1],
                "active_ctas": p["key"][2],
                "bytes_per_cta": p["key"][3],
                "steps": p["steps"],
            }
            for p in order["points"]
        ]
        mlp_order.check_result(order, {"points": ran})
        with self.assertRaisesRegex(ValueError, "differ"):
            mlp_order.check_result(order, {"points": list(reversed(copy.deepcopy(ran)))})
