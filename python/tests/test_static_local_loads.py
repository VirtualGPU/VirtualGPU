"""Static per-loop local loads from SASS (scripts/static_local_loads.py) on synthetic input."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import static_local_loads as s  # noqa: E402

SRC = Path("/x/attention.py")
SITES = {"main": (56, 59), "diagonal": (60, 63)}
SASS = """
\t//## File "/x/attention.py", line 51
        /*0000*/                   LDL R2, [R1] ;
\t//## File "/x/attention.py", line 20 inlined at "/x/attention.py", line 56
\t//## File "/x/attention.py", line 56
        /*0010*/                   LDL.64 R4, [R1+0x8] ;
.L_x_1:
        /*0020*/                   LDL.LU R6, [R1+0x10] ;
        /*0030*/                   STL [R1+0x14], R6 ;
        /*0040*/              @P0  BRA `(.L_x_1) ;
\t//## File "/x/attention.py", line 26 inlined at "/x/attention.py", line 60
\t//## File "/x/attention.py", line 60
        /*0050*/                   LDL.128 R8, [R1+0x20] ;
"""


class Parse(unittest.TestCase):
    def test_scopes_loop_body_and_widths(self):
        c = s.count(s.parse_sass(SASS, SITES, SRC))
        self.assertEqual(
            c["outside_all"], {"LDL_4B": 1, "LDL_bytes": 4, "LDL_count": 1, "LDL_predicated": 0}
        )
        self.assertEqual(c["main_peeled"]["LDL_8B"], 1)
        self.assertEqual(c["main_loop_body"]["LDL_count"], 1)
        self.assertEqual(c["main_loop_body"]["STL_count"], 1)
        self.assertEqual(c["diagonal_peeled"]["LDL_16B"], 1)
        self.assertEqual(c["diagonal_loop_body"], {})

    def test_per_iteration_weights_peeled_copies_by_entries(self):
        case = {"seq_len": 256, "batch": 1, "query_heads": 1}
        config = {"block_m": 128, "block_n": 64}
        self.assertEqual(s.trips(case, config), {"main": (2, 1), "diagonal": (4, 2)})
        c = s.count(s.parse_sass(SASS, SITES, SRC))
        est = s.per_iteration(c, case, config)
        self.assertEqual(est["main"], (1 * (2 - 1) + 1 * 1) / 2)
        self.assertEqual(est["diagonal"], (0 * 2 + 1 * 2) / 4)


if __name__ == "__main__":
    unittest.main()
