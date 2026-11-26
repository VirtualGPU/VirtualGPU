"""Summarize `vgpu_bench --mode stage-narrow` captures (H5, measurement only).

Per point: the median over repeats of each repeat's median sample and the
spread over repeats. No model form is scored here; forms for register staging
are proposed from these points and validated on others.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

FIELDS = ("variant", "width_bytes", "stage_bytes", "mma_per_warp", "ctas_per_sm", "num_warps")


def summarize(captures):
    """Points whose CTAs per SM did not fit as set are excluded and listed."""
    repeats, excluded = {}, {}
    for capture in captures:
        for rep in capture["repeats"]:
            for row in rep["result"]:
                key = json.dumps({k: row[k] for k in FIELDS}, sort_keys=True)
                if row["ctas_per_sm_fit"] != row["ctas_per_sm"]:
                    excluded[key] = row["ctas_per_sm_fit"]
                    continue
                repeats.setdefault(key, []).append(statistics.median(row["elapsed_ns"]))
    points = {k: {"measured_ns": round(statistics.median(v), 1),
                  "repeat_spread": round((max(v) - min(v)) / statistics.median(v), 4)}
              for k, v in repeats.items()}
    return points, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    raws = [path.read_bytes() for path in args.captures]
    points, excluded = summarize([json.loads(r) for r in raws])
    out = {"status": "post-hoc",
           "captures": {p.name: hashlib.sha256(r).hexdigest() for p, r in zip(args.captures, raws)},
           "points": points, "excluded_ctas_per_sm_fit": excluded}
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
