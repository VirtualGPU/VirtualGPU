"""Score the frozen H5 candidate forms against measured narrow-load points.

Inputs are `microbench/run_narrow.py` captures. A point's measurement is the
median over repeats of each repeat's median sample. The geometry the binary
reports must equal the geometry the predictions were made for.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validation import h5_candidates

B1_FIELDS = ("width_bytes", "mask", "aligned", "cache", "level", "ctas_per_sm", "num_warps",
             "loads_in_flight", "share", "column_step_bytes")
B2_FIELDS = ("width_bytes", "aligned", "level", "ctas_per_sm", "num_warps", "stage_bytes",
             "mma_per_warp")
FACTORS = ("bench", "width_bytes", "aligned", "mask", "cache", "level", "ctas_per_sm",
           "num_warps", "loads_in_flight", "column_step_bytes", "share", "stage_bytes",
           "mma_per_warp")


def point_of(bench, row):
    fields = B1_FIELDS if bench == "gmem-narrow" else B2_FIELDS
    return {"bench": bench, **{k: row[k] for k in fields}}


def check_geometry(bench, row, g):
    if row["ctas_per_sm_fit"] != row["ctas_per_sm"] or row["grid"] != g["grid"]:
        raise ValueError(f"{row}: occupancy or grid differs")
    if row["pitch_bytes"] != g["pitch"]:
        raise ValueError(f"{row}: pitch differs from the prediction")
    if bench == "gmem-narrow":
        same = (row["batches"], row["sweeps"]) == (g["iters"], g["sweeps"])
    else:
        same = (row["distinct_steps"], row["steps"]) == (g["iters"], g["iters"] * g["sweeps"])
    if not same:
        raise ValueError(f"{row}: iterations differ from the prediction")


def measurements(captures, frozen):
    repeats = {}
    for capture in captures:
        bench = capture["mode"]
        for rep in capture["repeats"]:
            for row in rep["result"]:
                key = h5_candidates.point_key(point_of(bench, row))
                check_geometry(bench, row, frozen["points"][key]["geometry"])
                repeats.setdefault(key, []).append(statistics.median(row["elapsed_ns"]))
    return ({k: statistics.median(v) for k, v in repeats.items()},
            {k: (max(v) - min(v)) / statistics.median(v) for k, v in repeats.items()})


def rounded(x):
    return None if x is None else round(x, 4)


def summarize(errors):
    values = sorted(abs(e) for e in errors.values())
    return {"points": len(values), "mape": statistics.mean(values),
            "p90_abs_error": values[int(0.9 * (len(values) - 1))],
            "median_error": statistics.median(errors.values())}


def by_factor(errors):
    out = {}
    for factor in FACTORS:
        groups = {}
        for key, e in errors.items():
            value = json.loads(key).get(factor)
            if value is not None:
                groups.setdefault(str(value), []).append(e)
        out[factor] = {v: {"points": len(es), "median_error": statistics.median(es)}
                       for v, es in sorted(groups.items())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    frozen_raw = args.frozen.read_bytes()
    frozen = json.loads(frozen_raw)
    raws = [path.read_bytes() for path in args.captures]
    captures = [json.loads(raw) for raw in raws]
    measured, spread = measurements(captures, frozen)
    result = h5_candidates.evaluate(frozen, measured)
    report = {
        "status": "post-hoc",
        "frozen_sha256": hashlib.sha256(frozen_raw).hexdigest(),
        "captures": {path.name: hashlib.sha256(raw).hexdigest()
                     for path, raw in zip(args.captures, raws)},
        "measured_points": len(measured),
        "max_repeat_spread": max(spread.values()),
        "forms": {},
        "points": {key: {"measured_ns": round(measured[key], 1),
                         "repeat_spread": round(spread[key], 4),
                         "error": {form: rounded(r["errors"].get(key)) for form, r in result.items()}}
                   for key in frozen["order"] if key in measured},
    }
    for form, r in result.items():
        report["forms"][form] = {"k": r["k"], "clamped": r["clamped"],
                                 "calibration": frozen["calibrations"].get(form),
                                 **summarize(r["errors"]), "by_factor": by_factor(r["errors"])}
        s = report["forms"][form]
        print(f"{form:20s} k={s['k']} clamped={s['clamped']} n={s['points']} "
              f"MAPE {s['mape']:.1%} p90 {s['p90_abs_error']:.1%} median {s['median_error']:+.1%}")
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
