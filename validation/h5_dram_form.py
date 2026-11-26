"""Frozen DRAM-side narrow request form (H5 follow-up) and its scoring.

Form A: T = max(T_core, S * 32 B / BW, N / R), where T_core is the current
core's prediction, S the 32 B sectors that DRAM-sourced warp loads touch, N
the number of those warp loads, BW the largest bandwidth of the profile's
strided DRAM curve with 128 B segments, and R, the only constant, a GPU-wide
rate of DRAM warp requests, calibrated at one point.

The form was proposed after the gmem-narrow and sync-loop-narrow points were
seen, so it is validated only on new points (`vgpu_bench --mode
gmem-narrow-validate` and `sync-loop-narrow-validate`), predicted here before
they are measured. Hot-L2 points have no DRAM-sourced loads, so the form leaves
their prediction at T_core; they are measured as a regression check.

`freeze` writes the predictions; `score` reads captures from
`microbench/run_narrow.py`.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validation import h5_candidates as h5
from validation.h5_evaluate import check_geometry

DEFAULT_DRAM_BYTES = 256 << 20
CALIBRATION = {"bench": "gmem-narrow", "width_bytes": 1, "mask": "none", "aligned": True,
               "cache": "cg", "level": "dram", "ctas_per_sm": 2, "num_warps": 8,
               "loads_in_flight": 16, "share": 1, "column_step_bytes": 32,
               "dram_request_bytes": DEFAULT_DRAM_BYTES}
PASS_MAPE = 0.15
PASS_GROUP_MEDIAN = 0.10


def gmem_point(width, depth, level, aligned, cps, warps, dram_bytes=DEFAULT_DRAM_BYTES):
    return {"bench": "gmem-narrow", "width_bytes": width, "mask": "none", "aligned": aligned,
            "cache": "cg", "level": level, "ctas_per_sm": cps, "num_warps": warps,
            "loads_in_flight": depth, "share": 1, "column_step_bytes": 32 * width,
            "dram_request_bytes": dram_bytes}


def gmem_points():
    """In the benchmark's order, the calibration point first."""
    points = [CALIBRATION]
    for width in (1, 2, 4, 16):
        for depth in (4, 12, 24):
            for mib in (128, 512):
                for aligned in (True, False):
                    for cps in (1, 3):
                        for warps in (4, 8):
                            points.append(gmem_point(width, depth, "dram", aligned, cps, warps,
                                                     mib << 20))
            for aligned in (True, False):
                for cps in (1, 3):
                    for warps in (4, 8):
                        points.append(gmem_point(width, depth, "l2", aligned, cps, warps))
    return points


def sync_points():
    points = []
    for width, rows, mma in ((2, 16, 0), (2, 16, 16), (2, 96, 0), (2, 96, 16),
                             (16, 16, 0), (16, 16, 16), (16, 96, 0), (16, 96, 16)):
        for aligned in ((True, False) if width == 2 else (True,)):
            for cps in (1, 2, 4):
                points.append({"bench": "sync-loop-narrow", "width_bytes": width, "aligned": aligned,
                               "level": "dram", "ctas_per_sm": cps, "num_warps": 4,
                               "stage_bytes": rows * 128, "mma_per_warp": mma,
                               "dram_request_bytes": DEFAULT_DRAM_BYTES})
    return points


def geometry(p, specs):
    g = h5.geometry(p, specs)
    if p["bench"] == "gmem-narrow" and p["level"] == "dram":
        seg = 32 * p["width_bytes"]
        iters = max(1, p["dram_request_bytes"] // (g["grid"] * g["rows_per_cta"] * seg))
        g = {**g, "iters": iters, "sweeps": 1,
             "pitch": h5.round_up(iters * seg + seg, h5.ROW_ALIGN) + (0 if p["aligned"] else p["width_bytes"])}
    return g


def dram_requests(p, g):
    if p["level"] != "dram":
        return 0
    return g["grid"] * p["num_warps"] * h5.requests_per_warp_iteration(p, g) * g["iters"] * g["sweeps"]


def strided_128_bw(profile):
    """GB/s: the largest point of every strided DRAM curve with 128 B segments."""
    return max(q["bw_gbps"]["value"] for q in profile["measured"]["memory"]["dram_bw_by_bytes_in_flight_strided"]
               if q["segment_bytes"] == 128)


def freeze(output):
    raw = h5.PROFILE.read_bytes()
    profile = json.loads(raw)
    specs = profile["specs"]
    base = h5.profile_text("base", None)
    points = gmem_points() + sync_points()
    frozen = {"status": "frozen before measurement", "profile_revision": profile["provenance"]["profile_revision"],
              "profile_sha256": hashlib.sha256(raw).hexdigest(), "calibration": CALIBRATION,
              "bw_strided_128_gbps": strided_128_bw(profile), "order": [], "points": {}}
    for p in points:
        g = geometry(p, specs)
        r = h5.run(h5.kernel_ir(p, specs, g=g), base, p["level"])
        if r["ctas_per_sm"] != p["ctas_per_sm"]:
            raise RuntimeError(f"{h5.point_key(p)}: core places {r['ctas_per_sm']} CTAs per SM")
        n = dram_requests(p, g)
        key = h5.point_key(p)
        frozen["order"].append(key)
        frozen["points"][key] = {"geometry": g, "t_core_ns": r["time_ns"], "dram_requests": n,
                                 "dram_sector_bytes": n * h5.mean_sectors(p, g) * h5.SECTOR}
    output.write_text(json.dumps(frozen, separators=(",", ":")) + "\n")


def predict(pred, bw_gbps, rate):
    return max(pred["t_core_ns"], pred["dram_sector_bytes"] / bw_gbps, pred["dram_requests"] / rate)


def measured_points(captures, frozen):
    repeats = {}
    for capture in captures:
        bench = "gmem-narrow" if capture["mode"].startswith("gmem") else "sync-loop-narrow"
        fields = ("width_bytes", "mask", "aligned", "cache", "level", "ctas_per_sm", "num_warps",
                  "loads_in_flight", "share", "column_step_bytes", "dram_request_bytes") \
            if bench == "gmem-narrow" else \
            ("width_bytes", "aligned", "level", "ctas_per_sm", "num_warps", "stage_bytes",
             "mma_per_warp")
        for rep in capture["repeats"]:
            for row in rep["result"]:
                # sync-loop-narrow always requests the default volume and does not report it.
                p = {"bench": bench, "dram_request_bytes": DEFAULT_DRAM_BYTES,
                     **{k: row[k] for k in fields}}
                key = h5.point_key(p)
                check_geometry(bench, row, frozen["points"][key]["geometry"])
                repeats.setdefault(key, []).append(statistics.median(row["elapsed_ns"]))
    return {k: statistics.median(v) for k, v in repeats.items()}


def group_summary(errors, by):
    groups = {}
    for key, e in errors.items():
        groups.setdefault(by(json.loads(key)), []).append(e)
    return {str(k): {"points": len(v), "median_error": statistics.median(v)} for k, v in sorted(groups.items())}


def score(frozen, measured):
    preds, bw = frozen["points"], frozen["bw_strided_128_gbps"]
    cal = h5.point_key(frozen["calibration"])
    c = preds[cal]
    floor = max(c["t_core_ns"], c["dram_sector_bytes"] / bw)
    identifiable = measured[cal] > floor
    rate = c["dram_requests"] / measured[cal] if identifiable else None
    out = {"calibration_measured_ns": measured[cal], "calibration_floor_ns": floor,
           "identifiable": identifiable, "rate_warp_requests_per_ns": rate, "groups": {}}
    groups = {
        "gmem_dram": lambda p: p["bench"] == "gmem-narrow" and p["level"] == "dram",
        "sync_dram": lambda p: p["bench"] == "sync-loop-narrow",
        "gmem_l2_regression": lambda p: p["level"] == "l2",
    }
    for name, member in groups.items():
        keys = [k for k in measured if k != cal and member(json.loads(k))]
        if not keys:
            out["groups"][name] = {"points": 0, "prediction_unchanged": True}
            continue
        form = {k: (predict(preds[k], bw, rate) if rate else preds[k]["t_core_ns"]) / measured[k] - 1 for k in keys}
        core = {k: preds[k]["t_core_ns"] / measured[k] - 1 for k in keys}
        out["groups"][name] = {
            "points": len(keys),
            "form_mape": statistics.mean(abs(e) for e in form.values()),
            "core_mape": statistics.mean(abs(e) for e in core.values()),
            "form_by_width_alignment": group_summary(form, lambda p: (p["width_bytes"], p["aligned"])),
            "core_by_width_alignment": group_summary(core, lambda p: (p["width_bytes"], p["aligned"])),
            "prediction_unchanged": all(abs(form[k] - core[k]) < 1e-12 for k in keys),
            "form_errors": form,
        }
    g = out["groups"]["gmem_dram"]
    out["passed"] = bool(identifiable and g["form_mape"] < PASS_MAPE
                         and all(abs(v["median_error"]) <= PASS_GROUP_MEDIAN
                                 for v in g["form_by_width_alignment"].values())
                         and out["groups"]["gmem_l2_regression"]["prediction_unchanged"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("output", type=Path)
    s = sub.add_parser("score")
    s.add_argument("frozen", type=Path)
    s.add_argument("output", type=Path)
    s.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
        return
    frozen_raw = args.frozen.read_bytes()
    frozen = json.loads(frozen_raw)
    raws = [path.read_bytes() for path in args.captures]
    result = score(frozen, measured_points([json.loads(r) for r in raws], frozen))
    result = {"status": "post-hoc", "frozen_sha256": hashlib.sha256(frozen_raw).hexdigest(),
              "captures": {p.name: hashlib.sha256(r).hexdigest() for p, r in zip(args.captures, raws)},
              **result}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, g in result["groups"].items():
        if g["points"]:
            print(f"{name}: n={g['points']} form MAPE {g['form_mape']:.1%} "
                  f"core MAPE {g['core_mape']:.1%} unchanged={g['prediction_unchanged']}")
    print(f"R={result['rate_warp_requests_per_ns']} identifiable={result['identifiable']} passed={result['passed']}")


if __name__ == "__main__":
    main()
