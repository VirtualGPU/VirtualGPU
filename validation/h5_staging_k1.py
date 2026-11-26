"""Last H5 register staging round: candidate K1 = b1 + b2, frozen and scored.

K1 is `b1_b2` of `h5_staging_forms` unchanged, with no new constant: Q (warp
shared memory stores per ns per SM) and tau (ns per load from L2 per thread)
come from the same two calibration points, each the median of 5 independent
processes (`vgpu_bench --mode stage-narrow-k1-calibrate`, one before each of
the 5 repeats). The new points (`--mode stage-narrow-k1`) were not measured
before: 4 warps, 6 and 10 KiB stages, 2 and 4 CTAs per SM, 1 to 16 B, the
three variants, MMA chains of 0 and 8.

Pass rule, as before: MAPE under 15% and every width's median error within
10%; 8-MMA points count only if the current model's median error on the 16 B
8-MMA points is within 10%. Reported besides, not judged: MAPE and width
medians over the points without MMA.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from validation import h5_candidates as h5
from validation import h5_staging_forms as sf

CALIBRATION_RUNS = 5
# More than 128 registers per thread with the Spark's CUDA 13.0 ptxas
# (sm_121a), so 4 CTAs of 4 warps do not fit; excluded before measurement.
EXCLUDED = (
    ("load_store", 2, 10240, 0, 4),
    ("load_store", 2, 10240, 8, 4),
    ("load_store", 1, 10240, 0, 4),
    ("load_store", 1, 10240, 8, 4),
)


def points():
    """In the benchmark's order."""
    out = []
    for variant in sf.VARIANTS:
        for width in (1, 2, 4, 16):
            for stage in (6144, 10240):
                for mma in (0, 8):
                    for cps in (2, 4):
                        out.append({"variant": variant, "width_bytes": width, "stage_bytes": stage,
                                    "mma_per_warp": mma, "ctas_per_sm": cps, "num_warps": 4})
    return out


def excluded(p):
    return (p["variant"], p["width_bytes"], p["stage_bytes"], p["mma_per_warp"], p["ctas_per_sm"]) in EXCLUDED


def freeze(output):
    raw = h5.PROFILE.read_bytes()
    profile = json.loads(raw)
    specs = profile["specs"]
    smem_bw = profile["measured"]["memory"]["smem_bw_per_sm_gbps"]["value"]
    base = h5.profile_text("base", None)
    frozen = {"status": "frozen before measurement", "profile_revision": profile["provenance"]["profile_revision"],
              "profile_sha256": hashlib.sha256(raw).hexdigest(), "calibration_b1": sf.CAL_B1,
              "calibration_b2": sf.CAL_B2, "calibration_runs": CALIBRATION_RUNS, "q_grid": sf.Q_GRID,
              "smem_bw_gbps": smem_bw, "excluded": [list(e) for e in EXCLUDED], "order": [], "points": {}}
    for p in [sf.CAL_B1, sf.CAL_B2] + points():
        g = h5.geometry(sf.as_sync_loop(p), specs)
        r = h5.run(sf.kernel_ir(p, specs), base, "l2")
        if r["ctas_per_sm"] != p["ctas_per_sm"]:
            raise RuntimeError(f"{sf.key(p)}: core places {r['ctas_per_sm']} CTAs per SM")
        steps = g["iters"] * g["sweeps"]
        stores = p["stage_bytes"] // (32 * p["width_bytes"]) if p["variant"] != "load_only" else 0
        loads = p["stage_bytes"] // (p["num_warps"] * 32 * p["width_bytes"]) if p["variant"] != "store_only" else 0
        b1 = [r["time_ns"]] * len(sf.Q_GRID)
        if stores:
            b1 = [h5.run(sf.kernel_ir(p, specs, max(p["stage_bytes"], round(stores * smem_bw / q))),
                         base, "l2")["time_ns"] for q in sf.Q_GRID]
        frozen["order"].append(sf.key(p))
        frozen["points"][sf.key(p)] = {"geometry": g, "c0_current": r["time_ns"], "b1_grid": b1,
                                       "l2_load_steps_x_loads_per_thread": steps * loads}
    output.write_text(json.dumps(frozen, separators=(",", ":")) + "\n")


def calibration_medians(captures):
    """Median over the dedicated calibration processes of each calibration point."""
    runs = {}
    for capture in captures:
        for run in capture.get("calibration_runs") or []:
            for row in run["result"]:
                p = {k: row[k] for k in ("variant", "width_bytes", "stage_bytes", "mma_per_warp",
                                         "ctas_per_sm", "num_warps")}
                runs.setdefault(sf.key(p), []).append(statistics.median(row["elapsed_ns"]))
    return runs


def score(frozen, measured, cal_runs):
    preds = frozen["points"]
    cal1, cal2 = sf.key(frozen["calibration_b1"]), sf.key(frozen["calibration_b2"])
    for cal in (cal1, cal2):
        if len(cal_runs.get(cal, [])) != frozen["calibration_runs"]:
            raise ValueError(f"expected {frozen['calibration_runs']} calibration runs of {cal}")
    m1, m2 = statistics.median(cal_runs[cal1]), statistics.median(cal_runs[cal2])
    q = sf.solve_q(frozen["q_grid"], preds[cal1]["b1_grid"], m1)
    tau = (m2 - preds[cal2]["c0_current"]) / preds[cal2]["l2_load_steps_x_loads_per_thread"]
    new = [k for k in frozen["order"] if k in measured and k not in (cal1, cal2)
           and not excluded(json.loads(k))]
    forms = {"c0_current": lambda k: preds[k]["c0_current"]}
    if q is not None and tau > 0:
        forms["k1_b1_b2"] = lambda k: (sf.interpolate_log(frozen["q_grid"], preds[k]["b1_grid"], q)
                                       + preds[k]["l2_load_steps_x_loads_per_thread"] * tau)
    errors = {name: {k: f(k) / measured[k] - 1 for k in new} for name, f in forms.items()}
    base16 = [e for k, e in errors["c0_current"].items()
              if json.loads(k)["width_bytes"] == 16 and json.loads(k)["mma_per_warp"] == 8]
    mma8_judged = bool(base16) and abs(statistics.median(base16)) <= sf.PASS_GROUP_MEDIAN
    judged = [k for k in new if mma8_judged or json.loads(k)["mma_per_warp"] == 0]
    no_mma = [k for k in new if json.loads(k)["mma_per_warp"] == 0]
    out = {"q_warp_stores_per_ns_per_sm": q, "tau_ns_per_load": tau,
           "calibration_run_ns": {"b1": cal_runs[cal1], "b2": cal_runs[cal2]},
           "mma8_c0_16b_median_error": statistics.median(base16) if base16 else None,
           "mma8_judged": mma8_judged, "forms": {}}
    for name, errs in errors.items():
        j = sf.summarize({k: errs[k] for k in judged})
        out["forms"][name] = {
            "judged": j,
            "passed": j["mape"] is not None and j["mape"] < sf.PASS_MAPE
            and all(abs(v) <= sf.PASS_GROUP_MEDIAN for v in j["median_by_width"].values()),
            "descriptive_without_mma": sf.summarize({k: errs[k] for k in no_mma}),
            "errors": {k: round(e, 4) for k, e in errs.items()},
        }
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
    raws = [path.read_bytes() for path in args.captures]
    captures = [json.loads(r) for r in raws]
    result = score(json.loads(frozen_raw), sf.measured_points(captures), calibration_medians(captures))
    result = {"status": "post-hoc", "frozen_sha256": hashlib.sha256(frozen_raw).hexdigest(),
              "captures": {p.name: hashlib.sha256(r).hexdigest() for p, r in zip(args.captures, raws)},
              **result}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Q={result['q_warp_stores_per_ns_per_sm']} tau={result['tau_ns_per_load']} "
          f"mma8_judged={result['mma8_judged']} ({result['mma8_c0_16b_median_error']})")
    for name, r in result["forms"].items():
        d = r["descriptive_without_mma"]
        print(f"{name}: judged n={r['judged']['points']} MAPE {r['judged']['mape']:.1%} passed={r['passed']}; "
              f"without MMA n={d['points']} MAPE {d['mape']:.1%} {d['median_by_width']}")


if __name__ == "__main__":
    main()
