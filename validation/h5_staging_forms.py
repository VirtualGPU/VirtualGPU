"""Frozen register staging forms (H5 follow-up) and their scoring.

Proposed from the `stage-narrow` and `stage-narrow-batched` points, so they
are validated only on new points (`vgpu_bench --mode stage-narrow-validate`:
8 warps, 8 and 12 KiB stages, 1 to 16 B, MMA chains of 0 and 8, 1 and 3 CTAs
per SM), predicted here before measurement.

- `c0_current`: the core as it is.
- `b1_store_issue`: shared memory stores cost per warp instruction on the
  shared memory pipe: the stage's store bytes are charged as
  max(bytes, store instructions x smem bandwidth / Q), Q (warp stores per ns
  per SM) the constant.
- `b2_load_chain`: loads served from L2 add a fixed time per load on each
  CTA's chain: T = T_core + steps x loads per thread per step x tau, tau the
  constant. Loads from DRAM get nothing (the DRAM-side form covers them).
- `b1_b2`: both, each constant from its own calibration point.

Each constant comes from one calibration point, remeasured in the same
session: Q where stores alone fill the shared pipe, tau where loads alone
bound one CTA's chain.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from validation import h5_candidates as h5

VARIANTS = ("load_only", "store_only", "load_store")
CAL_B1 = {"variant": "store_only", "width_bytes": 2, "stage_bytes": 16384, "mma_per_warp": 0,
          "ctas_per_sm": 3, "num_warps": 4}
CAL_B2 = {"variant": "load_only", "width_bytes": 2, "stage_bytes": 4096, "mma_per_warp": 0,
          "ctas_per_sm": 1, "num_warps": 4}
Q_GRID = [0.05 * 2 ** (i / 2) for i in range(17)]
# Points whose kernel needs more than 85 registers per thread (ptxas,
# sm_121a), so 3 CTAs of 8 warps do not fit; excluded before measurement.
EXCLUDED = (
    ("load_only", 2, 12288, 8, 3),
    ("load_only", 1, 12288, 0, 3),
    ("load_only", 1, 12288, 8, 3),
    ("load_store", 1, 12288, 0, 3),
    ("load_store", 1, 12288, 8, 3),
)
PASS_MAPE = 0.15
PASS_GROUP_MEDIAN = 0.10


def key(p):
    return json.dumps(p, sort_keys=True)


def points():
    """In the benchmark's order, the two calibration points first."""
    out = [CAL_B1, CAL_B2]
    for variant in VARIANTS:
        for width in (1, 2, 4, 16):
            for stage in (8192, 12288):
                for mma in (0, 8):
                    for cps in (1, 3):
                        out.append({"variant": variant, "width_bytes": width, "stage_bytes": stage,
                                    "mma_per_warp": mma, "ctas_per_sm": cps, "num_warps": 8})
    return out


def excluded(p):
    return (p["variant"], p["width_bytes"], p["stage_bytes"], p["mma_per_warp"], p["ctas_per_sm"]) in EXCLUDED


def as_sync_loop(p):
    return {"bench": "sync-loop-narrow", "width_bytes": p["width_bytes"], "aligned": True,
            "level": "l2", "ctas_per_sm": p["ctas_per_sm"], "num_warps": p["num_warps"],
            "stage_bytes": p["stage_bytes"], "mma_per_warp": p["mma_per_warp"]}


def kernel_ir(p, specs, store_bytes=None):
    q = as_sync_loop(p)
    ir = h5.kernel_ir(q, specs)
    ops = ir["body"]["pipeline_loop"]["ops"]
    if p["variant"] == "load_only":
        ops[:] = [o for o in ops if o["kind"] != "smem_store"]
    if p["variant"] == "store_only":
        ops[:] = [o for o in ops if o["kind"] != "gmem_load"]
    if store_bytes is not None:
        for o in ops:
            if o["kind"] == "smem_store":
                o["bytes_per_cta"] = store_bytes
    return ir


def freeze(output):
    raw = h5.PROFILE.read_bytes()
    profile = json.loads(raw)
    specs = profile["specs"]
    smem_bw = profile["measured"]["memory"]["smem_bw_per_sm_gbps"]["value"]
    base = h5.profile_text("base", None)
    frozen = {"status": "frozen before measurement", "profile_revision": profile["provenance"]["profile_revision"],
              "profile_sha256": hashlib.sha256(raw).hexdigest(), "calibration_b1": CAL_B1,
              "calibration_b2": CAL_B2, "q_grid": Q_GRID, "smem_bw_gbps": smem_bw,
              "excluded": [list(e) for e in EXCLUDED], "order": [], "points": {}}
    for p in points():
        g = h5.geometry(as_sync_loop(p), specs)
        r = h5.run(kernel_ir(p, specs), base, "l2")
        if r["ctas_per_sm"] != p["ctas_per_sm"] and not excluded(p):
            raise RuntimeError(f"{key(p)}: core places {r['ctas_per_sm']} CTAs per SM")
        steps = g["iters"] * g["sweeps"]
        stores = p["stage_bytes"] // (32 * p["width_bytes"]) if p["variant"] != "load_only" else 0
        loads = p["stage_bytes"] // (p["num_warps"] * 32 * p["width_bytes"]) if p["variant"] != "store_only" else 0
        b1 = [r["time_ns"]] * len(Q_GRID)
        if stores:
            b1 = [h5.run(kernel_ir(p, specs, max(p["stage_bytes"], round(stores * smem_bw / q))),
                         base, "l2")["time_ns"] for q in Q_GRID]
        frozen["order"].append(key(p))
        frozen["points"][key(p)] = {"geometry": g, "c0_current": r["time_ns"], "b1_grid": b1,
                                    "l2_load_steps_x_loads_per_thread": steps * loads}
    output.write_text(json.dumps(frozen, separators=(",", ":")) + "\n")


def interpolate_log(xs, ys, x):
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (math.log(x) - math.log(xs[i])) / (math.log(xs[i + 1]) - math.log(xs[i]))
            return math.exp(math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i])))
    raise ValueError("outside the grid")


def solve_q(xs, ys, target):
    """Q whose interpolated prediction equals `target`, or None outside the grid."""
    for i in range(len(xs) - 1):
        lo, hi = sorted((ys[i], ys[i + 1]))
        if lo <= target <= hi and ys[i] != ys[i + 1]:
            t = (math.log(target) - math.log(ys[i])) / (math.log(ys[i + 1]) - math.log(ys[i]))
            return math.exp(math.log(xs[i]) + t * (math.log(xs[i + 1]) - math.log(xs[i])))
    return None


def measured_points(captures):
    repeats = {}
    for capture in captures:
        for rep in capture["repeats"]:
            for row in rep["result"]:
                p = {k: row[k] for k in ("variant", "width_bytes", "stage_bytes", "mma_per_warp",
                                         "ctas_per_sm", "num_warps")}
                if row["ctas_per_sm_fit"] != row["ctas_per_sm"]:
                    continue
                repeats.setdefault(key(p), []).append(statistics.median(row["elapsed_ns"]))
    return {k: statistics.median(v) for k, v in repeats.items()}


def summarize(errors):
    values = [abs(e) for e in errors.values()]
    by_width = {}
    for k, e in errors.items():
        by_width.setdefault(json.loads(k)["width_bytes"], []).append(e)
    by_variant = {}
    for k, e in errors.items():
        by_variant.setdefault(json.loads(k)["variant"], []).append(e)
    return {"points": len(values), "mape": statistics.mean(values) if values else None,
            "median_by_width": {str(w): statistics.median(v) for w, v in sorted(by_width.items())},
            "median_by_variant": {v: statistics.median(es) for v, es in sorted(by_variant.items())}}


def score(frozen, measured):
    preds = frozen["points"]
    cal1, cal2 = key(frozen["calibration_b1"]), key(frozen["calibration_b2"])
    q = solve_q(frozen["q_grid"], preds[cal1]["b1_grid"], measured[cal1])
    c2 = preds[cal2]
    tau = (measured[cal2] - c2["c0_current"]) / c2["l2_load_steps_x_loads_per_thread"]
    if tau <= 0:
        tau = None
    new = [k for k in frozen["order"] if k in measured and k not in (cal1, cal2)
           and not excluded(json.loads(k))]

    def b1(k):
        return interpolate_log(frozen["q_grid"], preds[k]["b1_grid"], q)

    def b2(k, t_core):
        return t_core + preds[k]["l2_load_steps_x_loads_per_thread"] * tau

    forms = {"c0_current": lambda k: preds[k]["c0_current"]}
    if q is not None:
        forms["b1_store_issue"] = b1
    if tau is not None:
        forms["b2_load_chain"] = lambda k: b2(k, preds[k]["c0_current"])
    if q is not None and tau is not None:
        forms["b1_b2"] = lambda k: b2(k, b1(k))
    errors = {name: {k: f(k) / measured[k] - 1 for k in new} for name, f in forms.items()}
    base16 = [e for k, e in errors["c0_current"].items()
              if json.loads(k)["width_bytes"] == 16 and json.loads(k)["mma_per_warp"] == 8]
    mma8_judged = bool(base16) and abs(statistics.median(base16)) <= PASS_GROUP_MEDIAN
    judged = [k for k in new if mma8_judged or json.loads(k)["mma_per_warp"] == 0]
    out = {"q_warp_stores_per_ns_per_sm": q, "tau_ns_per_load": tau,
           "calibration_measured_ns": {"b1": measured[cal1], "b2": measured[cal2]},
           "mma8_c0_16b_median_error": statistics.median(base16) if base16 else None,
           "mma8_judged": mma8_judged, "forms": {}}
    for name, errs in errors.items():
        j = summarize({k: errs[k] for k in judged})
        out["forms"][name] = {"judged": j, "all_new_points": summarize(errs),
                              "passed": j["mape"] is not None and j["mape"] < PASS_MAPE
                              and all(abs(v) <= PASS_GROUP_MEDIAN for v in j["median_by_width"].values()),
                              "errors": {k: round(e, 4) for k, e in errs.items()}}
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
    result = score(json.loads(frozen_raw), measured_points([json.loads(r) for r in raws]))
    result = {"status": "post-hoc", "frozen_sha256": hashlib.sha256(frozen_raw).hexdigest(),
              "captures": {p.name: hashlib.sha256(r).hexdigest() for p, r in zip(args.captures, raws)},
              **result}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Q={result['q_warp_stores_per_ns_per_sm']} tau={result['tau_ns_per_load']} "
          f"mma8_judged={result['mma8_judged']} ({result['mma8_c0_16b_median_error']})")
    for name, r in result["forms"].items():
        print(f"{name}: judged n={r['judged']['points']} MAPE {r['judged']['mape']:.1%} passed={r['passed']}")


if __name__ == "__main__":
    main()
