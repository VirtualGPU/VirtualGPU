"""Post-hoc H5 counterfactuals: 2-byte register-staged loads on GB10 mask-mnk.

Run with the v0.2-final core (rebuilt from the tag) on the archived KernelIR and
the v0.2-final GB10 profile, nominal profile only. Unchanged, it reproduces the
archived nominal predictions exactly; the script checks that before anything else.
"""

import argparse
import copy
import itertools
import json
from pathlib import Path

from virtualgpu import _core as core

CASES = ("v02-gb10-mask-mnk-1153", "v02-gb10-mask-mnk-641")
CACHES = ("warm", "cold")
SM_COUNT = 48
CORE_SOURCE = ("v0.2-final core rebuilt from `git archive v0.2-final` (no git metadata, so its build_info "
               "reads unknown/diagnostic); checked to reproduce every archived nominal prediction exactly")


def kendall_tau_b(a, b):
    concordant = discordant = ties_a = ties_b = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 and db == 0:
            continue
        if da == 0:
            ties_a += 1
        elif db == 0:
            ties_b += 1
        elif da * db > 0:
            concordant += 1
        else:
            discordant += 1
    n0 = concordant + discordant
    return (concordant - discordant) / ((n0 + ties_a) * (n0 + ties_b)) ** 0.5


def top5_regret(predicted, measured):
    picks = sorted(range(len(predicted)), key=predicted.__getitem__)[:5]
    return min(measured[i] for i in picks) / min(measured) - 1


def scale_lsu(profile, factor):
    p = copy.deepcopy(profile)
    for point in p["measured"]["memory"]["gmem_load_issue_controlled"]:
        point["issue_rate_gwarp_inst_per_sm"]["value"] *= factor
    return p


def scale_mlp(profile, factor):
    p = copy.deepcopy(profile)
    for key in ("dram_bw_by_bytes_in_flight", "l2_bw_by_bytes_in_flight"):
        for point in p["measured"]["memory"][key]:
            if point["issue_mode"] == "ld_global_v4":
                point["bw_gbps"]["value"] *= factor
    return p


def scale_l2(profile, factor):
    p = copy.deepcopy(profile)
    p["specs"]["l2_size_bytes"] = int(p["specs"]["l2_size_bytes"] * factor)
    p["measured"]["memory"]["l2_effective_capacity_bytes"]["value"] *= factor
    return p


def load_width(ir, width):
    ir = copy.deepcopy(ir)
    for op in ir["body"]["pipeline_loop"]["ops"]:
        if op["kind"] == "gmem_load":
            op["transfer_width_bytes"] = width
    return ir


COUNTERFACTUALS = {
    "baseline": (None, None),
    "load_width_16": (None, lambda ir: load_width(ir, 16)),
    **{f"lsu_rate_x{f}": (lambda p, f=f: scale_lsu(p, f), None) for f in (0.5, 0.25, 0.15, 0.1)},
    **{f"mlp_ld_global_v4_bw_x{f}": (lambda p, f=f: scale_mlp(p, f), None) for f in (0.5, 0.33, 0.25)},
    "l2_capacity_x0.05": (lambda p: scale_l2(p, 0.05), None),
}


def predict(ir, profile, cache):
    return dict(core.predict(json.dumps(ir), json.dumps(profile), l2_state=cache, draws=0, seed=17))


def mem_warp_insts(ir):
    loop = ir["body"]["pipeline_loop"]
    per_iteration = sum(op["nominal_bytes_per_cta"] / (32 * op["transfer_width_bytes"])
                        for op in loop["ops"] if op["kind"] == "gmem_load")
    return per_iteration * loop["trip_counts"] * ir["launch"]["grid"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="v0.2 formal prediction archive directory")
    parser.add_argument("profile", type=Path, help="v0.2-final profiles/gb10.json")
    parser.add_argument("scores", type=Path, help="validation/v02-holdout-scores.json")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = json.loads((args.archive / "predictions.json").read_text())
    profile = json.loads(args.profile.read_text())
    scores = json.loads(args.scores.read_text())
    out = {"status": "post-hoc", "core_source": CORE_SOURCE, "core_build": None, "cases": {}}
    for case in CASES:
        ir_dir = args.archive / "ir" / "gb10-gemm" / case
        irs = {path.stem: json.loads(path.read_text()) for path in sorted(ir_dir.glob("*.json"))}
        keys = sorted(irs)
        out["cases"][case] = record = {"configs": {}, "counterfactuals": {}}
        for cache in CACHES:
            measured = [scores["cohorts"]["gb10-gemm"]["cases"][case]["cache"][cache]["core"]["configs"][k]
                        ["measured_median_ns"] for k in keys]
            for name, (edit_profile, edit_ir) in COUNTERFACTUALS.items():
                p = edit_profile(profile) if edit_profile else profile
                results = [predict(edit_ir(irs[k]) if edit_ir else irs[k], p, cache) for k in keys]
                predicted = [r["time_ns"] for r in results]
                if name == "baseline":
                    archived = [archive["cohorts"]["gb10-gemm"][case][cache]["core"][k]["nominal_latency_ns"]
                                for k in keys]
                    if predicted != archived:
                        raise SystemExit(f"{case} {cache}: core does not reproduce the archive")
                    out["core_build"] = results[0]["build"]
                    for k, r, m in zip(keys, results, measured):
                        cfg = record["configs"].setdefault(k, {
                            "ctas_per_sm": r["ctas_per_sm"],
                            "load_warp_insts": mem_warp_insts(irs[k]),
                            "spill": "spill" in r["flags"],
                        })
                        cfg[cache] = {
                            "measured_ns": m,
                            "predicted_ns": r["time_ns"],
                            "error": r["time_ns"] / m - 1,
                            "bound": max(r["bound_fraction"].items(), key=lambda t: t[1])[0],
                            "implied_load_issue_gwarp_inst_per_ns_per_sm":
                                mem_warp_insts(irs[k]) / m / SM_COUNT,
                        }
                errors = sorted(t / m - 1 for t, m in zip(predicted, measured))
                record["counterfactuals"].setdefault(name, {})[cache] = {
                    "kendall_tau_b": kendall_tau_b(predicted, measured),
                    "top5_regret": top5_regret(predicted, measured),
                    "error_min": errors[0],
                    "error_median": errors[len(errors) // 2],
                    "error_max": errors[-1],
                    "error_s1_nonspill": {k: t / m - 1 for k, t, m in zip(keys, predicted, measured)
                                          if k.endswith("-s1-g1") and not record["configs"][k]["spill"]},
                }
                print(case, cache, name, flush=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
