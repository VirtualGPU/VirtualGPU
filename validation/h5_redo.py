"""H5 on GB10, both forms redone under the long-launch protocol
(docs/results/analysis/v0.3-gb10-h5-redo-registration.md).

`microbench/long_capture.py pilot BINARY RUN_DIR --modes h5-dram h5-staging` sizes every point;
then, before the capture, `freeze` predicts every point at its frozen length from
profiles/gb10-0.23.0.json and the installed core, and lists the exclusions: a point that does
not fit its CTAs per SM in the pilot, or whose kernel has a stack frame or local memory in
`cuobjdump -res-usage BINARY | c++filt`. After `capture` and `reduce`, `score` judges each form
on the merged raws (sample lists are the 5 round medians; a point's value is their median) with
the pass rules of validation/h5_dram_form.py and validation/h5_staging_k1.py, unchanged.

    python -m validation.h5_redo freeze RUN_DIR RESOURCES.txt FROZEN.json
    python -m validation.h5_redo score FROZEN.json RUN_DIR RESULT.json
"""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from validation import h5_candidates as h5
from validation import h5_dram_form as dram
from validation import h5_staging_forms as sf
from validation import h5_staging_k1 as k1

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "gb10-0.23.0.json"
ROUNDS = 5
VARIANT_INDEX = {name: i for i, name in enumerate(sf.VARIANTS)}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def kernel_resources(text):
    """{(kernel, template arguments): (stack bytes, local bytes)} from demangled
    `cuobjdump -res-usage` output, for the three narrow kernels."""
    out = {}
    pattern = r"Function [^\n]*?(gmem_narrow_kernel|sync_loop_narrow_kernel|stage_batched_kernel)<([^>]*)>"
    for match in re.finditer(pattern + r"[^\n]*\n([^\n]*)", text):
        usage = match.group(3)
        stack = int(re.search(r"STACK:(\d+)", usage).group(1))
        local = int(re.search(r"LOCAL:(\d+)", usage).group(1))
        args = tuple(a.strip() for a in match.group(2).split(","))
        out[(match.group(1), args)] = (stack, local)
    return out


def gmem_key(row):
    registered = row["registered_request_bytes"] or dram.DEFAULT_DRAM_BYTES
    return dram.gmem_point(row["width_bytes"], row["loads_in_flight"], row["level"],
                           row["aligned"], row["ctas_per_sm"], row["num_warps"], registered)


def sync_key(row):
    return {"bench": "sync-loop-narrow", "width_bytes": row["width_bytes"], "aligned": row["aligned"],
            "level": "dram", "ctas_per_sm": row["ctas_per_sm"], "num_warps": row["num_warps"],
            "stage_bytes": row["stage_bytes"], "mma_per_warp": row["mma_per_warp"],
            "dram_request_bytes": row["registered_request_bytes"]}


def stage_key(row):
    return {k: row[k] for k in ("variant", "width_bytes", "stage_bytes", "mma_per_warp",
                                "ctas_per_sm", "num_warps")}


def kernel_of(field, row):
    """The narrow kernel instance a point runs, as cuobjdump names its template arguments."""
    w = str(row["width_bytes"])
    if "loads_in_flight" in row:
        return "gmem_narrow_kernel", (w, str(row["loads_in_flight"]), "true", "0")
    if field == "sync_loop_narrow":
        return "sync_loop_narrow_kernel", (w, str(row["stage_bytes"] // 128), "4",
                                           str(row["mma_per_warp"]))
    return "stage_batched_kernel", (str(VARIANT_INDEX[row["variant"]]), w,
                                    str(row["stage_bytes"] // 128), str(row["mma_per_warp"]), "4")


def exclusion(field, row, resources):
    """Why a point is excluded before the capture, or None."""
    reasons = []
    if row["ctas_per_sm_fit"] != row["ctas_per_sm"]:
        reasons.append(f"fits {row['ctas_per_sm_fit']} CTAs per SM, not {row['ctas_per_sm']}")
    instance = kernel_of(field, row)
    if instance not in resources:
        raise ValueError(f"no resource usage for {instance}")
    stack, local = resources[instance]
    if stack or local:
        reasons.append(f"stack frame {stack} B, local memory {local} B")
    return "; ".join(reasons) or None


def dram_prediction(p, g, specs, base, arch):
    ir = h5.kernel_ir(p, specs, g=g)
    ir["metadata"]["target_arch"] = arch
    r = h5.run(ir, base, p["level"])
    n = dram.dram_requests(p, g)
    return r, {"geometry": g, "t_core_ns": r["time_ns"], "dram_requests": n,
               "dram_sector_bytes": n * h5.mean_sectors(p, g) * h5.SECTOR}


def freeze_dram(pilot, specs, base, arch, resources):
    points, order, excluded = {}, [], {}
    for field in ("calibration", "gmem_load_narrow", "sync_loop_narrow"):
        for row in pilot[field]:
            p = sync_key(row) if field == "sync_loop_narrow" else gmem_key(row)
            g = h5.geometry(p, specs)
            if field == "sync_loop_narrow":
                g = {**g, "iters": row["distinct_steps"], "sweeps": 1, "pitch": row["pitch_bytes"]}
            elif p["level"] == "dram":
                g = {**g, "iters": row["batches"], "sweeps": 1, "pitch": row["pitch_bytes"]}
            else:
                g = {**g, "sweeps": row["sweeps"]}
            key = h5.point_key(p)
            reason = exclusion(field, row, resources)
            if reason:
                excluded[key] = reason
            r, points[key] = dram_prediction(p, g, specs, base, arch)
            if not reason and r["ctas_per_sm"] != p["ctas_per_sm"]:
                raise RuntimeError(f"{key}: core places {r['ctas_per_sm']} CTAs per SM")
            points[key]["launch_length"] = row["steps"]
            order.append(key)
    return {"calibration": dram.CALIBRATION, "calibration_runs": ROUNDS, "order": order,
            "points": points, "excluded": excluded}


def freeze_staging(pilot, profile, specs, base, resources):
    smem_bw = profile["measured"]["memory"]["smem_bw_per_sm_gbps"]["value"]
    points, order, excluded = {}, [], {}
    for field in ("calibration", "stage_narrow"):
        for row in pilot[field]:
            p = stage_key(row)
            g = h5.geometry(sf.as_sync_loop(p), specs)
            if g["iters"] != row["distinct_steps"]:
                raise ValueError(f"{sf.key(p)}: {g['iters']} distinct stages, benchmark {row['distinct_steps']}")
            g = {**g, "sweeps": row["stage_steps"] // g["iters"]}
            key = sf.key(p)
            reason = exclusion(field, row, resources)
            if reason:
                excluded[key] = reason
            t_core = k1.run_extrapolated(p, specs, g, base) if not reason else None
            steps = g["iters"] * g["sweeps"]
            stores = p["stage_bytes"] // (32 * p["width_bytes"]) if p["variant"] != "load_only" else 0
            loads = (p["stage_bytes"] // (p["num_warps"] * 32 * p["width_bytes"])
                     if p["variant"] != "store_only" else 0)
            grid = [t_core] * len(sf.Q_GRID)
            if stores and not reason:
                grid = [k1.run_extrapolated(p, specs, g, base, max(p["stage_bytes"], round(stores * smem_bw / q)))
                        for q in sf.Q_GRID]
            points[key] = {"geometry": g, "c0_current": t_core, "b1_grid": grid,
                           "l2_load_steps_x_loads_per_thread": steps * loads, "launch_length": row["steps"]}
            order.append(key)
    return {"calibration_b1": sf.CAL_B1, "calibration_b2": sf.CAL_B2, "calibration_runs": ROUNDS,
            "q_grid": sf.Q_GRID, "smem_bw_gbps": smem_bw, "order": order, "points": points,
            "excluded": excluded}


def freeze(run_dir, resources_path, output):
    from virtualgpu import _core

    raw = PROFILE.read_bytes()
    profile = json.loads(raw)
    specs, base, arch = profile["specs"], raw.decode(), profile["identity"]["arch"]
    resources_text = resources_path.read_text()
    resources = kernel_resources(resources_text)
    pilots = {m: (run_dir / f"pilot-{m}.json").read_bytes() for m in ("h5-dram", "h5-staging")}
    manifest = (run_dir / "freeze-manifest.json").read_bytes()
    frozen = {
        "status": "frozen before the capture",
        "registration": "docs/results/analysis/v0.3-gb10-h5-redo-registration.md",
        "profile": PROFILE.name, "profile_revision": profile["provenance"]["profile_revision"],
        "profile_sha256": sha256(raw), "core": _core.build_info(),
        "freeze_manifest_sha256": sha256(manifest),
        "pilot_sha256": {m: sha256(b) for m, b in pilots.items()},
        "resources_sha256": sha256(resources_text.encode()),
        "local_memory": "none" if not any(s or loc for s, loc in resources.values()) else "some",
        "bw_strided_128_gbps": dram.strided_128_bw(profile),
        "dram": freeze_dram(json.loads(pilots["h5-dram"]), specs, base, arch, resources),
        "staging": freeze_staging(json.loads(pilots["h5-staging"]), profile, specs, base, resources),
    }
    output.write_text(json.dumps(frozen, separators=(",", ":")) + "\n")


def merged_values(run_dir, mode):
    """{point key: round medians} from the merged raw of `mode`."""
    raw = json.loads((run_dir / f"merged-{mode}.json").read_text())
    keys = {"calibration": gmem_key if mode == "h5-dram" else stage_key,
            "gmem_load_narrow": gmem_key, "sync_loop_narrow": sync_key, "stage_narrow": stage_key}
    name = h5.point_key if mode == "h5-dram" else sf.key
    return {name(keys[field](row)): row["launch_ns_samples"]
            for field, rows in raw.items() if isinstance(rows, list) for row in rows}


def judged_modes(state):
    """Per mode, whether its capture qualifies: 5 valid rounds and rule 3 passed."""
    out = {}
    for mode in ("h5-dram", "h5-staging"):
        valid = sum(r["results"].get(mode, {}).get("status") == "valid" for r in state["rounds"])
        rule = state.get("rule_three", {}).get(mode, {}).get("status")
        out[mode] = valid >= ROUNDS and rule == "passed"
    return out


def score_dram(frozen, rounds):
    part = frozen["dram"]
    cal = h5.point_key(part["calibration"])
    measured = {k: statistics.median(v) for k, v in rounds.items() if k not in part["excluded"]}
    shaped = {"points": part["points"], "bw_strided_128_gbps": frozen["bw_strided_128_gbps"],
              "calibration": part["calibration"], "calibration_runs": part["calibration_runs"]}
    return dram.score(shaped, measured, rounds[cal])


def score_staging(frozen, rounds):
    """validation/h5_staging_k1.score with this freeze's exclusions in place of its fixed list."""
    part = frozen["staging"]
    preds = part["points"]
    cal1, cal2 = sf.key(part["calibration_b1"]), sf.key(part["calibration_b2"])
    m1, m2 = statistics.median(rounds[cal1]), statistics.median(rounds[cal2])
    q = sf.solve_q(part["q_grid"], preds[cal1]["b1_grid"], m1)
    tau = (m2 - preds[cal2]["c0_current"]) / preds[cal2]["l2_load_steps_x_loads_per_thread"]
    measured = {k: statistics.median(v) for k, v in rounds.items()}
    new = [k for k in part["order"] if k in measured and k not in (cal1, cal2) and k not in part["excluded"]]
    forms = {"c0_current": lambda k: preds[k]["c0_current"]}
    if q is not None and tau > 0:
        forms["k1_b1_b2"] = lambda k: (sf.interpolate_log(part["q_grid"], preds[k]["b1_grid"], q)
                                       + preds[k]["l2_load_steps_x_loads_per_thread"] * tau)
    errors = {name: {k: f(k) / measured[k] - 1 for k in new} for name, f in forms.items()}
    base16 = [e for k, e in errors["c0_current"].items()
              if json.loads(k)["width_bytes"] == 16 and json.loads(k)["mma_per_warp"] == 8]
    mma8_judged = bool(base16) and abs(statistics.median(base16)) <= sf.PASS_GROUP_MEDIAN
    judged = [k for k in new if mma8_judged or json.loads(k)["mma_per_warp"] == 0]
    no_mma = [k for k in new if json.loads(k)["mma_per_warp"] == 0]
    out = {"q_warp_stores_per_ns_per_sm": q, "tau_ns_per_load": tau,
           "calibration_round_ns": {"b1": rounds[cal1], "b2": rounds[cal2]},
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
    out["passed"] = "k1_b1_b2" in out["forms"] and out["forms"]["k1_b1_b2"]["passed"]
    return out


def score(frozen, run_dir):
    state = json.loads((run_dir / "capture-state.json").read_text())
    qualified = judged_modes(state)
    out = {"qualified": qualified, "rule_three": state.get("rule_three", {})}
    for mode, name, fn in (("h5-dram", "dram_form", score_dram), ("h5-staging", "k1", score_staging)):
        out[name] = fn(frozen, merged_values(run_dir, mode)) if qualified[mode] else {
            "verdict": "not judged (capture not qualified)"}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("run_dir", type=Path)
    f.add_argument("resources", type=Path)
    f.add_argument("output", type=Path)
    s = sub.add_parser("score")
    s.add_argument("frozen", type=Path)
    s.add_argument("run_dir", type=Path)
    s.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.run_dir, args.resources, args.output)
        return
    raw = args.frozen.read_bytes()
    result = {"frozen_sha256": sha256(raw), **score(json.loads(raw), args.run_dir)}
    args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
