"""Frozen candidate predictions for the H5 narrow-load microbenchmarks.

Every point of `vgpu_bench --mode gmem-narrow` and `--mode sync-loop-narrow`
becomes a KernelIR with the benchmark's geometry and is predicted by the
current core and GB10 profile under seven model forms:

- `c0_current`: the core as it is.
- `c1_issue_shared`: one cost per warp load instruction on the shared path
  (`lsu`), at rate `k` (Gwarp-inst/ns per SM).
- `c2_sector_shared`: the same per 32 B sector the request touches, scaled so
  an aligned full-warp 2 B load costs one instruction.
- `c3_issue_chain`: one issue slot per warp load instruction on the CTA's own
  dependent chain and the shared issue pipe, at issue rate `k`.
- `c4_sector_chain`: the same per sector.
- `c5_request_mlp`: the `ld_global_v4` MLP bandwidth scaled by
  `(width / 16) ** k`; `k = 0` is the current byte-indexed curve, `k = 1` a
  request-indexed one.
- `c6_sector_traffic`: loads move every byte of the sectors they touch,
  no parameter.

Each parametrized form gets one constant from one calibration point of its
own (`CALIBRATIONS`, chosen where its mechanism lives) and predicts every
other point. The predictions are frozen on
a grid of `k` before any point is measured; `evaluate` interpolates them
log-linearly at the `k` that matches its calibration point.
"""

import argparse
import copy
import functools
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

from virtualgpu import _core as core

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "gb10.json"
SECTOR = 32
ROW_ALIGN = 128
DRAM_BYTES = 256 << 20
L2_SET = 12 << 20
L2_BYTES = 256 << 20

# Each form is calibrated where its mechanism lives, from one point each:
# the SM-side forms on hot L2 with `.cg` and one line per batch (no L1 reuse),
# the shared-path forms with two CTAs per SM and the chain forms with one,
# the MLP form from DRAM above the knee of the MLP curve (768 KiB in flight).
_SM_SIDE = {"bench": "gmem-narrow", "width_bytes": 2, "mask": "none", "aligned": True,
            "cache": "cg", "level": "l2", "num_warps": 8, "loads_in_flight": 16, "share": 1,
            "column_step_bytes": 128}
CALIBRATIONS = {
    "c1_issue_shared": {**_SM_SIDE, "ctas_per_sm": 2},
    "c2_sector_shared": {**_SM_SIDE, "ctas_per_sm": 2},
    "c3_issue_chain": {**_SM_SIDE, "ctas_per_sm": 1},
    "c4_sector_chain": {**_SM_SIDE, "ctas_per_sm": 1},
    "c5_request_mlp": {**_SM_SIDE, "level": "dram", "ctas_per_sm": 2},
}

GRIDS = {
    "c1_issue_shared": [0.01 * 2 ** (i / 2) for i in range(17)],
    "c2_sector_shared": [0.01 * 2 ** (i / 2) for i in range(17)],
    "c3_issue_chain": [0.02 * 2 ** (i / 2) for i in range(17)],
    "c4_sector_chain": [0.02 * 2 ** (i / 2) for i in range(17)],
    "c5_request_mlp": [i / 10 for i in range(16)],
}


def gmem_narrow_points():
    """In the benchmark's order."""
    points = []
    for width, masks in ((1, ("none", "true")), (2, ("none", "true", "tail")),
                         (4, ("none", "true", "tail")), (16, ("none", "true"))):
        for mask in masks:
            for depth in (8, 16, 32):
                for cache in ("default", "cg"):
                    for level in ("l2", "dram"):
                        for aligned in (True, False):
                            for cps in (1, 2):
                                for warps in (4, 8):
                                    points.append(dict(
                                        bench="gmem-narrow", width_bytes=width, mask=mask,
                                        aligned=aligned, cache=cache, level=level, ctas_per_sm=cps,
                                        num_warps=warps, loads_in_flight=depth, share=1,
                                        column_step_bytes=32 * width))
    for width, masks in ((1, ("none", "true")), (2, ("none", "true", "tail"))):
        for mask in masks:
            for cache in ("default", "cg"):
                for level in ("l2", "dram"):
                    for aligned in (True, False):
                        for cps in (1, 2):
                            for warps in (4, 8):
                                points.append(dict(
                                    bench="gmem-narrow", width_bytes=width, mask=mask,
                                    aligned=aligned, cache=cache, level=level, ctas_per_sm=cps,
                                    num_warps=warps, loads_in_flight=16, share=1,
                                    column_step_bytes=128))
    for aligned in (True, False):
        for cps in (1, 2):
            for width in (2, 16):
                points.append(dict(bench="gmem-narrow", width_bytes=width, mask="none",
                                   aligned=aligned, cache="default", level="dram", ctas_per_sm=cps,
                                   num_warps=8, loads_in_flight=16, share=4,
                                   column_step_bytes=32 * width))
    return points


def sync_loop_narrow_points():
    points = []
    for width in (2, 16):
        for mma in (0, 8, 32):
            for rows in (32, 64, 128):
                for level in ("l2", "dram"):
                    for aligned in ((True, False) if width == 2 else (True,)):
                        for cps in (1, 2, 3):
                            points.append(dict(bench="sync-loop-narrow", width_bytes=width,
                                               aligned=aligned, level=level, ctas_per_sm=cps,
                                               num_warps=4, stage_bytes=rows * 128,
                                               mma_per_warp=mma))
    return points


def point_key(p):
    return json.dumps(p, sort_keys=True)


def round_up(x, a):
    return (x + a - 1) // a * a


def pad_bytes(specs, cps):
    per_sm = specs["shared_mem_per_sm_bytes"]
    reserved = specs["reserved_shared_mem_per_cta_bytes"]
    bytes_ = (per_sm // cps - reserved - (128 if cps > 1 else 0)) // 128 * 128
    return min(bytes_, specs["max_shared_mem_per_block_bytes"])


def sectors_per_request(width, lanes, offset):
    """32 B sectors a warp request touches, from its first byte offset."""
    first = offset % SECTOR
    return math.ceil((first + lanes * width) / SECTOR)


def geometry(p, specs):
    """Grid, rows, row bytes, pitch, iterations, outer sweeps and lanes, as the benchmark computes them."""
    sms = specs["sm_count"]
    w = p["width_bytes"]
    grid = sms * p["ctas_per_sm"]
    if p["bench"] == "gmem-narrow":
        rows_per_cta = p["num_warps"] * p["loads_in_flight"]
        seg = 32 * w
        step = p["column_step_bytes"]
        rows = grid // p["share"] * rows_per_cta
        if p["level"] == "dram":
            iters, sweeps = max(1, DRAM_BYTES // (grid * rows_per_cta * seg)), 1
        else:
            iters = max(1, L2_SET // (rows * step))
            sweeps = max(1, L2_BYTES // (grid * rows_per_cta * seg * iters))
        pitch = round_up(iters * step + seg, ROW_ALIGN) + (0 if p["aligned"] else w)
        lanes = 24 if p["mask"] == "tail" else 32
        return dict(grid=grid, rows_per_cta=rows_per_cta, row_bytes=lanes * w, inner=step,
                    pitch=pitch, iters=iters, sweeps=sweeps, lanes=lanes,
                    smem=pad_bytes(specs, p["ctas_per_sm"]))
    stage = p["stage_bytes"]
    requested = DRAM_BYTES if p["level"] == "dram" else L2_BYTES
    steps = max(1, requested // (grid * stage))
    distinct = steps if p["level"] == "dram" else max(1, L2_SET // (grid * stage))
    pitch = round_up(distinct * 128, ROW_ALIGN) + (0 if p["aligned"] else w)
    return dict(grid=grid, rows_per_cta=stage // 128, row_bytes=128, inner=128, pitch=pitch,
                iters=distinct, sweeps=max(1, steps // distinct), lanes=32,
                smem=max(stage, pad_bytes(specs, p["ctas_per_sm"])))


def requests_per_warp_iteration(p, g):
    if p["bench"] == "gmem-narrow":
        return p["loads_in_flight"]
    return p["stage_bytes"] // (p["num_warps"] * 32 * p["width_bytes"])


def mean_sectors(p, g):
    """Mean sectors per warp request over the rows of one iteration."""
    w, lanes = p["width_bytes"], g["lanes"]
    per_row = g["row_bytes"] // (lanes * w) if p["bench"] == "sync-loop-narrow" else 1
    total = n = 0
    for r in range(g["rows_per_cta"]):
        for j in range(max(1, per_row)):
            total += sectors_per_request(w, lanes, r * g["pitch"] + j * lanes * w)
            n += 1
    return total / n


def kernel_ir(p, specs, alu_per_warp=0, byte_scale=1.0):
    g = geometry(p, specs)
    w = p["width_bytes"]
    entries = []
    for cta in range(g["grid"]):
        band = cta // p.get("share", 1)
        base = band * g["rows_per_cta"] * g["pitch"]
        entries.extend([[cta, s, base, g["rows_per_cta"], g["row_bytes"], g["pitch"], g["inner"], False]
                        for s in range(g["sweeps"])])
    load_bytes = round(g["rows_per_cta"] * g["row_bytes"] * byte_scale)
    ops = [{"access_ref": "acc_X", "kind": "gmem_load", "nominal_bytes_per_cta": load_bytes,
            "op_id": "load_X", "stage_offset": 0, "tensor_id": "X", "transfer_width_bytes": w,
            "warp_group_id": 0}]
    if p["bench"] == "sync-loop-narrow":
        stage = p["stage_bytes"]
        ops += [{"bytes_per_cta": stage, "kind": "smem_store", "op_id": "smem_store_X",
                 "stage_offset": 0, "warp_group_id": 0},
                {"bytes_per_cta": stage, "kind": "smem_load", "op_id": "smem_load_X",
                 "stage_offset": 0, "warp_group_id": 0},
                {"count": 2, "kind": "barrier", "op_id": "stage_sync", "stage_offset": 0,
                 "warp_group_id": 0}]
    # The core needs an MMA; points without one get a single token MMA per
    # CTA and iteration (a few ns against round trips of microseconds).
    mma = p.get("mma_per_warp", 0) * p["num_warps"] or 1
    ops.append({"a_dtype": "f16", "acc_dtype": "f32", "b_dtype": "f16", "count": mma,
                "inst_mnk": [16, 8, 16], "kind": "mma", "op_class": "mma.sync", "op_id": "mma_0",
                "stage_offset": 0, "tile_mnk": [16, 8, 16 * mma], "warp_group_id": 0})
    if alu_per_warp:
        ops.append({"count": alu_per_warp, "dtype": "u32", "kind": "alu", "op_id": "load_issue",
                    "stage_offset": 0, "warp_group_id": 0})
    rows_total = g["grid"] // p.get("share", 1) * g["rows_per_cta"]
    return {
        "schema_version": "0.1",
        "metadata": {"kernel_name": p["bench"].replace("-", "_"), "frontend": "manual",
                     "target_arch": "sm_121", "target_vendor": "nvidia", "config": {}},
        "tensors": [{"id": "X", "dtype": "u8", "element_bits": 8, "shape": [rows_total * g["pitch"]],
                     "strides": [1], "base_addr": 0, "cache_policy": "default"}],
        "launch": {"grid": [g["grid"], 1, 1], "num_warps": p["num_warps"], "warp_size": 32,
                   "cluster_shape": [1, 1, 1], "dynamic_smem_bytes": g["smem"],
                   "warp_groups": [{"id": 0, "num_warps": p["num_warps"], "role": "unified"}]},
        "resources": {"registers_per_thread": 64, "static_smem_bytes": 0, "barriers": 1,
                      "spill_loads_bytes": 0, "spill_stores_bytes": 0, "stack_frame_bytes": 0,
                      "spill_scope": "none", "source": "estimated"},
        "body": {
            "prologue": {"ops": []},
            "pipeline_loop": {"async_copy": "none", "num_stages": 1, "prefetch_distance": 0,
                              "trip_count_mode": "uniform", "trip_counts": g["iters"], "ops": ops},
            "epilogue": {"ops": [], "overlaps_next_tile": False},
            "outer_loop": {"trip_count_mode": "uniform", "trip_counts": g["sweeps"]},
        },
        "access_map": {"acc_X": {"encoding": "strided_2d_per_outer_tile", "entries": entries,
                                 "sidecar_ref": None}},
        "unsupported_features": [],
        "diagnostics": [],
    }


@functools.lru_cache(maxsize=None)
def profile_text(kind, value, width=None):
    """The GB10 profile with one edit, serialized once per distinct edit."""
    base = json.loads(PROFILE.read_text())
    edit = {"base": lambda p: p, "lsu": lambda p: lsu_profile(p, value),
            "issue": lambda p: issue_profile(p, value), "mlp": lambda p: mlp_profile(p, width, value)}
    return json.dumps(edit[kind](base))


def lsu_profile(profile, rate):
    p = copy.deepcopy(profile)
    for point in p["measured"]["memory"]["gmem_load_issue_controlled"]:
        point["issue_rate_gwarp_inst_per_sm"]["value"] = rate
    return p


def issue_profile(profile, rate):
    p = copy.deepcopy(profile)
    p["measured"]["compute"]["non_mma_issue_rate_per_sm_gips"]["value"] = rate
    return p


def mlp_profile(profile, width, exponent):
    p = copy.deepcopy(profile)
    for key in ("dram_bw_by_bytes_in_flight", "l2_bw_by_bytes_in_flight"):
        for point in p["measured"]["memory"][key]:
            if point["issue_mode"] == "ld_global_v4":
                point["bw_gbps"]["value"] *= (width / 16) ** exponent
    return p


def run(ir, profile, level):
    text = profile if isinstance(profile, str) else json.dumps(profile)
    r = dict(core.predict(json.dumps(ir), text, l2_state="warm" if level == "l2" else "cold",
                          draws=0, seed=17))
    if r["status"] != "ok":
        raise RuntimeError(r)
    return r


def predict_point(p, profile):
    specs = profile["specs"]
    g = geometry(p, specs)
    sectors = mean_sectors(p, g)
    reqs = requests_per_warp_iteration(p, g)
    ref = sectors_per_request(2, 32, 0)
    ir = kernel_ir(p, specs)
    lvl = p["level"]
    base = run(ir, profile_text("base", None), lvl)
    if base["ctas_per_sm"] != p["ctas_per_sm"]:
        raise RuntimeError(f"{point_key(p)}: core places {base['ctas_per_sm']} CTAs per SM")
    out = {"geometry": g, "mean_sectors_per_request": sectors, "c0_current": base["time_ns"]}
    out["c1_issue_shared"] = [run(ir, profile_text("lsu", k), lvl)["time_ns"]
                              for k in GRIDS["c1_issue_shared"]]
    out["c2_sector_shared"] = [run(ir, profile_text("lsu", k * ref / sectors), lvl)["time_ns"]
                               for k in GRIDS["c2_sector_shared"]]
    issue_ir = kernel_ir(p, specs, alu_per_warp=reqs)
    out["c3_issue_chain"] = [run(issue_ir, profile_text("issue", k), lvl)["time_ns"]
                             for k in GRIDS["c3_issue_chain"]]
    out["c4_sector_chain"] = [run(issue_ir, profile_text("issue", k * ref / sectors), lvl)["time_ns"]
                              for k in GRIDS["c4_sector_chain"]]
    out["c5_request_mlp"] = [run(ir, profile_text("mlp", k, p["width_bytes"]), lvl)["time_ns"]
                             for k in GRIDS["c5_request_mlp"]]
    scale = sectors * SECTOR / (g["lanes"] * p["width_bytes"])
    out["c6_sector_traffic"] = run(kernel_ir(p, specs, byte_scale=scale), profile_text("base", None),
                                   lvl)["time_ns"]
    return out


def interpolate(xs, ys, x, log_x):
    fx = (lambda v: math.log(v)) if log_x else (lambda v: v)
    for i in range(len(xs) - 1):
        if fx(xs[i]) <= fx(x) <= fx(xs[i + 1]):
            t = (fx(x) - fx(xs[i])) / (fx(xs[i + 1]) - fx(xs[i]))
            return math.exp(math.log(ys[i]) + t * (math.log(ys[i + 1]) - math.log(ys[i])))
    raise ValueError("outside the grid")


def solve(xs, ys, target, log_x):
    """The grid value whose interpolated prediction equals `target`, clamped to the grid."""
    for i in range(len(xs) - 1):
        lo, hi = sorted((ys[i], ys[i + 1]))
        if lo <= target <= hi:
            fx = (lambda v: math.log(v)) if log_x else (lambda v: v)
            gx = (lambda v: math.exp(v)) if log_x else (lambda v: v)
            t = (math.log(target) - math.log(ys[i])) / (math.log(ys[i + 1]) - math.log(ys[i]))
            return gx(fx(xs[i]) + t * (fx(xs[i + 1]) - fx(xs[i]))), False
    return (xs[0] if abs(math.log(target / ys[0])) < abs(math.log(target / ys[-1])) else xs[-1]), True


def form_prediction(pred, form, k):
    if form not in GRIDS:
        return pred[form]
    return interpolate(GRIDS[form], pred[form], k, form != "c5_request_mlp")


def evaluate(frozen, measured):
    """measured: point key to median elapsed ns. Returns per-form calibration and held-out errors."""
    preds = frozen["points"]
    result = {}
    for form in ("c0_current", *GRIDS, "c6_sector_traffic"):
        cal = point_key(frozen["calibrations"][form]) if form in GRIDS else None
        if form in GRIDS:
            log_x = form != "c5_request_mlp"
            k, clamped = solve(GRIDS[form], preds[cal][form], measured[cal], log_x)
        else:
            k, clamped = None, False
        errors = {key: form_prediction(preds[key], form, k) / t - 1
                  for key, t in measured.items() if key != cal}
        result[form] = {"k": k, "clamped": clamped, "errors": errors}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    raw = PROFILE.read_bytes()
    profile = json.loads(raw)
    points = gmem_narrow_points() + sync_loop_narrow_points()
    if not {point_key(c) for c in CALIBRATIONS.values()} <= {point_key(p) for p in points}:
        raise SystemExit("a calibration point is not a benchmark point")
    frozen = {"status": "frozen before measurement", "profile_revision": profile["provenance"]["profile_revision"],
              "profile_sha256": hashlib.sha256(raw).hexdigest(), "calibrations": CALIBRATIONS,
              "grids": GRIDS, "order": [point_key(p) for p in points], "points": {}}
    with multiprocessing.Pool(args.jobs) as pool:
        results = pool.starmap(predict_point, [(p, profile) for p in points], chunksize=4)
    frozen["points"] = {point_key(p): r for p, r in zip(points, results)}
    args.output.write_text(json.dumps(frozen, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
