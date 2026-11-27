"""Re-check the core's mechanisms (design.md 6.4) on a GPU profile's own microbenchmark points.

Every point of a mechanism's basis in the profile is written as the synthetic KernelIR of the
kernel that ran (microbench/src/m2_kernels.cuh) and predicted with the installed core, which is
only read: no mechanism or parameter changes here. The per-step model time is the difference of
two launch lengths over their step difference, so launch and CTA dispatch costs drop out, as
they do from the measured per-step values. Per mechanism and profile: points, median absolute
error, MAPE, signed median error and the worst point.

    python scripts/recheck_mechanisms.py --profile old=profiles/gb10.json \\
        --profile new=NEW.json --out OUT.json [--markdown OUT.md]
    python scripts/recheck_mechanisms.py --check BASELINE.json   # after a core change

`--check` requires a clean core built from the current checkout and the exact
compiled-source SHA recorded in the baseline. Docs-only commits keep that SHA.
It then re-runs on the baseline's profiles and checks the point counts and
error tolerance. Refresh the baseline when the compiled inputs change.

What the IR cannot express is left out and listed in NOT_EXPRESSED; it is the same for every
profile, so a column-to-column change comes from the profile alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import subprocess
from pathlib import Path

from virtualgpu import _core

ROOT = Path(__file__).resolve().parents[1]
# The build identity in crates/vgpu-core/build.rs covers these same paths.
COMPILED_PATHS = ("crates", "schemas", "Cargo.toml", "Cargo.lock")
LENGTHS = (256, 512)
FLOPS_PER_MMA = 2 * 16 * 8 * 16
NOT_EXPRESSED = {
    "closed_loop": "compute_ns is a clock64 spin, written as issue work at the CTA's alone rate; "
    "with more than one CTA per SM the model would share that issue work and the spin does "
    "not, so those points (compute_ns > 0 and more CTAs than SMs) are left out",
    "miss_merging": "stagger_stages 16 delays the followers' start once; KernelIR has no per-CTA "
    "start offset, so only stagger 0 is checked (the basis in design.md)",
    "dependent_chain": "instruction counts from the CUDA source, not SASS: per warp per step "
    "2 x mma_per_half MMAs, softmax_ops / 2 ex2, 4 shuffles, softmax_ops + 15 other",
    "pipes": "sfu and shfl rates are the profile's own curves (their check is leave-one-out "
    "interpolation, not a core run) and are not re-run here",
    "cross_wave_reuse": "its microbenchmarks are not in the profile",
    "address_arithmetic": "loop and address instructions are not counted in any kernel",
}


def mma_op(count, stage_offset=0):
    return {
        "kind": "mma",
        "op_id": "mma",
        "op_class": "mma.sync",
        "a_dtype": "f16",
        "b_dtype": "f16",
        "acc_dtype": "f32",
        "inst_mnk": [16, 8, 16],
        "tile_mnk": [16, 8, 16 * count],
        "count": count,
        "stage_offset": stage_offset,
        "warp_group_id": 0,
    }


def counted(kind, op_id, count):
    return {
        "kind": kind,
        "op_id": op_id,
        "dtype": "f32",
        "count": count,
        "stage_offset": 0,
        "warp_group_id": 0,
    }


def smem(kind, nbytes):
    return {
        "kind": kind,
        "op_id": kind,
        "bytes_per_cta": int(nbytes),
        "stage_offset": 0,
        "warp_group_id": 0,
    }


def kernel(
    *,
    grid,
    warps,
    steps,
    smem_bytes=0,
    load=None,
    mma=0,
    alu=0,
    sfu=0,
    shfl=0,
    smem_load=0,
    smem_store=0,
    prefetch=0,
    asynchronous=True,
    spill=None,
):
    """One CTA loop of `steps` iterations. `load` is (bytes per CTA, width, stride per step,
    per-CTA base): stride 0 re-reads the same bytes (L2 resident), a grid-wide stride streams."""
    ops, access, tensors = [], {}, []
    if load:
        nbytes, width, stride, base = load
        ops.append(
            {
                "kind": "gmem_load",
                "op_id": "load",
                "tensor_id": "src",
                "access_ref": "acc",
                "nominal_bytes_per_cta": int(nbytes),
                "transfer_width_bytes": width,
                "stage_offset": prefetch,
                "warp_group_id": 0,
            }
        )
        access["acc"] = {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": [
                [0, b, base(b), 1, int(nbytes), int(nbytes), int(stride), False]
                for b in range(grid)
            ],
        }
        tensors.append(
            {
                "id": "src",
                "dtype": "u8",
                "element_bits": 8,
                "shape": [1 << 40],
                "strides": [1],
                "base_addr": 0,
                "cache_policy": "bypass_l1",
            }
        )
    if smem_load:
        ops.append(smem("smem_load", smem_load))
    if smem_store:
        ops.append(smem("smem_store", smem_store))
    for kind, count in (("alu", alu), ("sfu", sfu), ("shfl", shfl)):
        if count:
            ops.append(counted(kind, kind, int(round(count))))
    loop_mma = int(mma)
    if loop_mma:
        ops.append(mma_op(loop_mma))
    resources = {
        "registers_per_thread": 64,
        "static_smem_bytes": int(smem_bytes),
        "spill_loads_bytes": 0,
        "spill_stores_bytes": 0,
        "stack_frame_bytes": 0,
        "spill_scope": "none",
        "barriers": 1,
        "source": "assembler_report",
    }
    if spill:
        resources.update(spill)
    return {
        "schema_version": "0.1",
        "metadata": {
            "kernel_name": "microbenchmark",
            "frontend": "triton",
            "target_vendor": "nvidia",
            "target_arch": "sm_80",
            "config": {},
        },
        "launch": {
            "grid": [grid, 1, 1],
            "num_warps": warps,
            "warp_size": 32,
            "warp_groups": [{"id": 0, "num_warps": warps, "role": "unified"}],
            "cluster_shape": [1, 1, 1],
            "dynamic_smem_bytes": 0,
        },
        "resources": resources,
        "tensors": tensors,
        "body": {
            # One MMA outside the loop names the MMA family for loops without MMAs; it cancels
            # in the two-length difference.
            "prologue": {"ops": [] if loop_mma else [mma_op(1)]},
            "pipeline_loop": {
                "async_copy": "async_element" if asynchronous else "none",
                "num_stages": prefetch + 1,
                "prefetch_distance": prefetch,
                "trip_count_mode": "uniform",
                "trip_counts": steps,
                "ops": ops,
            },
            "outer_loop": {"trip_count_mode": "uniform", "trip_counts": 1},
            "epilogue": {"ops": [], "overlaps_next_tile": False},
        },
        "access_map": access,
        "diagnostics": [],
        "unsupported_features": [],
    }


def value(entry, name):
    return entry[name]["value"]


def stream(grid, nbytes):
    """A grid-strided stream of `nbytes` per CTA per step: every step reads new bytes."""
    return (nbytes, 16, grid * nbytes, lambda b: b * nbytes)


def resident(nbytes, width=16):
    """The same `nbytes` per CTA every step: L2 resident once warm."""
    return (nbytes, width, 0, lambda b: b * nbytes)


# Each mechanism: (name, design.md 6.4 paragraph, profile section and field, point filter,
# builder(profile, point) -> (kernel kwargs without steps, l2 state), measured(point),
# predicted per-step ns -> the measured quantity, point label).


def closed_loop(p, q, level):
    issue = value(p["measured"]["compute"], "non_mma_issue_rate_per_sm_gips")
    grid, nbytes, tokens = q["active_ctas"], q["stage_bytes"], q["tokens"]
    load = stream(grid, nbytes) if level == "dram" else resident(nbytes)
    return dict(
        grid=grid,
        warps=8,
        smem_bytes=tokens * nbytes,
        load=load,
        alu=q["compute_ns"] * issue / 8,
        prefetch=max(tokens - 1, 0),
        asynchronous=tokens > 0,
    ), ("cold" if level == "dram" else "warm")


def closed_loop_mma(p, q):
    grid, warps = p["specs"]["sm_count"] * q["ctas_per_sm"], q["num_warps"]
    return dict(
        grid=grid,
        warps=warps,
        smem_bytes=q["tokens"] * 8192,
        load=stream(grid, 8192),
        mma=warps * 4 * q["mma_iters_per_stage"],
        smem_load=q.get("ldmatrix_bytes_per_stage", 0),
        prefetch=q["tokens"] - 1,
    ), "cold"


def l2_merge(p, q):
    n, r, k = q["active_ctas"], q["sharing"], q["bytes_per_cta"]
    return dict(
        grid=n,
        warps=8,
        smem_bytes=k,
        load=(k, 16, (n // r) * k, lambda b: (b // r) * k),
    ), "cold"


def serial_fill(p, q):
    nbytes, cp = q["stage_bytes"], q["transfer_mode"] == "cp_async"
    return dict(
        grid=q["active_ctas"],
        warps=q["num_warps"],
        smem_bytes=nbytes,
        load=resident(nbytes),
        smem_store=0 if cp else nbytes,
        smem_load=nbytes if q["read_mode"] == "fill_and_ldmatrix" else 0,
        asynchronous=cp,
    ), "warm"


def sync_loop(p, q):
    grid, nbytes, warps = q["active_ctas"], q["stage_bytes"], q["num_warps"]
    cp, dram = q["transfer_mode"] == "cp_async_s1", q["mem_level"] == "dram"
    return dict(
        grid=grid,
        warps=warps,
        smem_bytes=nbytes,
        load=stream(grid, nbytes) if dram else resident(nbytes),
        mma=warps * q["mma_chain_len"],
        smem_load=nbytes,
        smem_store=0 if cp else nbytes,
        asynchronous=cp,
    ), ("cold" if dram else "warm")


def softmax_chain(p, q):
    warps, m, k = q["num_warps"], q["mma_per_half"], q["softmax_ops"]
    return dict(
        grid=p["specs"]["sm_count"] * q["ctas_per_sm"],
        warps=warps,
        mma=warps * 2 * m,
        sfu=k / 2,
        shfl=4,
        alu=k + 15,
    ), "warm"


def issue_controlled(p, q):
    warps, width, batch = q["num_warps"], q["load_width_bytes"], q["batch_loads_per_thread"]
    return dict(
        grid=1,
        warps=warps,
        load=resident(warps * 32 * batch * width, width),
        asynchronous=False,
    ), "warm"


def spill_boundary(p, q):
    frame, count = q["frame_bytes_per_thread"], q["reload_count"]
    return dict(
        grid=p["specs"]["sm_count"],
        warps=4,
        alu=14,
        spill={
            "stack_frame_bytes": frame,
            "spill_loads_bytes": 4 * count,
            "spill_stores_bytes": 0,
            "spill_scope": "inner_loop_confirmed",
        },
    ), "warm"


def spin_alone(q, points, profile):
    """No spin compute, or at most one CTA per SM (see NOT_EXPRESSED)."""
    return q["compute_ns"] == 0 or q["active_ctas"] <= profile["specs"]["sm_count"]


def lsu_calibration(points):
    narrow = min(q["load_width_bytes"] for q in points)
    return max(
        (q for q in points if q["load_width_bytes"] == narrow),
        key=lambda q: value(q, "issue_rate_gwarp_inst_per_sm"),
    )


MECHANISMS = [
    {
        "name": "closed_loop_dram",
        "paragraph": "Pipelined loop as a closed loop",
        "path": ("memory", "dram_closed_loop"),
        "keep": spin_alone,
        "build": lambda p, q: closed_loop(p, q, "dram"),
        "measured": lambda q: value(q, "mean_stage_turnaround_ns"),
        "label": lambda q: (
            f"{q['issue_mode']} t{q['tokens']} n{q['active_ctas']} "
            f"{q['stage_bytes']}B c{q['compute_ns']:g}"
        ),
    },
    {
        "name": "closed_loop_l2",
        "paragraph": "Pipelined loop as a closed loop",
        "path": ("memory", "l2_closed_loop"),
        "keep": spin_alone,
        "build": lambda p, q: closed_loop(p, q, "l2"),
        "measured": lambda q: value(q, "mean_stage_turnaround_ns"),
        "label": lambda q: (
            f"{q['issue_mode']} t{q['tokens']} n{q['active_ctas']} "
            f"{q['stage_bytes']}B c{q['compute_ns']:g}"
        ),
    },
    {
        "name": "closed_loop_mma",
        "paragraph": "Pipelined loop as a closed loop; per-stage synchronization",
        "path": ("memory", "dram_closed_loop_mma"),
        "build": closed_loop_mma,
        "measured": lambda q: value(q, "mean_stage_turnaround_ns"),
        "label": lambda q: (
            f"cps{q['ctas_per_sm']} t{q['tokens']} w{q['num_warps']} "
            f"ld{q.get('ldmatrix_bytes_per_stage', 0)} m{q['mma_iters_per_stage']}"
        ),
    },
    {
        "name": "miss_merging",
        "paragraph": "Miss merging",
        "path": ("memory", "l2_merge_by_sharing"),
        "keep": lambda q, points, profile: q["stagger_stages"] == 0,
        "build": l2_merge,
        "measured": lambda q: value(q, "turnaround_ns"),
        "label": lambda q: f"n{q['active_ctas']} r{q['sharing']} {q['bytes_per_cta']}B",
    },
    {
        "name": "port_floor_serial_fill",
        "paragraph": "Round-trip port floor",
        "path": ("pipeline", "serial_stage_fill"),
        "build": serial_fill,
        "measured": lambda q: value(q, "stage_time_ns"),
        "label": lambda q: (
            f"{q['transfer_mode']} {q['read_mode']} w{q['num_warps']} "
            f"{q['stage_bytes']}B n{q['active_ctas']}"
        ),
    },
    {
        "name": "port_floor_sync_loop",
        "paragraph": "Round-trip port floor",
        "path": ("pipeline", "sync_stage_loop"),
        "build": sync_loop,
        "measured": lambda q: value(q, "mean_stage_turnaround_ns"),
        "label": lambda q: (
            f"{q['mem_level']} {q['transfer_mode']} cps{q['ctas_per_sm']} "
            f"w{q['num_warps']} {q['stage_bytes']}B mma{q['mma_chain_len']}"
        ),
    },
    {
        "name": "dependent_chain",
        "paragraph": "A CTA's dependent chain",
        "path": ("compute", "mma_softmax_chain"),
        "build": softmax_chain,
        "measured": lambda q: value(q, "iter_time_ns"),
        "label": lambda q: (
            f"cps{q['ctas_per_sm']} w{q['num_warps']} m{q['mma_per_half']} k{q['softmax_ops']}"
        ),
    },
    {
        "name": "lsu_pipe",
        "paragraph": "Pipes (lsu)",
        "path": ("memory", "gmem_load_issue_controlled"),
        "keep": lambda q, points, profile: q is not lsu_calibration(points),
        "build": issue_controlled,
        "measured": lambda q: value(q, "issue_rate_gwarp_inst_per_sm"),
        # warp instructions per step over the step time
        "convert": lambda q, ns: q["num_warps"] * q["batch_loads_per_thread"] / ns,
        "label": lambda q: (
            f"{q['load_width_bytes']}B b{q['batch_loads_per_thread']} w{q['num_warps']}"
        ),
    },
    {
        "name": "spill_l1_boundary",
        "paragraph": "Register spills",
        "path": ("memory", "spill_reload_latency_by_count"),
        "keep": lambda q, points, profile: (
            q["regime"] == "l1_boundary"
            and q["dependency_mode"] == "batch_before_dep"
            and q["reload_count"] > 0
        ),
        "build": spill_boundary,
        # The extra time per step over the same frame with no reloads, as in the profile.
        "baseline": lambda p, q: spill_boundary(p, q | {"reload_count": 0})[0],
        "measured": lambda q: value(q, "extra_ns_vs_zero"),
        "label": lambda q: f"{q['frame_bytes_per_thread']}B x{q['reload_count']}",
    },
]

# Systematic shifts of the measured values from short to long launches (long-launch-capture.md,
# section 3). A new long-launch profile moves the measured side by about these amounts, so the
# signed error is expected to move the other way even with an unchanged model.
EXPECTED = {
    "closed_loop_dram": "turnaround +0.2% (DRAM -1.0% to +2.8%)",
    "closed_loop_l2": "turnaround +0.2% (L2 -2.7% to +1.2%)",
    "closed_loop_mma": "turnaround +0.1% (-0.9% to +1.3%)",
    "miss_merging": "turnaround -2.9% with stagger 0 (-5.2% to -1.4%); bandwidth +2.8%",
    "port_floor_serial_fill": "stage time -0.1% (-0.3% to +0.3%)",
    "port_floor_sync_loop": "sync-loop: L2 under 100 us -5.4%, 100 to 300 us -2.4%, 300 us and "
    "more -0.8%; DRAM 100 to 300 us -1.7%, 300 us and more +0.2% to +0.5%; multicta -0.2% "
    "(under 100 us -5.2%, 100 to 300 us -1.5%); reconcile -0.1% (up to -5.0% and +2.4%)",
    "dependent_chain": "iteration time -0.6% (-2.1% to +3.3%) at a 0.56% lower SM clock",
    "lsu_pipe": "issue rate +0.7% (-0.5% to +2.3%), up to +2.6% for legacy launches of 160 to "
    "210 us",
    "spill_l1_boundary": "spill-latency: per-step times fall with the frame's fixed cost; the "
    "1 KiB l1_boundary points are unstable at any length",
}


def per_step(irs, profile_text, cache):
    out = _core.predict_batch(
        [json.dumps(k) for k in irs], profile_text, l2_state=cache, draws=0, seed=17
    )
    for r in out:
        if r["status"] != "ok":
            raise ValueError(r["reason"])
    times = [r["time_ns"] for r in out]
    span = LENGTHS[1] - LENGTHS[0]
    return [(b - a) / span for a, b in zip(times[0::2], times[1::2], strict=True)]


def check(mech, profile, profile_text):
    section, field = mech["path"]
    points = profile["measured"][section].get(field) or []
    keep = mech.get("keep", lambda q, points, profile: True)
    chosen = [q for q in points if keep(q, points, profile)]
    if not chosen:
        return None
    by_cache = {}
    for q in chosen:
        spec, cache = mech["build"](profile, q)
        by_cache.setdefault(cache, []).append((q, spec))
    rows = []
    for cache, items in by_cache.items():
        irs = [kernel(steps=n, **spec) for _, spec in items for n in LENGTHS]
        steps = per_step(irs, profile_text, cache)
        if "baseline" in mech:
            base = [
                kernel(steps=n, **mech["baseline"](profile, q)) for q, _ in items for n in LENGTHS
            ]
            steps = [a - b for a, b in zip(steps, per_step(base, profile_text, cache), strict=True)]
        for (q, _), ns in zip(items, steps, strict=True):
            predicted = mech.get("convert", lambda q, ns: ns)(q, ns)
            measured = mech["measured"](q)
            rows.append(
                {
                    "point": mech["label"](q),
                    "measured": measured,
                    "predicted": predicted,
                    "error": predicted / measured - 1,
                }
            )
    errors = [r["error"] for r in rows]
    worst = max(rows, key=lambda r: abs(r["error"]))
    return {
        "points": len(rows),
        "median_abs_error": st.median(abs(e) for e in errors),
        "mape": st.mean(abs(e) for e in errors),
        "signed_median": st.median(errors),
        "worst": {"point": worst["point"], "error": worst["error"]},
        "rows": rows,
    }


def markdown(result):
    names = list(result["profiles"])
    head = "| Mechanism | " + " | ".join(names) + " | Expected shift of the measured side |"
    lines = [head, "|" + "---|" * (len(names) + 2)]
    for mech in MECHANISMS:
        cells = []
        for name in names:
            r = result["profiles"][name]["mechanisms"].get(mech["name"])
            cells.append(
                "not in profile"
                if r is None
                else f"{r['points']} pts: median {r['median_abs_error'] * 100:.1f}%, "
                f"MAPE {r['mape'] * 100:.1f}%, signed {r['signed_median'] * 100:+.1f}%, "
                f"worst {r['worst']['error'] * 100:+.1f}% ({r['worst']['point']})"
            )
        lines.append(f"| {mech['name']} | " + " | ".join(cells) + f" | {EXPECTED[mech['name']]} |")
    return "\n".join(lines) + "\n"


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def compiled_source_sha(root, commit):
    entries = git_output(root, "ls-tree", "-z", commit, "--", *COMPILED_PATHS)
    return hashlib.sha256(entries).hexdigest()


def clean_build(build):
    return (
        isinstance(build, dict)
        and build.get("dirty") is False
        and build.get("result_class") == "formal"
        and isinstance(build.get("commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", build["commit"]) is not None
    )


def check_core_revision(baseline, build, root=ROOT):
    if not clean_build(build):
        raise ValueError("The installed core must be a clean formal build.")
    if not clean_build(baseline.get("core")) or "core_source_sha256" not in baseline:
        raise ValueError(
            "The baseline lacks a clean core build or compiled-source SHA. Refresh it."
        )

    try:
        head = git_output(root, "rev-parse", "HEAD").decode().strip()
        if build["commit"] != head:
            raise ValueError(
                f"Installed core {build['commit']} differs from checkout {head}. Rebuild the core."
            )
        if git_output(root, "status", "--porcelain", "--untracked-files=no", "--", *COMPILED_PATHS):
            raise ValueError("The checkout has uncommitted compiled-source changes.")
        current_sha = compiled_source_sha(root, head)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Unable to verify the compiled sources in the current Git checkout."
        ) from exc

    recorded_sha = baseline["core_source_sha256"]
    if current_sha != recorded_sha:
        raise ValueError(
            f"Baseline core SHA {recorded_sha} differs from current {current_sha}. Refresh it."
        )

    # If the recorded commit is available, also bind the SHA to its origin.
    # Shallow checkouts can still verify the exact compiled inputs against HEAD.
    try:
        origin_sha = compiled_source_sha(root, baseline["core"]["commit"])
    except subprocess.CalledProcessError:
        return
    if origin_sha != recorded_sha:
        raise ValueError("The baseline core SHA does not match its recorded commit. Refresh it.")


def run(profiles):
    """The re-check of every mechanism on each (name, path) profile."""
    build = _core.build_info()
    result = {
        "core": build,
        "lengths": LENGTHS,
        "not_expressed": NOT_EXPRESSED,
        "expected_shift": EXPECTED,
        "profiles": {},
    }
    if clean_build(build):
        try:
            result["core_source_sha256"] = compiled_source_sha(ROOT, build["commit"])
        except (OSError, subprocess.CalledProcessError):
            # An archived build may come from another checkout. It can still
            # produce a report, while --check requires a bound baseline.
            pass
    for name, path in profiles:
        text = Path(path).read_text()
        profile = json.loads(text)
        result["profiles"][name] = {
            "path": str(path),
            "revision": profile["provenance"].get("profile_revision"),
            "mechanisms": {
                m["name"]: r for m in MECHANISMS if (r := check(m, profile, text)) is not None
            },
        }
    return result


def compare(baseline, result, tolerance):
    """Per profile and mechanism, the change from the baseline; failures are a changed point
    set or a median absolute error or MAPE that grew by more than `tolerance`."""
    lines, failures = [], []
    for name, base in baseline["profiles"].items():
        now = result["profiles"][name]["mechanisms"]
        for mech, b in base["mechanisms"].items():
            r = now.get(mech)
            if r is None or r["points"] != b["points"]:
                failures.append(f"{name} {mech}: points {b['points']} -> {r and r['points']}")
                continue
            delta = {k: r[k] - b[k] for k in ("median_abs_error", "mape", "signed_median")}
            lines.append(
                f"{name:10} {mech:24} median {b['median_abs_error'] * 100:5.1f}% -> "
                f"{r['median_abs_error'] * 100:5.1f}%, MAPE {b['mape'] * 100:5.1f}% -> "
                f"{r['mape'] * 100:5.1f}%, signed {b['signed_median'] * 100:+5.1f}% -> "
                f"{r['signed_median'] * 100:+5.1f}%"
            )
            for k in ("median_abs_error", "mape"):
                if delta[k] > tolerance:
                    failures.append(f"{name} {mech}: {k} +{delta[k] * 100:.2f} points")
        for mech in set(now) - set(base["mechanisms"]):
            failures.append(f"{name} {mech}: not in the baseline")
    return lines, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="append", metavar="NAME=PATH")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--markdown", type=Path)
    ap.add_argument(
        "--check",
        type=Path,
        metavar="BASELINE",
        help="check the baseline core SHA and re-run on the baseline's profiles",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="allowed growth of the median absolute error and the MAPE (fraction, 0.005 = "
        "half a percentage point)",
    )
    args = ap.parse_args()
    if args.check:
        baseline = json.loads(args.check.read_text())
        try:
            check_core_revision(baseline, _core.build_info())
        except ValueError as exc:
            raise SystemExit(f"FAIL {exc}") from None
        profiles = [(n, ROOT / b["path"]) for n, b in baseline["profiles"].items()]
        lines, failures = compare(baseline, run(profiles), args.tolerance)
        print("\n".join(lines))
        for f in failures:
            print(f"FAIL {f}")
        raise SystemExit(1 if failures else 0)
    if not args.profile or not args.out:
        ap.error("--profile and --out are required without --check")
    result = run([spec.split("=", 1) for spec in args.profile])
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    table = markdown(result)
    if args.markdown:
        args.markdown.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
