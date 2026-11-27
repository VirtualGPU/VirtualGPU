"""Pilot, freeze and verify launch schedules for any `vgpu_bench` mode.

A mode run with `--launch-schedule pilot` sizes each point until its
fastest launch reaches 1.25 ms and prints every trial; `freeze` turns that
output into a schedule (one length per point, in emission order) plus a
record of each point's identity; a capture run with the schedule file is
checked by `verify`: same points in the same order, the frozen lengths, and
every launch (and every in-kernel `window_ns_samples` value) at least 1.0 ms.
A point is any object in the output that has `launch_ns_samples`, in document
order (see docs/components/long-launch-capture.md).

    python microbench/long_launch.py freeze MODE PILOT.json SCHEDULE.txt FREEZE.json
    python microbench/long_launch.py check-schedule FREEZE.json SCHEDULE.txt   # before a capture
    python microbench/long_launch.py verify FREEZE.json CAPTURE.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PILOT_MIN_NS = 1_250_000
CAPTURE_MIN_NS = 1_000_000
# The fields that define a point, per mode and per top-level output field. Everything
# else in a point (timing results, run-time diagnostics such as observed_sms) stays
# out of the freeze.
CLOSED_LOOP = ("issue_mode", "tokens", "active_ctas", "stage_bytes", "compute_ns")
MLP = ("issue_mode", "bytes_in_flight", "active_ctas", "bytes_per_cta")
DESIGN_FIELDS = {
    "mlp": {"dram_bw_by_bytes_in_flight": MLP, "l2_bw_by_bytes_in_flight": MLP},
    "strided-mlp": {
        "dram_bw_by_bytes_in_flight_strided": (
            "segment_bytes",
            "stride_bytes",
            "bytes_in_flight",
            "active_ctas",
            "bytes_per_cta",
        )
    },
    "dram-bw": {"read_curve": ("target_sms",), "write_full": ("target_sms",)},
    "l2": {
        "read_curve": ("target_sms",),
        "read_curve_cp_async": ("target_sms",),
        "write_full": ("working_set_bytes",),
        "capacity_sweep": ("working_set_bytes",),
    },
    "closed-loop": {"dram_closed_loop": CLOSED_LOOP, "l2_closed_loop": CLOSED_LOOP},
    "l2-merge": {
        "l2_merge_by_sharing": ("active_ctas", "sharing", "bytes_per_cta", "stagger_stages")
    },
    "mma": {"mma": ("mma_warps_per_sm",), "fp32_fma": ("warps_per_sm",)},
    "gmem-issue": {"gmem_load_issue_per_sm": ("active_sms", "load_width_bytes", "num_warps")},
    "gmem-issue-controlled": {
        "gmem_load_issue_controlled": ("load_width_bytes", "batch_loads_per_thread", "num_warps")
    },
    "sync-loop": {
        "sync_stage_loop": (
            "mem_level",
            "transfer_mode",
            "ctas_per_sm",
            "num_warps",
            "stage_bytes",
            "mma_chain_len",
        )
    },
    "closed-loop-mma": {
        "dram_closed_loop_mma": (
            "ctas_per_sm",
            "tokens",
            "num_warps",
            "stage_bytes",
            "ldmatrix_bytes_per_stage",
            "mma_iters_per_stage",
        )
    },
    "spill-latency": {
        "reload_latency_by_count": (
            "dependency_mode",
            "num_warps",
            "resident_threads_per_sm",
            "frame_bytes_per_thread",
            "reload_count",
        ),
        "launch_cost_by_frame": ("dependency_mode", "frame_bytes_per_thread"),
    },
    "sfu-shfl": {
        **{
            key: ("warps_per_sm",)
            for key in (
                "sfu_issue_rate_per_sm_gips",
                "sfu_rcp_issue_rate_per_sm_gips",
                "shfl_issue_rate_per_sm_gips",
                "shfl_idx_issue_rate_per_sm_gips",
                "alu_issue_rate_per_sm_gips",
            )
        },
        "shfl_latency": ("warps",),
    },
    "cp-async": {"cp_async": ("threads_per_sm",), "smem": ("threads_per_sm",)},
    "launch": {
        "graph_launch": ("ctas",),
        "per_cta": ("ctas", "large_ctas"),
        "stream_launch": ("ctas",),
    },
    "latency": {
        key: ("probe",)
        for key in (
            "smem_latency",
            "l2_latency",
            "l2_latency_loaded",
            "dram_latency",
            "dram_latency_loaded",
        )
    },
    "latency-curve": {
        "dram_latency_by_bw": ("bg_blocks", "bg_threads"),
        "l2_latency_by_bw": ("bg_blocks", "bg_threads"),
    },
    "l1-carveout": {"l1_carveout_sweep": ("smem_carveout_bytes", "ws_bytes")},
    "l2-rescan": {"points": ("working_set_bytes",)},
    "mma-chain": {
        "mma_chain_latency": (
            "variant",
            "num_warps",
            "tile_bytes",
            "ldmatrix_x4_per_warp",
            "mma_chain_len",
        )
    },
    "sync-loop-multicta": {
        "sync_stage_loop": (
            "mem_level",
            "transfer_mode",
            "ctas_per_sm",
            "num_warps",
            "stage_bytes",
            "mma_chain_len",
        )
    },
    "sync-loop-ablation": {
        "sync_stage_loop_ablation": (
            "variant",
            "ctas_per_sm",
            "num_warps",
            "stage_bytes",
            "mma_chain_len",
        )
    },
    "sync-loop-reconcile": {
        "sync_stage_loop": (
            "mem_level",
            "transfer_mode",
            "ctas_per_sm",
            "num_warps",
            "stage_bytes",
            "mma_chain_len",
        )
    },
    "mma-softmax-chain": {
        key: ("ctas_per_sm", "num_warps", "mma_per_half", "softmax_ops")
        for key in ("solo_calibration", "mma_softmax_chain")
    },
    "serial-fill": {
        "serial_stage_fill": (
            "transfer_mode",
            "read_mode",
            "num_warps",
            "stage_bytes",
            "active_ctas",
        )
    },
    "stage-overhead": {
        "stage_sync_overhead_ns": ("issue_mode", "num_warps", "tokens"),
        "fixed_sync_ns": ("num_warps", "tokens"),
        "min_async_roundtrip": ("num_warps",),
    },
}


def points(node, path=()):
    """Yield (path, point) for every timed point in document order."""
    if isinstance(node, dict):
        if "launch_ns_samples" in node:
            yield path, node
            return
        for key, value in node.items():
            yield from points(value, path + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from points(value, path)


def identity(mode, path, point):
    """A point's geometry: where it sits and the mode's design fields."""
    fields = DESIGN_FIELDS.get(mode, {}).get(path[0] if path else None)
    if fields is None:
        raise ValueError(
            f"{mode} {'/'.join(path)}: no design fields registered for launch schedules"
        )
    return {"field": "/".join(path), **{k: point[k] for k in fields}}


# Point parameters derived at run time that the schedule freezes, on lines ahead of each
# point's length, in this order (LaunchSchedule::parameter); they are design fields too.
FROZEN_PARAMETERS = {"closed-loop-mma": {"dram_closed_loop_mma": ("mma_iters_per_stage",)}}


def schedule_lines(mode, path, point):
    """The schedule lines of one point: its frozen parameters, then its length."""
    names = FROZEN_PARAMETERS.get(mode, {}).get(path[0], ())
    return "".join(f"{point[name]}\n" for name in names) + f"{point['steps']}\n"


# Work a frozen length implies; bound in the freeze so the same steps cannot do other work.
WORK_FIELDS = ("transfer_bytes", "source_buffer_bytes")


def work(point):
    return {k: point[k] for k in WORK_FIELDS if k in point}


def check_diagnostics(mode, point):
    """Run-time diagnostics that must agree with the design, checked apart from the freeze."""
    if "observed_sms" in point and "target_sms" in point:
        if point["observed_sms"] != point["target_sms"]:
            raise ValueError(f"{mode}: {point['observed_sms']} SMs ran for {point['target_sms']}")


def check_samples(mode, point, count):
    """A complete list of finite, positive CUDA event times."""
    samples = point["launch_ns_samples"]
    if not samples:
        raise ValueError(f"{mode}: no CUDA event samples")
    if not isinstance(samples, list) or len(samples) != count:
        raise ValueError(f"{mode}: expected {count} CUDA event samples")
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0
        for x in samples
    ):
        raise ValueError(f"{mode}: CUDA event samples must be finite and positive")


def freeze(mode, pilot):
    rows = list(points(pilot))
    if not rows:
        raise ValueError(f"{mode}: no timed points in the pilot output")
    frozen = []
    for path, point in rows:
        check_samples(mode, point, len(point["launch_ns_samples"]))
        check_diagnostics(mode, point)
        # A probe timed inside the kernel is sized by the shorter of its launch and window.
        fastest = min(point["launch_ns_samples"] + point.get("window_ns_samples", []))
        if fastest < PILOT_MIN_NS:
            raise ValueError(
                f"{mode} {identity(mode, path, point)}: pilot reached only {fastest:.0f} ns"
            )
        frozen.append(
            {
                "point": identity(mode, path, point),
                "steps": point["steps"],
                "pilot_min_ns": fastest,
                **work(point),
            }
        )
    schedule = "".join(schedule_lines(mode, path, point) for path, point in rows)
    record = {
        "mode": mode,
        "pilot_min_ns": PILOT_MIN_NS,
        "capture_min_ns": CAPTURE_MIN_NS,
        "schedule_sha256": hashlib.sha256(schedule.encode()).hexdigest(),
        "points": frozen,
    }
    return schedule, record


def check_schedule(record, schedule_bytes):
    """Before a capture: the schedule file handed to the binary is the frozen one."""
    if hashlib.sha256(schedule_bytes).hexdigest() != record["schedule_sha256"]:
        raise ValueError(f"{record['mode']}: schedule file differs from the freeze")


# Modes whose capture sample is the median of a group of launches, with every launch kept in
# `launch_ns_each` (docs/results/analysis/v0.3-l2-merge-groups-registration.md).
GROUPS = {"l2-merge": 9}


# Modes whose points with reloads are read in several local memory allocations
# (v0.3-spill-allocation-design-registration.md): the number of allocations, kept in
# `allocation_launch_ns` as one list of samples per allocation.
ALLOCATIONS = {"spill-latency": 15}


def check_allocations(mode, point, samples):
    """Every launch of every allocation is present, finite and positive; returns the shortest."""
    n = ALLOCATIONS.get(mode)
    lists = point.get("allocation_launch_ns")
    wanted = n is not None and point.get("reload_count", 0) > 0
    if not wanted:
        if lists is not None:
            raise ValueError(f"{mode}: allocation_launch_ns on a point read in one allocation")
        return None
    if not isinstance(lists, list) or len(lists) != n:
        raise ValueError(f"{mode}: expected {n} allocations in allocation_launch_ns")
    for one in lists:
        check_samples(mode, {"launch_ns_samples": one}, samples)
    return min(min(one) for one in lists)


def check_group(mode, point, samples):
    """Every launch of a grouped point is present and at the capture floor; returns the shortest."""
    group = GROUPS.get(mode)
    each = point.get("launch_ns_each")
    if group is None:
        if each is not None:
            raise ValueError(f"{mode}: launch_ns_each without a registered group")
        return None
    if not isinstance(each, list) or len(each) != samples * group:
        raise ValueError(f"{mode}: expected {samples} x {group} launches in launch_ns_each")
    return min(each)


def verify(record, capture):
    """After a capture: same points in order, frozen lengths, complete samples, launches of 1 ms."""
    mode = record["mode"]
    rows = list(points(capture))
    if len(rows) != len(record["points"]):
        raise ValueError(f"{mode}: {len(rows)} points captured, {len(record['points'])} frozen")
    shortest = None
    for (path, point), frozen in zip(rows, record["points"], strict=True):
        if identity(mode, path, point) != frozen["point"]:
            raise ValueError(f"{mode}: point {identity(mode, path, point)} differs from the freeze")
        check_samples(mode, point, capture["samples"])
        check_diagnostics(mode, point)
        if point["steps"] != frozen["steps"]:
            raise ValueError(
                f"{mode} {frozen['point']}: ran {point['steps']} steps, froze {frozen['steps']}"
            )
        if work(point) != {k: frozen[k] for k in WORK_FIELDS if k in frozen}:
            raise ValueError(f"{mode} {frozen['point']}: work differs from the freeze")
        fastest = min(point["launch_ns_samples"])
        for each in (
            check_group(mode, point, capture["samples"]),
            check_allocations(mode, point, capture["samples"]),
        ):
            if each is not None:
                fastest = min(fastest, each)
        if fastest < record["capture_min_ns"]:
            raise ValueError(f"{mode} {frozen['point']}: a launch took {fastest:.0f} ns")
        # A probe timed inside the kernel also needs its own window at the floor.
        window = min(point.get("window_ns_samples", [record["capture_min_ns"]]))
        if window < record["capture_min_ns"]:
            raise ValueError(f"{mode} {frozen['point']}: an in-kernel window took {window:.0f} ns")
        shortest = fastest if shortest is None else min(shortest, fastest)
    return {"mode": mode, "points": len(rows), "shortest_launch_ns": shortest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("mode")
    f.add_argument("pilot", type=Path)
    f.add_argument("schedule", type=Path)
    f.add_argument("record", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("record", type=Path)
    v.add_argument("capture", type=Path)
    c = sub.add_parser("check-schedule")
    c.add_argument("record", type=Path)
    c.add_argument("schedule", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        for path in (args.schedule, args.record):
            if path.exists():
                raise FileExistsError(path)
        schedule, record = freeze(args.mode, json.loads(args.pilot.read_text()))
        args.schedule.write_text(schedule, newline="\n")
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    elif args.action == "check-schedule":
        check_schedule(json.loads(args.record.read_text()), args.schedule.read_bytes())
    else:
        print(
            json.dumps(
                verify(json.loads(args.record.read_text()), json.loads(args.capture.read_text()))
            )
        )


if __name__ == "__main__":
    main()
