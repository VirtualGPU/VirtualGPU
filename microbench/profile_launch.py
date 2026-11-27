"""Attach the long-launch records of merged raws (microbench/long_capture.py) to a profile.

For each timed point of a merged raw, the profile entry with the same design fields
gets, on every value built from one of the point's sample lists: `launch`, the range
of the round medians, and for compute rates the per-cycle value (median over rounds
of each round's rate over its mean SM clock). Entries a device exception covers get
the exception's flag and an uncertainty that covers the round medians. A point
without its profile entry, or a value that is not the median of its list, is an error.
"""

from __future__ import annotations

import statistics

try:
    import long_launch
except ImportError:  # imported as microbench.profile_launch
    from microbench import long_launch

# (mode, raw field) -> (profile section, profile field)
PROFILE_PATH = {
    ("mlp", "dram_bw_by_bytes_in_flight"): ("memory", "dram_bw_by_bytes_in_flight"),
    ("mlp", "l2_bw_by_bytes_in_flight"): ("memory", "l2_bw_by_bytes_in_flight"),
    ("strided-mlp", "dram_bw_by_bytes_in_flight_strided"): (
        "memory",
        "dram_bw_by_bytes_in_flight_strided",
    ),
    ("dram-bw", "read_curve"): ("memory", "dram_bw_by_active_sms"),
    ("dram-bw", "write_full"): ("memory", "dram_bw_write_gbps"),
    ("l2", "read_curve"): ("memory", "l2_bw_by_active_sms"),
    ("l2", "read_curve_cp_async"): ("memory", "l2_bw_by_active_sms_cp_async"),
    ("l2", "write_full"): ("memory", "l2_bw_write_gbps"),
    ("closed-loop", "dram_closed_loop"): ("memory", "dram_closed_loop"),
    ("closed-loop", "l2_closed_loop"): ("memory", "l2_closed_loop"),
    ("l2-merge", "l2_merge_by_sharing"): ("memory", "l2_merge_by_sharing"),
    ("mma", "mma"): ("compute", "mma_tflops_per_sm_by_warps"),
    ("mma", "fp32_fma"): ("compute", "fp32_fma_tflops"),
    ("gmem-issue", "gmem_load_issue_per_sm"): ("memory", "gmem_load_issue_per_sm"),
    ("gmem-issue-controlled", "gmem_load_issue_controlled"): (
        "memory",
        "gmem_load_issue_controlled",
    ),
    ("sync-loop", "sync_stage_loop"): ("pipeline", "sync_stage_loop"),
    ("closed-loop-mma", "dram_closed_loop_mma"): ("memory", "dram_closed_loop_mma"),
    ("spill-latency", "reload_latency_by_count"): ("memory", "spill_reload_latency_by_count"),
    ("mma-chain", "mma_chain_latency"): ("compute", "mma_chain_latency"),
    ("sync-loop-multicta", "sync_stage_loop"): ("pipeline", "sync_stage_loop"),
    ("sync-loop-reconcile", "sync_stage_loop"): ("pipeline", "sync_stage_loop"),
    ("sync-loop-ablation", "sync_stage_loop_ablation"): ("pipeline", "sync_stage_loop_ablation"),
    ("serial-fill", "serial_stage_fill"): ("pipeline", "serial_stage_fill"),
    ("mma-softmax-chain", "mma_softmax_chain"): ("compute", "mma_softmax_chain"),
    ("stage-overhead", "stage_sync_overhead_ns"): ("pipeline", "stage_sync_overhead_ns"),
    ("stage-overhead", "fixed_sync_ns"): ("pipeline", "fixed_sync_ns"),
    ("stage-overhead", "min_async_roundtrip"): ("pipeline", "min_async_roundtrip_ns"),
    **{
        ("sfu-shfl", key): ("compute", key)
        for key in (
            "sfu_issue_rate_per_sm_gips",
            "sfu_rcp_issue_rate_per_sm_gips",
            "shfl_issue_rate_per_sm_gips",
            "shfl_idx_issue_rate_per_sm_gips",
        )
    },
    ("sfu-shfl", "shfl_latency"): ("compute", "shfl_latency_ns"),
    ("cp-async", "cp_async"): ("async_pipeline", "cp_async_bw_per_sm_gbps"),
    ("cp-async", "smem"): ("memory", "smem_bw_per_sm_gbps"),
    ("launch", "graph_launch"): ("overhead", "launch_overhead_us"),
    ("launch", "per_cta"): ("overhead", "per_cta_launch_latency_ns"),
    ("launch", "stream_launch"): ("overhead", "host_launch_overhead_us"),
    **{
        ("latency", key): ("memory", f"{key}_ns")
        for key in (
            "smem_latency",
            "l2_latency",
            "l2_latency_loaded",
            "dram_latency",
            "dram_latency_loaded",
        )
    },
    ("latency-curve", "dram_latency_by_bw"): ("memory", "dram_latency_by_bw"),
    ("latency-curve", "l2_latency_by_bw"): ("memory", "l2_latency_by_bw"),
    ("l1-carveout", "l1_carveout_sweep"): ("memory", "l1_to_l2_spill_threshold_by_carveout"),
    ("l2-rescan", "points"): ("memory", "l2_rescan_hit_rate_by_working_set"),
}
# Raw sample lists whose profile value has another name than the list without "_samples".
STAGE_LOOP = {
    "bw_gbps_samples": "delivered_bw_gbps",
    "turnaround_ns_samples": "mean_stage_turnaround_ns",
    "mma_tflops_per_sm_samples": "achieved_mma_tflops_per_sm",
    "tensor_util_samples": "tensor_utilization",
}
METRIC = {
    "closed-loop": STAGE_LOOP,
    "sync-loop": STAGE_LOOP,
    "closed-loop-mma": STAGE_LOOP,
    "sync-loop-multicta": STAGE_LOOP,
    "sync-loop-reconcile": STAGE_LOOP,
    "sync-loop-ablation": STAGE_LOOP,
    "serial-fill": {"effective_bytes_per_ns_samples": "effective_bytes_per_ns_per_cta"},
    "sfu-shfl": {"rate_gwarp_inst_per_sm_samples": "rate"},
}
# Points that are one scalar profile value, and where their rate list sits: a top-level
# list, or None for the point's own list.
SCALARS = {
    ("mma", "fp32_fma"): "fp32_fma_tflops_samples",
    ("dram-bw", "write_full"): None,
    ("l2", "write_full"): None,
    ("stage-overhead", "min_async_roundtrip"): "min_async_roundtrip_ns_samples",
    ("sfu-shfl", "shfl_latency"): "shfl_latency_ns_samples",
    ("cp-async", "cp_async"): "cp_async_bw_per_sm_gbps_samples",
    ("cp-async", "smem"): "smem_bw_per_sm_gbps_samples",
    ("launch", "graph_launch"): "device_launch_overhead_us_samples",
    ("launch", "per_cta"): "per_cta_launch_latency_ns_samples",
    ("launch", "stream_launch"): "host_launch_overhead_us_samples",
    **{
        ("latency", key): f"{key}_ns_samples"
        for key in (
            "smem_latency",
            "l2_latency",
            "l2_latency_loaded",
            "dram_latency",
            "dram_latency_loaded",
        )
    },
}
# Top-level compatibility lists that must equal a point's own list: (mode, raw field) -> list.
VIEWS = {("l2", "write_full"): "write_full_gbps_samples"}
# Timed points with no profile value of their own (their result is derived).
# The mma-softmax-chain calibrations are timed points whose medians become every point's
# solo_mma_ns and solo_softmax_ns, plain numbers in the profile. l1-carveout's profile
# values (miss rates and the threshold per carveout) are derived from every point of a
# carveout together.
NOT_IN_PROFILE = {
    ("l2", "capacity_sweep"),
    # Only its peak may enter the profile, as non_mma_issue_rate_per_sm_gips (see
    # attach_non_mma).
    ("sfu-shfl", "alu_issue_rate_per_sm_gips"),
    # Taken out of every spill point's per-step time; not a profile value itself.
    ("spill-latency", "launch_cost_by_frame"),
    ("mma-softmax-chain", "solo_calibration"),
    ("l1-carveout", "l1_carveout_sweep"),
}
# Sample lists of a point that are not profile values: (mode, list name).
# stage-overhead's in-kernel windows are checked by long_launch.verify; the per-step
# overhead_ns is the profile value.
NOT_A_VALUE = {
    ("stage-overhead", "window_ns_samples"),
    ("latency-curve", "window_ns_samples"),
    # The profile stores the median background bandwidth as a plain number.
    ("latency-curve", "achieved_bw_gbps_samples"),
    ("l2-rescan", "window_ns_samples"),
    # Pass 1 enters the derived hit rate; the profile stores pass 2.
    ("l2-rescan", "pass1_bw_gbps_samples"),
}
# Modes whose points build_profile merges into one profile field, later overriding earlier
# on the same design fields; only the last source of each point is annotated.
OVERRIDE_ORDER = ("sync-loop", "sync-loop-multicta", "sync-loop-reconcile")
# Profile values that are compute rates and get a per-cycle value.
COMPUTE = {"tflops_per_sm", "fp32_fma_tflops", "issue_rate_gwarp_inst_per_sm", "rate"}
# Design fields the profile names differently.
ALIAS = {"target_sms": "active_sms"}
FLAG = {
    "low_in_flight_unstable": "low_in_flight_unstable",
    "bistable_band": "bistable_band",
    "l1_boundary_two_state": "l1_boundary_two_state",
}
# Exceptions whose registration widens the uncertainty to cover the round medians.
WIDEN = {"low_in_flight_unstable", "bistable_band"}


def metric(mode, key):
    if key == "gbps_samples":
        return "bw_gbps"
    return METRIC.get(mode, {}).get(key, key.removesuffix("_samples"))


def entry_for(profile, mode, path, point):
    """The profile entry (or, for a scalar, its container and key) of a merged point."""
    section, field = PROFILE_PATH[(mode, path[0])]
    target = profile["measured"][section][field]
    if (mode, path[0]) in SCALARS:
        return profile["measured"][section], field
    for key in path[1:]:
        target = target[key]
    names = long_launch.DESIGN_FIELDS[mode][path[0]]
    matches = [e for e in target if all(e[ALIAS.get(k, k)] == point[k] for k in names)]
    if len(matches) != 1:
        raise ValueError(
            f"{mode} {path}: {len(matches)} profile entries for {[point[k] for k in names]}"
        )
    return matches[0], None


def per_cycle(rates, launch, unit):
    clocks = [
        (b + a) / 2 for b, a in zip(launch["sm_mhz_before"], launch["sm_mhz_after"], strict=True)
    ]
    if None in clocks or len(clocks) != len(rates):
        raise ValueError("per-cycle values need every round's SM clock")
    return {
        "value": statistics.median(r / c for r, c in zip(rates, clocks)),
        "unit": f"{unit} per MHz",
    }


def annotate(value, rounds, launch, compute):
    """Attach one value's records; `rounds` is the list of its round medians."""
    if abs(value["value"] - statistics.median(rounds)) > 1e-3 * max(abs(value["value"]), 1e-9):
        raise ValueError(f"value {value['value']} is not the median of its round medians")
    value["launch"] = {
        k: launch[k] for k in ("length", "min_ns", "median_ns", "sm_mhz_before", "sm_mhz_after")
    }
    value["round_median_range"] = [min(rounds), max(rounds)]
    if compute:
        value["per_cycle"] = per_cycle(rounds, launch, value["unit"])


def cover_allocations(entry, point):
    """A spill point read in several local memory allocations: store the span of its
    per-allocation medians (per step, pooled over the valid rounds) over their median, and
    widen the uncertainty of its per-step time and of the values derived from it to at least
    half that span (v0.3-spill-allocation-design-registration.md)."""
    steps = point["steps"]
    medians = [
        statistics.median(one) / steps for rnd in point["allocation_launch_ns"] for one in rnd
    ]
    half = (max(medians) - min(medians)) / 2
    entry["allocation_span"] = round((max(medians) - min(medians)) / statistics.median(medians), 4)

    def widen(value, rel):
        value["rel_uncertainty"] = round(min(1.0, max(value["rel_uncertainty"], rel)), 4)

    widen(entry["iter_time_ns"], half / entry["iter_time_ns"]["value"])
    extra = entry["extra_ns_vs_zero"]["value"]
    rel = half / extra if extra > 0 else 1.0
    widen(entry["extra_ns_vs_zero"], rel)
    widen(entry["ns_per_reload"], rel)  # extra over the reload count: the same relative spread


def cover(value):
    """Widen the uncertainty so value +/- uncertainty spans the round medians."""
    low, high = value["round_median_range"]
    spread = max(value["value"] - low, high - value["value"]) / value["value"]
    value["rel_uncertainty"] = round(min(1.0, max(value["rel_uncertainty"], spread)), 4)


def key(mode, path, point):
    """A point's profile field and design fields, comparable across modes."""
    names = long_launch.DESIGN_FIELDS[mode][path[0]]
    return PROFILE_PATH[(mode, path[0])], tuple(point[k] for k in names)


def overridden(merged_by_mode):
    """Per mode, the keys of its points a later mode in OVERRIDE_ORDER overrides."""
    out, later = {}, set()
    for mode in reversed([m for m in OVERRIDE_ORDER if m in merged_by_mode]):
        mine = {key(mode, p, pt) for p, pt in long_launch.points(merged_by_mode[mode])}
        out[mode] = mine & later
        later |= mine
    return out


def attach_all(profile, merged_by_mode):
    """Annotate `profile` from every mode's merged raw."""
    skip = overridden(merged_by_mode)
    for mode, merged in merged_by_mode.items():
        attach(profile, mode, merged, skip.get(mode, set()))
    attach_non_mma(profile, merged_by_mode)
    return profile


def attach_non_mma(profile, merged_by_mode):
    """Annotate non_mma_issue_rate_per_sm_gips from the source collect_profile chose: the
    `mma` FP32 FMA point, or the peak of the `sfu-shfl` ALU curve."""
    try:
        import collect_profile as cp
    except ImportError:  # imported as microbench.profile_launch
        from microbench import collect_profile as cp

    value = profile["measured"]["compute"]["non_mma_issue_rate_per_sm_gips"]
    if value["source"] != "measured":
        return
    if value["method"] == cp.FFMA_ISSUE_METHOD:
        mma = merged_by_mode["mma"]
        point = mma["fp32_fma"]
        rounds = cp.ffma_issue_samples(mma["fp32_fma_tflops_samples"], profile["specs"]["sm_count"])
    else:
        point = cp.peak_point(merged_by_mode["sfu-shfl"]["alu_issue_rate_per_sm_gips"])
        rounds = point["rate_gwarp_inst_per_sm_samples"]
    annotate(value, rounds, point["launch"], True)


def attach(profile, mode, merged, skip=frozenset()):
    """Annotate `profile` in place from one mode's merged raw, except the points whose
    key is in `skip` (overridden by a later source)."""
    for path, point in long_launch.points(merged):
        if (mode, path[0]) in NOT_IN_PROFILE or (skip and key(mode, path, point) in skip):
            continue
        entry, scalar = entry_for(profile, mode, path, point)
        lists = {
            metric(mode, k): v
            for k, v in point.items()
            if k.endswith("_samples") and k != "launch_ns_samples" and (mode, k) not in NOT_A_VALUE
        }
        view = VIEWS.get((mode, path[0]))
        if view and [merged[view]] != list(lists.values()):
            raise ValueError(f"{mode}: {view} differs from {path[0]}'s own samples")
        if scalar is not None:
            outside = SCALARS[(mode, path[0])]
            (rounds,) = [merged[outside]] if outside else lists.values()
            values = {scalar: rounds}
        else:
            values = lists
        absent = [name for name in values if name not in entry]
        if absent:
            raise ValueError(
                f"{mode} {path}: no profile value {absent}; list it in NOT_A_VALUE if it is none"
            )
        targets = {name: entry[name] for name in values}
        for name, rounds in values.items():
            annotate(targets[name], rounds, point["launch"], name in COMPUTE)
        if "allocation_launch_ns" in point:
            cover_allocations(entry, point)
        for name in point.get("exceptions", ()):
            if scalar is None:
                entry[FLAG[name]] = True
            if name in WIDEN:
                for value in targets.values():
                    cover(value)
    return profile
