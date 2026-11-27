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
}
# Points that are one scalar profile value, and where their rate list sits: a top-level
# list, or None for the point's own list.
SCALARS = {
    ("mma", "fp32_fma"): "fp32_fma_tflops_samples",
    ("dram-bw", "write_full"): None,
    ("l2", "write_full"): None,
}
# Top-level compatibility lists that must equal a point's own list: (mode, raw field) -> list.
VIEWS = {("l2", "write_full"): "write_full_gbps_samples"}
# Timed points with no profile value of their own (their result is derived).
NOT_IN_PROFILE = {("l2", "capacity_sweep")}
# Sample lists of a point that are not profile values: (mode, list name).
NOT_A_VALUE = set()
# Modes whose points build_profile merges into one profile field, later overriding earlier
# on the same design fields; only the last source of each point is annotated.
OVERRIDE_ORDER = ()
# Profile values that are compute rates and get a per-cycle value.
COMPUTE = {"tflops_per_sm", "fp32_fma_tflops", "issue_rate_gwarp_inst_per_sm"}
# Design fields the profile names differently.
ALIAS = {"target_sms": "active_sms"}
FLAG = {"low_in_flight_unstable": "low_in_flight_unstable", "bistable_band": "bistable_band"}


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
    return profile


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
        for name in point.get("exceptions", ()):
            if scalar is None:
                entry[FLAG[name]] = True
            for value in targets.values():
                cover(value)
    return profile
