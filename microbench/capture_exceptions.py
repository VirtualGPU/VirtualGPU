"""Device-specific exceptions to the capture rules (docs/components/long-launch-capture.md,
section 4), registered per device and per profile field, each with its evidence.

A point is exempt from a rule when its device has an exception for that rule and
the point's field, and the point meets every condition the exception lists for the
field: `(key, value)` for equality, `(key, low, high)` for a closed range, in the key's
unit or in multiples of the L2 size when `per_l2` is set.
Exempt points are removed from a rule's lists or points, and from its count,
before the rule is judged; rule 2 has no exception.
"""

from __future__ import annotations

KiB = 1024

EXCEPTIONS = {
    "NVIDIA GB10": (
        {
            "name": "low_in_flight_unstable",
            "rules": (1, 3),
            "evidence": "docs/results/analysis/v0.3-gb10-mlp-wall-order-posthoc.md",
            "per_l2": False,
            "fields": {
                "dram_bw_by_bytes_in_flight": [("bytes_in_flight", 0, 256 * KiB)],
                "dram_bw_by_bytes_in_flight_strided": [("bytes_in_flight", 0, 384 * KiB)],
                "dram_bw_by_active_sms": [("target_sms", 0, 2)],
            },
        },
        {
            "name": "bistable_band",
            "rules": (3,),
            "evidence": "docs/results/analysis/v0.3-gb10-l2-bistability-observation.md",
            "per_l2": True,
            "fields": {
                "l2_rescan_hit_rate_by_working_set": [("working_set_bytes", 0.75, 1.25)],
            },
        },
        {
            "name": "l2_near_capacity_unstable",
            "rules": (3,),
            "evidence": "docs/results/analysis/v0.3-gb10-l2-near-capacity-exception-registration.md",
            "per_l2": True,
            "fields": {
                "l2_capacity_sweep": [("working_set_bytes", 0.75, 2.0)],
            },
        },
        {
            "name": "l1_boundary_two_state",
            "rules": (1, 3),
            "evidence": "docs/results/analysis/v0.3-gb10-spill-serial-boundary-exception-registration.md",
            "per_l2": False,
            "fields": {
                "spill_reload_latency_by_count": [
                    ("dependency_mode", "serial_chain"),
                    ("frame_bytes_per_thread", 1 * KiB, 1 * KiB),
                    ("reload_count", 1, 1 << 30),
                ],
            },
        },
    )
}


def meets(condition, point, scale):
    key, *bounds = condition
    if len(bounds) == 1:
        return point[key] == bounds[0]
    low, high = bounds
    return low * scale <= point[key] <= high * scale


def exceptions_for(device, field, point, l2_bytes=0):
    """The exceptions that cover this point, in table order."""
    out = []
    for entry in EXCEPTIONS.get(device, ()):
        if field not in entry["fields"]:
            continue
        scale = l2_bytes if entry["per_l2"] else 1
        if all(meets(c, point, scale) for c in entry["fields"][field]):
            out.append(entry)
    return out


def exempt(device, field, point, rule, l2_bytes=0):
    """Whether this point leaves `rule` (1, 2 or 3)."""
    return any(rule in entry["rules"] for entry in exceptions_for(device, field, point, l2_bytes))
