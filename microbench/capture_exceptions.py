"""Device-specific exceptions to the capture rules (docs/components/long-launch-capture.md,
section 4), registered per device and per profile field, each with its evidence.

A point is exempt from a rule when its device has an exception for that rule and
the point's field, and the point's key value lies within the exception's bounds.
Bounds are in the key's unit, or in multiples of the L2 size when `per_l2` is set.
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
                "dram_bw_by_bytes_in_flight": ("bytes_in_flight", 0, 256 * KiB),
                "dram_bw_by_bytes_in_flight_strided": ("bytes_in_flight", 0, 384 * KiB),
                "dram_bw_by_active_sms": ("target_sms", 0, 2),
            },
        },
        {
            "name": "bistable_band",
            "rules": (3,),
            "evidence": "docs/results/analysis/v0.3-gb10-l2-bistability-observation.md",
            "per_l2": True,
            "fields": {
                "l2_rescan_hit_rate_by_working_set": ("working_set_bytes", 0.75, 1.25),
                "l2_capacity_sweep": ("working_set_bytes", 0.75, 1.25),
            },
        },
    )
}


def exceptions_for(device, field, point, l2_bytes=0):
    """The exceptions that cover this point, in table order."""
    out = []
    for entry in EXCEPTIONS.get(device, ()):
        if field not in entry["fields"]:
            continue
        key, low, high = entry["fields"][field]
        scale = l2_bytes if entry["per_l2"] else 1
        if low * scale <= point[key] <= high * scale:
            out.append(entry)
    return out


def exempt(device, field, point, rule, l2_bytes=0):
    """Whether this point leaves `rule` (1, 2 or 3)."""
    return any(rule in entry["rules"] for entry in exceptions_for(device, field, point, l2_bytes))
