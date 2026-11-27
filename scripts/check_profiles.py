"""Validate every GPU profile in profiles/ against the schema, provenance, and sanity rules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from virtualgpu.profile import ProfileValidationError, load_profile  # noqa: E402

LEAF_KEYS = {"value", "rel_uncertainty", "unit", "method"}

# Profiles awaiting hardware re-collection for https://github.com/VirtualGPU/VirtualGPU/issues/8
SANITY_CHECK_ALLOWLIST = {
    "profiles/a100-80gb.json",
}

# Datasheet theoretical peak DRAM bandwidth (10^9 B/s) for known SKUs.
KNOWN_SKU_DRAM_PEAK_GBPS: dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100 80GB PCIe": 1935.0,
    "NVIDIA GB10": 273.1,
    "NVIDIA L4": 300.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA GeForce RTX 3070": 461.0,
    "NVIDIA GeForce RTX 4070": 504.2,
    "NVIDIA RTX PRO 5000 72GB Blackwell": 1344.0,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 1792.0,
}

# Measured values that are derived from other timed values rather than timed themselves
# (docs/components/long-launch-capture.md, section 5): no launch of their own.
UNTIMED_LEAVES = {
    "l2_effective_capacity_bytes",
    "estimated_hit_rate",
    "serial_stage_overhead_ns",
    "extra_ns_vs_zero",
    "ns_per_reload",
}

DRAM_WRITE_TO_READ_MAX_RATIO = 1.15
ACTIVE_SM_STEP_FLOOR = 0.90
FULL_CHIP_PEAK_FLOOR = 0.90


def leaf_sources(obj: Any, path: str = "measured") -> list[tuple[str, Any]]:
    if isinstance(obj, dict):
        if LEAF_KEYS.issubset(obj):
            return [(path, obj.get("source"))]
        return [leaf for k, v in obj.items() for leaf in leaf_sources(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [leaf for i, v in enumerate(obj) for leaf in leaf_sources(v, f"{path}[{i}]")]
    return []


def measured_leaves(obj: Any, path: str = "measured", name: str = "") -> list[tuple[str, str, Any]]:
    """(path, field name, value) for every leaf value under `obj`."""
    if isinstance(obj, dict):
        if LEAF_KEYS.issubset(obj):
            return [(path, name, obj)]
        return [leaf for k, v in obj.items() for leaf in measured_leaves(v, f"{path}.{k}", k)]
    if isinstance(obj, list):
        return [
            leaf for i, v in enumerate(obj) for leaf in measured_leaves(v, f"{path}[{i}]", name)
        ]
    return []


def launch_errors(profile: dict[str, Any]) -> list[str]:
    """A long-launch profile: every timed measured value records launches of at least the
    protocol's minimum."""
    minimum = profile["provenance"]["capture_protocol"]["minimum_launch_ns"]
    errors = []
    for path, name, leaf in measured_leaves(profile["measured"]):
        if leaf.get("source") != "measured" or name in UNTIMED_LEAVES:
            continue
        launch = leaf.get("launch")
        if launch is None:
            errors.append(f"{path}: no launch record")
        elif launch["min_ns"] < minimum:
            errors.append(f"{path}: launch of {launch['min_ns']:.0f} ns under {minimum:.0f} ns")
    return errors


def sanity_errors(profile: dict[str, Any]) -> list[str]:
    """Check physical and cross-field measurement invariants beyond schema structure."""
    errors: list[str] = []
    ident = profile["identity"]
    specs = profile["specs"]
    mem = profile["measured"]["memory"]

    sku = ident["sku"]
    sm_count = specs["sm_count"]

    dram_read = float(mem["dram_bw_read_gbps"]["value"])
    dram_write = float(mem["dram_bw_write_gbps"]["value"])
    l2_read = float(mem["l2_bw_read_gbps"]["value"])

    sku_peak = KNOWN_SKU_DRAM_PEAK_GBPS.get(sku)
    if sku_peak is not None:
        if dram_read > sku_peak:
            errors.append(
                f"measured.memory.dram_bw_read_gbps ({dram_read:.1f} GB/s) "
                f"exceeds theoretical DRAM peak for {sku!r} ({sku_peak:.1f} GB/s)"
            )
        if dram_write > sku_peak:
            errors.append(
                f"measured.memory.dram_bw_write_gbps ({dram_write:.1f} GB/s) "
                f"exceeds theoretical DRAM peak for {sku!r} ({sku_peak:.1f} GB/s)"
            )

    if dram_read > 0 and dram_write > DRAM_WRITE_TO_READ_MAX_RATIO * dram_read:
        errors.append(
            f"measured.memory.dram_bw_write_gbps ({dram_write:.1f} GB/s) exceeds "
            f"{DRAM_WRITE_TO_READ_MAX_RATIO:.2f}x dram_bw_read_gbps ({dram_read:.1f} GB/s)"
        )

    for curve_key in (
        "dram_bw_by_active_sms",
        "l2_bw_by_active_sms",
        "l2_bw_by_active_sms_cp_async",
    ):
        if curve_key not in mem:
            continue
        curve = mem[curve_key]
        end_sms = curve[-1]["active_sms"]
        if end_sms != sm_count:
            errors.append(
                f"measured.memory.{curve_key}[-1].active_sms ({end_sms}) != sm_count ({sm_count})"
            )
        for idx in range(1, len(curve)):
            prev_bw = float(curve[idx - 1]["bw_gbps"]["value"])
            cur_bw = float(curve[idx]["bw_gbps"]["value"])
            if prev_bw > 0 and cur_bw < ACTIVE_SM_STEP_FLOOR * prev_bw:
                errors.append(
                    f"measured.memory.{curve_key}[{idx}].bw_gbps ({cur_bw:.1f} GB/s at "
                    f"{curve[idx]['active_sms']} SMs) drops below {ACTIVE_SM_STEP_FLOOR:.2f}x "
                    f"previous point ({prev_bw:.1f} GB/s at {curve[idx - 1]['active_sms']} SMs)"
                )

    dram_end_bw = float(mem["dram_bw_by_active_sms"][-1]["bw_gbps"]["value"])
    if dram_read > 0 and dram_end_bw < FULL_CHIP_PEAK_FLOOR * dram_read:
        errors.append(
            f"measured.memory.dram_bw_by_active_sms[-1].bw_gbps ({dram_end_bw:.1f} GB/s) "
            f"is below {FULL_CHIP_PEAK_FLOOR:.2f}x dram_bw_read_gbps ({dram_read:.1f} GB/s)"
        )

    l2_end_candidates = [float(mem["l2_bw_by_active_sms"][-1]["bw_gbps"]["value"])]
    if "l2_bw_by_active_sms_cp_async" in mem:
        l2_end_candidates.append(float(mem["l2_bw_by_active_sms_cp_async"][-1]["bw_gbps"]["value"]))
    best_l2_end = max(l2_end_candidates)
    if l2_read > 0 and best_l2_end < FULL_CHIP_PEAK_FLOOR * l2_read:
        errors.append(
            f"full-chip L2 active-SM curve ceiling ({best_l2_end:.1f} GB/s) "
            f"is below {FULL_CHIP_PEAK_FLOOR:.2f}x l2_bw_read_gbps ({l2_read:.1f} GB/s)"
        )

    l1_size = int(specs["l1_size_bytes"])
    l1_unified = int(specs["l1_smem_unified_bytes"])
    smem_per_sm = int(specs["shared_mem_per_sm_bytes"])
    if l1_size != l1_unified:
        errors.append(
            f"specs.l1_size_bytes ({l1_size}) != specs.l1_smem_unified_bytes ({l1_unified})"
        )
    if ident["vendor"] == "nvidia" and ident["arch"].startswith("sm_") and smem_per_sm > 0:
        if l1_size <= smem_per_sm:
            errors.append(
                f"specs.l1_size_bytes ({l1_size}) must exceed specs.shared_mem_per_sm_bytes "
                f"({smem_per_sm}) to include reserved unified L1 capacity"
            )
    elif l1_size < smem_per_sm:
        errors.append(
            f"specs.l1_size_bytes ({l1_size}) < specs.shared_mem_per_sm_bytes ({smem_per_sm})"
        )

    return errors


def check(path: Path | str, *, skip_sanity: bool = False) -> tuple[list[str], str | None]:
    try:
        profile = load_profile(path, schema_path=ROOT / "schemas" / "gpu_profile.schema.json")
    except (OSError, ValueError, ProfileValidationError) as exc:
        return [str(exc)], None

    errors: list[str] = []
    prov = profile["provenance"]
    source = prov["source"]
    if source not in ("measured", "partial", "spec_derived"):
        return [f"invalid provenance.source={source!r}"], None

    required = ["profile_revision", "calibration_run_id", "timestamp_utc"]
    if source in ("measured", "partial"):
        required += ["driver_version", "toolkit_version", "suite_version", "host_os"]
    for field in required:
        value = prov.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"provenance.{field} must be a non-empty string for source={source!r}")

    leaves = leaf_sources(profile["measured"])
    if not leaves:
        return errors + ["no measured values found"], None
    bad = [(p, s) for p, s in leaves if s not in ("measured", "estimated")]
    if bad:
        sample = ", ".join(f"{p}={s!r}" for p, s in bad[:5])
        return errors + [f"invalid or missing leaf source: {sample}"], None

    measured = sum(1 for _, s in leaves if s == "measured")
    estimated = len(leaves) - measured
    if source == "measured" and estimated:
        errors.append(f"source is 'measured' but {estimated} values are 'estimated'")
    elif source == "spec_derived" and measured:
        errors.append(f"source is 'spec_derived' but {measured} values are 'measured'")
    elif source == "partial" and not (measured and estimated):
        errors.append(f"source is 'partial' but found {measured} measured, {estimated} estimated")

    if "capture_protocol" in prov:
        launch = launch_errors(profile)
        errors.extend(
            launch[:20] + ([f"{len(launch) - 20} more launch errors"] if len(launch) > 20 else [])
        )

    if not skip_sanity:
        errors.extend(sanity_errors(profile))

    ident = profile["identity"]
    summary = (
        f"{ident['sku']}, {ident['arch']}, {source}, {measured} measured, {estimated} estimated"
    )
    return errors, summary


def main() -> int:
    paths = sorted((ROOT / "profiles").glob("*.json"))
    if not paths:
        print("No profiles found in profiles/", file=sys.stderr)
        return 1
    failed = False
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        skip_sanity = name in SANITY_CHECK_ALLOWLIST
        errors, summary = check(path, skip_sanity=skip_sanity)
        if errors:
            failed = True
            for err in errors:
                print(f"{name}: {err}", file=sys.stderr)
        else:
            note = "; sanity checks skipped, see #8" if skip_sanity else ""
            print(f"OK {name} ({summary}{note})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
