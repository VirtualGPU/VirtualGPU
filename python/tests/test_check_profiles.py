"""Tests for profile collection L1 mapping and scripts/check_profiles.py sanity gates."""

from __future__ import annotations

import copy
from pathlib import Path

from virtualgpu.profile import load_profile

from microbench.collect_profile import arch_unified_l1_bytes
from scripts.check_profiles import SANITY_CHECK_ALLOWLIST, check, launch_errors, sanity_errors

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE = REPO_ROOT / "schemas" / "examples" / "profile_example.json"


def test_arch_unified_l1_bytes_includes_reserved_l1() -> None:
    assert arch_unified_l1_bytes(7, 5, 65536) == 98304
    assert arch_unified_l1_bytes(8, 0, 167936) == 196608
    assert arch_unified_l1_bytes(8, 6, 102400) == 131072
    assert arch_unified_l1_bytes(8, 7, 167936) == 196608
    assert arch_unified_l1_bytes(8, 9, 102400) == 131072
    assert arch_unified_l1_bytes(9, 0, 233472) == 262144
    assert arch_unified_l1_bytes(10, 0, 233472) == 262144
    assert arch_unified_l1_bytes(10, 3, 233472) == 262144
    assert arch_unified_l1_bytes(12, 0, 102400) == 131072
    assert arch_unified_l1_bytes(12, 1, 102400) == 131072
    assert arch_unified_l1_bytes(99, 0, 524288) == 524288


def test_example_and_clean_profiles_pass_sanity_errors() -> None:
    example = load_profile(EXAMPLE_PROFILE)
    assert sanity_errors(example) == []

    for name in (
        "ascend910b3.json",
        "gb10.json",
        "l4.json",
        "l40s.json",
        "rtx3070.json",
        "rtx4070.json",
        "rtxpro5000-72gb.json",
        "rtxpro6000.json",
    ):
        path = REPO_ROOT / "profiles" / name
        if path.is_file():
            errors, summary = check(path)
            assert errors == [], f"{name}: {errors}"
            assert summary is not None


def test_sanity_rejects_inflated_dram_write_bw() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    profile["identity"]["sku"] = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    mem = profile["measured"]["memory"]
    mem["dram_bw_read_gbps"]["value"] = 1510.4
    mem["dram_bw_read_gbps"]["ci95_low"] = 1500.0
    mem["dram_bw_read_gbps"]["ci95_high"] = 1520.0
    mem["dram_bw_by_active_sms"][-1]["bw_gbps"]["value"] = 1510.4
    mem["dram_bw_write_gbps"]["value"] = 1962.7
    mem["dram_bw_write_gbps"]["ci95_low"] = 1950.0
    mem["dram_bw_write_gbps"]["ci95_high"] = 1970.0

    errs = sanity_errors(profile)
    assert any("exceeds theoretical DRAM peak" in e for e in errs)
    assert any("exceeds 1.15x dram_bw_read_gbps" in e for e in errs)


def test_sanity_rejects_active_sm_curve_drop_and_shortfall() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = profile["measured"]["memory"]["dram_bw_by_active_sms"]
    curve[-1]["bw_gbps"]["value"] = curve[-2]["bw_gbps"]["value"] * 0.65

    errs = sanity_errors(profile)
    assert any("drops below 0.90x previous point" in e for e in errs)
    assert any("is below 0.90x dram_bw_read_gbps" in e for e in errs)


def test_sanity_accepts_cp_async_l2_ceiling_when_ld_global_is_capped() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    mem = profile["measured"]["memory"]
    mem["l2_bw_read_gbps"]["value"] = 5000.0
    mem["l2_bw_by_active_sms"][-1]["bw_gbps"]["value"] = 2980.0

    errs_without_cp = sanity_errors(profile)
    assert any("full-chip L2 active-SM curve ceiling" in e for e in errs_without_cp)

    cp_curve = copy.deepcopy(mem["l2_bw_by_active_sms"])
    cp_curve[-1]["bw_gbps"]["value"] = 4950.0
    mem["l2_bw_by_active_sms_cp_async"] = cp_curve
    assert sanity_errors(profile) == []


def test_sanity_rejects_missing_reserved_l1_capacity() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    smem = profile["specs"]["shared_mem_per_sm_bytes"]
    profile["specs"]["l1_size_bytes"] = smem
    profile["specs"]["l1_smem_unified_bytes"] = smem

    errs = sanity_errors(profile)
    assert any("must exceed specs.shared_mem_per_sm_bytes" in e for e in errs)


def test_issue8_allowlist_skips_sanity_until_recollected() -> None:
    for rel in ("profiles/a100-80gb.json",):
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        assert rel in SANITY_CHECK_ALLOWLIST
        raw_errs = sanity_errors(load_profile(path))
        assert len(raw_errs) >= 2
        errs_skipped, summary = check(path, skip_sanity=True)
        assert errs_skipped == []
        assert summary is not None


def test_long_launch_profile_requires_launch_of_every_timed_value() -> None:
    profile = load_profile(REPO_ROOT / "profiles" / "gb10.json")
    profile["provenance"]["capture_protocol"] = {
        "minimum_launch_ns": 1e6,
        "freeze_manifest_sha256": "0" * 64,
        "rounds": [],
        "rule_three": {},
    }
    errors = launch_errors(profile)
    assert errors and all("no launch record" in e for e in errors)
    for leaf in [profile["measured"]["compute"]["fp32_fma_tflops"]]:
        leaf["launch"] = {"length": 40000, "min_ns": 0.9e6, "median_ns": 1.0e6}
    assert any("fp32_fma_tflops: launch of 900000 ns" in e for e in launch_errors(profile))
    spill = profile["measured"]["memory"]["spill_reload_latency_by_count"][0]
    assert not any("extra_ns_vs_zero" in e for e in launch_errors(profile))
    assert any(
        e.startswith("measured.memory.spill_reload_latency_by_count[0].iter_time_ns")
        for e in launch_errors(profile)
    )
    assert "extra_ns_vs_zero" in spill


def test_long_launch_check_skips_every_value_of_a_derived_field() -> None:
    # serial_stage_overhead_ns holds an intercept and a slope per point, fitted, not timed.
    profile = load_profile(REPO_ROOT / "profiles" / "gb10.json")
    profile["provenance"]["capture_protocol"] = {
        "minimum_launch_ns": 1e6,
        "freeze_manifest_sha256": "0" * 64,
        "rounds": [],
        "rule_three": {},
    }
    assert profile["measured"]["compute"]["serial_stage_overhead_ns"][0]["intercept_ns"]
    assert not any("serial_stage_overhead_ns" in e for e in launch_errors(profile))


def test_sanity_rejects_mma_drop_with_more_warps() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = next(iter(profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"].values()))
    curve[-1]["tflops_per_sm"]["value"] = curve[-2]["tflops_per_sm"]["value"] * 0.85

    errs = sanity_errors(profile)
    assert any("mma_tflops_per_sm_by_warps" in e and "drops below 0.90x" in e for e in errs)


def test_sanity_rejects_rescan_hit_rate_rising_past_l2() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))

    def point(ratio: float, hit: float) -> dict:
        return {"working_set_ratio_to_l2": ratio, "estimated_hit_rate": {"value": hit}}

    rescan = [point(0.25, 0.5), point(0.5, 1.0), point(1.0, 1.0), point(2.0, 0.0)]
    profile["measured"]["memory"]["l2_rescan_hit_rate_by_working_set"] = rescan
    assert sanity_errors(profile) == []

    rescan.append(point(3.0, 0.15))
    errs = sanity_errors(profile)
    assert any("l2_rescan_hit_rate_by_working_set[4]" in e for e in errs)
