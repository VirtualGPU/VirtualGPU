"""Tests for GPU profile schema and validator."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from virtualgpu.profile import (
    DEFAULT_SCHEMA_PATH,
    ProfileValidationError,
    load_profile,
    validate_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PROFILE = REPO_ROOT / "schemas" / "examples" / "profile_example.json"
RTX4070_PROFILE = REPO_ROOT / "profiles" / "rtx4070.json"
GB10_PROFILE = REPO_ROOT / "profiles" / "gb10.json"


def test_example_profile_is_valid() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    assert profile["schema_version"] == "0.1"
    assert profile["identity"]["arch"] == "sm_89"
    assert DEFAULT_SCHEMA_PATH.is_file()


def test_rtx4070_profile_is_valid() -> None:
    if not RTX4070_PROFILE.is_file():
        pytest.skip("profiles/rtx4070.json not present")
    profile = load_profile(RTX4070_PROFILE)
    assert profile["identity"]["arch"] == "sm_89"
    assert profile["specs"]["sm_count"] == 46
    mem = profile["measured"]["memory"]
    assert mem["dram_bw_read_gbps"]["source"] == "measured"
    assert mem["l2_bw_read_gbps"]["source"] == "measured"
    assert len(mem["l2_bw_by_active_sms"]) >= 8
    assert (
        mem["l2_bw_by_active_sms"][0]["bw_gbps"]["value"]
        > mem["dram_bw_by_active_sms"][0]["bw_gbps"]["value"]
    )
    assert mem["l2_effective_capacity_bytes"]["source"] == "measured"
    assert mem["l2_effective_capacity_bytes"]["value"] <= profile["specs"]["l2_size_bytes"]
    assert mem["dram_latency_loaded_ns"]["source"] == "measured"
    assert len(mem["dram_latency_by_bw"]) >= 5
    assert len(mem["l2_latency_by_bw"]) >= 5
    assert len(mem["dram_bw_by_bytes_in_flight"]) >= 40
    assert len(mem["dram_bw_by_bytes_in_flight_strided"]) >= 40
    assert len(mem["l2_bw_by_bytes_in_flight"]) >= 40
    assert max(pt["bytes_in_flight"] for pt in mem["dram_bw_by_bytes_in_flight"]) >= 6 * 1024 * 1024
    assert max(pt["bytes_in_flight"] for pt in mem["l2_bw_by_bytes_in_flight"]) >= 6 * 1024 * 1024
    assert len(mem["dram_closed_loop"]) >= 100
    assert len(mem["dram_closed_loop_mma"]) >= 140
    assert any(
        pt["num_warps"] == 8 and pt.get("ldmatrix_bytes_per_stage", 0) >= 16384
        for pt in mem["dram_closed_loop_mma"]
    )
    assert len(mem["l2_closed_loop"]) >= 100
    assert len(mem["l2_merge_by_sharing"]) >= 20
    assert len(mem["l2_rescan_hit_rate_by_working_set"]) >= 12
    assert len(mem["gmem_load_issue_per_sm"]) >= 24
    assert len(mem["gmem_load_issue_controlled"]) >= 36
    assert len(mem["spill_reload_latency_by_count"]) >= 60
    assert len(mem["l1_to_l2_spill_threshold_by_carveout"]) >= 6
    assert all(
        len(entry.get("miss_rate_by_working_set", [])) >= 30
        and all(
            0.0 <= pt["estimated_miss_rate"]["value"] <= 1.0
            for pt in entry["miss_rate_by_working_set"]
        )
        for entry in mem["l1_to_l2_spill_threshold_by_carveout"]
    )
    assert (
        profile["measured"]["spill"]["l1_to_l2_spill_threshold_bytes_per_sm"]["source"]
        == "measured"
    )
    assert len(profile["measured"]["pipeline"]["stage_sync_overhead_ns"]) >= 8
    assert len(profile["measured"]["pipeline"]["fixed_sync_ns"]) >= 8
    assert profile["measured"]["pipeline"]["min_async_roundtrip_ns"]["source"] == "measured"
    assert len(profile["measured"]["pipeline"]["serial_stage_fill"]) >= 32
    assert len(profile["measured"]["pipeline"]["sync_stage_loop"]) >= 564
    assert len(profile["measured"]["pipeline"]["sync_stage_loop_ablation"]) >= 64
    assert profile["measured"]["async_pipeline"]["cp_async_bw_per_sm_gbps"]["source"] == "measured"
    assert profile["measured"]["overhead"]["launch_overhead_us"]["source"] == "measured"
    mma_f16 = profile["measured"]["compute"]["mma_tflops_per_sm_by_warps"][
        "mma.sync:f16:f16:f32:16x8x16"
    ]
    assert len(mma_f16) >= 4
    assert all(pt["tflops_per_sm"]["source"] == "measured" for pt in mma_f16)
    assert len(profile["measured"]["compute"]["mma_chain_latency"]) >= 128
    assert len(profile["measured"]["compute"]["serial_stage_overhead_ns"]) >= 16
    assert profile["measured"]["compute"]["sfu_tflops"]["source"] == "measured"
    assert len(profile["measured"]["compute"]["sfu_issue_rate_per_sm_gips"]) >= 6
    assert len(profile["measured"]["compute"]["shfl_issue_rate_per_sm_gips"]) >= 6
    assert profile["measured"]["compute"]["shfl_latency_ns"]["source"] == "measured"
    assert len(profile["measured"]["compute"]["mma_softmax_chain"]) >= 72


def test_gb10_profile_is_valid() -> None:
    if not GB10_PROFILE.is_file():
        pytest.skip("profiles/gb10.json not present")
    profile = load_profile(GB10_PROFILE)
    assert profile["identity"]["arch"] == "sm_121"
    assert profile["identity"]["memory_type"] == "unified"
    assert profile["specs"]["sm_count"] == 48
    assert profile["specs"]["capabilities"]["tma"] is True
    assert profile["specs"]["capabilities"]["tcgen05_tmem"] is False
    mem = profile["measured"]["memory"]
    assert mem["dram_bw_read_gbps"]["source"] == "measured"
    assert mem["l2_bw_read_gbps"]["source"] == "measured"
    assert len(mem["l2_bw_by_active_sms"]) >= 8
    assert (
        mem["l2_bw_by_active_sms"][0]["bw_gbps"]["value"]
        > mem["dram_bw_by_active_sms"][0]["bw_gbps"]["value"]
    )
    assert len(mem["dram_bw_by_bytes_in_flight"]) >= 40
    assert len(mem["dram_bw_by_bytes_in_flight_strided"]) >= 40
    assert len(mem["gmem_load_issue_per_sm"]) >= 24
    assert len(mem["gmem_load_issue_controlled"]) >= 36
    assert len(mem["spill_reload_latency_by_count"]) >= 60
    assert len(mem["l1_to_l2_spill_threshold_by_carveout"]) >= 6
    assert all(
        len(entry.get("miss_rate_by_working_set", [])) >= 30
        and all(
            0.0 <= pt["estimated_miss_rate"]["value"] <= 1.0
            for pt in entry["miss_rate_by_working_set"]
        )
        for entry in mem["l1_to_l2_spill_threshold_by_carveout"]
    )
    assert (
        profile["measured"]["spill"]["l1_to_l2_spill_threshold_bytes_per_sm"]["source"]
        == "measured"
    )
    assert len(mem["dram_closed_loop_mma"]) >= 140
    assert len(profile["measured"]["pipeline"]["serial_stage_fill"]) >= 32
    assert len(profile["measured"]["pipeline"]["sync_stage_loop"]) >= 564
    assert len(profile["measured"]["pipeline"]["sync_stage_loop_ablation"]) >= 64
    assert len(profile["measured"]["compute"]["mma_chain_latency"]) >= 128


def test_rejects_missing_required_field() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    del profile["specs"]["sm_count"]
    with pytest.raises(ProfileValidationError, match="sm_count"):
        validate_profile(profile)


def test_rejects_unknown_field() -> None:
    profile = load_profile(EXAMPLE_PROFILE)
    profile["identity"]["unexpected_field"] = True
    with pytest.raises(ProfileValidationError, match="unexpected_field"):
        validate_profile(profile)


def test_rejects_invalid_ci_bounds() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    mv = profile["measured"]["memory"]["dram_bw_read_gbps"]
    mv["ci95_low"] = mv["value"] + 10.0
    mv["ci95_high"] = mv["value"] + 20.0
    with pytest.raises(ProfileValidationError, match="outside"):
        validate_profile(profile)


def test_rejects_non_monotonic_active_sms_curve() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = profile["measured"]["memory"]["dram_bw_by_active_sms"]
    curve[1]["active_sms"] = curve[0]["active_sms"]
    with pytest.raises(ProfileValidationError, match="strictly increasing"):
        validate_profile(profile)


def test_rejects_active_sms_exceeding_sm_count() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    curve = profile["measured"]["memory"]["dram_bw_by_active_sms"]
    curve[-1]["active_sms"] = profile["specs"]["sm_count"] + 1
    with pytest.raises(ProfileValidationError, match="exceeds hardware limit"):
        validate_profile(profile)


def test_rejects_dispatch_policy_not_in_candidates() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    profile["measured"]["cta_dispatch"]["observed_policy"] = "grouped"
    profile["measured"]["cta_dispatch"]["candidate_policies"] = ["round_robin"]
    with pytest.raises(ProfileValidationError, match="candidate_policies"):
        validate_profile(profile)


def test_rejects_latency_curve_bw_exceeding_peak() -> None:
    profile = copy.deepcopy(load_profile(EXAMPLE_PROFILE))
    peak_mv = profile["measured"]["memory"]["dram_bw_read_gbps"]
    max_allowed = peak_mv["value"] * (1.0 + peak_mv["rel_uncertainty"])
    profile["measured"]["memory"]["dram_latency_by_bw"] = [
        {"achieved_bw_gbps": 0.0, "latency_ns": profile["measured"]["memory"]["dram_latency_ns"]},
        {
            "achieved_bw_gbps": max_allowed + 1.0,
            "latency_ns": profile["measured"]["memory"]["dram_latency_ns"],
        },
    ]
    with pytest.raises(ProfileValidationError, match="dram_bw_read_gbps"):
        validate_profile(profile)


def test_bundled_profiles_load_by_name_as_str_or_path():
    from pathlib import Path

    import virtualgpu
    from virtualgpu.api import _load_profile

    assert {"rtx4070", "gb10"} <= set(virtualgpu.list_profiles())
    for name in ("rtx4070", Path("rtx4070"), "rtx4070.json"):
        _, data = _load_profile(name)
        assert data["measured"]
    with pytest.raises(FileNotFoundError, match="Available bundled profiles"):
        _load_profile("no-such-gpu")
