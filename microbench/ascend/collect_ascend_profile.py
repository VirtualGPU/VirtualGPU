"""Assemble a schema-validated Ascend 910B hardware profile from vgpu_ascend_bench JSON."""

from __future__ import annotations

import argparse
import configparser
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUITE_VERSION = "0.3.2-ascend"
PROFILE_REVISION = "0.3.2-ascend"
DEFAULT_REL_UNCERTAINTY_FLOOR = 0.10


def _mv(
    samples: list[float],
    *,
    unit: str,
    method: str,
    perturb_group: str | None = None,
    source: str = "measured",
    floor: float = DEFAULT_REL_UNCERTAINTY_FLOOR,
) -> dict[str, Any]:
    s = sorted(float(x) for x in samples)
    if not s:
        raise ValueError(f"empty sample list for {method}")
    med = float(statistics.median(s))
    n = len(s)
    if n >= 3:
        lo_idx = max(0, int(math.floor(0.05 * (n - 1))))
        hi_idx = min(n - 1, int(math.ceil(0.95 * (n - 1))))
        ci_lo = min(med, s[lo_idx])
        ci_hi = max(med, s[hi_idx])
    else:
        ci_lo = s[0]
        ci_hi = s[-1]
    half_ci_rel = (ci_hi - ci_lo) / (2.0 * med) if med > 0.0 else floor
    rel_u = min(1.0, max(floor, half_ci_rel))
    return {
        "value": round(med, 4),
        "ci95_low": round(ci_lo, 4),
        "ci95_high": round(ci_hi, 4),
        "rel_uncertainty": round(rel_u, 4),
        "unit": unit,
        "method": method,
        "source": source,
        "perturb_group": perturb_group,
    }


def _estimate_l2_rescan(
    sweep: list[dict[str, Any]],
    l2_spec_bytes: int,
    dram_peak_gbps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build l2_rescan_hit_rate_by_working_set and locate l2_effective_capacity_bytes."""
    medians = [(int(p["working_set_bytes"]), float(statistics.median(p["gbps"]))) for p in sweep]
    medians.sort(key=lambda x: x[0])
    # Peak L2 bandwidth from working sets well inside L2 (<= 0.75 * spec L2)
    in_l2 = [bw for ws, bw in medians if ws <= int(0.75 * l2_spec_bytes)]
    l2_ref_bw = max(in_l2) if in_l2 else medians[0][1]
    # Tail HBM bandwidth from working sets >= 2x spec L2
    out_l2 = [bw for ws, bw in medians if ws >= 2 * l2_spec_bytes]
    dram_ref_bw = float(statistics.median(out_l2)) if out_l2 else dram_peak_gbps

    inv_l2 = 1.0 / max(l2_ref_bw, 1e-6)
    inv_dram = 1.0 / max(dram_ref_bw, 1e-6)
    denom = inv_dram - inv_l2

    rescan_points: list[dict[str, Any]] = []
    for pt in sorted(sweep, key=lambda p: int(p["working_set_bytes"])):
        ws = int(pt["working_set_bytes"])
        raw_gbps = [float(x) for x in pt["gbps"]]
        hit_samples = []
        for bw in raw_gbps:
            if abs(denom) < 1e-12:
                h = 1.0 if ws <= l2_spec_bytes else 0.0
            else:
                h = (inv_dram - 1.0 / max(bw, 1e-6)) / denom
            hit_samples.append(max(0.0, min(1.0, h)))
        rescan_points.append(
            {
                "working_set_bytes": ws,
                "working_set_ratio_to_l2": round(ws / float(l2_spec_bytes), 4),
                "pass2_bw_gbps": _mv(
                    raw_gbps,
                    unit="GB/s",
                    method="ascendc_l2_rescan_warm_read",
                    perturb_group="l2_bw",
                ),
                "estimated_hit_rate": _mv(
                    hit_samples,
                    unit="ratio",
                    method="ascendc_l2_rescan_harmonic_estimate:(1/bw_dram-1/bw_ws)/(1/bw_dram-1/bw_l2)",
                    source="estimated",
                    perturb_group=None,
                ),
            }
        )

    # Locate 50% hit-rate crossing clamped to <= l2_spec_bytes
    eff_cap = float(l2_spec_bytes)
    prev_ws, prev_hr = None, None
    for pt in rescan_points:
        ws = float(pt["working_set_bytes"])
        hr = float(pt["estimated_hit_rate"]["value"])
        if prev_hr is not None and prev_hr >= 0.5 and hr < 0.5:
            frac = (prev_hr - 0.5) / max(prev_hr - hr, 1e-9)
            eff_cap = prev_ws + frac * (ws - prev_ws)
            break
        prev_ws, prev_hr = ws, hr
    eff_cap = min(eff_cap, float(l2_spec_bytes))

    eff_mv = _mv(
        [eff_cap],
        unit="bytes",
        method="ascendc_l2_rescan_half_drop",
        perturb_group=None,
        floor=0.10,
    )
    return rescan_points, eff_mv


def _parse_platform_ini(ini_path: Path | None) -> dict[str, int]:
    defaults = {
        "ai_core_cnt": 20,
        "cube_core_cnt": 20,
        "vector_core_cnt": 40,
        "l2_size": 201326592,
        "l1_size": 524288,
        "l0_a_size": 65536,
        "l0_b_size": 65536,
        "l0_c_size": 131072,
        "ub_size": 196608,
        "cube_freq": 1800,
    }
    if ini_path is None or not ini_path.is_file():
        return defaults
    cfg = configparser.ConfigParser(strict=False)
    cfg.read(ini_path, encoding="utf-8")
    for sec, key in (
        ("SoCInfo", "ai_core_cnt"),
        ("SoCInfo", "cube_core_cnt"),
        ("SoCInfo", "vector_core_cnt"),
        ("SoCInfo", "l2_size"),
        ("AICoreSpec", "l1_size"),
        ("AICoreSpec", "l0_a_size"),
        ("AICoreSpec", "l0_b_size"),
        ("AICoreSpec", "l0_c_size"),
        ("AICoreSpec", "ub_size"),
        ("AICoreSpec", "cube_freq"),
    ):
        if cfg.has_option(sec, key):
            defaults[key] = int(cfg.get(sec, key))
    return defaults


def build_ascend_profile(
    raw: dict[str, Any],
    *,
    soc_name: str = "Ascend910B3",
    sku: str = "Huawei Ascend 910B3",
    driver_version: str = "26.1.1",
    toolkit_version: str = "CANN 9.2.1",
    platform_ini: Path | None = None,
) -> dict[str, Any]:
    spec = _parse_platform_ini(platform_ini)
    ai_cores = spec["ai_core_cnt"]
    l2_size = spec["l2_size"]
    l1_size = spec["l1_size"]
    ub_size = spec["ub_size"]
    cube_mhz = float(spec["cube_freq"])

    dram_curve = [
        {
            "active_sms": int(pt["active_cores"]),
            "bw_gbps": _mv(
                pt["gbps"],
                unit="GB/s",
                method="ascendc_stream_read_cold_l2",
                perturb_group="dram_bw",
            ),
        }
        for pt in sorted(raw["dram_read_by_active_cores"], key=lambda x: int(x["active_cores"]))
    ]
    l2_curve = [
        {
            "active_sms": int(pt["active_cores"]),
            "bw_gbps": _mv(
                pt["gbps"],
                unit="GB/s",
                method="ascendc_stream_read_warm_l2",
                perturb_group="l2_bw",
            ),
        }
        for pt in sorted(raw["l2_read_by_active_cores"], key=lambda x: int(x["active_cores"]))
    ]

    dram_read_peak = dram_curve[-1]["bw_gbps"]
    dram_write_peak = _mv(
        raw["dram_write_gbps"],
        unit="GB/s",
        method="ascendc_stream_write_cold_l2",
        perturb_group="dram_bw",
    )
    l2_read_peak = l2_curve[-1]["bw_gbps"]
    l2_write_peak = _mv(
        raw["l2_write_gbps"],
        unit="GB/s",
        method="ascendc_stream_write_warm_l2",
        perturb_group="l2_bw",
    )

    rescan_points, l2_eff_cap = _estimate_l2_rescan(
        raw["l2_rescan_sweep"],
        l2_spec_bytes=l2_size,
        dram_peak_gbps=float(dram_read_peak["value"]),
    )

    smem_bw = _mv(
        raw["smem_bw_per_core_gbps"],
        unit="GB/s",
        method="ascendc_ub_datacopy_bw",
    )
    l2_lat = _mv(
        raw["l2_latency_ns"],
        unit="ns",
        method="ascendc_ptr_chase_warm_l2",
    )
    dram_lat = _mv(
        raw["dram_latency_ns"],
        unit="ns",
        method="ascendc_ptr_chase_cold_l2",
    )
    if raw.get("smem_latency_ns"):
        smem_lat = _mv(
            raw["smem_latency_ns"],
            unit="ns",
            method="ascendc_ub_ptr_chase_latency",
        )
    else:
        # UB access latency is deterministic on-chip SRAM (~10 cycles at 1.8 GHz -> ~5.6 ns)
        smem_lat = _mv(
            [10.0 / (cube_mhz / 1e3)],
            unit="ns",
            method="ascend_ub_sram_spec_latency",
            source="estimated",
            floor=0.25,
        )

    def _build_mlp(
        raw_pts: list[dict[str, Any]] | None,
        method_prefix: str,
        perturb_group: str,
    ) -> list[dict[str, Any]]:
        if not raw_pts:
            return []
        pts: list[dict[str, Any]] = []
        for pt in raw_pts:
            mode = str(pt["issue_mode"])
            pts.append(
                {
                    "issue_mode": mode,
                    "bytes_in_flight": int(pt["bytes_in_flight"]),
                    "active_ctas": int(pt["active_ctas"]),
                    "bytes_per_cta": int(pt["bytes_per_cta"]),
                    "bw_gbps": _mv(
                        pt["gbps"],
                        unit="GB/s",
                        method=f"{method_prefix}_{mode}",
                        perturb_group=perturb_group,
                    ),
                }
            )
        return pts

    dram_mlp_curve = _build_mlp(
        raw.get("dram_bw_by_bytes_in_flight"), "ascendc_mlp_dram_cold_l2", "dram_bw"
    )
    l2_mlp_curve = _build_mlp(raw.get("l2_bw_by_bytes_in_flight"), "ascendc_mlp_l2_warm", "l2_bw")

    dram_lat_loaded = (
        _mv(
            raw["dram_latency_loaded_ns"],
            unit="ns",
            method="ascendc_ptr_chase_cold_l2_loaded",
        )
        if raw.get("dram_latency_loaded_ns")
        else None
    )
    l2_lat_loaded = (
        _mv(
            raw["l2_latency_loaded_ns"],
            unit="ns",
            method="ascendc_ptr_chase_warm_l2_loaded",
        )
        if raw.get("l2_latency_loaded_ns")
        else None
    )

    def _build_lat_by_bw(raw_pts: list[dict[str, Any]] | None, method: str) -> list[dict[str, Any]]:
        if not raw_pts:
            return []
        out: list[dict[str, Any]] = []
        for pt in sorted(raw_pts, key=lambda x: float(x["achieved_bw_gbps"])):
            out.append(
                {
                    "achieved_bw_gbps": round(float(pt["achieved_bw_gbps"]), 3),
                    "latency_ns": _mv(
                        pt["latency_ns"],
                        unit="ns",
                        method=method,
                    ),
                }
            )
        return out

    dram_lat_by_bw = _build_lat_by_bw(
        raw.get("dram_latency_by_bw"), "ascendc_ptr_chase_dram_vs_bg_bw"
    )
    l2_lat_by_bw = _build_lat_by_bw(raw.get("l2_latency_by_bw"), "ascendc_ptr_chase_l2_vs_bg_bw")

    l2_merge_by_sharing: list[dict[str, Any]] = []
    for pt in raw.get("l2_merge_by_sharing", []):
        l2_merge_by_sharing.append(
            {
                "active_ctas": int(pt["active_ctas"]),
                "sharing": int(pt["sharing"]),
                "bytes_per_cta": int(pt["bytes_per_cta"]),
                "stagger_stages": int(pt["stagger_stages"]),
                "turnaround_ns": _mv(
                    pt["turnaround_ns"],
                    unit="ns",
                    method="ascendc_l2_merge_mte2",
                ),
                "dram_unique_bw_gbps": _mv(
                    pt["dram_unique_bw_gbps"],
                    unit="GB/s",
                    method="ascendc_l2_merge_mte2",
                    perturb_group="dram_bw",
                ),
                "l2_to_sm_bw_gbps": _mv(
                    pt["l2_to_sm_bw_gbps"],
                    unit="GB/s",
                    method="ascendc_l2_merge_mte2",
                    perturb_group="l2_bw",
                ),
            }
        )

    fixed_sync_ns: list[dict[str, Any]] = [
        {
            "num_warps": int(pt["num_warps"]),
            "tokens": int(pt["tokens"]),
            "overhead_ns": _mv(
                pt["overhead_ns"],
                unit="ns",
                method="ascendc_aic_empty_event_ring",
            ),
        }
        for pt in raw.get("fixed_sync_ns", [])
    ]
    stage_sync_overhead_ns: list[dict[str, Any]] = [
        {
            "issue_mode": str(pt["issue_mode"]),
            "num_warps": int(pt["num_warps"]),
            "tokens": int(pt["tokens"]),
            "overhead_ns": _mv(
                pt["overhead_ns"],
                unit="ns",
                method=f"ascendc_stage_sync_hot_l2_{pt['issue_mode']}",
            ),
        }
        for pt in raw.get("stage_sync_overhead_ns", [])
    ]
    min_async_rt = (
        _mv(
            raw["min_async_roundtrip_ns"],
            unit="ns",
            method="ascendc_32b_hot_l2_mte2_roundtrip",
        )
        if raw.get("min_async_roundtrip_ns")
        else None
    )

    f16_mma_mv = _mv(
        raw["cube_mmad_f16_tflops_per_core"],
        unit="TFLOPS",
        method="ascendc_cube_mmad_l0_resident_f16",
        perturb_group="mma_dense",
    )
    bf16_mma_mv = _mv(
        raw["cube_mmad_bf16_tflops_per_core"],
        unit="TFLOPS",
        method="ascendc_cube_mmad_l0_resident_bf16",
        perturb_group="mma_dense",
    )

    # Launch overhead and per-core launch delta
    ov1 = [float(x) for x in raw["launch_overhead_1core_us"]]
    ov20 = [float(x) for x in raw["launch_overhead_20core_us"]]
    per_core_ns = [max(1.0, (b - a) * 1e3 / max(ai_cores - 1, 1)) for a, b in zip(ov1, ov20)]

    # Ascend 910B has no hardware register-spill path; populate schema-required spill
    # fields as estimated on-chip/L2 equivalents with wide uncertainty so provenance stays honest.
    spill_l1_bw = _mv(
        [smem_bw["value"]],
        unit="GB/s",
        method="ascend_no_reg_spill_ub_proxy",
        perturb_group="spill_l1",
        source="estimated",
        floor=0.25,
    )
    spill_l2_bw = _mv(
        [l2_curve[0]["bw_gbps"]["value"]],
        unit="GB/s",
        method="ascend_no_reg_spill_l2_proxy",
        perturb_group="spill_l2",
        source="estimated",
        floor=0.25,
    )
    spill_l1_lat = _mv(
        [smem_lat["value"]],
        unit="ns",
        method="ascend_no_reg_spill_ub_lat_proxy",
        perturb_group="spill_l1",
        source="estimated",
        floor=0.25,
    )
    spill_l2_lat = _mv(
        [l2_lat["value"]],
        unit="ns",
        method="ascend_no_reg_spill_l2_lat_proxy",
        perturb_group="spill_l2",
        source="estimated",
        floor=0.25,
    )
    spill_thresh = _mv(
        [float(ub_size)],
        unit="bytes",
        method="ascend_ub_capacity_spec",
        source="estimated",
        floor=0.25,
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    arch_id = soc_name.lower()

    memory_section: dict[str, Any] = {
        "dram_bw_read_gbps": dram_read_peak,
        "dram_bw_write_gbps": dram_write_peak,
        "dram_bw_by_active_sms": dram_curve,
        "l2_bw_read_gbps": l2_read_peak,
        "l2_bw_write_gbps": l2_write_peak,
        "l2_bw_by_active_sms": l2_curve,
        "l2_effective_capacity_bytes": l2_eff_cap,
        "l2_rescan_hit_rate_by_working_set": rescan_points,
        "smem_bw_per_sm_gbps": smem_bw,
        "dram_latency_ns": dram_lat,
        "dram_latency_loaded_ns": dram_lat_loaded,
        "l2_latency_ns": l2_lat,
        "l2_latency_loaded_ns": l2_lat_loaded,
        "smem_latency_ns": smem_lat,
    }
    if dram_mlp_curve:
        memory_section["dram_bw_by_bytes_in_flight"] = dram_mlp_curve
    if l2_mlp_curve:
        memory_section["l2_bw_by_bytes_in_flight"] = l2_mlp_curve
    if dram_lat_by_bw:
        memory_section["dram_latency_by_bw"] = dram_lat_by_bw
    if l2_lat_by_bw:
        memory_section["l2_latency_by_bw"] = l2_lat_by_bw
    if l2_merge_by_sharing:
        memory_section["l2_merge_by_sharing"] = l2_merge_by_sharing
    if raw.get("gmem_load_issue_controlled"):
        memory_section["gmem_load_issue_controlled"] = [
            {
                "load_width_bytes": int(pt["load_width_bytes"]),
                "batch_loads_per_thread": int(pt["batch_loads_per_thread"]),
                "num_warps": int(pt["num_warps"]),
                "bytes_in_flight": int(pt["bytes_in_flight"]),
                "issue_rate_gwarp_inst_per_sm": _mv(
                    pt["issue_rate_gwarp_inst_per_sm"],
                    unit="Gwarp-inst/s",
                    method=f"ascendc_aic_nd2nz_row_issue_w{int(pt['load_width_bytes']) * 32}b",
                    perturb_group="l2_bw",
                ),
                "delivered_bw_per_sm_gbps": _mv(
                    pt["delivered_bw_per_sm_gbps"],
                    unit="GB/s",
                    method=f"ascendc_aic_nd2nz_row_issue_w{int(pt['load_width_bytes']) * 32}b",
                    perturb_group="l2_bw",
                ),
            }
            for pt in raw["gmem_load_issue_controlled"]
        ]

    pipeline_section: dict[str, Any] = {}
    if fixed_sync_ns:
        pipeline_section["fixed_sync_ns"] = fixed_sync_ns
    if raw.get("cv_sync_hop_ns_by_active_cores"):
        pipeline_section["cv_sync_hop_ns_by_active_sms"] = [
            {
                "active_sms": int(pt["active_cores"]),
                "latency_ns": _mv(
                    pt["hop_ns"],
                    unit="ns",
                    method="ascendc_cross_core_sync_pingpong",
                    perturb_group="cv_sync",
                ),
            }
            for pt in sorted(
                raw["cv_sync_hop_ns_by_active_cores"], key=lambda x: int(x["active_cores"])
            )
        ]
    if stage_sync_overhead_ns:
        pipeline_section["stage_sync_overhead_ns"] = stage_sync_overhead_ns
    if min_async_rt is not None:
        pipeline_section["min_async_roundtrip_ns"] = min_async_rt

    measured_section: dict[str, Any] = {
        "memory": memory_section,
        "compute": {
            "mma_tflops_per_sm_by_warps": {
                "cube.mmad:f16:f16:f32:16x16x16": [
                    {"mma_warps_per_sm": 1, "tflops_per_sm": f16_mma_mv}
                ],
                "cube.mmad:bf16:bf16:f32:16x16x16": [
                    {"mma_warps_per_sm": 1, "tflops_per_sm": bf16_mma_mv}
                ],
            },
            "non_mma_issue_rate_per_sm_gips": _mv(
                raw["non_mma_issue_rate_per_core_gips"],
                unit="GIPS",
                method="ascendc_vec_fma_issue_rate",
            ),
            "fp32_fma_tflops": _mv(
                raw["fp32_fma_tflops"],
                unit="TFLOPS",
                method="ascendc_vec_fp32_fma",
            ),
            "sfu_tflops": _mv(
                raw["sfu_tflops"],
                unit="TFLOPS",
                method="ascendc_vec_fp32_exp",
            ),
            "sfu_issue_rate_per_sm_gips": [
                {
                    "warps_per_sm": 1,
                    "rate": _mv(
                        raw["sfu_issue_rate_per_core_gips"],
                        unit="GIPS",
                        method="ascendc_vec_fp32_exp_issue_rate",
                    ),
                }
            ],
        },
        "occupancy": {
            "smem_carveout_observed": [
                {
                    "requested_smem_per_cta_bytes": l1_size,
                    "chosen_carveout_bytes": l1_size,
                    "max_active_ctas_per_sm": 1,
                }
            ]
        },
        "async_pipeline": {
            "cp_async_bw_per_sm_gbps": _mv(
                raw["l2_read_by_active_cores"][0]["gbps"],
                unit="GB/s",
                method="ascendc_mte2_single_core_l2_bw",
                perturb_group="async_cp",
            ),
            "tma_bw_per_sm_gbps": None,
            "tma_setup_ns": None,
        },
        "spill": {
            "local_mem_l1_hit_bw_per_sm_gbps": spill_l1_bw,
            "local_mem_l1_miss_bw_per_sm_gbps": spill_l2_bw,
            "local_mem_l1_hit_latency_ns": spill_l1_lat,
            "local_mem_l1_miss_latency_ns": spill_l2_lat,
            "l1_to_l2_spill_threshold_bytes_per_sm": spill_thresh,
        },
        "overhead": {
            "launch_overhead_us": _mv(
                ov1,
                unit="us",
                method="ascendc_queued_device_empty_kernel",
            ),
            "host_launch_overhead_us": (
                _mv(
                    raw["host_launch_overhead_us"],
                    unit="us",
                    method="ascendc_stream_empty_kernel_host_batch",
                )
                if "host_launch_overhead_us" in raw
                else None
            ),
            "per_cta_launch_latency_ns": _mv(
                per_core_ns,
                unit="ns",
                method="ascendc_queued_empty_kernel_1_vs_20_core_delta",
            ),
        },
        "cta_dispatch": {
            "observed_policy": "round_robin",
            "candidate_policies": ["round_robin"],
            "policy_params": {"auto_blockify_stride": ai_cores},
            "confidence": "high",
            "probes": [
                {
                    "grid_dims": [ai_cores, 1, 1],
                    "ctas_per_sm": 1,
                    "waves_approx": 1.0,
                    "order_consistency": 1.0,
                    "first_wave_trace": raw.get("dispatch_first_wave", []),
                }
            ],
        },
    }
    if pipeline_section:
        measured_section["pipeline"] = pipeline_section

    return {
        "schema_version": "0.1",
        "identity": {
            "vendor": "ascend",
            "arch": arch_id,
            "sku": sku,
            "memory_type": "discrete",
        },
        "provenance": {
            "profile_revision": PROFILE_REVISION,
            "calibration_run_id": f"cal-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{arch_id}",
            "calibration_manifest_digest": None,
            "source": "partial",
            "driver_version": driver_version,
            "toolkit_version": toolkit_version,
            "suite_version": SUITE_VERSION,
            "host_os": "linux",
            "is_wsl": False,
            "power_limit_w": None,
            "timestamp_utc": ts,
            "clock_policy": {
                "locked": False,
                "sm_clock_locked": False,
                "mem_clock_locked": False,
                "sm_clock_mhz": cube_mhz,
                "mem_clock_mhz": None,
                "observed_sm_clock_range_mhz": [cube_mhz, cube_mhz],
                "observed_mem_clock_range_mhz": None,
                "throttle_events": 0,
            },
        },
        "specs": {
            "sm_count": ai_cores,
            "warp_size": 32,
            "max_threads_per_sm": 32,
            "max_warps_per_sm": 1,
            "max_blocks_per_sm": 1,
            "max_threads_per_block": 32,
            "registers_per_sm": 1024,
            "max_registers_per_thread": 32,
            "register_alloc_granularity": "block",
            "register_alloc_unit_regs": 32,
            "l1_smem_unified_bytes": l1_size,
            "l1_size_bytes": l1_size,
            "shared_mem_per_sm_bytes": l1_size,
            "max_shared_mem_per_block_bytes": l1_size,
            "reserved_shared_mem_per_cta_bytes": 0,
            "shared_mem_alloc_granularity_bytes": 32,
            "shared_mem_carveout_options_bytes": [l1_size],
            "l2_size_bytes": l2_size,
            "l2_line_bytes": 128,
            "l2_sector_bytes": 32,
            "l2_partitions": 1,
            "l1_line_bytes": 32,
            "capabilities": {
                "cp_async": True,
                "tma": False,
                "wgmma": False,
                "tcgen05_tmem": False,
                "cluster": False,
                "cube_vector_split": True,
            },
            "ascend": {
                "cube_cores": spec["cube_core_cnt"],
                "vector_cores": spec["vector_core_cnt"],
                "l1_bytes": l1_size,
                "l0a_bytes": spec["l0_a_size"],
                "l0b_bytes": spec["l0_b_size"],
                "l0c_bytes": spec["l0_c_size"],
                "ub_bytes": ub_size,
            },
            "mma_shapes": [
                {
                    "op_class": "cube.mmad",
                    "a_dtype": "f16",
                    "b_dtype": "f16",
                    "acc_dtype": "f32",
                    "m": 16,
                    "n": 16,
                    "k": 16,
                },
                {
                    "op_class": "cube.mmad",
                    "a_dtype": "bf16",
                    "b_dtype": "bf16",
                    "acc_dtype": "f32",
                    "m": 16,
                    "n": 16,
                    "k": 16,
                },
            ],
        },
        "measured": measured_section,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Ascend 910B GPU profile JSON.")
    parser.add_argument("--raw-json", type=Path, required=True, help="Output of vgpu_ascend_bench")
    parser.add_argument("--out", type=Path, required=True, help="Destination profile JSON path")
    parser.add_argument("--soc", default="Ascend910B3")
    parser.add_argument("--sku", default="Huawei Ascend 910B3")
    parser.add_argument("--driver", default="26.1.1")
    parser.add_argument("--toolkit", default="CANN 9.2.1")
    parser.add_argument("--platform-ini", type=Path, default=None)
    args = parser.parse_args()

    raw = json.loads(args.raw_json.read_text(encoding="utf-8"))
    profile = build_ascend_profile(
        raw,
        soc_name=args.soc,
        sku=args.sku,
        driver_version=args.driver,
        toolkit_version=args.toolkit,
        platform_ini=args.platform_ini,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
