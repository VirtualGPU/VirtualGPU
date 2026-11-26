"""Run M1 CUDA microbenchmarks and assemble a validated VirtualGPU profile JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from virtualgpu.profile import validate_profile  # noqa: E402

DEFAULT_BIN = Path(__file__).resolve().parent / "build" / "vgpu_bench"
SUITE_VERSION = "0.21.0"
PROFILE_REVISION = "0.21.0-posthoc"


def find_nvidia_smi() -> str | None:
    for candidate in (shutil.which("nvidia-smi"), "/usr/lib/wsl/lib/nvidia-smi"):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def sample_nvidia_smi(device: int) -> dict[str, Any]:
    smi = find_nvidia_smi()
    if smi is None:
        return {}
    fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.limit",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
    ]
    cmd = [
        smi,
        f"--id={device}",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return {}

    def parse_float(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    throttled = any(p.lower() == "active" for p in parts[4:])
    return {
        "driver_version": parts[0],
        "sm_clock_mhz": parse_float(parts[1]),
        "mem_clock_mhz": parse_float(parts[2]),
        "power_limit_w": parse_float(parts[3]),
        "throttled": throttled,
    }


def run_bench_with_clock_polling(
    binary: Path,
    mode: str,
    device: int,
    extra_args: list[str] | None = None,
    poll_interval_s: float = 0.15,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run vgpu_bench while polling nvidia-smi under active load (avoids idle P8 clocks)."""
    import tempfile
    import time

    cmd = [str(binary), "--mode", mode, "--device", str(device)]
    if extra_args:
        cmd.extend(extra_args)
    # Write stdout to a temp file so large JSON outputs (>64 KB) never deadlock the pipe.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as out_f,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as err_f,
    ):
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
        samples: list[dict[str, Any]] = []
        time.sleep(poll_interval_s)
        while proc.poll() is None:
            snap = sample_nvidia_smi(device)
            if snap:
                samples.append(snap)
            time.sleep(poll_interval_s)
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read()
        stderr = err_f.read()
        if proc.returncode != 0:
            if stderr.strip():
                print(stderr.strip(), file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    if not samples:
        snap = sample_nvidia_smi(device)
        if snap:
            samples.append(snap)
    return json.loads(stdout), samples


def bootstrap_median_ci(
    samples: list[float], n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    med = float(statistics.median(samples))
    if len(samples) < 2:
        return med, med, med
    rng = random.Random(seed)
    n = len(samples)
    boots = [statistics.median(rng.choices(samples, k=n)) for _ in range(n_boot)]
    boots.sort()
    lo = float(boots[int(0.025 * n_boot)])
    hi = float(boots[int(0.975 * n_boot)])
    return med, min(lo, med), max(hi, med)


def unlocked_rel_uncertainty(
    med: float,
    ci_lo: float,
    ci_hi: float,
    clock_range: list[float] | None,
    locked: bool,
    throttle_events: int,
) -> float:
    half_ci_rel = (ci_hi - ci_lo) / (2.0 * med) if med > 0 else 0.10
    unc = max(0.10, half_ci_rel)
    if not locked:
        if clock_range is not None and clock_range[0] > 0 and clock_range[1] >= clock_range[0]:
            mid = 0.5 * (clock_range[0] + clock_range[1])
            drift_rel = (clock_range[1] - clock_range[0]) / (2.0 * mid)
            unc = max(unc, drift_rel + 0.05)
        else:
            unc = max(unc, 0.15)
    if throttle_events > 0:
        unc += 0.05
    return round(min(unc, 1.0), 4)


def make_measured(
    samples: list[float],
    unit: str,
    method: str,
    perturb_group: str | None,
    clock_range: list[float] | None,
    locked: bool,
    throttle_events: int,
) -> dict[str, Any]:
    med, lo, hi = bootstrap_median_ci(samples)
    rel_unc = unlocked_rel_uncertainty(med, lo, hi, clock_range, locked, throttle_events)
    return {
        "value": round(med, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "rel_uncertainty": rel_unc,
        "unit": unit,
        "method": method,
        "source": "measured",
        "perturb_group": perturb_group,
    }


def make_estimated(
    value: float,
    unit: str,
    method: str,
    perturb_group: str | None,
    rel_uncertainty: float = 0.30,
) -> dict[str, Any]:
    return {
        "value": round(value, 4),
        "ci95_low": None,
        "ci95_high": None,
        "rel_uncertainty": rel_uncertainty,
        "unit": unit,
        "method": method,
        "source": "estimated",
        "perturb_group": perturb_group,
    }


def classify_dispatch(dispatch_raw: dict[str, Any]) -> dict[str, Any]:
    sm_count = dispatch_raw["sm_count"]
    probes = dispatch_raw["probes"]
    if not probes:
        return {
            "observed_policy": "unknown",
            "candidate_policies": ["round_robin", "grouped", "unknown"],
            "policy_params": {},
            "confidence": "low",
            "probes": [],
        }

    consistencies = [p["order_consistency"] for p in probes]
    avg_consistency = sum(consistencies) / len(consistencies)

    # Check whether CTA i maps to (i % sm_count) or contiguous groups on 1-CTA/SM first wave.
    rr_matches = 0
    rr_total = 0
    for p in probes:
        trace = p["first_wave_trace"]
        if p["ctas_per_sm"] == 1 and len(trace) >= sm_count:
            for rec in trace[:sm_count]:
                rr_total += 1
                if rec["sm_id"] == (rec["cta_id"] % sm_count):
                    rr_matches += 1
    rr_ratio = (rr_matches / rr_total) if rr_total > 0 else 0.0

    if rr_ratio >= 0.90 and avg_consistency >= 0.85:
        policy = "round_robin"
        candidates = ["round_robin"]
        confidence = "high"
    elif avg_consistency >= 0.85:
        policy = "measured_table"
        candidates = ["measured_table", "round_robin"]
        confidence = "medium"
    else:
        policy = "unknown"
        candidates = ["round_robin", "grouped", "unknown"]
        confidence = "low"

    return {
        "observed_policy": policy,
        "candidate_policies": candidates,
        "policy_params": {"rr_exact_match_ratio": round(rr_ratio, 4)},
        "confidence": confidence,
        "probes": probes,
    }


def is_wsl() -> bool:
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def arch_carveout_options(major: int, minor: int, smem_per_sm: int) -> list[int]:
    if major == 8 and minor == 9:
        return [0, 8192, 16384, 32768, 65536, 102400]
    if major == 12:
        return [0, 8192, 16384, 32768, 65536, 102400]
    return sorted({0, smem_per_sm // 4, smem_per_sm // 2, smem_per_sm})


def arch_unified_l1_bytes(major: int, minor: int, smem_per_sm: int) -> int:
    """Physical unified L1/Tex + shared memory bytes per SM, including reserved L1."""
    by_sm = {
        (7, 5): 98304,
        (8, 0): 196608,
        (8, 6): 131072,
        (8, 7): 196608,
        (8, 9): 131072,
        (9, 0): 262144,
    }
    if (major, minor) in by_sm:
        return by_sm[(major, minor)]
    if major == 10:
        return 262144
    if major == 12:
        return 131072
    return smem_per_sm


def estimate_l2_knee(sweep: list[dict[str, Any]], l2_spec_bytes: int) -> tuple[float, float]:
    """Locate the effective L2 capacity where bandwidth first drops below 85% of the L2 plateau."""
    if len(sweep) < 3:
        return float(l2_spec_bytes) * 0.85, 0.20
    bw_l2 = statistics.median(pt["median_gbps"] for pt in sweep[:2])
    bw_dram = statistics.median(pt["median_gbps"] for pt in sweep[-2:])
    if bw_l2 <= bw_dram * 1.1:
        return float(l2_spec_bytes) * 0.85, 0.20
    thresh = bw_l2 * 0.85
    for i in range(1, len(sweep)):
        prev_ws, prev_bw = (
            float(sweep[i - 1]["working_set_bytes"]),
            float(sweep[i - 1]["median_gbps"]),
        )
        cur_ws, cur_bw = float(sweep[i]["working_set_bytes"]), float(sweep[i]["median_gbps"])
        if cur_bw <= thresh <= prev_bw and prev_bw > cur_bw:
            frac = (prev_bw - thresh) / (prev_bw - cur_bw)
            knee = min(float(l2_spec_bytes), prev_ws + frac * (cur_ws - prev_ws))
            step_rel = (cur_ws - prev_ws) / (2.0 * knee) if knee > 0 else 0.10
            return knee, round(max(0.10, min(step_rel, 0.50)), 4)
    return float(l2_spec_bytes) * 0.85, 0.20


def build_profile(
    query: dict[str, Any],
    dram_raw: dict[str, Any],
    dispatch_raw: dict[str, Any],
    mma_raw: dict[str, Any],
    l2_raw: dict[str, Any],
    cp_async_raw: dict[str, Any],
    latency_raw: dict[str, Any],
    launch_raw: dict[str, Any],
    lat_curve_raw: dict[str, Any],
    l2_rescan_raw: dict[str, Any],
    mlp_raw: dict[str, Any],
    closed_loop_raw: dict[str, Any],
    l2_merge_raw: dict[str, Any],
    stage_oh_raw: dict[str, Any],
    closed_loop_mma_raw: dict[str, Any],
    mma_chain_raw: dict[str, Any],
    serial_fill_raw: dict[str, Any],
    sync_loop_raw: dict[str, Any],
    gmem_issue_raw: dict[str, Any],
    spill_lat_raw: dict[str, Any],
    strided_mlp_raw: dict[str, Any],
    sfu_shfl_raw: dict[str, Any],
    mma_sf_raw: dict[str, Any],
    sync_abl_raw: dict[str, Any],
    gmem_ctrl_raw: dict[str, Any],
    l1_carve_raw: dict[str, Any],
    sync_multicta_raw: dict[str, Any],
    sync_reconcile_raw: dict[str, Any],
    smi_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    major = int(query["major"])
    minor = int(query["minor"])
    arch = f"sm_{major}{minor}"
    sm_count = int(query["sm_count"])
    max_warps_per_sm = int(query["max_warps_per_sm"])
    l2_size = int(query["l2_size_bytes"])
    smem_per_sm = int(query["shared_mem_per_sm_bytes"])
    max_smem_block = int(
        query["max_shared_mem_per_block_optin_bytes"] or query["shared_mem_per_block_bytes"]
    )
    reserved_smem = int(query["reserved_shared_mem_per_cta_bytes"])

    # Aggregate active-load clocks from nvidia-smi snapshots (exclude idle P8 transitions < 50% peak).
    raw_sm_clocks = [s["sm_clock_mhz"] for s in smi_samples if s.get("sm_clock_mhz")]
    raw_mem_clocks = [s["mem_clock_mhz"] for s in smi_samples if s.get("mem_clock_mhz")]
    sm_clocks = [c for c in raw_sm_clocks if c >= 0.5 * max(raw_sm_clocks)] if raw_sm_clocks else []
    mem_clocks = (
        [c for c in raw_mem_clocks if c >= 0.5 * max(raw_mem_clocks)] if raw_mem_clocks else []
    )
    sm_range = [min(sm_clocks), max(sm_clocks)] if sm_clocks else None
    mem_range = [min(mem_clocks), max(mem_clocks)] if mem_clocks else None
    throttle_events = sum(1 for s in smi_samples if s.get("throttled"))

    driver_str = next(
        (s["driver_version"] for s in smi_samples if s.get("driver_version")),
        str(query["driver_version"]),
    )
    power_limit = next((s["power_limit_w"] for s in smi_samples if s.get("power_limit_w")), None)
    runtime_ver = int(query["runtime_version"])
    toolkit_str = f"{runtime_ver // 1000}.{(runtime_ver % 1000) // 10}"

    # Build measured DRAM read curve (keyed by target_sms; use observed_sms if strictly increasing).
    read_curve_points: list[dict[str, Any]] = []
    seen_sms: set[int] = set()
    for pt in dram_raw["read_curve"]:
        active = int(pt["target_sms"])
        if active in seen_sms:
            continue
        seen_sms.add(active)
        mv = make_measured(
            pt["gbps_samples"],
            unit="GB/s",
            method="cuda_stream_read_cold_l2",
            perturb_group="dram_bw",
            clock_range=mem_range,
            locked=False,
            throttle_events=throttle_events,
        )
        read_curve_points.append({"active_sms": active, "bw_gbps": mv})
    read_curve_points.sort(key=lambda x: x["active_sms"])

    full_read_mv = read_curve_points[-1]["bw_gbps"]
    full_write_samples = dram_raw["write_full"][-1]["gbps_samples"]
    full_write_mv = make_measured(
        full_write_samples,
        unit="GB/s",
        method="cuda_stream_write_cold_l2",
        perturb_group="dram_bw",
        clock_range=mem_range,
        locked=False,
        throttle_events=throttle_events,
    )

    # Build measured L2 read curves (ld.global.cg and cp.async.cg), write BW, and capacity knee.
    l2_curve_points: list[dict[str, Any]] = []
    seen_l2_sms: set[int] = set()
    for pt in l2_raw["read_curve"]:
        active = int(pt["target_sms"])
        if active in seen_l2_sms:
            continue
        seen_l2_sms.add(active)
        mv = make_measured(
            pt["gbps_samples"],
            unit="GB/s",
            method="cuda_l2_cg_read_warm_ilp8_16w",
            perturb_group="l2_bw",
            clock_range=sm_range,
            locked=False,
            throttle_events=throttle_events,
        )
        l2_curve_points.append({"active_sms": active, "bw_gbps": mv})
    l2_curve_points.sort(key=lambda x: x["active_sms"])

    l2_cp_async_curve_points: list[dict[str, Any]] = []
    seen_l2_cp_sms: set[int] = set()
    for pt in l2_raw.get("read_curve_cp_async", []):
        active = int(pt["target_sms"])
        if active in seen_l2_cp_sms:
            continue
        seen_l2_cp_sms.add(active)
        mv = make_measured(
            pt["gbps_samples"],
            unit="GB/s",
            method="cuda_l2_cp_async_cg_read_warm",
            perturb_group="l2_bw",
            clock_range=sm_range,
            locked=False,
            throttle_events=throttle_events,
        )
        l2_cp_async_curve_points.append({"active_sms": active, "bw_gbps": mv})
    l2_cp_async_curve_points.sort(key=lambda x: x["active_sms"])

    full_l2_read_mv = l2_curve_points[-1]["bw_gbps"]
    if (
        l2_cp_async_curve_points
        and l2_cp_async_curve_points[-1]["bw_gbps"]["value"] > full_l2_read_mv["value"]
    ):
        full_l2_read_mv = l2_cp_async_curve_points[-1]["bw_gbps"]
    full_l2_write_mv = make_measured(
        l2_raw["write_full_gbps_samples"],
        unit="GB/s",
        method="cuda_l2_cg_write_warm",
        perturb_group="l2_bw",
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    l2_knee_bytes, l2_knee_unc = estimate_l2_knee(l2_raw["capacity_sweep"], l2_size)
    l2_eff_mv = {
        "value": round(l2_knee_bytes),
        "ci95_low": None,
        "ci95_high": None,
        "rel_uncertainty": l2_knee_unc,
        "unit": "bytes",
        "method": "cuda_l2_working_set_sweep",
        "source": "measured",
        "perturb_group": None,
    }

    # Build measured MMA curves per SM by issuing warps.
    mma_curves: dict[str, list[dict[str, Any]]] = {}
    for op_key, pts in mma_raw["mma"].items():
        curve: list[dict[str, Any]] = []
        for pt in pts:
            mv = make_measured(
                pt["tflops_per_sm_samples"],
                unit="TFLOP/s",
                method="cuda_ptx_mma_sync_unrolled",
                perturb_group="mma_dense",
                clock_range=sm_range,
                locked=False,
                throttle_events=throttle_events,
            )
            curve.append({"mma_warps_per_sm": int(pt["mma_warps_per_sm"]), "tflops_per_sm": mv})
        curve.sort(key=lambda x: x["mma_warps_per_sm"])
        mma_curves[op_key] = curve

    fp32_fma_mv = make_measured(
        mma_raw["fp32_fma_tflops_samples"],
        unit="TFLOP/s",
        method="cuda_fp32_fma_unrolled",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )

    # Measured cp.async (from L2), smem BW (128-bit read-only), latency hierarchy, and launch overhead.
    cp_async_mv = make_measured(
        cp_async_raw["cp_async_bw_per_sm_gbps_samples"],
        unit="GB/s",
        method="cuda_ptx_cp_async_cg_16b_from_l2",
        perturb_group="async_cp",
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    smem_bw_mv = make_measured(
        cp_async_raw["smem_bw_per_sm_gbps_samples"],
        unit="GB/s",
        method="cuda_smem_128b_readonly_16warps",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    smem_lat_mv = make_measured(
        latency_raw["smem_latency_ns_samples"],
        unit="ns",
        method="cuda_smem_pchase",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    l2_lat_mv = make_measured(
        latency_raw["l2_latency_ns_samples"],
        unit="ns",
        method="cuda_l2_cg_pchase_unloaded",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    l2_lat_loaded_mv = make_measured(
        latency_raw["l2_latency_loaded_ns_samples"],
        unit="ns",
        method="cuda_l2_cg_pchase_loaded",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    dram_lat_mv = make_measured(
        latency_raw["dram_latency_ns_samples"],
        unit="ns",
        method="cuda_dram_cg_pchase_unloaded",
        perturb_group=None,
        clock_range=mem_range,
        locked=False,
        throttle_events=throttle_events,
    )
    dram_lat_loaded_mv = make_measured(
        latency_raw["dram_latency_loaded_ns_samples"],
        unit="ns",
        method="cuda_dram_cg_pchase_loaded",
        perturb_group=None,
        clock_range=mem_range,
        locked=False,
        throttle_events=throttle_events,
    )
    launch_oh_mv = make_measured(
        launch_raw["device_launch_overhead_us_samples"],
        unit="us",
        method="cuda_graph_empty_kernel_device_overhead",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    host_launch_oh_mv = make_measured(
        launch_raw["host_launch_overhead_us_samples"],
        unit="us",
        method="cuda_stream_empty_kernel_host_batch",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    per_cta_lat_mv = make_measured(
        launch_raw["per_cta_launch_latency_ns_samples"],
        unit="ns",
        method="cuda_graph_empty_kernel_grid_slope",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )

    # Build latency-vs-achieved-BW curves (DRAM & L2) for Little's Law concurrency model.
    def build_lat_curve(
        raw_points: list[dict[str, Any]], method: str, clk_rng: list[float] | None
    ) -> list[dict[str, Any]]:
        pts: list[dict[str, Any]] = []
        for pt in raw_points:
            bw_med = float(statistics.median(pt["achieved_bw_gbps_samples"]))
            lat_mv = make_measured(
                pt["latency_ns_samples"],
                unit="ns",
                method=method,
                perturb_group=None,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            )
            pts.append({"achieved_bw_gbps": round(bw_med, 3), "latency_ns": lat_mv})
        pts.sort(key=lambda x: x["achieved_bw_gbps"])
        return pts

    dram_lat_curve = build_lat_curve(
        lat_curve_raw["dram_latency_by_bw"], "cuda_dram_cg_pchase_vs_bg_bw", mem_range
    )
    l2_lat_curve = build_lat_curve(
        lat_curve_raw["l2_latency_by_bw"], "cuda_l2_cg_pchase_vs_bg_bw", sm_range
    )

    # Build L2 rescan hit-rate curve (0.25x to 4.0x L2).
    # Use each working set's own cold pass-1 BW as the 0%-hit baseline (cancels launch overhead
    # and clean-flush effects), and the peak <=0.5x L2 pass-2 BW as the 100%-hit baseline.
    rescan_pts_raw = l2_rescan_raw["points"]
    bw_l2_ref = max(
        float(statistics.median(pt["pass2_bw_gbps_samples"]))
        for pt in rescan_pts_raw
        if pt["ratio_to_l2"] <= 0.6
    )
    l2_rescan_curve: list[dict[str, Any]] = []
    for pt in rescan_pts_raw:
        p1_med = float(statistics.median(pt["pass1_bw_gbps_samples"]))
        p2_samples = pt["pass2_bw_gbps_samples"]
        p2_mv = make_measured(
            p2_samples,
            unit="GB/s",
            method="cuda_l2_rescan_pass2_cg_read",
            perturb_group="l2_bw",
            clock_range=sm_range,
            locked=False,
            throttle_events=throttle_events,
        )
        # Harmonic time model: 1/bw2 = h/bw_l2 + (1-h)/bw_cold
        # => h = (1/bw_cold - 1/bw2) / (1/bw_cold - 1/bw_l2)
        bw_cold = p1_med
        denom = (1.0 / bw_cold) - (1.0 / bw_l2_ref)
        hr_samples: list[float] = []
        for bw2 in p2_samples:
            if denom > 0 and bw2 > 0:
                h = ((1.0 / bw_cold) - (1.0 / bw2)) / denom
                hr_samples.append(max(0.0, min(1.0, h)))
            else:
                hr_samples.append(0.0)
        hr_mv = make_measured(
            hr_samples,
            unit="ratio",
            method="cuda_l2_rescan_harmonic_hit_rate",
            perturb_group=None,
            clock_range=sm_range,
            locked=False,
            throttle_events=throttle_events,
        )
        l2_rescan_curve.append(
            {
                "working_set_bytes": int(pt["working_set_bytes"]),
                "working_set_ratio_to_l2": round(float(pt["ratio_to_l2"]), 4),
                "pass2_bw_gbps": p2_mv,
                "estimated_hit_rate": hr_mv,
            }
        )

    # Build MLP curves: delivered BW vs bytes in flight (cp_async & ld_global_v4).
    def build_mlp_curve(
        raw_points: list[dict[str, Any]],
        method_prefix: str,
        perturb_group: str,
        clk_rng: list[float] | None,
    ) -> list[dict[str, Any]]:
        pts: list[dict[str, Any]] = []
        for pt in raw_points:
            mode = pt["issue_mode"]
            mv = make_measured(
                pt["gbps_samples"],
                unit="GB/s",
                method=f"{method_prefix}_{mode}",
                perturb_group=perturb_group,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            )
            pts.append(
                {
                    "issue_mode": mode,
                    "bytes_in_flight": int(pt["bytes_in_flight"]),
                    "active_ctas": int(pt["active_ctas"]),
                    "bytes_per_cta": int(pt["bytes_per_cta"]),
                    "bw_gbps": mv,
                }
            )
        return pts

    dram_mlp_curve = build_mlp_curve(
        mlp_raw["dram_bw_by_bytes_in_flight"], "cuda_mlp_dram_clean", "dram_bw", mem_range
    )
    l2_mlp_curve = build_mlp_curve(
        mlp_raw["l2_bw_by_bytes_in_flight"], "cuda_mlp_l2_warm", "l2_bw", sm_range
    )

    # Cross-check peak read BW against MLP saturation points (use the higher measured ceiling).
    best_dram_mlp = max(dram_mlp_curve, key=lambda pt: pt["bw_gbps"]["value"])["bw_gbps"]
    if best_dram_mlp["value"] > full_read_mv["value"]:
        full_read_mv = best_dram_mlp
    best_l2_mlp = max(l2_mlp_curve, key=lambda pt: pt["bw_gbps"]["value"])["bw_gbps"]
    if best_l2_mlp["value"] > full_l2_read_mv["value"]:
        full_l2_read_mv = best_l2_mlp

    def build_closed_loop(
        raw_pts: list[dict[str, Any]],
        method_prefix: str,
        perturb_group: str,
        clk_rng: list[float] | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pt in raw_pts:
            mode = pt["issue_mode"]
            bw_mv = make_measured(
                pt["bw_gbps_samples"],
                unit="GB/s",
                method=f"{method_prefix}_{mode}",
                perturb_group=perturb_group,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            )
            turn_mv = make_measured(
                pt["turnaround_ns_samples"],
                unit="ns",
                method=f"{method_prefix}_{mode}",
                perturb_group=None,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            )
            out.append(
                {
                    "issue_mode": mode,
                    "tokens": int(pt["tokens"]),
                    "active_ctas": int(pt["active_ctas"]),
                    "stage_bytes": int(pt["stage_bytes"]),
                    "compute_ns": float(pt["compute_ns"]),
                    "delivered_bw_gbps": bw_mv,
                    "mean_stage_turnaround_ns": turn_mv,
                }
            )
        return out

    dram_closed_loop = build_closed_loop(
        closed_loop_raw["dram_closed_loop"], "cuda_closed_loop_dram", "dram_bw", mem_range
    )
    l2_closed_loop = build_closed_loop(
        closed_loop_raw["l2_closed_loop"], "cuda_closed_loop_l2", "l2_bw", sm_range
    )

    l2_merge_by_sharing: list[dict[str, Any]] = []
    for pt in l2_merge_raw["l2_merge_by_sharing"]:
        l2_merge_by_sharing.append(
            {
                "active_ctas": int(pt["active_ctas"]),
                "sharing": int(pt["sharing"]),
                "bytes_per_cta": int(pt["bytes_per_cta"]),
                "stagger_stages": int(pt["stagger_stages"]),
                "turnaround_ns": make_measured(
                    pt["turnaround_ns_samples"],
                    unit="ns",
                    method="cuda_l2_merge_cp_async",
                    perturb_group=None,
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "dram_unique_bw_gbps": make_measured(
                    pt["dram_unique_bw_gbps_samples"],
                    unit="GB/s",
                    method="cuda_l2_merge_cp_async",
                    perturb_group="dram_bw",
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "l2_to_sm_bw_gbps": make_measured(
                    pt["l2_to_sm_bw_gbps_samples"],
                    unit="GB/s",
                    method="cuda_l2_merge_cp_async",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    stage_sync_overhead_ns: list[dict[str, Any]] = []
    for pt in stage_oh_raw["stage_sync_overhead_ns"]:
        mode = pt["issue_mode"]
        stage_sync_overhead_ns.append(
            {
                "issue_mode": mode,
                "num_warps": int(pt["num_warps"]),
                "tokens": int(pt["tokens"]),
                "overhead_ns": make_measured(
                    pt["overhead_ns_samples"],
                    unit="ns",
                    method=f"cuda_stage_sync_overhead_{mode}",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    fixed_sync_ns: list[dict[str, Any]] = []
    for pt in stage_oh_raw["fixed_sync_ns"]:
        fixed_sync_ns.append(
            {
                "num_warps": int(pt["num_warps"]),
                "tokens": int(pt["tokens"]),
                "overhead_ns": make_measured(
                    pt["overhead_ns_samples"],
                    unit="ns",
                    method="cuda_empty_commit_wait_bar_sync",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    min_async_rt_mv = make_measured(
        stage_oh_raw["min_async_roundtrip_ns_samples"],
        unit="ns",
        method="cuda_16b_hot_l2_cp_async_d1",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )

    dram_closed_loop_mma: list[dict[str, Any]] = []
    for pt in closed_loop_mma_raw["dram_closed_loop_mma"]:
        dram_closed_loop_mma.append(
            {
                "ctas_per_sm": int(pt["ctas_per_sm"]),
                "active_ctas": int(pt["active_ctas"]),
                "tokens": int(pt["tokens"]),
                "num_warps": int(pt["num_warps"]),
                "stage_bytes": int(pt["stage_bytes"]),
                "ldmatrix_bytes_per_stage": int(pt.get("ldmatrix_bytes_per_stage", 0)),
                "mma_iters_per_stage": int(pt["mma_iters_per_stage"]),
                "solo_compute_ns": float(pt["solo_compute_ns"]),
                "delivered_bw_gbps": make_measured(
                    pt["bw_gbps_samples"],
                    unit="GB/s",
                    method="cuda_closed_loop_dram_mma_cp_async",
                    perturb_group="dram_bw",
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "mean_stage_turnaround_ns": make_measured(
                    pt["turnaround_ns_samples"],
                    unit="ns",
                    method="cuda_closed_loop_dram_mma_cp_async",
                    perturb_group=None,
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "achieved_mma_tflops_per_sm": make_measured(
                    pt["mma_tflops_per_sm_samples"],
                    unit="TFLOP/s",
                    method="cuda_closed_loop_dram_mma_cp_async",
                    perturb_group="mma_dense",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "tensor_utilization": make_measured(
                    pt["tensor_util_samples"],
                    unit="ratio",
                    method="cuda_closed_loop_dram_mma_cp_async",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    mma_chain_latency: list[dict[str, Any]] = []
    chain_groups: dict[tuple[str, int, int, int], list[tuple[int, list[float]]]] = {}
    for pt in mma_chain_raw["mma_chain_latency"]:
        variant = str(pt["variant"])
        warps = int(pt["num_warps"])
        tb = int(pt["tile_bytes"])
        ld_n = int(pt["ldmatrix_x4_per_warp"])
        m_len = int(pt["mma_chain_len"])
        t_samples = [float(x) for x in pt["stage_time_ns_samples"]]
        mma_chain_latency.append(
            {
                "variant": variant,
                "num_warps": warps,
                "tile_bytes": tb,
                "ldmatrix_x4_per_warp": ld_n,
                "mma_chain_len": m_len,
                "stage_time_ns": make_measured(
                    t_samples,
                    unit="ns",
                    method=f"cuda_mma_chain_{variant}",
                    perturb_group="mma_dense",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )
        chain_groups.setdefault((variant, warps, tb, ld_n), []).append((m_len, t_samples))

    serial_stage_overhead_ns: list[dict[str, Any]] = []
    for (variant, warps, tb, ld_n), pts in chain_groups.items():
        pts.sort(key=lambda x: x[0])
        xs = [float(p[0]) for p in pts]
        n_samp = min(len(p[1]) for p in pts)
        mx = sum(xs) / len(xs)
        denom = sum((x - mx) ** 2 for x in xs)
        slope_samples: list[float] = []
        inter_samples: list[float] = []
        for si in range(n_samp):
            ys = [p[1][si] for p in pts]
            my = sum(ys) / len(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom > 0 else 0.0
            inter = max(0.0, my - slope * mx)
            slope_samples.append(max(0.0, slope))
            inter_samples.append(inter)
        med_ys = [float(statistics.median(p[1])) for p in pts]
        my_med = sum(med_ys) / len(med_ys)
        sl_med = (
            sum((x - mx) * (y - my_med) for x, y in zip(xs, med_ys)) / denom if denom > 0 else 0.0
        )
        in_med = my_med - sl_med * mx
        ss_res = sum((y - (sl_med * x + in_med)) ** 2 for x, y in zip(xs, med_ys))
        ss_tot = sum((y - my_med) ** 2 for y in med_ys)
        r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot)) if ss_tot > 0 else 1.0
        serial_stage_overhead_ns.append(
            {
                "variant": variant,
                "num_warps": warps,
                "tile_bytes": tb,
                "ldmatrix_x4_per_warp": ld_n,
                "intercept_ns": make_measured(
                    inter_samples,
                    unit="ns",
                    method=f"cuda_mma_chain_ols_intercept_{variant}",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "slope_ns_per_mma": make_measured(
                    slope_samples,
                    unit="ns",
                    method=f"cuda_mma_chain_ols_slope_{variant}",
                    perturb_group="mma_dense",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "r_squared": round(r2, 6),
            }
        )

    serial_stage_fill: list[dict[str, Any]] = []
    for pt in serial_fill_raw["serial_stage_fill"]:
        t_mode = str(pt["transfer_mode"])
        r_mode = str(pt["read_mode"])
        serial_stage_fill.append(
            {
                "transfer_mode": t_mode,
                "read_mode": r_mode,
                "num_warps": int(pt["num_warps"]),
                "stage_bytes": int(pt["stage_bytes"]),
                "active_ctas": int(pt["active_ctas"]),
                "stage_time_ns": make_measured(
                    [float(x) for x in pt["stage_time_ns_samples"]],
                    unit="ns",
                    method=f"cuda_serial_fill_{t_mode}_{r_mode}",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "effective_bytes_per_ns_per_cta": make_measured(
                    [float(x) for x in pt["effective_bytes_per_ns_samples"]],
                    unit="B/ns",
                    method=f"cuda_serial_fill_{t_mode}_{r_mode}",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    sync_stage_loop_map: dict[tuple[str, str, int, int, int, int], dict[str, Any]] = {}
    for pt in (
        list(sync_loop_raw["sync_stage_loop"])
        + list(sync_multicta_raw["sync_stage_loop"])
        + list(sync_reconcile_raw["sync_stage_loop"])
    ):
        level = str(pt["mem_level"])
        t_mode = str(pt["transfer_mode"])
        clk_rng = mem_range if level == "dram" else sm_range
        p_grp = "dram_bw" if level == "dram" else "l2_bw"
        key = (
            level,
            t_mode,
            int(pt["ctas_per_sm"]),
            int(pt["num_warps"]),
            int(pt["stage_bytes"]),
            int(pt["mma_chain_len"]),
        )
        sync_stage_loop_map[key] = {
            "mem_level": level,
            "transfer_mode": t_mode,
            "ctas_per_sm": int(pt["ctas_per_sm"]),
            "active_ctas": int(pt["active_ctas"]),
            "num_warps": int(pt["num_warps"]),
            "stage_bytes": int(pt["stage_bytes"]),
            "mma_chain_len": int(pt["mma_chain_len"]),
            "solo_compute_ns": float(pt["solo_compute_ns"]),
            "delivered_bw_gbps": make_measured(
                [float(x) for x in pt["bw_gbps_samples"]],
                unit="GB/s",
                method=f"cuda_sync_loop_{level}_{t_mode}",
                perturb_group=p_grp,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            ),
            "mean_stage_turnaround_ns": make_measured(
                [float(x) for x in pt["turnaround_ns_samples"]],
                unit="ns",
                method=f"cuda_sync_loop_{level}_{t_mode}",
                perturb_group=None,
                clock_range=clk_rng,
                locked=False,
                throttle_events=throttle_events,
            ),
            "achieved_mma_tflops_per_sm": make_measured(
                [float(x) for x in pt["mma_tflops_per_sm_samples"]],
                unit="TFLOP/s",
                method=f"cuda_sync_loop_{level}_{t_mode}",
                perturb_group="mma_dense",
                clock_range=sm_range,
                locked=False,
                throttle_events=throttle_events,
            ),
        }
    sync_stage_loop = list(sync_stage_loop_map.values())

    gmem_load_issue_per_sm: list[dict[str, Any]] = []
    for pt in gmem_issue_raw["gmem_load_issue_per_sm"]:
        w_bytes = int(pt["load_width_bytes"])
        gmem_load_issue_per_sm.append(
            {
                "active_sms": int(pt["active_sms"]),
                "load_width_bytes": w_bytes,
                "num_warps": int(pt["num_warps"]),
                "issue_rate_gwarp_inst_per_sm": make_measured(
                    [float(x) for x in pt["issue_rate_gwarp_inst_per_sm_samples"]],
                    unit="Gwarp-inst/s",
                    method=f"cuda_gmem_cg_load_issue_{w_bytes}b",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "delivered_bw_per_sm_gbps": make_measured(
                    [float(x) for x in pt["delivered_bw_per_sm_gbps_samples"]],
                    unit="GB/s",
                    method=f"cuda_gmem_cg_load_issue_{w_bytes}b",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    # Compute spill reload extra_ns_vs_zero and ns_per_reload relative to reload_count=0 baseline.
    spill_zero_med: dict[tuple[int, str], float] = {}
    for pt in spill_lat_raw["reload_latency_by_count"]:
        if int(pt["reload_count"]) == 0:
            spill_zero_med[
                (int(pt.get("frame_bytes_per_thread", 0)), str(pt["dependency_mode"]))
            ] = float(statistics.median([float(x) for x in pt["iter_time_ns_samples"]]))

    reload_latency_by_count: list[dict[str, Any]] = []
    for pt in spill_lat_raw["reload_latency_by_count"]:
        regime = str(pt["regime"])
        dep_mode = str(pt["dependency_mode"])
        fb = int(pt.get("frame_bytes_per_thread", 0))
        res_t = int(pt.get("resident_threads_per_sm", int(pt["num_warps"]) * 32))
        rc = int(pt["reload_count"])
        raw_s = [float(x) for x in pt["iter_time_ns_samples"]]
        base_ns = spill_zero_med.get((fb, dep_mode), 0.0)
        extra_s = [max(0.0, x - base_ns) for x in raw_s]
        per_r_s = [e / rc for e in extra_s] if rc > 0 else [0.0 for _ in extra_s]
        p_grp = "spill_l1" if regime == "l1_hit" else "spill_l2"
        entry: dict[str, Any] = {
            "regime": regime,
            "dependency_mode": dep_mode,
            "num_warps": int(pt["num_warps"]),
            "resident_threads_per_sm": res_t,
            "frame_bytes_per_thread": fb,
            "reload_count": rc,
            "iter_time_ns": make_measured(
                raw_s,
                unit="ns",
                method=f"cuda_spill_reload_uniform_{regime}_{dep_mode}",
                perturb_group=p_grp,
                clock_range=sm_range,
                locked=False,
                throttle_events=throttle_events,
            ),
            "extra_ns_vs_zero": make_measured(
                extra_s,
                unit="ns",
                method=f"cuda_spill_reload_uniform_{regime}_{dep_mode}",
                perturb_group=p_grp,
                clock_range=sm_range,
                locked=False,
                throttle_events=throttle_events,
            ),
            "ns_per_reload": make_measured(
                per_r_s,
                unit="ns",
                method=f"cuda_spill_reload_uniform_{regime}_{dep_mode}",
                perturb_group=p_grp,
                clock_range=sm_range,
                locked=False,
                throttle_events=throttle_events,
            ),
        }
        reload_latency_by_count.append(entry)

    dram_strided_mlp: list[dict[str, Any]] = []
    for pt in strided_mlp_raw["dram_bw_by_bytes_in_flight_strided"]:
        seg = int(pt["segment_bytes"])
        stride = int(pt["stride_bytes"])
        dram_strided_mlp.append(
            {
                "segment_bytes": seg,
                "stride_bytes": stride,
                "bytes_in_flight": int(pt["bytes_in_flight"]),
                "active_ctas": int(pt["active_ctas"]),
                "bytes_per_cta": int(pt["bytes_per_cta"]),
                "bw_gbps": make_measured(
                    [float(x) for x in pt["gbps_samples"]],
                    unit="GB/s",
                    method=f"cuda_strided_mlp_dram_seg{seg}_str{stride}",
                    perturb_group="dram_bw",
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    def build_warp_issue_curve(raw_pts: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for pt in raw_pts:
            out.append(
                {
                    "warps_per_sm": int(pt["warps_per_sm"]),
                    "rate": make_measured(
                        [float(x) for x in pt["rate_gwarp_inst_per_sm_samples"]],
                        unit="Gwarp-inst/s",
                        method=method,
                        perturb_group=None,
                        clock_range=sm_range,
                        locked=False,
                        throttle_events=throttle_events,
                    ),
                }
            )
        out.sort(key=lambda x: x["warps_per_sm"])
        return out

    sfu_issue_curve = build_warp_issue_curve(
        sfu_shfl_raw["sfu_issue_rate_per_sm_gips"], "cuda_ptx_ex2_approx_f32"
    )
    sfu_rcp_issue_curve = build_warp_issue_curve(
        sfu_shfl_raw["sfu_rcp_issue_rate_per_sm_gips"], "cuda_ptx_rcp_approx_f32"
    )
    shfl_issue_curve = build_warp_issue_curve(
        sfu_shfl_raw["shfl_issue_rate_per_sm_gips"], "cuda_ptx_shfl_sync_bfly_b32"
    )
    shfl_idx_issue_curve = build_warp_issue_curve(
        sfu_shfl_raw["shfl_idx_issue_rate_per_sm_gips"], "cuda_ptx_shfl_sync_idx_b32"
    )
    shfl_lat_mv = make_measured(
        [float(x) for x in sfu_shfl_raw["shfl_latency_ns_samples"]],
        unit="ns",
        method="cuda_ptx_shfl_sync_bfly_chain",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )
    peak_sfu_pt = max(
        sfu_shfl_raw["sfu_issue_rate_per_sm_gips"],
        key=lambda p: statistics.median(p["rate_gwarp_inst_per_sm_samples"]),
    )
    sfu_tflops_mv = make_measured(
        [
            float(x) * sm_count * 32.0 / 1000.0
            for x in peak_sfu_pt["rate_gwarp_inst_per_sm_samples"]
        ],
        unit="TFLOP/s",
        method="cuda_ptx_ex2_approx_f32",
        perturb_group=None,
        clock_range=sm_range,
        locked=False,
        throttle_events=throttle_events,
    )

    mma_softmax_chain: list[dict[str, Any]] = []
    for pt in mma_sf_raw["mma_softmax_chain"]:
        mma_softmax_chain.append(
            {
                "ctas_per_sm": int(pt["ctas_per_sm"]),
                "num_warps": int(pt["num_warps"]),
                "mma_per_half": int(pt["mma_per_half"]),
                "softmax_ops": int(pt["softmax_ops"]),
                "solo_mma_ns": float(pt["solo_mma_ns"]),
                "solo_softmax_ns": float(pt["solo_softmax_ns"]),
                "iter_time_ns": make_measured(
                    [float(x) for x in pt["iter_time_ns_samples"]],
                    unit="ns",
                    method="cuda_mma_softmax_chain",
                    perturb_group="mma_dense",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    sync_stage_loop_ablation: list[dict[str, Any]] = []
    for pt in sync_abl_raw["sync_stage_loop_ablation"]:
        v_name = str(pt["variant"])
        sync_stage_loop_ablation.append(
            {
                "variant": v_name,
                "ctas_per_sm": int(pt["ctas_per_sm"]),
                "active_ctas": int(pt["active_ctas"]),
                "num_warps": int(pt["num_warps"]),
                "stage_bytes": int(pt["stage_bytes"]),
                "mma_chain_len": int(pt["mma_chain_len"]),
                "delivered_bw_gbps": make_measured(
                    [float(x) for x in pt["bw_gbps_samples"]],
                    unit="GB/s",
                    method=f"cuda_sync_ablation_{v_name}",
                    perturb_group="dram_bw",
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "mean_stage_turnaround_ns": make_measured(
                    [float(x) for x in pt["turnaround_ns_samples"]],
                    unit="ns",
                    method=f"cuda_sync_ablation_{v_name}",
                    perturb_group=None,
                    clock_range=mem_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    gmem_load_issue_controlled: list[dict[str, Any]] = []
    for pt in gmem_ctrl_raw["gmem_load_issue_controlled"]:
        w_b = int(pt["load_width_bytes"])
        bn = int(pt["batch_loads_per_thread"])
        gmem_load_issue_controlled.append(
            {
                "load_width_bytes": w_b,
                "batch_loads_per_thread": bn,
                "num_warps": int(pt["num_warps"]),
                "bytes_in_flight": int(pt["bytes_in_flight"]),
                "issue_rate_gwarp_inst_per_sm": make_measured(
                    [float(x) for x in pt["issue_rate_gwarp_inst_per_sm_samples"]],
                    unit="Gwarp-inst/s",
                    method=f"cuda_gmem_cg_load_ctrl_{w_b}b_n{bn}",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "delivered_bw_per_sm_gbps": make_measured(
                    [float(x) for x in pt["delivered_bw_per_sm_gbps_samples"]],
                    unit="GB/s",
                    method=f"cuda_gmem_cg_load_ctrl_{w_b}b_n{bn}",
                    perturb_group="l2_bw",
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
        )

    l1_to_l2_spill_threshold_by_carveout: list[dict[str, Any]] = []
    for entry in l1_carve_raw["l1_carveout_sweep"]:
        smem_b = int(entry["smem_carveout_bytes"])
        pts = sorted(entry["points"], key=lambda p: int(p["ws_bytes"]))
        n_samp = min(len(p["latency_ns_samples"]) for p in pts)
        thresh_samples: list[float] = []
        miss_samples_by_pt: list[list[float]] = [[] for _ in pts]
        for si in range(n_samp):
            lats = [float(p["latency_ns_samples"][si]) for p in pts]
            wss = [float(p["ws_bytes"]) for p in pts]
            l1_lat = statistics.median(lats[:3])
            l2_lat = statistics.median(lats[-4:])
            denom = max(l2_lat - l1_lat, 1e-6)
            for pi, lat in enumerate(lats):
                miss = max(0.0, min(1.0, (lat - l1_lat) / denom))
                miss_samples_by_pt[pi].append(miss)
            mid = 0.5 * (l1_lat + l2_lat)
            knee = wss[0]
            for i in range(1, len(pts)):
                if lats[i - 1] <= mid <= lats[i] and lats[i] > lats[i - 1]:
                    frac = (mid - lats[i - 1]) / (lats[i] - lats[i - 1])
                    knee = wss[i - 1] + frac * (wss[i] - wss[i - 1])
                    break
            else:
                if lats[-1] <= mid:
                    knee = wss[-1]
            thresh_samples.append(float(round(knee)))
        miss_curve = [
            {
                "working_set_bytes": int(p["ws_bytes"]),
                "estimated_miss_rate": make_measured(
                    miss_samples_by_pt[pi],
                    unit="fraction",
                    method="cuda_l1_ca_pchase_carveout_miss_rate",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
            }
            for pi, p in enumerate(pts)
        ]
        l1_to_l2_spill_threshold_by_carveout.append(
            {
                "smem_carveout_bytes": smem_b,
                "threshold_bytes_per_sm": make_measured(
                    thresh_samples,
                    unit="bytes",
                    method="cuda_l1_ca_pchase_carveout_cliff",
                    perturb_group=None,
                    clock_range=sm_range,
                    locked=False,
                    throttle_events=throttle_events,
                ),
                "miss_rate_by_working_set": miss_curve,
            }
        )
    l1_to_l2_spill_threshold_by_carveout.sort(key=lambda x: x["smem_carveout_bytes"])
    l1_spill_thresh_mv = l1_to_l2_spill_threshold_by_carveout[0]["threshold_bytes_per_sm"]

    sm_clock_ghz = (
        (statistics.median(sm_clocks) / 1000.0)
        if sm_clocks
        else (query["clock_khz"] / 1e6 if query["clock_khz"] > 0 else 2.4)
    )

    l1_unified = arch_unified_l1_bytes(major, minor, smem_per_sm)
    tma_supported = major >= 9

    now_utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"cal-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{arch}"

    memory_measured: dict[str, Any] = {
        "dram_bw_read_gbps": full_read_mv,
        "dram_bw_write_gbps": full_write_mv,
        "dram_bw_by_active_sms": read_curve_points,
        "dram_bw_by_bytes_in_flight": dram_mlp_curve,
        "dram_bw_by_bytes_in_flight_strided": dram_strided_mlp,
        "dram_closed_loop": dram_closed_loop,
        "dram_closed_loop_mma": dram_closed_loop_mma,
        "l2_bw_read_gbps": full_l2_read_mv,
        "l2_bw_write_gbps": full_l2_write_mv,
        "l2_bw_by_active_sms": l2_curve_points,
    }
    if l2_cp_async_curve_points:
        memory_measured["l2_bw_by_active_sms_cp_async"] = l2_cp_async_curve_points
    memory_measured.update(
        {
            "l2_bw_by_bytes_in_flight": l2_mlp_curve,
            "l2_closed_loop": l2_closed_loop,
            "l2_merge_by_sharing": l2_merge_by_sharing,
            "l2_effective_capacity_bytes": l2_eff_mv,
            "smem_bw_per_sm_gbps": smem_bw_mv,
            "dram_latency_ns": dram_lat_mv,
            "dram_latency_loaded_ns": dram_lat_loaded_mv,
            "dram_latency_by_bw": dram_lat_curve,
            "l2_latency_ns": l2_lat_mv,
            "l2_latency_loaded_ns": l2_lat_loaded_mv,
            "l2_latency_by_bw": l2_lat_curve,
            "l2_rescan_hit_rate_by_working_set": l2_rescan_curve,
            "smem_latency_ns": smem_lat_mv,
            "gmem_load_issue_per_sm": gmem_load_issue_per_sm,
            "gmem_load_issue_controlled": gmem_load_issue_controlled,
            "spill_reload_latency_by_count": reload_latency_by_count,
            "l1_to_l2_spill_threshold_by_carveout": l1_to_l2_spill_threshold_by_carveout,
        }
    )

    profile: dict[str, Any] = {
        "schema_version": "0.1",
        "identity": {
            "vendor": "nvidia",
            "arch": arch,
            "sku": query["name"],
            "memory_type": "unified" if query["integrated"] else "discrete",
        },
        "provenance": {
            "profile_revision": PROFILE_REVISION,
            "calibration_run_id": run_id,
            "calibration_manifest_digest": None,
            "source": "partial",
            "driver_version": driver_str,
            "toolkit_version": toolkit_str,
            "suite_version": SUITE_VERSION,
            "host_os": platform.system().lower(),
            "is_wsl": is_wsl(),
            "power_limit_w": power_limit,
            "timestamp_utc": now_utc,
            "clock_policy": {
                "locked": False,
                "sm_clock_locked": False,
                "mem_clock_locked": False,
                "sm_clock_mhz": round(statistics.median(sm_clocks), 1)
                if sm_clocks
                else (round(query["clock_khz"] / 1000.0, 1) if query["clock_khz"] > 0 else None),
                "mem_clock_mhz": round(statistics.median(mem_clocks), 1)
                if mem_clocks
                else (
                    round(query["mem_clock_khz"] / 1000.0, 1)
                    if query["mem_clock_khz"] > 0
                    else None
                ),
                "observed_sm_clock_range_mhz": sm_range,
                "observed_mem_clock_range_mhz": mem_range,
                "throttle_events": throttle_events,
            },
        },
        "specs": {
            "sm_count": sm_count,
            "warp_size": int(query["warp_size"]),
            "max_threads_per_sm": int(query["max_threads_per_sm"]),
            "max_warps_per_sm": max_warps_per_sm,
            "max_blocks_per_sm": int(query["max_blocks_per_sm"]),
            "max_threads_per_block": int(query["max_threads_per_block"]),
            "registers_per_sm": int(query["registers_per_sm"]),
            "max_registers_per_thread": 255,
            "register_alloc_granularity": "warp",
            "register_alloc_unit_regs": 256,
            "l1_smem_unified_bytes": l1_unified,
            "l1_size_bytes": l1_unified,
            "shared_mem_per_sm_bytes": smem_per_sm,
            "max_shared_mem_per_block_bytes": max_smem_block,
            "reserved_shared_mem_per_cta_bytes": reserved_smem,
            "shared_mem_alloc_granularity_bytes": 128,
            "shared_mem_carveout_options_bytes": arch_carveout_options(major, minor, smem_per_sm),
            "l2_size_bytes": l2_size,
            "l2_line_bytes": 128,
            "l2_sector_bytes": 32,
            "l2_partitions": 1,
            "l1_line_bytes": 128,
            "capabilities": {
                "cp_async": major >= 8,
                "tma": tma_supported,
                "wgmma": major == 9,
                "tcgen05_tmem": major == 10,
                "cluster": major in (9, 10),
            },
            "mma_shapes": [
                {
                    "op_class": "mma.sync",
                    "a_dtype": "f16",
                    "b_dtype": "f16",
                    "acc_dtype": "f32",
                    "m": 16,
                    "n": 8,
                    "k": 16,
                },
                {
                    "op_class": "mma.sync",
                    "a_dtype": "bf16",
                    "b_dtype": "bf16",
                    "acc_dtype": "f32",
                    "m": 16,
                    "n": 8,
                    "k": 16,
                },
            ],
        },
        "measured": {
            "memory": memory_measured,
            "compute": {
                "mma_tflops_per_sm_by_warps": mma_curves,
                "non_mma_issue_rate_per_sm_gips": make_estimated(
                    4.0 * sm_clock_ghz, "Ginst/s", "m2_estimated", None
                ),
                "fp32_fma_tflops": fp32_fma_mv,
                "sfu_tflops": sfu_tflops_mv,
                "mma_chain_latency": mma_chain_latency,
                "serial_stage_overhead_ns": serial_stage_overhead_ns,
                "sfu_issue_rate_per_sm_gips": sfu_issue_curve,
                "sfu_rcp_issue_rate_per_sm_gips": sfu_rcp_issue_curve,
                "shfl_issue_rate_per_sm_gips": shfl_issue_curve,
                "shfl_idx_issue_rate_per_sm_gips": shfl_idx_issue_curve,
                "shfl_latency_ns": shfl_lat_mv,
                "mma_softmax_chain": mma_softmax_chain,
            },
            "occupancy": {
                "smem_carveout_observed": [],
            },
            "async_pipeline": {
                "cp_async_bw_per_sm_gbps": cp_async_mv,
                "tma_bw_per_sm_gbps": make_estimated(150.0, "GB/s", "m2_estimated", "async_tma")
                if tma_supported
                else None,
                "tma_setup_ns": make_estimated(80.0, "ns", "m2_estimated", None)
                if tma_supported
                else None,
            },
            "pipeline": {
                "stage_sync_overhead_ns": stage_sync_overhead_ns,
                "fixed_sync_ns": fixed_sync_ns,
                "min_async_roundtrip_ns": min_async_rt_mv,
                "serial_stage_fill": serial_stage_fill,
                "sync_stage_loop": sync_stage_loop,
                "sync_stage_loop_ablation": sync_stage_loop_ablation,
            },
            "spill": {
                "local_mem_l1_hit_bw_per_sm_gbps": make_estimated(
                    150.0, "GB/s", "m2_estimated", "spill_l1"
                ),
                "local_mem_l1_miss_bw_per_sm_gbps": make_estimated(
                    40.0, "GB/s", "m2_estimated", "spill_l2"
                ),
                "local_mem_l1_hit_latency_ns": make_estimated(
                    25.0, "ns", "m2_estimated", "spill_l1"
                ),
                "local_mem_l1_miss_latency_ns": make_estimated(
                    160.0, "ns", "m2_estimated", "spill_l2"
                ),
                "l1_to_l2_spill_threshold_bytes_per_sm": l1_spill_thresh_mv,
            },
            "overhead": {
                "launch_overhead_us": launch_oh_mv,
                "host_launch_overhead_us": host_launch_oh_mv,
                "per_cta_launch_latency_ns": per_cta_lat_mv,
            },
            "cta_dispatch": classify_dispatch(dispatch_raw),
        },
    }

    validate_profile(profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect M1 GPU profile via CUDA microbenchmarks.")
    parser.add_argument(
        "--binary", type=Path, default=DEFAULT_BIN, help="Path to compiled vgpu_bench binary."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--samples", type=int, default=15, help="Timed iterations per DRAM BW point."
    )
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per CTA dispatch probe.")
    parser.add_argument("--output", type=Path, required=True, help="Output profile JSON path.")
    args = parser.parse_args()

    query, _ = run_bench_with_clock_polling(args.binary, "query", args.device, poll_interval_s=0.05)
    dram_raw, smi_dram = run_bench_with_clock_polling(
        args.binary, "dram-bw", args.device, ["--samples", str(args.samples)]
    )
    mma_raw, smi_mma = run_bench_with_clock_polling(
        args.binary, "mma", args.device, ["--samples", str(args.samples)]
    )
    l2_raw, smi_l2 = run_bench_with_clock_polling(
        args.binary, "l2", args.device, ["--samples", str(args.samples)]
    )
    cp_async_raw, smi_cp = run_bench_with_clock_polling(
        args.binary, "cp-async", args.device, ["--samples", str(args.samples)]
    )
    latency_raw, smi_lat = run_bench_with_clock_polling(
        args.binary, "latency", args.device, ["--samples", str(args.samples)]
    )
    launch_raw, smi_launch = run_bench_with_clock_polling(
        args.binary, "launch", args.device, ["--samples", str(args.samples)]
    )
    lat_curve_raw, smi_lc = run_bench_with_clock_polling(
        args.binary, "latency-curve", args.device, ["--samples", "11"]
    )
    l2_rescan_raw, smi_rs = run_bench_with_clock_polling(
        args.binary, "l2-rescan", args.device, ["--samples", str(args.samples)]
    )
    mlp_raw, smi_mlp = run_bench_with_clock_polling(
        args.binary, "mlp", args.device, ["--samples", "11"]
    )
    closed_loop_raw, smi_cl = run_bench_with_clock_polling(
        args.binary, "closed-loop", args.device, ["--samples", "5", "--warmup", "1"]
    )
    l2_merge_raw, smi_mg = run_bench_with_clock_polling(
        args.binary, "l2-merge", args.device, ["--samples", "7", "--warmup", "1"]
    )
    stage_oh_raw, smi_so = run_bench_with_clock_polling(
        args.binary,
        "stage-overhead",
        args.device,
        ["--samples", str(args.samples), "--warmup", "2"],
    )
    closed_loop_mma_raw, smi_clm = run_bench_with_clock_polling(
        args.binary, "closed-loop-mma", args.device, ["--samples", "5", "--warmup", "1"]
    )
    mma_chain_raw, smi_mc = run_bench_with_clock_polling(
        args.binary, "mma-chain", args.device, ["--samples", "9", "--warmup", "2"]
    )
    serial_fill_raw, smi_sf = run_bench_with_clock_polling(
        args.binary, "serial-fill", args.device, ["--samples", "11", "--warmup", "2"]
    )
    sync_loop_raw, smi_sl = run_bench_with_clock_polling(
        args.binary, "sync-loop", args.device, ["--samples", "5", "--warmup", "1"]
    )
    gmem_issue_raw, smi_gi = run_bench_with_clock_polling(
        args.binary, "gmem-issue", args.device, ["--samples", "11", "--warmup", "2"]
    )
    spill_lat_raw, smi_sp = run_bench_with_clock_polling(
        args.binary, "spill-latency", args.device, ["--samples", "11", "--warmup", "2"]
    )
    strided_mlp_raw, smi_sm = run_bench_with_clock_polling(
        args.binary, "strided-mlp", args.device, ["--samples", "7", "--warmup", "1"]
    )
    sfu_shfl_raw, smi_ss = run_bench_with_clock_polling(
        args.binary, "sfu-shfl", args.device, ["--samples", "11", "--warmup", "2"]
    )
    mma_sf_raw, smi_ms = run_bench_with_clock_polling(
        args.binary, "mma-softmax-chain", args.device, ["--samples", "9", "--warmup", "2"]
    )
    sync_abl_raw, smi_sa = run_bench_with_clock_polling(
        args.binary, "sync-loop-ablation", args.device, ["--samples", "9", "--warmup", "2"]
    )
    gmem_ctrl_raw, smi_gc = run_bench_with_clock_polling(
        args.binary, "gmem-issue-controlled", args.device, ["--samples", "11", "--warmup", "2"]
    )
    l1_carve_raw, smi_l1c = run_bench_with_clock_polling(
        args.binary, "l1-carveout", args.device, ["--samples", "7", "--warmup", "1"]
    )
    sync_multicta_raw, smi_slm = run_bench_with_clock_polling(
        args.binary, "sync-loop-multicta", args.device, ["--samples", "7", "--warmup", "2"]
    )
    sync_reconcile_raw, smi_slr = run_bench_with_clock_polling(
        args.binary, "sync-loop-reconcile", args.device, ["--samples", "9", "--warmup", "2"]
    )
    dispatch_raw, smi_disp = run_bench_with_clock_polling(
        args.binary, "cta-dispatch", args.device, ["--repeats", str(args.repeats)]
    )
    smi_samples = (
        smi_dram
        + smi_mma
        + smi_l2
        + smi_cp
        + smi_lat
        + smi_launch
        + smi_lc
        + smi_rs
        + smi_mlp
        + smi_cl
        + smi_mg
        + smi_so
        + smi_clm
        + smi_mc
        + smi_sf
        + smi_sl
        + smi_gi
        + smi_sp
        + smi_sm
        + smi_ss
        + smi_ms
        + smi_sa
        + smi_gc
        + smi_l1c
        + smi_slm
        + smi_slr
        + smi_disp
    )

    profile = build_profile(
        query,
        dram_raw,
        dispatch_raw,
        mma_raw,
        l2_raw,
        cp_async_raw,
        latency_raw,
        launch_raw,
        lat_curve_raw,
        l2_rescan_raw,
        mlp_raw,
        closed_loop_raw,
        l2_merge_raw,
        stage_oh_raw,
        closed_loop_mma_raw,
        mma_chain_raw,
        serial_fill_raw,
        sync_loop_raw,
        gmem_issue_raw,
        spill_lat_raw,
        strided_mlp_raw,
        sfu_shfl_raw,
        mma_sf_raw,
        sync_abl_raw,
        gmem_ctrl_raw,
        l1_carve_raw,
        sync_multicta_raw,
        sync_reconcile_raw,
        smi_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(profile, fh, indent=2)
        fh.write("\n")

    ident = profile["identity"]
    full_bw = profile["measured"]["memory"]["dram_bw_read_gbps"]
    disp = profile["measured"]["cta_dispatch"]
    print(
        f"Wrote {args.output}: {ident['sku']} ({ident['arch']}, {profile['specs']['sm_count']} SMs) | "
        f"DRAM read={full_bw['value']:.1f} GB/s (±{full_bw['rel_uncertainty'] * 100:.1f}%) | "
        f"dispatch={disp['observed_policy']} ({disp['confidence']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
