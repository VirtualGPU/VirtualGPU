"""Single-launch durations of the measurements in a GPU profile.

For each timed profile field, the work one timed launch does (from the run_*
functions of vgpu_bench as they were when the v0.2 profiles were captured)
divided by the value the profile stores gives that launch's duration. Used to
find the fields captured with launches under 1 ms. CPU only.

    python microbench/launch_audit.py profiles/gb10.json [profiles/rtx4070.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MiB = 1 << 20
TARGET_MS = 1.0


def v(x):
    return x["value"] if isinstance(x, dict) else x


def mlp_bytes(q, is_dram):
    ctas, mult, bif = q["active_ctas"], q["bytes_per_cta"] // 4096, q["bytes_in_flight"]
    target = min(256 * MiB, max(16 * MiB, bif * 128)) if is_dram else max(64 * MiB, bif * 256)
    if q["issue_mode"] == "cp_async":
        sub, stages = (1, mult) if mult <= 4 else (mult // 4, 4)
        return ctas * 4096 * sub * max(stages * 4, target // (ctas * 4096 * sub))
    return ctas * 4096 * mult * max(4, target // (ctas * 4096 * mult))


def strided_bytes(q):
    ctas, mult, bif = q["active_ctas"], q["bytes_per_cta"] // 4096, q["bytes_in_flight"]
    sub, stages = (1, mult) if mult <= 4 else (mult // 4, 4)
    target = min(128 * MiB, max(16 * MiB, bif * 64))
    return ctas * 4096 * sub * max(stages * 4, target // (ctas * 4096 * sub))


def closed_loop_steps(q, is_dram):
    bw_ref = 450.0 if is_dram else 1400.0
    est = max(
        q["compute_ns"], q["active_ctas"] * q["stage_bytes"] / bw_ref, 250.0 if is_dram else 120.0
    )
    return max(256, min(1024, int(500000.0 / est)))


def spill_steps(q):
    likely_l2 = q["frame_bytes_per_thread"] * 128 >= 131072
    serial = q["dependency_mode"] == "serial_chain"
    est = 30 + q["reload_count"] * ((120 if serial else 20) if likely_l2 else (20 if serial else 3))
    return max(128, min(4096, 400000 // est))


def issue_iters(warps):
    return 16384 if warps <= 4 else (8192 if warps <= 16 else 4096)


def durations(profile):
    """Field name to the list of single-launch durations in ms, one per stored point."""
    m, c, p = (
        profile["measured"]["memory"],
        profile["measured"]["compute"],
        profile["measured"]["pipeline"],
    )
    sms = profile["specs"]["sm_count"]
    out = {
        "dram_bw_by_bytes_in_flight": [
            mlp_bytes(q, True) / v(q["bw_gbps"]) / 1e6 for q in m["dram_bw_by_bytes_in_flight"]
        ],
        "l2_bw_by_bytes_in_flight": [
            mlp_bytes(q, False) / v(q["bw_gbps"]) / 1e6 for q in m["l2_bw_by_bytes_in_flight"]
        ],
        "dram_bw_by_bytes_in_flight_strided": [
            strided_bytes(q) / v(q["bw_gbps"]) / 1e6
            for q in m["dram_bw_by_bytes_in_flight_strided"]
        ],
        "dram_bw_by_active_sms": [
            256 * MiB / v(q["bw_gbps"]) / 1e6 for q in m["dram_bw_by_active_sms"]
        ],
        "l2_bw_by_active_sms": [
            q["active_sms"]
            * 512
            * 128
            * (4096 if q["active_sms"] <= 4 else 2048)
            / v(q["bw_gbps"])
            / 1e6
            for q in m["l2_bw_by_active_sms"]
        ],
        "dram_bw_write_gbps": [256 * MiB / v(m["dram_bw_write_gbps"]) / 1e6],
        "dram_closed_loop": [
            closed_loop_steps(q, True) * v(q["mean_stage_turnaround_ns"]) / 1e6
            for q in m["dram_closed_loop"]
        ],
        "l2_closed_loop": [
            closed_loop_steps(q, False) * v(q["mean_stage_turnaround_ns"]) / 1e6
            for q in m["l2_closed_loop"]
        ],
        "dram_closed_loop_mma": [
            256 * v(q["mean_stage_turnaround_ns"]) / 1e6 for q in m["dram_closed_loop_mma"]
        ],
        "l2_merge_by_sharing": [
            q["active_ctas"] * q["bytes_per_cta"] * 256 / v(q["l2_to_sm_bw_gbps"]) / 1e6
            for q in m["l2_merge_by_sharing"]
        ],
        "gmem_load_issue_controlled": [
            q["num_warps"]
            * q["batch_loads_per_thread"]
            * max(256, 32768 // (q["batch_loads_per_thread"] * (q["num_warps"] // 4)))
            / v(q["issue_rate_gwarp_inst_per_sm"])
            / 1e6
            for q in m["gmem_load_issue_controlled"]
        ],
        "gmem_load_issue_per_sm": [
            q["num_warps"]
            * (4096 if q["active_sms"] == 1 else 2048)
            * 8
            / v(q["issue_rate_gwarp_inst_per_sm"])
            / 1e6
            for q in m["gmem_load_issue_per_sm"]
        ],
        "spill_reload_latency_by_count": [
            spill_steps(q) * v(q["iter_time_ns"]) / 1e6 for q in m["spill_reload_latency_by_count"]
        ],
        "fp32_fma_tflops": [sms * 512 * 40000 * 16 / (v(c["fp32_fma_tflops"]) * 1e12) * 1e3],
        "serial_stage_fill": [4096 * v(q["stage_time_ns"]) / 1e6 for q in p["serial_stage_fill"]],
        "sync_stage_loop_ablation": [
            128 * v(q["mean_stage_turnaround_ns"]) / 1e6 for q in p["sync_stage_loop_ablation"]
        ],
    }
    for op, pts in c["mma_tflops_per_sm_by_warps"].items():
        out[f"mma_tflops_per_sm_by_warps[{op}]"] = [
            q["mma_warps_per_sm"]
            * max(4000, 32000 // min(q["mma_warps_per_sm"], 8))
            * 4
            * 4096
            / (v(q["tflops_per_sm"]) * 1e12)
            * 1e3
            for q in pts
        ]
    for key in (
        "sfu_issue_rate_per_sm_gips",
        "sfu_rcp_issue_rate_per_sm_gips",
        "shfl_issue_rate_per_sm_gips",
        "shfl_idx_issue_rate_per_sm_gips",
    ):
        if key in c:
            out[key] = [
                q["warps_per_sm"] * issue_iters(q["warps_per_sm"]) * 16 / (v(q["rate"]) * 1e9) * 1e3
                for q in c[key]
            ]
    out["mma_chain_latency"] = [
        max(
            128, min(16384, 350000 // max(20, (q["num_warps"] // 4) * 12 * q["mma_chain_len"] + 30))
        )
        * v(q["stage_time_ns"])
        / 1e6
        for q in c["mma_chain_latency"]
    ]
    fixed = [q for q in p["sync_stage_loop"] if q["mem_level"] != "l2" and q["stage_bytes"] < 12288]
    out["sync_stage_loop[dram, stage < 12 KiB]"] = [
        128 * v(q["mean_stage_turnaround_ns"]) / 1e6 for q in fixed
    ]
    return out


def summary(profile):
    rows = {}
    for field, ms in durations(profile).items():
        rows[field] = {
            "points": len(ms),
            "min_ms": min(ms),
            "max_ms": max(ms),
            "under_1_ms": sum(d < TARGET_MS for d in ms),
            "scale_to_1_ms": max(1.0, TARGET_MS / min(ms)),
        }
    return rows


def main():
    out = {}
    for path in sys.argv[1:]:
        profile = json.loads(Path(path).read_text())
        out[path] = {
            "profile_revision": profile["provenance"]["profile_revision"],
            "fields": summary(profile),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
