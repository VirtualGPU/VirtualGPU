# Microbenchmarks and GPU profiles

The profile format and the CUDA suite that measures it. Schema: `schemas/gpu_profile.schema.json`. Code: `microbench/`. Profiles: `profiles/`. To measure your own GPU, see [contributing a profile](../guides/contributing-a-profile.md). The mechanisms the core runs today are listed in [design.md section 6.4](../design.md#64-mechanisms-in-the-core). If this file and the code disagree, the code is right and this file gets fixed.

## 1. Scope

This document defines two components:

1. **GPU Profile Format (`v0.1` JSON):** A versioned, self-describing hardware profile consumed by the L2 model (the L2 model), performance model core (the model core), and validation harness (validation).
2. **Microbenchmark Suite:** Isolated, public-API-only kernels that measure effective hardware rates, latencies, spill penalties, and CTA dispatch behavior on a target GPU to populate a profile.

**Out of scope:** Reverse engineering vendor drivers, firmware, or ISA encodings; fitting parameters to target workloads (GEMM/attention/MoE); and benchmark validation datasets (owned by validation).

**Current hardware reality:** Sealed holdout validation covers two NVIDIA GPUs:
- **RTX 4070 (`sm_89`, Ada):** Discrete GDDR6X, warp-level `mma.sync.aligned`, `cp.async`, no TMA.
- **DGX Spark GB10 (`sm_121`, consumer/workstation Blackwell):** Unified LPDDR5x memory shared with Grace CPU. Per NVIDIA PTX ISA and CUDA documentation, `sm_121` supports TMA and warp-level `mma.sync.aligned` (including block-scaled `m16n8k32`/`m16n8k64` for low-precision formats) and **does not** support `tcgen05` or Tensor Memory (TMEM).

Additional measured profiles (A100, L4, L40S, RTX 3070, RTX PRO 5000, RTX PRO 6000, and Huawei Ascend 910B3) are included for offline use but have not been through a sealed holdout. The Ascend 910B3 profile (`profiles/ascend910b3.json`) is collected by the separate AscendC microbenchmark suite in `microbench/ascend/` (`bisheng -xcce --npu-arch=dav-2201`), populates optional `specs.ascend` on-chip SRAM limits (`l1_bytes`, `l0a_bytes`, `l0b_bytes`, `l0c_bytes`, `ub_bytes`) and `specs.capabilities.cube_vector_split`, reuses the `cp_async` and `ld_global_v4` MLP issue-mode names for 2-buffer and 1-buffer MTE2 DMA transfers, and tags harmonic-interpolated `estimated_hit_rate` values as `source: "estimated"` with the formula recorded in `method`.

No Hopper (`sm_90`) GPU has been measured yet. Hopper `wgmma` and data center Blackwell `tcgen05`/TMEM are represented by capability flags in the schema, but are not calibrated on hardware.

---

## 2. Input and Output Interfaces

### 2.1 Microbenchmark Suite Interface

- **Inputs:** `device_id`, `clock_policy` (`lock` | `record_only`), `samples` (default `>= 15`), `l2_state` (`cold` | `warm`), `output_path`.
- **Outputs:** One validated GPU Profile JSON file, plus optional raw per-sample JSONL for auditability (kept outside the profile).

### 2.2 GPU Profile JSON Schema (`schema_version: "0.1"`)

Every measured quantity uses a uniform **MeasuredValue** envelope:

```json
{
  "value": 312.4,
  "ci95_low": 308.1,
  "ci95_high": 316.9,
  "rel_uncertainty": 0.10,
  "unit": "GB/s",
  "method": "stream_read_cold_l2",
  "perturb_group": "dram_bw"
}
```

`rel_uncertainty` is `max(default_floor, (ci95_high - ci95_low) / (2 * value))`, where `default_floor` is `0.10` for measured fields (the perturbation floor the core uses) and higher (for example `0.25`) for spec-derived estimates. `perturb_group` is optional (`null` = independent draw; fields sharing the same group name share one multiplicative draw, e.g., `dram_bw_read_gbps` and `dram_bw_write_gbps`). Curves (`*_by_active_sms`, `*_by_warps`) share a single scalar multiplier across all points so the curve shape is preserved under perturbation.

Top-level structure (**note:** all GB10 numbers below, such as `sm_count`, `l2_size_bytes`, and clocks, are **illustrative placeholders only**; real profiles must populate `specs.*` from runtime API queries and `measured.*` from hardware runs):

```json
{
  "schema_version": "0.1",
  "identity": {
    "vendor": "nvidia",
    "arch": "sm_121",
    "sku": "NVIDIA GB10",
    "memory_type": "unified"
  },
  "provenance": {
    "profile_revision": "0.1.0",
    "calibration_run_id": "cal-20261125-gb10-001",
    "calibration_manifest_digest": "sha256:...",
    "source": "measured",
    "driver_version": "575.xx",
    "toolkit_version": "12.9",
    "suite_version": "0.1.0",
    "host_os": "linux",
    "is_wsl": false,
    "power_limit_w": null,
    "timestamp_utc": "2026-11-25T00:00:00Z",
    "clock_policy": {
      "locked": false,
      "sm_clock_locked": true,
      "mem_clock_locked": false,
      "sm_clock_mhz": 2400,
      "mem_clock_mhz": 4266,
      "observed_sm_clock_range_mhz": [2385, 2400],
      "observed_mem_clock_range_mhz": [4266, 4266],
      "throttle_events": 0
    }
  },
  "specs": {
    "sm_count": 48,
    "warp_size": 32,
    "max_threads_per_sm": 1536,
    "max_warps_per_sm": 48,
    "max_blocks_per_sm": 16,
    "max_threads_per_block": 1024,
    "registers_per_sm": 65536,
    "max_registers_per_thread": 255,
    "register_alloc_granularity": "warp",
    "register_alloc_unit_regs": 256,
    "l1_smem_unified_bytes": 131072,
    "l1_size_bytes": 131072,
    "shared_mem_per_sm_bytes": 131072,
    "max_shared_mem_per_block_bytes": 101376,
    "reserved_shared_mem_per_cta_bytes": 1024,
    "shared_mem_alloc_granularity_bytes": 128,
    "shared_mem_carveout_options_bytes": [0, 8192, 16384, 32768, 65536, 102400],
    "l2_size_bytes": 25165824,
    "l2_line_bytes": 128,
    "l2_sector_bytes": 32,
    "l2_partitions": 1,
    "l1_line_bytes": 128,
    "capabilities": {
      "cp_async": true,
      "tma": true,
      "wgmma": false,
      "tcgen05_tmem": false,
      "cluster": false
    },
    "mma_shapes": [
      { "op_class": "mma.sync", "a_dtype": "f16", "b_dtype": "f16", "acc_dtype": "f32", "m": 16, "n": 8, "k": 16 }
    ]
  },
  "measured": {
    "memory": {
      "dram_bw_read_gbps": { "...": "MeasuredValue" },
      "dram_bw_write_gbps": { "...": "MeasuredValue" },
      "dram_bw_by_active_sms": [
        { "active_sms": 1, "bw_gbps": { "...": "MeasuredValue" } }
      ],
      "l2_bw_read_gbps": { "...": "MeasuredValue" },
      "l2_bw_write_gbps": { "...": "MeasuredValue" },
      "l2_bw_by_active_sms": [
        { "active_sms": 1, "bw_gbps": { "...": "MeasuredValue" } }
      ],
      "l2_bw_by_active_sms_cp_async": [
        { "active_sms": 1, "bw_gbps": { "...": "MeasuredValue" } }
      ],
      "l2_effective_capacity_bytes": { "...": "MeasuredValue" },
      "smem_bw_per_sm_gbps": { "...": "MeasuredValue" },
      "dram_latency_ns": { "...": "MeasuredValue" },
      "dram_latency_loaded_ns": { "...": "MeasuredValue" },
      "l2_latency_ns": { "...": "MeasuredValue" },
      "l2_latency_loaded_ns": { "...": "MeasuredValue" },
      "smem_latency_ns": { "...": "MeasuredValue" }
    },
    "compute": {
      "mma_tflops_per_sm_by_warps": {
        "mma.sync:f16:f16:f32:16x8x16": [
          { "mma_warps_per_sm": 1, "tflops_per_sm": { "...": "MeasuredValue" } },
          { "mma_warps_per_sm": 2, "tflops_per_sm": { "...": "MeasuredValue" } },
          { "mma_warps_per_sm": 4, "tflops_per_sm": { "...": "MeasuredValue" } },
          { "mma_warps_per_sm": 8, "tflops_per_sm": { "...": "MeasuredValue" } }
        ]
      },
      "non_mma_issue_rate_per_sm_gips": { "...": "MeasuredValue" },
      "fp32_fma_tflops": { "...": "MeasuredValue" },
      "sfu_tflops": { "...": "MeasuredValue" }
    },
    "occupancy": {
      "smem_carveout_observed": [
        { "requested_smem_per_cta_bytes": 32768, "chosen_carveout_bytes": 65536, "max_active_ctas_per_sm": 2 }
      ]
    },
    "async_pipeline": {
      "cp_async_bw_per_sm_gbps": { "...": "MeasuredValue" },
      "tma_bw_per_sm_gbps": { "...": "MeasuredValue" },
      "tma_setup_ns": { "...": "MeasuredValue" }
    },
    "spill": {
      "local_mem_l1_hit_bw_per_sm_gbps": { "...": "MeasuredValue" },
      "local_mem_l1_miss_bw_per_sm_gbps": { "...": "MeasuredValue" },
      "local_mem_l1_hit_latency_ns": { "...": "MeasuredValue" },
      "local_mem_l1_miss_latency_ns": { "...": "MeasuredValue" },
      "l1_to_l2_spill_threshold_bytes_per_sm": { "...": "MeasuredValue" }
    },
    "overhead": {
      "launch_overhead_us": { "...": "MeasuredValue" },
      "per_cta_launch_latency_ns": { "...": "MeasuredValue" }
    },
    "cta_dispatch": {
      "observed_policy": "round_robin",
      "candidate_policies": ["round_robin"],
      "policy_params": {},
      "confidence": "high",
      "probes": [
        {
          "grid_dims": [128, 1, 1],
          "ctas_per_sm": 2,
          "waves_approx": 1.33,
          "order_consistency": 0.96,
          "first_wave_trace": [
            { "cta_id": 0, "sm_id": 0, "start_ns": 0 }
          ]
        }
      ]
    }
  }
}
```

### 2.3 Default `perturb_group` Assignments (RTX 4070 & DGX Spark GB10)

Fields sharing a `perturb_group` receive the same multiplicative draw per perturbation sample; `null` fields are drawn independently. Every curve (`*_by_active_sms`, `*_by_warps`) is scaled by a single multiplier across all points to preserve its shape.

| `perturb_group` | Fields Included | RTX 4070 (`sm_89`) | DGX Spark GB10 (`sm_121`) |
|---|---|---|---|
| `"dram_bw"` | `dram_bw_read_gbps`, `dram_bw_write_gbps`, `dram_bw_by_active_sms` | Shared GDDR6X memory controllers | Shared unified LPDDR5x bus |
| `"l2_bw"` | `l2_bw_read_gbps`, `l2_bw_write_gbps`, `l2_bw_by_active_sms`, `l2_bw_by_active_sms_cp_async` | Shared L2 fabric | Shared L2 fabric |
| `"mma_dense"` | `mma_tflops_per_sm_by_warps` for dense `f16`/`bf16`/`tf32`/`f8` `mma.sync` | Same SM tensor pipe | Same SM tensor pipe |
| `"mma_block_scaled"` | `mma_tflops_per_sm_by_warps` for block-scaled `fp4`/`fp8` `mma.sync` | N/A | Separate group from dense MMA |
| `"async_cp"` | `cp_async_bw_per_sm_gbps` | Independent async copy pipe | Independent async copy pipe |
| `"async_tma"` | `tma_bw_per_sm_gbps` | N/A (`tma: false`) | Separate from `cp.async` |
| `"spill_l1"` | `local_mem_l1_hit_bw_per_sm_gbps`, `local_mem_l1_hit_latency_ns` | SM-local L1 spill path | SM-local L1 spill path |
| `"spill_l2"` | `local_mem_l1_miss_bw_per_sm_gbps`, `local_mem_l1_miss_latency_ns` | L2/DRAM overflow spill path | L2/DRAM overflow spill path |
| `null` (independent) | `l2_effective_capacity_bytes`, `smem_bw_per_sm_gbps`, `dram_latency_*`, `l2_latency_*`, `smem_latency_ns`, `non_mma_issue_rate_per_sm_gips`, `fp32_fma_tflops`, `sfu_tflops`, `tma_setup_ns`, `l1_to_l2_spill_threshold_bytes_per_sm`, `launch_overhead_us`, `per_cta_launch_latency_ns` | Independent draws | Independent draws |

### 2.4 Downstream Consumer Alignment

- **Model core:** Reads `specs` (occupancy, carveout options, allocation granularity, MMA shape fit), `measured.*` (active-SM bandwidth curves for DRAM/L2, per-SM MMA throughput vs warp count, unloaded/loaded latency, non-MMA issue rate, spill, launch/per-CTA overhead, observed carveout, and `rel_uncertainty` for perturbation), and `cta_dispatch` (`observed_policy`, `candidate_policies`, `first_wave_trace`).
- **L2 model:** Reads `specs.l2_size_bytes`, `specs.l2_line_bytes`, `specs.l2_sector_bytes`, `specs.l1_size_bytes`, `specs.l2_partitions`, `measured.memory.l2_effective_capacity_bytes`, `measured.memory.l2_bw_*` / `dram_bw_*` / `*_latency_ns`, and `cta_dispatch`.
- **Validation:** Shares the measurement hygiene protocol and inspects `provenance` (`profile_revision`, `calibration_run_id`, clock/WSL/power metadata) to enforce strict calibration/validation separation.

---

## 3. Key Decisions

### D1. Uniform `MeasuredValue` envelope with built-in uncertainty
- **Alternatives:** (A) Plain floats in JSON, with a hardcoded ±10% constant inside the model core; (B) Store raw timing arrays in the profile.
- **Decision:** Structured `MeasuredValue` per scalar field. Keeps profiles compact, preserves observed hardware noise (CI95), and lets the model core sample perturbations directly from `rel_uncertainty` (with a 10% floor for measured fields and wider bands for spec-derived or noisy fields).

### D2. Runtime API queries + synthetic isolated microbenchmarks (no workload regression)
- **Alternatives:** (A) Fit profile parameters by regressing against Triton GEMM timings; (B) Use vendor datasheet numbers only.
- **Decision:** Query all static limits (`specs.*`) from CUDA/HIP runtime APIs (`cudaGetDeviceProperties` / `cudaDeviceGetAttribute`), and measure all effective rates (`measured.*`) using small, isolated synthetic kernels. Regression against GEMMs would leak validation data into calibration (violating Risk #9) and overfit to one operator; datasheet numbers miss achievable throughput and unified-memory effects on GB10.

### D3. Capability-flagged single schema instead of per-architecture schemas
- **Alternatives:** Separate JSON schemas per architecture (`sm_89`, `sm_121`, `sm_90`, `gfx942`).
- **Decision:** One versioned schema with `capabilities` booleans and keyed `tensor_tflops` entries (`<op_class>:<a_dtype>:<b_dtype>:<acc_dtype>:<MxNxK>`). Cleanly handles the `sm_121` split (TMA without TMEM/`tcgen05`), `sm_89` (`cp.async` + `mma.sync`), future `sm_90`/`sm_100`, and AMD CDNA without schema forks.

### D4. Compact scaling curves for DRAM/L2 concurrency, MMA warps, and L2 working set
- **Alternatives:** Single peak DRAM/L2/MMA scalars and raw spec L2 size.
- **Decision:** Measure `dram_bw_by_active_sms` and `l2_bw_by_active_sms` at 4 to 6 SM counts (1, 2, 4, 25%, 50%, 100% SMs), emit optional `l2_bw_by_active_sms_cp_async` on `sm_80+` GPUs so `cp.async.cg` L2 scaling is recorded separately when register scoreboard limits cap `ld.global.cg`, sweep working-set size across `0.25x` to `4x` spec L2 to locate `l2_effective_capacity_bytes`, and measure `mma_tflops_per_sm_by_warps` across `1, 2, 4, 8+` issuing warps per SM. Tail waves with few active SMs do not achieve peak bandwidth, `num_warps` enters the model through the per-SM MMA curve, and GB10's unified LPDDR5x + Ada's large L2 make effective L2 capacity decisive for ranking.

### D5. Dedicated register-spill microbenchmark
- **Alternatives:** Model register pressure purely via occupancy reduction.
- **Decision:** Include a forced-spill kernel (allocating register-indexed thread-local arrays that compile to `LDL`/`STL` local memory traffic) across increasing local frame sizes to measure L1-hit vs L1-miss bandwidth/latency and the empirical `l1_to_l2_spill_threshold_bytes_per_sm`. Gives Agents 3 and 4 calibrated penalties for `ptxas` spill bytes.

### D6. Empirical CTA dispatch probing via `%smid` and `%globaltimer`
- **Alternatives:** Assume strict round-robin CTA-to-SM assignment.
- **Decision:** Run lightweight probe kernels that record `(blockIdx, %smid, %globaltimer_start, %globaltimer_end)` across 1D/2D grids at 1 to 4 waves and multiple occupancies, repeated across launches. Classify dispatch into `round_robin`, `grouped`, `measured_table`, or `unknown`, emit `candidate_policies` when ambiguous (so the model core treats policy ambiguity as a perturbation dimension), and record `order_consistency` (`0.0 to 1.0`) and `first_wave_trace`.

### D7. Microbenchmark suite implemented in CUDA C++
- **Alternatives:** Python + Triton, or CuPy/NVRTC.
- **Decision:** Implement the NVIDIA microbenchmark suite in **CUDA C++** (`nvcc`), with a thin Python CLI wrapper for orchestration and JSON emission. Direct inline PTX gives exact control over `mma.sync.aligned`, `cp.async`, TMA, `%smid`, `%globaltimer`, and forced register spills without Triton compiler optimizations interfering with measurements. AMD HIP parity can reuse the same C++ harness if AMD enters scope later.

### D8. Unlocked-clock degradation policy (widen `rel_uncertainty`, do not reject)
- **Alternatives:** Reject profiles whenever `nvidia-smi` clock locking fails (common on consumer WSL2 or unified-memory SoCs).
- **Decision:** Accept unlocked profiles, record `clock_policy.locked: false` (plus per-domain `sm_clock_locked` / `mem_clock_locked` and `observed_*_clock_range_mhz`), and automatically widen `rel_uncertainty`:
  - Define observed clock drift half-width `drift_rel = (clock_max - clock_min) / (2 * clock_median)` over the benchmark run.
  - For SM-clock-bound fields (`compute.*`, `l2_bw_*`, `smem_*`, `async_pipeline.*`, `spill.*`), if `sm_clock_locked == false`: `rel_uncertainty = max(0.10, half_ci_rel, drift_rel + 0.05)` (or floor `0.15` if clock telemetry is unavailable).
  - For memory-clock-bound fields (`dram_bw_*`, `dram_latency_*`), if `mem_clock_locked == false`: apply the same formula using `observed_mem_clock_range_mhz` (or floor `0.15` if unavailable).
  - If `throttle_events > 0`, add `+0.05` to the affected domain's `rel_uncertainty` and flag it in provenance.

---

## Post-hoc spill counter capture

After counter permission takes effect on the Bifrost RTX 4070, run this from
a clean commit with Nsight Compute 2026.4 on PATH and CUDA nvcc, PyTorch and
Triton in the project environment:

```sh
flock /tmp/virtualgpu-gpu.lock .venv/bin/python -m microbench.spill_counters
```

Use `--nvcc /path/to/nvcc` or `--ncu /path/to/ncu` to select an installed
toolchain. The collector profiles one launch before the sweep to check
hardware counter permission. An `ERR_NVGPUCTRPERM` error exits with no result.
The JSON appears in `results/post-hoc/` only after every capture succeeds.
The record binds the checkout, driver, NCU, compiler, kernel artifacts and
metric names. This capture leaves the frozen evaluations unchanged.

The 60 points from `spill_reload_latency_by_count` each use the existing
kernel with zero and 128 steps. Subtracting the zero-step launch removes
frame initialization from the average per-iteration counts. Launch totals
include warp-level local load and store instructions and hardware L1TEX
sectors. Source counters also count predicated-on thread instructions.
These measurements have different units.

The attention diagnostic profiles three configurations from the published
causal GQA development manifest. It gives the prefix and diagonal helper
calls separate source locations and checks normalized PTX, register count,
shared memory and total local thread counts against the original kernel.
Per-loop averages use the CTA trip counts and attributed predicated-on
thread instructions. Local instructions whose source is outside either
helper or has no location are reported separately and stay in the launch
totals. Launch-level L1TEX sectors have no loop attribution. See the
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
for metric definitions.

## Independent MLP repeat reduction

The MLP refresh captures five separate processes for each of `mlp`
and `strided-mlp`, with three warmups and 15 samples per point.
`microbench.repeat_mlp` verifies geometry against the existing profile
and stores the raw samples and active-load telemetry. Run each capture
under the shared GPU lock. The reducer uses the median of the five
sweep medians, stores the pooled statistics as descriptive data, and
binds the new profile to a calibration manifest digest. See the
[measurement plan and archive](../results/analysis/v0.3-mlp-repeat-plan.md)
for the protocol and evidence.
