# VirtualGPU Hardware Profile Library

This directory stores versioned GPU hardware profiles conforming to [`schemas/gpu_profile.schema.json`](../schemas/gpu_profile.schema.json) (`schema_version: "0.1"`). Each profile captures one SKU's static architectural limits and isolated microbenchmark measurements, allowing VirtualGPU to simulate and rank kernels for that GPU on a CPU without owning the hardware.

To contribute a profile for a GPU you have access to, see [`docs/contributing-a-profile.md`](../docs/contributing-a-profile.md).

## Available Profiles

| File | SKU | Arch | Memory | Revision | Source | Clocks | Driver / Toolkit |
|---|---|---|---|---|---|---|---|
| [`l40s.json`](l40s.json) | NVIDIA L40S | `sm_89` | Discrete (GDDR6) | `0.21.0-posthoc` | `partial` | Locked with `nvidia-smi` on native Linux (2100 MHz SM / 9001 MHz Mem). The collector still records `locked: false`. | `595.91.07` / `13.5` |
| [`rtx3070.json`](rtx3070.json) | NVIDIA GeForce RTX 3070 | `sm_86` | Discrete (GDDR6) | `0.21.0-posthoc` | `partial` | Unlocked (WSL2, 1770 to 2025 MHz SM / 7001 to 7201 MHz Mem) | `610.88` / `13.5` |
| [`rtx4070.json`](rtx4070.json) | NVIDIA GeForce RTX 4070 | `sm_89` | Discrete (GDDR6X) | `0.9.0-m3` | `partial` | Unlocked (WSL2, 2550–2865 MHz SM / 10251 MHz Mem) | `610.88` / `13.5` |

## Where Each Field Comes From

Every profile is built from public APIs and isolated synthetic microbenchmarks — never from reverse engineering and never by fitting parameters to GEMM or LLM kernel timings.

| Section | Source | Contents |
|---|---|---|
| `identity` | CUDA Runtime API | `vendor`, `arch` (e.g., `sm_89`), `sku`, `memory_type` (`discrete` or `unified`). |
| `provenance` | `nvidia-smi`, OS & collector | Suite/driver/toolkit versions, UTC timestamp, `source` label, `power_limit_w`, and `clock_policy`. |
| `specs` | CUDA Runtime API (`cudaGetDeviceProperties` / `cudaDeviceGetAttribute`) | Exact hardware limits: `sm_count`, `warp_size`, `registers_per_sm`, `shared_mem_per_sm_bytes`, carveout options, `l2_size_bytes`, `l2_line_bytes`, `l2_sector_bytes`, `capabilities`, and supported `mma_shapes`. |
| `measured.memory` | `vgpu_bench` (`dram-bw`, `l2`, `latency`, `latency-curve`, `l2-rescan`, `mlp`, `closed-loop`, `closed-loop-mma`, `l2-merge`) | Achievable DRAM/L2/SMEM bandwidth, active-SM scaling curves, bytes-in-flight MLP curves, closed-loop latency/bandwidth tables, L2 effective capacity, L2 rescan hit-rate curve, and MSHR miss-merging rates. |
| `measured.compute` | `vgpu_bench` (`mma`, `mma-chain`) + estimates | Measured `mma_tflops_per_sm_by_warps`, `mma_chain` dependency throughput, and `fp32_fma_tflops`. Currently `non_mma_issue_rate_per_sm_gips` and `sfu_tflops` are tagged `source: "estimated"`. |
| `measured.async_pipeline` & `measured.pipeline` | `vgpu_bench` (`cp-async`, `stage-overhead`) | `cp.async` per-SM bandwidth, per-stage synchronization overhead, and minimum async roundtrip latency (TMA fields are `null` on architectures without TMA). |
| `measured.spill` | Estimates (pending spill microbenchmark) | Per-SM local-memory L1 hit/miss bandwidth, latency, and L1-to-L2 spill threshold; currently tagged `source: "estimated"`. |
| `measured.overhead` | `vgpu_bench` (`launch`) | Device launch overhead (CUDA Graph), host launch overhead, and per-CTA launch latency. |
| `measured.cta_dispatch` | `vgpu_bench` (`cta-dispatch`) | Empirical CTA-to-SM first-wave mapping (`%smid`, `%globaltimer`), `observed_policy`, `candidate_policies`, `order_consistency`, and `confidence`. |

## Reading `provenance`

`provenance` records the exact conditions under which a profile was collected:

- `profile_revision` / `suite_version`: Version of the profile format and microbenchmark suite (`microbench/collect_profile.py`).
- `calibration_run_id` / `timestamp_utc`: Unique identifier and UTC timestamp of the collection run.
- `source`: Honest summary of how `measured.*` values were obtained:
  - `measured`: Every `MeasuredValue` leaf under `measured` was collected directly on hardware (`source: "measured"`).
  - `partial`: Core hardware parameters were measured on hardware, while one or more secondary fields remain `source: "estimated"`.
  - `spec_derived`: All `MeasuredValue` leaves are derived from public specifications (`source: "estimated"`), with no direct hardware run.
- `driver_version` / `toolkit_version` / `host_os` / `is_wsl` / `power_limit_w`: Software stack and host environment.
- `clock_policy`:
  - `locked`, `sm_clock_locked`, `mem_clock_locked`: Whether SM and memory clocks were locked during collection.
  - `sm_clock_mhz`, `mem_clock_mhz`: Representative active-load clocks.
  - `observed_sm_clock_range_mhz`, `observed_mem_clock_range_mhz`: `[min, max]` clocks polled via `nvidia-smi` under active benchmark load.
  - `throttle_events`: Count of thermal slowdown samples observed during collection (`0` is expected for a clean profile).

## Understanding Confidence (`rel_uncertainty`, `locked`, `source`)

Every scalar measurement uses a `MeasuredValue` envelope:

```json
{
  "value": 434.7,
  "ci95_low": 431.2,
  "ci95_high": 437.8,
  "rel_uncertainty": 0.1,
  "unit": "GB/s",
  "method": "cuda_mlp_dram_clean_ld_global_v4",
  "source": "measured",
  "perturb_group": "dram_bw"
}
```

How VirtualGPU uses these fields during simulation:

- **`rel_uncertainty` drives perturbation-based ranking resolution.** The Rust core draws shared Latin Hypercube samples (default 32 draws plus nominal draw 0) across all measured parameters, perturbing each by `±rel_uncertainty` (grouped by `perturb_group` so correlated quantities move together). A pair of kernel variants is only declared ordered if all draws agree by at least 5%; otherwise it is reported as `indistinguishable`.
- **Floor and widening rules:**
  - Directly measured fields with locked clocks carry a `0.10` (±10%) floor (or wider if the bootstrap 95% CI half-width is larger).
  - When `clock_policy.locked` is `false` (for example on WSL2, where Linux cannot lock GPU clocks), `collect_profile.py` widens `rel_uncertainty` for clock-sensitive fields based on the observed SM clock drift range (e.g., `0.1082` on `rtx4070.json`).
  - Fields with `source: "estimated"` carry a wider `0.30` (±30%) uncertainty so comparisons that depend heavily on unmeasured parameters naturally fall back to `indistinguishable` instead of producing false confidence.

## Validating Profiles

Run the built-in schema and semantic validator on any profile:

```bash
python -m virtualgpu.profile profiles/rtx4070.json
```

CI validates every `profiles/*.json` file against `schemas/gpu_profile.schema.json`, all semantic invariants in `python/virtualgpu/profile.py`, `provenance` completeness, and strict consistency between `provenance.source` and leaf `MeasuredValue.source` tags.
