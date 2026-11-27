# Ascend 910B3 Sealed Holdout Preregistration

This document fixes the holdout workload manifests, frozen model/profile revision, independent baselines, hardware capture protocol, and pass criteria for Huawei Ascend 910B3 (`profiles/ascend910b3.json`, `0.3.2-ascend`) before any holdout timings are collected or opened.

## 1. Frozen Inputs

- **Core and frontend revision:** `main` plus two zero-fitting physical refinements under `cube_vector_split`:
  1. Single in-order `PIPE_MTE2` + `L0A/L0B`-decoupled Cube pipeline in `crates/vgpu-core/src/sim.rs` (`round_trips = n - d`, `d = 0` serializing only `MTE1` (`smem_load`) with `MTE2` while `d >= 1` overlaps both; NVIDIA/AMD untouched).
  2. Isolated AscendC `Nd2Nz` 2D row-issue microbenchmark (`gmem_load_issue_controlled` in `profiles/ascend910b3.json`, `0.3.2-ascend`) paired with 2D row-width `transfer_width_bytes = row_bytes // 32` in `triton_ascend_gemm.py` and `triton_ascend_attention.py`.
  Zero modifications to `crates/vgpu-core/`, `python/virtualgpu/`, or `profiles/ascend910b3.json` during capture or scoring.
- **Hardware profile:** `profiles/ascend910b3.json` (`0.3.2-ascend`, M1 + M2 microbenchmarks including isolated FFTS+ cross-core sync handoff latency and isolated AIC `Nd2Nz` 2D row-issue rate, zero GEMM or Attention fitting).
- **Simulation settings:** `draws = 32`, `seed = 17`, `pair_margin = 0.05` (matching `validation/v02_inputs.py`), evaluated independently for `warm` and `cold` L2 states.

## 2. Disjoint Holdout Cohorts (v3)

All holdout workloads are strictly disjoint from the informal dev sanity checks (`2048^3` GEMM, `4096^3` GEMM, `S=2048` Attention), the v1 holdout (PR #13), and the v2 holdout (PR #14):

- **GEMM (`validation/ascend-910b3-gemm-holdout.json`):** 8 unseen shapes x 16 configs = 128 variants using `validation/ascend_kernels.py:dense_gemm` (FP16 inputs/output, FP32 accumulator, `input_seed = 37`). Covers small-M decode (`24x5632x3328`, `80x7168x3328`), a 20-AI-Core three-wave boundary triplet (`59 / 60 / 61` CTAs at `128x128`: `128x7552x3328`, `128x7680x3328`, `128x7808x3328`), medium and large prefill (`896x5632x3328`, `2560x5632x3328`), and unaligned odd dimensions (`1281x3713x2945`).
- **Attention (`validation/ascend-910b3-attention-holdout.json`):** 8 unseen shapes x 10 configs = 80 variants using `validation/attention.py:fused_attention` (FP16 Q/K/V/O, FP32 LSE/accumulator, `input_seed = 37`). Every `seq_len` (`768, 1088, 1409, 1664, 2177, 2432`) is strictly disjoint from dev, v1, and v2. Covers all 8 strata across `{non-causal, causal} x {MHA, GQA} x {D=64, D=128}` including aligned and odd sequence lengths.

## 3. Independent Baselines (Zero Ascend Labels)

1. **`roofline_wave`:** Analytical roofline + CTA wave quantization (`validation/baseline.py` and `validation/attention_baseline.py`), using `profiles/ascend910b3.json` peak `cube.mmad:f16:f16:f32:16x16x16` TFLOPS, peak HBM read bandwidth, `sm_count = 20`, and `launch_overhead_us`.
2. **`neusight_style`:** The published frozen ridge regression fits in `validation/v02-frozen-fit.json` (trained exclusively on NVIDIA RTX 4070 and GB10 historical labels, zero Ascend timing labels), evaluated with Ascend 910B3 profile hardware features (`sm_count = 20`, `l2_size_bytes`, HBM/L2 bandwidth, Cube FP16 TFLOPS, `num_warps = 1`, `num_stages = 2 if multibuffer else 1`).

## 4. Blind Execution Sequence and Capture Protocol

1. **Freeze predictions first (CPU-only):** Generate and archive `core`, `roofline_wave`, and `neusight_style` predictions for all 16 cases (warm + cold) and record SHA-256 digests and UTC timestamp in `validation/ascend-910b3-freeze.json` before launching any holdout kernel on NPU hardware.
2. **Hardware capture on `npu:0`:**
   - Software stack: Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`.
   - 3 independent sessions per case (order seeds `101, 102, 103`), 10 randomized complete blocks per session (30 paired blocks per `(case, config, cache)`).
   - Warm L2 preparation: one pre-warm launch of the timed kernel followed by a zero-L2-traffic register-only spacer kernel (`stream_spacer`, ~420 us) so `ev0 -> fn -> ev1` executes back-to-back on-device without `torch_npu` host-dispatch bubbles.
   - Clean-cold L2 preparation: read-only 384 MiB (2x L2 capacity) reduction kernel (`read_flush`, no dirty writebacks into the timed window).
   - Hard session L2 probe gate: every session measures a 64 MiB warm-vs-cold L2 read probe and aborts if `median(cold_ns) / median(warm_ns) <= 1.1`.
   - Numerical correctness check against PyTorch FP32 reference before timing each config.
   - `npu-smi` temperature, power, and clock telemetry recorded before and after every session.
   - Publish SHA-256 receipts in `validation/ascend-910b3-seal.json`.
3. **Single frozen scoring pass:** Verify all freeze and seal SHA-256 digests, then score once using `validation/score.py` (`kendall_tau_b`, `top_k_regret` with `k=5`, 5% resolution band pair confusion/precision, latency error).

## 5. Pass Criteria

Per [docs/design.md section 8](../docs/design.md#8-validation), evaluated independently on each of the four `(family, cache)` cohorts (`gemm-warm`, `gemm-cold`, `attention-warm`, `attention-cold`):

- Median Kendall tau-b is at least `+0.05` above both `roofline_wave` and `neusight_style`.
- Mean top-5 regret is at most `1.0%` and at most `0.5` percentage points worse than either baseline.
- Removing `unvalidated_target_ascend` / `unvalidated_feature` gating for `ascend910b3` requires passing all four cohorts; otherwise the report is published as-is and the `unvalidated_target_ascend` flag stays in place.
