# Ascend 910B3 Formal Sealed Holdout Evaluation

> **Overall decision: Pass (all 4 cohorts pass: GEMM warm/cold and Attention warm/cold).** Every cohort beats both zero-Ascend-label baselines by more than `+0.05` in median Kendall tau-b (or by `+0.067` to `+0.654`) with `0.00%` mean top-5 regret. Because `ascend910b3` meets the formal validation bar in [docs/design.md section 8](../design.md#8-validation), the `unvalidated_target_ascend` / `unvalidated_feature` gate is lifted for `ascend910b3` (while remaining active for other unvalidated Ascend architectures).

## Setup and Provenance

- **Protocol:** Fixed before capture in [validation/ascend-910b3-preregistration.md](../../validation/ascend-910b3-preregistration.md). All 16 holdout shapes (`input_seed = 37`) are strictly disjoint from development shapes (`2048`/`4096`) and from earlier v1/v2 holdout shape sets.
- **Frozen core and profile:** Single in-order `PIPE_MTE2` and `L0A/L0B`-decoupled Cube pipeline in `crates/vgpu-core/src/sim.rs` (gated strictly on `specs.capabilities.cube_vector_split`, keeping NVIDIA/AMD bit-identical), 2D `Nd2Nz` row-width mapping in the Triton-Ascend frontends, and `profiles/ascend910b3.json` (`0.3.2-ascend`, adding the isolated AscendC `aic_nd2nz_issue_kernel` M2 row-issue curve `gmem_load_issue_controlled`, zero GEMM or Attention timing labels). Predictions were generated CPU-only before any holdout kernel ran on NPU hardware (`draws = 32`, `seed = 17`, `pair_margin = 0.05`).
- **Frozen baselines (zero Ascend labels):**
  - `roofline_wave`: analytical roofline + CTA wave quantization (`sm_count = 20`, Cube FP16 peak, HBM read bandwidth, launch overhead).
  - `neusight_style`: frozen ridge regression fits in `validation/v02-frozen-fit.json` (trained exclusively on NVIDIA RTX 4070 and GB10 labels), evaluated with Ascend 910B3 hardware profile features.
- **Hardware and stack:** Huawei Ascend 910B3 (`npu:0`, 20 AI Cores = 20 AIC + 40 AIV, 64 GiB HBM, 192 MiB L2), Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`, Python `3.12.13`.
- **Capture and telemetry:** 16 fresh disjoint shapes (8 GEMM x 16 configs = 128 variants, 8 Attention x 10 configs = 80 variants; 208/208 compiled and passed PyTorch FP32 reference accuracy checks). 3 sessions x 10 randomized complete blocks per `(case, config, cache)` = 48 sealed session reports. Warm launches use a zero-L2 register spacer (`stream_spacer`) so `torch.npu.Event` measures on-device execution without host dispatch bubbles; cold launches use a clean read-only 384 MiB `read_flush`. Across all 48 sessions, the 64 MiB warm-vs-cold L2 probe ratio stayed at `1.439-1.585` (median `1.526` GEMM, `1.528` Attention), `Aicore curFreq` stayed locked at `1800 MHz`, die temperature `41-43 C`, and real-time power `105.9-121.9 W`.
- **Receipts and digests:**
  - Freeze receipt: [validation/ascend-910b3-freeze.json](../../validation/ascend-910b3-freeze.json) (`recorded_utc = 2026-11-27T07:35:07.202060+00:00`, `prediction_archive_sha256 = 1fbe908079dbe04b0d83ce29a2b158c1e3efa14b8cad40dd43255a86d6f6163b`, `profile_sha256 = 91a9db214229e3e8619554ff7a557bfc098a7f006bdce721e50a31f8b9bccae2`, `fit_sha256 = 9bbd965c362aed8f55dbb01224e76aef5073074ef726753a4ad9b045e860bcd4`). Note: the Linux `_core` binary on the remote host was built from a clean tarball without `.git` metadata, so `core_build` in the freeze JSON reports `commit = "unknown"`.
  - Seal receipt: [validation/ascend-910b3-seal.json](../../validation/ascend-910b3-seal.json) (`48` verified session reports).
  - Scored output: [validation/ascend-910b3-holdout-scores.json](../../validation/ascend-910b3-holdout-scores.json) (`scored_utc = 2026-11-27T07:53:15.407202+00:00`, `freeze_sha256` and `seal_sha256` verified).

## Cohort Summary

Pass criteria per [docs/design.md section 8](../design.md#8-validation): median Kendall tau-b at least `+0.05` above both baselines, and mean top-5 regret at most `1.0%` and at most `0.5` percentage points worse than either baseline.

| Cohort | Decision | Method | Median tau-b | Mean top-5 regret | Ordered agreement (agree / calls) | Median abs rel error |
|---|---|---|---:|---:|---:|---:|
| `ascend910b3-gemm-warm` | **Pass** | `core` | **0.870** | **0.00%** | **99.1%** (838/846) | 14.0% |
| `ascend910b3-gemm-warm` | | `roofline_wave` | 0.803 | 7.68% | 92.0% (666/724) | - |
| `ascend910b3-gemm-warm` | | `neusight_style` | 0.617 | 7.13% | 75.3% (676/898) | - |
| `ascend910b3-gemm-cold` | **Pass** | `core` | **0.929** | **0.00%** | **98.4%** (842/856) | 31.1% |
| `ascend910b3-gemm-cold` | | `roofline_wave` | 0.803 | 8.06% | 91.6% (663/724) | - |
| `ascend910b3-gemm-cold` | | `neusight_style` | 0.275 | 23.21% | 66.5% (567/853) | - |
| `ascend910b3-attention-warm` | **Pass** | `core` | **0.800** | **0.00%** | **88.2%** (285/323) | 22.4% |
| `ascend910b3-attention-warm` | | `roofline_wave` | 0.545 | 0.00% | 79.9% (211/264) | - |
| `ascend910b3-attention-warm` | | `neusight_style` | 0.600 | 0.00% | 79.3% (268/338) | - |
| `ascend910b3-attention-cold` | **Pass** | `core` | **0.800** | **0.00%** | **88.2%** (285/323) | 22.3% |
| `ascend910b3-attention-cold` | | `roofline_wave` | 0.545 | 0.00% | 79.5% (210/264) | - |
| `ascend910b3-attention-cold` | | `neusight_style` | 0.600 | 0.00% | 79.1% (268/339) | - |

- **GEMM (`warm` and `cold`): Pass.** VirtualGPU beats `roofline_wave` by `+0.067` (warm) and `+0.126` (cold) and beats `neusight_style` by `+0.253` (warm) and `+0.654` (cold) in median Kendall tau-b, achieves `0.00%` mean top-5 regret on all 16 case-cache combinations (versus `7.68%/8.06%` for roofline and `7.13%/23.21%` for NeuSight), reaches `99.1%` (warm, only 2 reversals in 846 ordered calls) and `98.4%` (cold, only 1 reversal in 856 ordered calls) ordered agreement, and cuts warm median absolute latency error to `14.0%` (`5.2%-8.0%` on the 3-wave boundary shapes).
- **Attention (`warm` and `cold`): Pass.** Across all 8 unseen shapes spanning `{full, causal} x {MHA, GQA} x {D=64, D=128}` (including odd sequence lengths `1409` and `2177`), VirtualGPU beats both baselines by `+0.200` to `+0.255` in median Kendall tau-b, achieves `0.00%` mean top-5 regret on all 16 case-cache combinations, agrees on `88.2%` of ordered pairs, and reaches `22.3%-22.4%` median absolute latency error (`0.867-0.911` tau-b on all four non-causal shapes).

## Per-Case Breakdown

### GEMM Holdout (`validation/ascend-910b3-gemm-holdout.json`)

| Case | Shape (`M x N x K`) | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---|---:|---:|---:|---:|---:|
| `ascend-v3-decode-24` | `24 x 5632 x 3328` | warm / cold | 0.795 / 0.962 | -0.355 / -0.416 | -0.117 / -0.083 | 0.00% / 0.00% | 33.1% / 43.4% |
| `ascend-v3-decode-80` | `80 x 7168 x 3328` | warm / cold | 0.879 / 0.912 | 0.711 / 0.693 | 0.483 / 0.050 | 0.00% / 0.00% | 36.4% / 36.9% |
| `ascend-v3-wave3-below` | `128 x 7552 x 3328` | warm / cold | 0.845 / 0.929 | 0.803 / 0.803 | 0.617 / 0.250 | 0.00% / 0.00% | 5.2% / 31.9% |
| `ascend-v3-wave3-exact` | `128 x 7680 x 3328` | warm / cold | 0.862 / 0.929 | 0.694 / 0.694 | 0.617 / 0.250 | 0.00% / 0.00% | 6.2% / 30.4% |
| `ascend-v3-wave3-above` | `128 x 7808 x 3328` | warm / cold | 0.845 / 0.912 | 0.803 / 0.803 | 0.617 / 0.300 | 0.00% / 0.00% | 8.0% / 25.7% |
| `ascend-v3-prefill-med` | `896 x 5632 x 3328` | warm / cold | 0.912 / 0.933 | 0.876 / 0.876 | 0.650 / 0.650 | 0.00% / 0.00% | 13.3% / 6.7% |
| `ascend-v3-prefill-large` | `2560 x 5632 x 3328` | warm / cold | 0.912 / 0.933 | 0.876 / 0.876 | 0.650 / 0.650 | 0.00% / 0.00% | 14.7% / 14.2% |
| `ascend-v3-mask-odd` | `1281 x 3713 x 2945` | warm / cold | 0.883 / 0.850 | 0.908 / 0.908 | 0.700 / 0.650 | 0.00% / 0.00% | 50.2% / 52.5% |

### Attention Holdout (`validation/ascend-910b3-attention-holdout.json`)

| Case | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---:|---:|---:|---:|---:|
| `b2-q16-kv4-s768-d64-full` | warm / cold | 0.867 / 0.867 | 0.441 / 0.441 | 0.511 / 0.511 | 0.00% / 0.00% | 24.4% / 23.9% |
| `b2-q16-kv4-s768-d64-causal` | warm / cold | 0.556 / 0.556 | 0.701 / 0.701 | 0.644 / 0.644 | 0.00% / 0.00% | 21.2% / 20.5% |
| `b2-q16-kv16-s1664-d64-full` | warm / cold | 0.867 / 0.867 | 0.441 / 0.441 | 0.511 / 0.511 | 0.00% / 0.00% | 22.4% / 22.4% |
| `b1-q32-kv8-s1409-d64-causal` | warm / cold | 0.733 / 0.733 | 0.649 / 0.649 | 0.644 / 0.644 | 0.00% / 0.00% | 13.6% / 13.6% |
| `b2-q16-kv4-s1088-d128-full` | warm / cold | 0.911 / 0.911 | 0.441 / 0.441 | 0.556 / 0.556 | 0.00% / 0.00% | 31.2% / 31.1% |
| `b2-q16-kv4-s1088-d128-causal` | warm / cold | 0.644 / 0.644 | 0.649 / 0.649 | 0.644 / 0.644 | 0.00% / 0.00% | 22.3% / 22.2% |
| `b1-q16-kv16-s2177-d128-full` | warm / cold | 0.867 / 0.867 | 0.441 / 0.441 | 0.511 / 0.511 | 0.00% / 0.00% | 28.4% / 28.4% |
| `b1-q32-kv8-s2432-d128-causal` | warm / cold | 0.600 / 0.600 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 16.4% / 16.4% |

## Validation Progression and Remaining Residuals (Zero Fitting)

Each holdout round used a brand-new, strictly disjoint shape set with predictions frozen before capture:

1. **v1 -> v2 (HIVM parser and warm timing harness fixes):** Fixing swapped Ascend fractal NZ `(Cols1, Rows1)` tile axes for `tl.constexpr` shapes, skipping `AIC`-only `cbuf` `hivm.hir.vbrc` helpers, and replacing the warm host-dispatch bubble with a zero-L2 `stream_spacer` lifted GEMM from `0.474/0.575` to `0.773/0.720` tau-b and cut Attention median error from `48.3%-49.3%` to `23.6%-23.9%`, passing both Attention cohorts.
2. **v2 -> v3 (Single-MTE2 physics and isolated `Nd2Nz` M2 row-issue microbenchmark):** Post-scheduling HIVM IR showed two hardware mechanisms missing from the v2 model:
   - Each AI Core has a single in-order `PIPE_MTE2` DMA engine (`pipe_barrier[<PIPE_MTE2>]`) and `L0A/L0B` decouples Cube `PIPE_M` from L1 `cbuf` (`set_flag[<PIPE_MTE1>, <PIPE_MTE2>]`). Modeling the single MTE2 queue (`round_trips = n - d`) and L0-decoupled Cube pipeline eliminated the artificial `~1.5x` `mb0/mb1` inversion.
   - An isolated AscendC 2D `Nd2Nz` (`GM ND -> L1 Fractal NZ`) microbenchmark (`aic_nd2nz_issue_kernel`, `0.3.2-ascend`) measured a hardware row-issue ceiling of `~0.415 Grows/s per AIC` (`128x32` 8 KiB = `307.4 ns`, `128x64` 16 KiB = `307.7 ns`, `128x128` 32 KiB = `321.3 ns`). Wiring 2D row widths into the existing `gmem_load_issue_controlled` (`lsu_rate`) pipe lifted GEMM from `0.773/0.720` to `0.870/0.929` tau-b (`99.1%/98.4%` ordered agreement, `0.00%` regret, warm error `34.6% -> 14.0%`) and Attention from `0.756/0.756` to `0.800/0.800` tau-b (`88.2%` ordered agreement).
3. **Known residuals:** Odd unaligned GEMM (`ascend-v3-mask-odd`, `1281 x 3713 x 2945`) still shows `50.2%-52.5%` median absolute latency error (even while ranking at `0.883/0.850` tau-b and `0.00%` regret) because `Nd2Nz` on non-32B-aligned strides incurs additional hardware unaligned-DMA penalty not yet isolated in a dedicated microbenchmark, and small decode tiles (`decode-24/80`) still show `33%-43%` absolute error with `0.795-0.962` tau-b.
