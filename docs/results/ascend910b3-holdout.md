# Ascend 910B3 Formal Sealed Holdout Evaluation

> **Overall decision: Fail (Attention warm/cold Pass; GEMM warm/cold Fail).** Because a device passes only when both kernel families pass in both cache regimes, `ascend910b3` remains gated with `unvalidated_feature` and `unvalidated_target_ascend`.

## Setup and Provenance

- **Protocol:** Fixed before capture in [validation/ascend-910b3-preregistration.md](../../validation/ascend-910b3-preregistration.md).
- **Frozen core and profile:** Commit `4618e2c` (merged into `main` via PR #10) with `profiles/ascend910b3.json` (`0.3.1-ascend`, zero GEMM or Attention timing labels). Predictions were generated CPU-only before any holdout kernel ran on NPU hardware (`draws = 32`, `seed = 17`, `pair_margin = 0.05`).
- **Frozen baselines (zero Ascend labels):**
  - `roofline_wave`: analytical roofline + CTA wave quantization (`sm_count = 20`, Cube FP16 peak, HBM read bandwidth, launch overhead).
  - `neusight_style`: frozen ridge regression fits in `validation/v02-frozen-fit.json` (trained exclusively on NVIDIA RTX 4070 and GB10 labels), evaluated with Ascend 910B3 hardware profile features.
- **Hardware and stack:** Huawei Ascend 910B3 (`npu:0`, 20 AI Cores = 20 AIC + 40 AIV, 64 GiB HBM, 192 MiB L2), Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`, Python `3.12.13`.
- **Capture and telemetry:** 16 unseen shapes (8 GEMM x 16 configs = 128 variants, 8 Attention x 10 configs = 80 variants; 208/208 compiled and passed PyTorch FP32 reference accuracy checks). 3 sessions x 10 randomized complete blocks per `(case, config, cache)` = 48 sealed session reports. Across all 96 pre/post `npu-smi` samples, `Aicore curFreq` stayed locked at `1800 MHz`, die temperature `40-43 C`, real-time power `105.6-121.5 W`.
- **Receipts and digests:**
  - Freeze receipt: [validation/ascend-910b3-freeze.json](../../validation/ascend-910b3-freeze.json) (`recorded_utc = 2026-11-26T23:26:53.484153+00:00`, `prediction_archive_sha256 = 8453160302d605a57a1d169a68e6474e4b28ee931c47e2d6854a6f063e04c0e3`, `profile_sha256 = 8846f4baf8e10d708def63e238899bf8ddd3cdb7c7df430fac5a7577ee25f91c`, `fit_sha256 = 9bbd965c362aed8f55dbb01224e76aef5073074ef726753a4ad9b045e860bcd4`). Note: the Linux `_core` wheel on the remote host was built from a clean tarball of commit `4618e2c` without `.git` metadata, so `core_build` in the freeze JSON reports `commit = "unknown"`.
  - Seal receipt: [validation/ascend-910b3-seal.json](../../validation/ascend-910b3-seal.json) (`48` verified session reports).
  - Scored output: [validation/ascend-910b3-holdout-scores.json](../../validation/ascend-910b3-holdout-scores.json) (`scored_utc = 2026-11-26T23:49:08.975828+00:00`, `freeze_sha256` and `seal_sha256` verified).

## Cohort Summary

Pass criteria per [docs/design.md section 8](../design.md#8-validation): median Kendall tau-b at least `+0.05` above both baselines, and mean top-5 regret at most `1.0%` and at most `0.5` percentage points worse than either baseline.

| Cohort | Decision | Method | Median tau-b | Mean top-5 regret | Ordered agreement (agree / calls) | Median abs rel error |
|---|---|---|---:|---:|---:|---:|
| `ascend910b3-attention-warm` | **Pass** | `core` | **0.667** | **0.00%** | **89.0%** (282/317) | 48.3% |
| `ascend910b3-attention-warm` | | `roofline_wave` | 0.545 | 0.00% | 79.5% (210/264) | - |
| `ascend910b3-attention-warm` | | `neusight_style` | 0.578 | 0.00% | 79.5% (271/341) | - |
| `ascend910b3-attention-cold` | **Pass** | `core` | **0.689** | **0.00%** | **85.7%** (282/329) | 49.3% |
| `ascend910b3-attention-cold` | | `roofline_wave` | 0.571 | 0.00% | 79.5% (210/264) | - |
| `ascend910b3-attention-cold` | | `neusight_style` | 0.600 | 0.00% | 79.5% (271/341) | - |
| `ascend910b3-gemm-warm` | **Fail** | `core` | 0.474 | 5.89% | 39.7% (323/814) | 54.7% |
| `ascend910b3-gemm-warm` | | `roofline_wave` | 0.586 | 0.33% | 48.1% (358/745) | - |
| `ascend910b3-gemm-warm` | | `neusight_style` | 0.517 | 0.06% | 38.5% (344/893) | - |
| `ascend910b3-gemm-cold` | **Fail** | `core` | 0.575 | 10.63% | 68.7% (571/831) | 40.4% |
| `ascend910b3-gemm-cold` | | `roofline_wave` | 0.831 | 12.75% | 86.8% (647/745) | - |
| `ascend910b3-gemm-cold` | | `neusight_style` | 0.250 | 29.04% | 64.9% (538/829) | - |

- **Attention (`warm` and `cold`): Pass.** On all 8 unseen shapes spanning `{full, causal} x {MHA, GQA} x {D=64, D=128}` (including odd sequence lengths `1537` and `1793`), VirtualGPU beats both baselines by `+0.089` to `+0.122` in median Kendall tau-b (above the `+0.05` requirement), achieves `0.00%` mean top-5 regret, and agrees with hardware on `85.7%` to `89.0%` of ordered verdicts (vs `79.5%` for baselines).
- **GEMM (`warm` and `cold`): Fail.** Both cache regimes failed rank gain against `roofline_wave` and exceeded the `1.0%` mean top-5 regret cap (`5.89%` warm, `10.63%` cold).

## Per-Case Breakdown

### Attention Holdout (`validation/ascend-910b3-attention-holdout.json`)

| Case | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---:|---:|---:|---:|---:|
| `b2-q16-kv4-s1024-d64-full` | warm / cold | 0.689 / 0.689 | 0.441 / 0.389 | 0.511 / 0.467 | 0.00% / 0.00% | 37.6% / 39.6% |
| `b2-q16-kv4-s1024-d64-causal` | warm / cold | 0.600 / 0.600 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 22.3% / 22.2% |
| `b2-q16-kv16-s1536-d64-full` | warm / cold | 0.644 / 0.733 | 0.389 / 0.441 | 0.467 / 0.511 | 0.00% / 0.00% | 39.9% / 40.8% |
| `b1-q32-kv8-s1537-d64-causal` | warm / cold | 0.689 / 0.689 | 0.649 / 0.649 | 0.644 / 0.644 | 0.00% / 0.00% | 23.5% / 23.9% |
| `b2-q16-kv4-s1280-d128-full` | warm / cold | 0.733 / 0.733 | 0.389 / 0.389 | 0.467 / 0.467 | 0.00% / 0.00% | 56.8% / 57.7% |
| `b2-q16-kv4-s1280-d128-causal` | warm / cold | 0.644 / 0.644 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 116.9% / 117.0% |
| `b1-q16-kv16-s1793-d128-full` | warm / cold | 0.689 / 0.733 | 0.441 / 0.493 | 0.511 / 0.556 | 0.00% / 0.00% | 61.6% / 63.3% |
| `b1-q32-kv8-s2560-d128-causal` | warm / cold | 0.644 / 0.644 | 0.701 / 0.701 | 0.689 / 0.689 | 0.00% / 0.00% | 93.6% / 94.0% |

### GEMM Holdout (`validation/ascend-910b3-gemm-holdout.json`)

| Case | Shape (`M x N x K`) | Cache | Core tau-b | Roofline tau-b | NeuSight tau-b | Core top-5 regret | Core med abs err |
|---|---|---|---:|---:|---:|---:|---:|
| `ascend-decode-16` | `16 x 3584 x 3072` | warm / cold | -0.017 / -0.207 | -0.042 / -0.335 | 0.083 / -0.150 | 0.00% / 28.78% | 60.7% / 46.0% |
| `ascend-decode-64` | `64 x 5120 x 3584` | warm / cold | 0.017 / -0.035 | 0.624 / 0.822 | 0.183 / -0.083 | 1.09% / 26.46% | 45.5% / 40.0% |
| `ascend-wave-below` | `128 x 2432 x 3072` | warm / cold | 0.500 / 0.610 | 0.420 / 0.840 | 0.400 / 0.267 | 0.00% / 0.00% | 63.5% / 29.9% |
| `ascend-wave-exact` | `128 x 2560 x 3072` | warm / cold | 0.449 / 0.541 | 0.548 / 0.566 | 0.467 / 0.233 | 0.00% / 0.00% | 64.5% / 32.0% |
| `ascend-wave-above` | `128 x 2688 x 3072` | warm / cold | 0.276 / 0.249 | 0.548 / 0.712 | 0.567 / 0.233 | 0.11% / 0.00% | 50.7% / 40.7% |
| `ascend-prefill-med` | `1024 x 3584 x 3072` | warm / cold | 0.725 / 0.734 | 0.876 / 0.876 | 0.650 / 0.650 | 0.00% / 0.00% | 54.8% / 44.2% |
| `ascend-prefill-large` | `3072 x 3584 x 3072` | warm / cold | 0.742 / 0.734 | 0.876 / 0.876 | 0.650 / 0.650 | 0.00% / 0.00% | 54.5% / 55.0% |
| `ascend-mask-odd` | `1537 x 2945 x 2497` | warm / cold | 0.582 / 0.650 | 0.908 / 0.908 | 0.700 / 0.683 | 45.90% / 29.78% | 36.0% / 36.2% |

## Post-Hoc Root-Cause Diagnosis (No Refitting Applied)

Inspection of the frozen predictions and sealed captures after scoring identified two genuine `hivm.py` / `triton_ascend_gemm.py` frontend bugs, one known Attention bias, and one Ascend `torch.npu.Event` harness artifact. None of these were patched prior to scoring; all fixes are deferred to a future revision with a fresh disjoint holdout.

1. **Static-shape `mmadL1` tile misparse when `M, N, K` are `tl.constexpr`.** Unlike the informal dev sanity check (which passed runtime `int32` `M, N, K`), the holdout GEMM kernel (`validation/ascend_kernels.py:dense_gemm`) mirrors `validation/gemm.py` by passing `M, N, K` as `tl.constexpr`. Under static shapes, `hivm.py` parsed `mmadL1` operand memrefs with swapped `(BM, BN)` and collapsed `BK` to `128` (e.g., `m128n256k64` became `tile_mnk = [256, 128, 128]`, `m128n64k64` became `[64, 128, 128]`, and `m128n128k{32,64,128}` all produced identical `[128, 128, 128]` with `k_trips = 24`). `triton_ascend_gemm.py` then overwrote the user tile config with that misparsed `tile_mnk`, inverting rectangular decode configs (`ascend-decode-16`: `m128n256k64` predicted `131.0 us` vs `54.9 us` measured, while `m128n64k64` predicted `37.2 us` vs `111.0 us` measured) and erasing all sensitivity to `BK in {32, 64, 128}`.
2. **Spurious `loop_vec_ops` issue bottleneck on `AIC_ONLY` GEMM.** On pure Cube GEMM (`ub_bytes = 0`, `workspace_bytes = 0`), `hivm.py` still incremented `loop_vec_ops` inside the loop, causing `triton_ascend_gemm.py` to emit `512-2048` vector ALU instructions per K-iteration against `non_mma_issue_rate_per_sm_gips = 1.18 GIPS`. This made large-tile GEMM configs `98%` `issue`-bound (e.g., `ascend-prefill-large` `m128n256k64`: predicted `1012.3 us` vs `487.0 us` measured) and pushed `m128n256k64` to rank 6 on `ascend-mask-odd` (`29.8%` cold regret).
3. **Ascend `torch.npu.Event` warm-path host-dispatch floor (`< 160 us` kernels).** Every session's warm-vs-cold 64 MiB L2 read probe recorded `cold_probe_ratio = 0.47-0.56` (`warm ~ 175 us`, `cold ~ 83 us`). A follow-up isolation test confirmed why: `torch_npu`'s async `TaskQueue` has a `~110-160 us` host dispatch latency. In `cold`, the preceding 384 MiB `read_flush` (`~250 us`) kept the NPU queue busy so `ev0 -> fn -> ev1` ran back-to-back on-device (`30.6 us` measured on `ascend-wave-exact`). In `warm`, a single pre-warm `fn()` under `~160 us` finished before Python submitted the timed `fn()`, adding the host-dispatch bubble into `warm` (`158-168 us` flat across short decode/wave configs, explaining `gemm-warm`'s `371/814` hardware-equivalent pairs). Longer kernels (`prefill-large`, `mask-odd`, and all 8 Attention shapes) ran well above the floor and showed normal `warm < cold`. Future Ascend harnesses should pack multiple back-to-back pre-warm launches (or device-side loop repetition) before `ev0.record()` and fail fast if `cold_probe_ratio <= 1.0`.
4. **Attention conservative latency bias on `D=128`.** Although Attention passed all formal ranking and regret criteria (`0.00%` top-5 regret across all 8 shapes), absolute predicted latency remains `+22%` to `+41%` high at `D=64` and `+57%` to `+117%` high at `D=128` due to serial addition of AIV issue time and FFTS+ cross-core sync latency in `Parts::compute()` (also noted in [docs/design.md section 9.4](../design.md#94-ascend-910b3-mapping-and-known-limits)).
