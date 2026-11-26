# Attention development set: model core diagnosis

**Diagnosis only.** This is a diagnosis of the core on the RTX 4070 attention development set (`fused_attention`, eight cases, measured at `fde371d`). No model parameter was changed or fitted to it. The attention holdout stays sealed (`validation/holdout-seal.json`) and is not used here.

## Setup

- Core: `main` `8a83947`, formal build (not dirty). Profile `rtx4070` `0.14.0-posthoc`.
- KernelIR from the `main` attention extractor (`7f71e56`, tile `(BM, BN)` from the TTGIR `tt.dot`), with `TRITON_DEFAULT_FP_FUSION=1`.
- Scoring: `validation.score` with `--aot` for every config and `--core-revision`. All 16 scores (eight cases, warm and cold) are `formal`; all 1128 artifact checks are `matched` with `ttgir_match = true`.
- 228 configs per cache state, of which 40 exceed the 99 KB shared-memory limit. The core reports them `unsupported` and they were not measured, so 188 are scored.
- An earlier run that passed `BLOCK_M`/`BLOCK_N` to the extractor by hand gave KernelIR that is byte-identical outside metadata, and the same predictions to the last bit.

## Result: systematic underprediction, poor ranking

| Case | Cache | Scored | Median error | Worst error | Kendall tau-b | Top-5 regret |
|---|---|---|---|---|---|---|
| `b1-q16-kv16-s1025-d64-causal` | warm | 24 of 24 | -19.3% | -60.4% | 0.08 | 14.3% |
| `b1-q16-kv16-s512-d64-full` | warm | 24 of 24 | -17.7% | -53.2% | 0.38 | 0.0% |
| `b1-q32-kv8-s1021-d64-full` | warm | 24 of 24 | -7.6% | -54.0% | 0.08 | 0.0% |
| `b1-q32-kv8-s2048-d64-causal` | warm | 24 of 24 | -11.7% | -59.1% | -0.03 | 0.0% |
| `b2-q16-kv16-s2048-d128-full` | warm | 24 of 24 | -20.7% | -56.9% | -0.07 | 26.0% |
| `b2-q24-kv24-s1536-d128-causal` | warm | 22 of 22 | -18.6% | -57.9% | -0.10 | 15.5% |
| `b2-q32-kv4-s1536-d128-full` | warm | 24 of 24 | -21.4% | -54.7% | 0.36 | 1.6% |
| `b2-q32-kv4-s768-d128-causal` | warm | 22 of 22 | -25.2% | -59.4% | -0.38 | 14.5% |
| `b1-q16-kv16-s1025-d64-causal` | cold | 24 of 24 | -18.9% | -59.2% | 0.20 | 3.1% |
| `b1-q16-kv16-s512-d64-full` | cold | 24 of 24 | -19.3% | -52.5% | 0.56 | 0.0% |
| `b1-q32-kv8-s1021-d64-full` | cold | 24 of 24 | -6.5% | -53.8% | 0.07 | 1.3% |
| `b1-q32-kv8-s2048-d64-causal` | cold | 24 of 24 | -11.9% | -58.8% | 0.12 | 0.0% |
| `b2-q16-kv16-s2048-d128-full` | cold | 24 of 24 | -20.6% | -57.0% | -0.07 | 26.4% |
| `b2-q24-kv24-s1536-d128-causal` | cold | 22 of 22 | -19.3% | -57.9% | -0.14 | 16.1% |
| `b2-q32-kv4-s1536-d128-full` | cold | 24 of 24 | -20.3% | -54.8% | 0.14 | 1.6% |
| `b2-q32-kv4-s768-d128-causal` | cold | 22 of 22 | -21.8% | -59.0% | -0.23 | 14.5% |

Error is predicted over measured median, minus one. The core labels 185 of the 188 configs MMA bound and predicts most of them near the tensor peak. Warm and cold are nearly the same, so memory is not what separates the configs.

The error depends on the tile, all cases pooled:

| `BM` | Warps | Points (warm) | Median error warm | Median error cold | Range (warm) |
|---|---|---|---|---|---|
| 128 | 4 | 48 | -11.2% | -11.1% | -34.2% to +12.1% |
| 128 | 8 | 48 | -12.4% | -10.3% | -23.4% to +7.7% |
| 64 | 4 | 46 | -20.5% | -20.9% | -38.4% to +8.6% |
| 64 | 8 | 46 | -50.8% | -47.8% | -60.4% to -31.5% |

## Candidate 1: softmax and MMA are serial within a CTA

Each iteration of the attention loop is a dependency chain: the `QK^T` MMA, then the softmax (row max, `exp2`, row sum, rescaling the accumulator: ALU, SFU and shuffles), then the `PV` MMA. Within a CTA the three cannot overlap, and the warps meet at `bar.sync`. Almost every config runs one CTA per SM, so no other CTA fills the gap, and the iteration takes close to the MMA time plus the softmax time. The core takes the maximum over the pipes of the shared compute path. GEMM has little non-MMA work, so the maximum and the sum were close there, and this did not show.

Diagnostic (temporary source change in a diagnostic build on `e3d667f` with profile `0.13.0-posthoc`; reverted, never committed): adding the MMA time to the largest of the other pipes, instead of taking the maximum, narrows the warm median errors from -7.6%..-25.2% to -14.4%..+3.3%, raises tau in seven of the eight cases (for example `b1-q32-kv8-s2048-d64-causal` from -0.03 to 0.40) and lowers most regrets (`b2-q16-kv16-s2048-d128-full` from 26% to 3.4%).

This is development-set evidence of a direction, not a parameter. Before it enters the model it needs a microbenchmark: one CTA per SM (then two and three), a loop of `m` dependent MMAs, then `k` ALU and SFU instructions that depend on the MMA result plus a few shuffle reductions, then `m` MMAs and a `bar.sync`, swept over `m`, `k` and warps, to measure whether the iteration time follows the maximum or the sum and how much a second CTA hides.

## Candidate 2: the KernelIR misses work at `BM = 64`

The sum diagnostic still leaves `BM = 64` with eight warps about 45% fast. Compiling `fused_attention` offline (`b2-q16-kv16-s2048-d128-full`, `num_stages = 3`, Triton 3.9, `sm_89`) and counting static instructions in the loop body:

| Config | PTX `mma.sync` per warp | Times warps | KernelIR MMA count per CTA iteration | PTX `bar.sync` | PTX `st.shared` / `ld.shared` | KernelIR barriers |
|---|---|---|---|---|---|---|
| `BM=64 BN=32 w4` | 64 | 256 | 256 | 16 | 30 / 13 | 1 |
| `BM=64 BN=32 w8` | 48 | 384 | 256 | 14 | 22 / 9 | 1 |
| `BM=128 BN=32 w4` | 128 | 512 | 512 | 12 | 34 / 1 | 1 |
| `BM=128 BN=32 w8` | 64 | 512 | 512 | 6 | 17 / 1 | 1 |

1. With `BM = 64` and eight warps the compiled code issues 50% more MMAs than the KernelIR records (384 against 256). The KernelIR derives the count from the tile shape; Triton's warp layout for 64 rows over eight warps repeats part of the work.
2. With `BM = 64` the loop body has more `bar.sync` and shared-memory traffic, which looks like the softmax row reductions crossing warps through shared memory. The KernelIR records one barrier per iteration and no shared-memory traffic for it.

Both are frontend counts and are reported to the frontend; the fix is to count MMAs, barriers and shared-memory operations from the compiled loop body. They match the group with the largest error.

## Parts that depend on unvalidated inputs

All 188 scored predictions carry `unvalidated_feature`: the attention KernelIR has SFU and shuffle operations, and the core does not yet use measured rates for them. In this core the SFU rate comes from the profile's `sfu_tflops` and the shuffle rate equals the issue rate; profile `0.14.0-posthoc` has measured SFU and shuffle issue rates that are not yet wired in.

- The results table and the tile table do not depend on them in practice: 185 of 188 configs are MMA bound, so the SFU and shuffle pipes do not set the prediction under the maximum.
- Candidate 1 does depend on them. Its sum adds the largest non-MMA pipe, which can be the SFU or shuffle pipe, so its numbers will move once the measured rates are in. Its conclusion should be rechecked then, and in any case it waits for its own microbenchmark.
- Candidate 2 does not depend on them: it is an instruction count from the compiled code.

With the measured curves wired in (commit "Use the measured SFU and shuffle throughput curves by warps per SM"), all 188 predictions and all 16 scores are unchanged, and `unvalidated_feature` remains only on the 28 configs whose KernelIR carries `ptxas_only_spill`. Candidate 1's diagnostic has not been rerun on that core.

Other flags: 188 configs carry `dispatch_ambiguous`, and 28 (`BM = 128`, four warps, and `m128n128-w8-s2`) carry `spill`.
