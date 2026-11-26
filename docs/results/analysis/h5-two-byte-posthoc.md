# Post-hoc analysis: 2-byte register-staged loads on GB10 (H5)

**Post-hoc.** Done after the v0.2 holdout labels were opened and scored. It cannot change the v0.2 decisions and does not choose a model form: it only says what the GB10 microbenchmark for H5 ([v0.2-posthoc.md](v0.2-posthoc.md), item 8) has to measure. Data: [h5-two-byte-posthoc.json](h5-two-byte-posthoc.json), written by [validation/h5_counterfactual.py](../../../validation/h5_counterfactual.py).

## Method

- Cases: `v02-gb10-mask-mnk-1153` (M, N, K = 1153, 9727, 6653), and `v02-gb10-mask-mnk-641` (641, 5123, 3581) for comparison. Measured medians from [validation/v02-holdout-scores.json](../../../validation/v02-holdout-scores.json).
- Core: v0.2-final rebuilt from `git archive v0.2-final`, run on the archived KernelIR of the formal prediction archive (`74caeeeb...`) and the v0.2-final GB10 profile (`0.21.0-posthoc`), nominal profile only. The script first checks that it reproduces every archived nominal prediction exactly (144 of 144 for the two cases and both cache regimes).
- Each counterfactual changes one IR or profile field for all 36 configs of a case. Kendall tau-b and top-5 regret are computed as in the scores.
- Load layouts come from the offline TTGIR and PTX of the same compiled artifacts (matched by the IR's `ptx_sha256`).

## Confirmed

1. **What the kernel does.** All 36 configs load A and B with `transfer_width_bytes = 2`, no asynchronous copy and prefetch distance 0; the loop has two global loads, a shared memory store and load, `mma.sync` and a barrier, and no ALU ops in the IR. The TTGIR layout of every load is `sizePerThread = [1, 1]`, `threadsPerWarp = [1, 32]`: a warp load is 32 consecutive f16 of one row, 64 B, so the loads are coalesced. K and N are odd, so every row starts only 2-byte aligned and a 64 B warp request usually spans three 32 B sectors. The PTX is `@p ld.global.b16` without a cache qualifier, so the loads go through L1. Each thread issues 10 to 128 such loads per iteration.
2. **The errors are larger than v0.2-posthoc item 8 says.** On `mask-mnk-1153` every config is predicted fast by 26.8% to 81.6% warm (26.6% to 81.9% cold); the worst is `m16n64k32-w4` (-81.5%), then `m16n64k32-w8` (-74%). On `mask-mnk-641` the range is 34.6% to 85.1% warm, `m16n64k32-w4` again the worst; that case has zero regret only because the order at the top survives.
3. **The model puts the time in DRAM latency, and the load width does not enter it.** On `mask-mnk-1153` the bounding cause of every non-spilling config is `dram_latency` (80% to 97% of the time); `lsu` gets about 1e-4. The width only reaches the model through the `lsu` pipe, which bounds the shared compute path only: on the round-trip path, which decides a synchronous loop, memory issue is set to zero, and the latency comes from the `ld_global_v4` MLP curves, measured with 16 B loads. Setting `transfer_width_bytes` to 16 changes no tau or regret and no error by more than 1 point.
4. **The `lsu` rate is not an issue ceiling for these loads.** It is the peak of `gmem_load_issue_controlled` at 2 B (0.859 Gwarp-inst/ns per SM): one SM, a 4 MB hot L2 buffer, `ld.global.cg`, aligned. Its rate rises with bytes in flight at every batch depth, so what it measured is throughput under latency, not an issue limit. Over the full GPU, loads alone at that rate take 1/5 to 1/12 of the measured times.
5. **The measured rates per instruction vary too much for one constant.** Load warp instructions over measured time and 48 SMs, non-spilling configs: 0.071 to 0.162 Gwarp-inst/ns per SM on `mask-mnk-1153`, 0.091 to 0.317 on `mask-mnk-641`. The same config is faster per instruction on the smaller case (its B is 36 MB, against 129 MB, with a 24 MB L2), so the cost depends on where the data comes from, not only on issue.
6. **A slower `lsu` alone does not fit.** Scaling the 2 B issue rates in the profile:

   | `lsu` rate | 1153 tau-b / regret (warm) | 1153 error min / median / max | 641 tau-b / regret (warm) |
   |---|---|---|---|
   | x1 (v0.2) | 0.588 / 3.57% | -81.6% / -63.1% / -26.8% | 0.811 / 0% |
   | x0.25 | 0.588 / 0% | -72.3% / -38.4% / -12.0% | 0.425 / 30.5% |
   | x0.15 | 0.471 / 0% | -72.2% / -5.1% / +44.2% | 0.307 / 30.5% |
   | x0.1 | 0.441 / 0.60% | -61.1% / +41.5% / +114.5% | 0.248 / 30.5% |

   A factor that removes the `mask-mnk-1153` regret wrecks `mask-mnk-641`, and even on `mask-mnk-1153` the `m16n64k32` configs stay 40% fast while `m64n128k32-w4` goes 41% slow.
7. **Slower `ld_global_v4` MLP curves alone do not fit either.** Scaling their bandwidth by 0.5, 0.33 and 0.25 gives `mask-mnk-1153` tau-b 0.353 to 0.383 (below the unchanged 0.588) and errors from -72% to +195%. The failures are the reverse of item 6: the one-CTA-per-SM `m128n128k64-w8` goes +193% while `m16n64k32` stays 17% to 36% fast.
8. **The `m16n64k32` configs carry something else.** On `mask-mnk-1153`, `-w4` (9 CTAs per SM) and `-w8` (6) measure 54.3 and 53.3 ms, and requested load bytes over measured time are 218 and 222 GB/s, 85% of the profile's DRAM read peak (258 GB/s). The model's L2 model fetches only 2.6 and 3.5 GB of their 11.8 GB from DRAM. One possible reading is that the misaligned 2 B loads lose the L2 reuse (in-flight merging) that the model credits. Shrinking the L2 capacity in the profile to 5% changes no prediction, since the reuse comes from merging, so the CPU cannot decide this.

## Hypotheses for the microbenchmark

- **H5a.** The cost of a narrow load is per request or per sector, and it sits on the round-trip path, not only on the shared path.
- **H5b.** Misaligned narrow loads from DRAM lose in-flight merging or fetch more sectors than they use.
- **H5c.** The default cache operator (L1) and `.cg` cost differently for narrow loads; the profile was measured with `.cg` only.

The microbenchmarks (`gmem-narrow` and `sync-loop-narrow` in `microbench/src/narrow_kernels.cuh`) measure 1, 2, 4 and 16 B loads, masked and unmasked, aligned and one element off, default and `.cg`, from hot L2 and from DRAM, at 1 and 2 CTAs per SM, plus a sharing variant for H5b. Which model form follows is decided on those points alone, from one calibration point with the rest held out; `mask-mnk-1153` and `mask-mnk-641` stay post-hoc comparisons.

## Frozen candidate predictions

Before any point is measured, [validation/h5_candidates.py](../../../validation/h5_candidates.py) turns all 1076 benchmark points (968 `gmem-narrow`, 108 `sync-loop-narrow`) into KernelIR with the benchmark's geometry and predicts them with the current core and GB10 profile (`0.22.0-posthoc`) under seven forms: the current model, per instruction and per sector on the shared `lsu` path, per instruction and per sector on the CTA's dependent chain, a request-indexed `ld_global_v4` MLP curve, and sector-inflated traffic. The core requires an MMA, so points without one carry a single token MMA per CTA and iteration. Each parametrized form takes its one constant from the calibration point (2 B, aligned, unmasked, default cache, DRAM, 1 CTA per SM, 8 warps, 16 loads in flight); every other point is held out. The predictions are frozen on a grid of each constant in [validation/h5-candidates-frozen.json](../../../validation/h5-candidates-frozen.json) (SHA-256 `ae49b684ae8ca9e9ca5b60aa7c12fb3ef9dd2c3f52da0a5ad0b1317e06ac7f70`), and `evaluate` interpolates them at the calibrated constant. A form that misses the held-out points by more than 15% MAPE does not go into the core.
