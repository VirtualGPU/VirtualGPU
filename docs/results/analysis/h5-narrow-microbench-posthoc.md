# Narrow-load microbenchmarks on GB10 (H5): results

**Post-hoc.** The candidate forms below were frozen before these points were measured, but the analysis is part of the H5 post-hoc investigation ([h5-two-byte-posthoc.md](h5-two-byte-posthoc.md)) and changes no model. The observations in the last sections were made after the data was seen; any form built on them must be validated on new points.

## Protocol

- Benchmarks: `vgpu_bench --mode gmem-narrow` (1128 points) and `--mode sync-loop-narrow` (162 points) from `microbench/src/narrow_kernels.cuh`, captured with `microbench/run_narrow.py` (thermal gate, quiet GPU and unified memory checks, telemetry), 5 repeats of 15 samples each. A point's value is the median over repeats of each repeat's median.
- Predictions: [validation/h5-candidates-frozen.json](../../../validation/h5-candidates-frozen.json), SHA-256 `4bb2ee23bc29866888a684f99ef726295e7966be0089b6fde227eefb33ff6f3f`, written by [validation/h5_candidates.py](../../../validation/h5_candidates.py) before any point was measured. Each parametrized form has one calibration point of its own (listed in the frozen file and in the previous document) and is scored on all other points.
- Scoring: [validation/h5_evaluate.py](../../../validation/h5_evaluate.py), committed before any capture existed. It checks that every captured point has the grid, occupancy, pitch and iteration count the predictions were made for. Its output is [h5-narrow-microbench-posthoc.json](h5-narrow-microbench-posthoc.json): per point the measurement, the spread over repeats and each form's error, and per form the calibrated constant, MAPE, p90 and median error by factor.
- Kernel checks in SASS (GB10, CUDA 13.0): unmasked loads carry no predicate; the always-true and tail masks compile to predicated `LDG.E.U16`, without branches or widening, and the always-true predicate is not folded; default-cache loads have no cache qualifier and `.cg` loads compile to `LDG.E.*.STRONG.GPU`; every `sync-loop-narrow` instance keeps its `ldmatrix`; no narrow kernel uses local memory or stack.

## Capture

DGX Spark GB10, 2026-11-27, under the shared GPU lock, one session per benchmark.

| Benchmark | Thermal gate (two 30 s windows) | During capture | Duration |
|---|---|---|---|
| `gmem-narrow` | 50.7 / 51.1 C, 2509 / 2515 MHz | 51 to 63 C, 2411 to 2515 MHz, no throttle reason active | 5 repeats, about 32 s each |
| `sync-loop-narrow` | 55.8 / 55.8 C, 2502 / 2509 MHz | 56 to 63 C, 2470 to 2502 MHz, no throttle reason active | 5 repeats |

About 115 GiB of memory was available before every repeat and no other process held more than 110 MB. In each repeat about 7% of the points ran more than 5% above their median, with no repeat shifted as a whole, so the median over repeats removes them; the largest spread over repeats at one point is 58%.

The raw captures are not in the repository. They are archived as `h5-narrow-captures.tar.gz` (SHA-256 `d0cb9350f364c7aa0bf7f685410e10e35a9b91de3dff2ca5a448b30416c0b9a2`), holding `gmem-narrow-1.json` (`6863e68b...`) and `sync-loop-narrow-1.json` (`6d5c93fd...`), kept on the Bifrost workstation in `~/scratch/vgpu8/` and on the Spark in `~/virtualgpu/s8-narrow/runs/`. The summary JSON records the same digests.

## Result: no candidate passes

| Form | Constant | Held-out points | MAPE | p90 | Median error |
|---|---|---:|---:|---:|---:|
| Current model | none | 1290 | 22.8% | 50.6% | -14.3% |
| Per instruction, shared `lsu` | 2.56 (grid limit) | 1289 | 23.7% | 50.7% | -16.3% |
| Per sector, shared `lsu` | 2.56 (grid limit) | 1289 | 23.7% | 50.7% | -16.3% |
| Per instruction, CTA chain | 5.12 (grid limit) | 1289 | 23.4% | 50.0% | -14.2% |
| Per sector, CTA chain | 5.12 (grid limit) | 1289 | 23.4% | 50.0% | -14.1% |
| Request-indexed MLP | 0.142 | 1289 | 21.2% | 41.2% | -5.0% |
| Sector traffic | none | 1290 | 23.2% | 50.8% | -13.7% |

None reaches the 15% MAPE the design required, so none goes into the core.

The four SM-side forms stop at the fast end of their grids because their calibration points measure faster than the current model predicts without any extra cost: 125.7 us against 130.9 us for the shared-path point and 150.0 us against 179.2 us for the chain point (2 B, aligned, `.cg`, hot L2, one line per batch). At hot L2, narrow loads leave no SM-side cost to calibrate.

## Observations (after the data was seen)

**1. From DRAM, narrow loads are slower than the model in proportion to their narrowness, and misalignment roughly doubles it.** `gmem-narrow`, current model, median error by level, width and row alignment:

| Width | DRAM aligned | DRAM offset | Hot L2 aligned | Hot L2 offset |
|---|---:|---:|---:|---:|
| 1 B | -50% | -51% to -66% | 0% | -7% to -10% |
| 2 B | -21% | -40% to -51% | +6% | -4% to -9% |
| 4 B | -12% | -28% | +4% | -15% |
| 16 B | -8% | -14% | -4% | -8% |

(Two values where the two column steps differ.) Counting every byte of each touched sector (the sector traffic form) changes no DRAM prediction, because the model holds these points latency bound, not bandwidth bound. The request-indexed MLP form scales the L2 curve with the DRAM curve, which pushes the hot-L2 points to +21%.

**2. The default cache operator and `.cg` cost the same.** Over the 560 pairs that differ only in the cache operator, default over `.cg` time is +0.00% at the median (5th to 95th percentile -4.0% to +1.6%); the current model's median error is -3.5% and -4.0% at hot L2, -26.5% and -27.0% from DRAM. Calibrating `gmem_load_issue_controlled` with `.cg` is therefore not why narrow GEMM loads are mispredicted.

**3. Staging narrow elements through registers into shared memory costs per element.** `sync-loop-narrow` at hot L2: 2 B stages are 45% to 54% faster in the model than measured, aligned or not, while 16 B stages are 9% to 22% fast. For the same stage, 2 B takes 1.1 to 3.6 times as long as 16 B; the model predicts the same time for both. The implied rate of 2 B shared memory stores grows almost linearly with CTAs per SM (4 KiB stages, no MMA: 0.17, 0.32 and 0.41 warp stores per ns per SM at 1, 2 and 3 CTAs), which points to a cost on each CTA's own chain rather than a shared throughput limit. `gmem-narrow` has no shared memory stores and shows none of it at hot L2. From DRAM, `sync-loop-narrow` at 2 B is 24% to 26% fast aligned and 36% to 38% offset, close to observation 1.

**4. `mask-mnk` has both.** Its loop is 2 B global loads, each followed by a 2 B shared memory store, with data from L2 and DRAM, which fits the earlier finding that neither a slower `lsu` nor slower MLP curves can fit it alone.

## Next steps

Both follow from observations made on these points, so these points are no longer held out for them.

- **DRAM-side narrow request cost.** Fix the form and one calibration point first, then validate on new GB10 points (in-flight depths, CTAs per SM and working sets not measured here, listed and predicted before measurement), with the hot-L2 predictions required to stay unchanged; then on the RTX 4070 as a cross-GPU holdout.
- **Per-element register staging.** A new benchmark that separates shared memory store width from global load width (stores only, loads only, both; 2, 4 and 16 B; 1 to 3 CTAs per SM; with and without MMA), measured first to place the cost on the shared memory pipe or the CTA's chain. Candidate forms are proposed and frozen only after that, and validated on another set of points.
