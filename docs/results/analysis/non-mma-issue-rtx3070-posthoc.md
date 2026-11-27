# Post-hoc: measured non-MMA issue rate on the RTX 3070

**Post-hoc.** Evidence for measuring `non_mma_issue_rate_per_sm_gips` instead of estimating it at 4 warp instructions per clock per SM. It uses the RTX 3070, which is not a reference GPU. No bundled profile changes; the field changes when a profile is re-collected.

## What the field means

The core charges every non-MMA instruction against this rate, so it is a ceiling. `collect_profile.py` fills it with the highest non-MMA rate measured on the card: the pure FFMA rate from `mma` (`fp32_fma_tflops` over 64 FLOP per warp instruction) or the peak over warps of the 1:1 FFMA and LOP3 mix from `sfu-shfl`, whichever is higher. That is a lower bound on the real ceiling, never below a rate the card was seen to reach. The mix curve stays in the raw output.

## Data and method

- Capture: [non-mma-issue-rtx3070-posthoc.json](non-mma-issue-rtx3070-posthoc.json), five independent processes each of `vgpu_bench --mode sfu-shfl --samples 15 --warmup 3` and `vgpu_bench --mode mma --samples 15 --warmup 3`, each started after desktop GPU use fell below 5%. RTX 3070 (`sm_86`), driver 610.88, CUDA 13.2, WSL2.
- Not run through `collect_profile.py`, so there is no section 6 gate report. Clocks were not locked: the SM clock was 1770 to 1965 MHz during `sfu-shfl` and about 1770 MHz during `mma` (sampled every 100 ms).
- Mix kernel: `alu_issue_kernel`, four independent `fma.rn.f32` chains and four independent `lop3.b32` chains per thread, one CTA per SM, 1 to 32 warps. The rate counts 128 instructions per iteration and leaves out the loop counter and branch, so it reads about 2% low.
- Every mix launch is 1.37 to 2.72 ms. The other `sfu-shfl` curves and `mma` used their legacy lengths.

## Capture rules

Checked with `dispersion_check` and `reproducibility_check` from `microbench/capture_posthoc.py`, with suffix `_samples` since the rate lists do not end in `_ns_samples` or `gbps_samples`.

| Scope | Rule 1 (IQR) | Rule 2 (spike) | Rule 3 (5 processes) |
|---|---|---|---|
| Mix curve, 6 lists | 0 failing in every run | 0 failing in every run | 0 of 6 points |
| Whole `sfu-shfl`, 31 lists | at most 1 failing (1 allowed) | 0 failing | 1 of 31 points (1 allowed) |
| `fp32_fma`, 1 list | 0 failing in every run | 0 failing in every run | 0 of 1 point |

## Result

| Warps per SM | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Mix, Gwarp-inst/s per SM, median of runs | 1.474 | 2.947 | 5.894 | 6.118 | 6.218 | 6.173 |

Pure FFMA reached 6.31 to 6.33 Gwarp-inst/s per SM (18.57 to 18.63 TFLOP/s), about 3.6 per clock at 1770 MHz. The mix peaks at 6.18 to 6.23, about 3.2 per clock at 1950 MHz. Pure FFMA is higher, so the field is 6.3291. The bundled estimate is 7.98 (4 per clock at its recorded clock).

## Prediction diff

- Base: `rtx3070.json` as bundled, sha256 `1878fbfd4f74bdc69e542ac3196aafd80189977ad8a57e3d73cc8195ce2019d7`. Candidate: the same file with only this field regenerated from run 1 of each mode (`value` 6.3291, `rel_uncertainty` 0.1022, `method` `cuda_fp32_fma_unrolled`), sha256 `a27894f392e316c06379b6c15938fc41d39ad8e0a6377b5f68a43a2ba5ab8b44`.
- Revisions: base `42e039d`, candidate `3fa5752`. Core, package and schemas are identical between them (`git diff 42e039d 3fa5752 -- crates python/virtualgpu schemas` is empty), so both sides ran in one build.
- IRs: the archived v0.2 development set for `sm_89` and `sm_121`, 26 cases and 747 configs per architecture, 1494 files, digest `09cec6b8981846671d1b85f2c81b3d0fffef87592af7b2e2a76a62a5a9775f91` (sha256 of the sorted `sha256sum` list). They are not `sm_86`, so this shows how sensitive predictions are to the field, not an RTX 3070 prediction.
- Warm and cold, 32 draws, seed 17. Raw predictions for both sides, with both profiles and the IR digest: [non-mma-issue-rtx3070-predictions.tar.xz](non-mma-issue-rtx3070-predictions.tar.xz).

"Draws only" rows keep their nominal prediction and differ only in the perturbed draws, because `rel_uncertainty` goes from 0.30 (estimate) to 0.1022. Medians are over the rows whose nominal prediction changed.

| IR arch | Cache | Family | Predictions | Bit-identical | Draws only | Nominal changed | Median of nominal changes | Max | Status changes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sm_89 | warm | gemm | 291 | 228 | 63 | 0 | n/a | n/a | 0 |
| sm_89 | warm | attention | 456 | 390 | 30 | 36 | +3.25% | +7.91% | 0 |
| sm_89 | cold | gemm | 291 | 228 | 63 | 0 | n/a | n/a | 0 |
| sm_89 | cold | attention | 456 | 390 | 30 | 36 | +3.21% | +7.71% | 0 |
| sm_121 | warm | gemm | 291 | 227 | 64 | 0 | n/a | n/a | 0 |
| sm_121 | warm | attention | 456 | 420 | 12 | 24 | +1.31% | +4.87% | 0 |
| sm_121 | cold | gemm | 291 | 227 | 64 | 0 | n/a | n/a | 0 |
| sm_121 | cold | attention | 456 | 420 | 12 | 24 | +1.30% | +5.26% | 0 |

No GEMM nominal prediction moves. Attention, where softmax puts ALU work on the critical path, gets up to 8% slower.

## Validation

The field is meant to be validated, but not on its own and not here. It enters the model when the reference profiles (RTX 4070, GB10) are re-collected for 0.3.0, and the sealed v0.3 GEMM and attention holdouts judge the model with it. Before that capture, a microbenchmark check on the same card would reject it: if pure FFMA or the mix, run as a kernel through the core, is predicted more than 10% off its measured rate. Effects on the already opened v0.2 data above are post-hoc and validate nothing.

## Limits

- One GPU that is not a reference GPU, without locked clocks, and the two modes ran at different clocks.
- A lower bound: a kernel with another instruction mix can issue faster than both measured rates, or slower.
