# Post-hoc: measured non-MMA issue rate on the RTX 3070

**Post-hoc.** Evidence for measuring `non_mma_issue_rate_per_sm_gips` instead of estimating it at 4 warp instructions per clock per SM. It uses the RTX 3070, which is not a reference GPU. No bundled profile changes; the field changes when a profile is re-collected.

## Data and method

- Capture: [non-mma-issue-rtx3070-posthoc.json](non-mma-issue-rtx3070-posthoc.json), five independent processes of `vgpu_bench --mode sfu-shfl --samples 15 --warmup 3`, each started after desktop GPU use fell below 5%. RTX 3070 (`sm_86`), driver 610.88, CUDA 13.2, WSL2, clocks not locked (SM clock 1770 to 1965 MHz during the runs).
- Kernel: `alu_issue_kernel`, four independent `fma.rn.f32` chains and four independent `lop3.b32` chains per thread, 1:1, one CTA per SM, 1 to 32 warps. `collect_profile.py` takes the samples of the warp count with the highest median.
- Every ALU launch is 1.37 to 2.72 ms. The other `sfu-shfl` curves still use their legacy lengths.

## Capture rules

Checked with `dispersion_check` and `reproducibility_check` from `microbench/capture_posthoc.py`, with suffix `_samples` since the rate lists do not end in `_ns_samples` or `gbps_samples`.

| Scope | Rule 1 (IQR) | Rule 2 (spike) | Rule 3 (5 processes) |
|---|---|---|---|
| ALU curve, 6 lists | 0 failing in every run | 0 failing in every run | 0 of 6 points |
| Whole mode, 31 lists | at most 1 failing (1 allowed) | 0 failing | 1 of 31 points (1 allowed) |

## Result

| Warps per SM | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Gwarp-inst/s per SM, median of runs | 1.474 | 2.947 | 5.894 | 6.118 | 6.218 | 6.173 |

The peak, 6.18 to 6.23 across processes, is about 3.2 warp instructions per clock at 1.95 GHz. The estimate in the bundled profile is 7.98 (4 per clock at its recorded clock), so the measured value is 23% lower.

## Prediction diff

`rtx3070.json` as bundled against a copy where only this field is regenerated from run 1 (`value` 6.1826, `rel_uncertainty` 0.1022). Core and IRs fixed. The IRs are the archived v0.2 development set for `sm_89` and `sm_121` (26 cases, 747 configs each), so this measures sensitivity to the field, not an RTX 3070 prediction. 32 draws, seed 17.

| IR arch | Cache | Family | Predictions | Bit-identical | Changed | Median change | Max change | Status changes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sm_89 | warm | gemm | 291 | 227 | 64 | +0.00% | +0.00% | 0 |
| sm_89 | warm | attention | 456 | 375 | 81 | +0.00% | +8.15% | 0 |
| sm_89 | cold | gemm | 291 | 227 | 64 | +0.00% | +0.00% | 0 |
| sm_89 | cold | attention | 456 | 375 | 81 | +0.00% | +8.13% | 0 |
| sm_121 | warm | gemm | 291 | 226 | 65 | +0.00% | +0.00% | 0 |
| sm_121 | warm | attention | 456 | 414 | 42 | +1.14% | +5.30% | 0 |
| sm_121 | cold | gemm | 291 | 226 | 65 | +0.00% | +0.00% | 0 |
| sm_121 | cold | attention | 456 | 414 | 42 | +0.91% | +5.63% | 0 |

GEMM changes stay under 0.005%, mostly in draws. Attention, where softmax puts ALU work on the critical path, gets up to 8% slower.

## Limits

- One GPU that is not a reference GPU, without locked clocks. The 1:1 mix is one choice; a kernel with a different FP32 to integer ratio can issue faster or slower.
- No validation: the change is not checked against kernel timings here.
