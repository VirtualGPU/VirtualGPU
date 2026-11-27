# H7 ldmatrix byte correction: registered as a unit correction

Registered with the frontend change (`fix: count the ldmatrix bytes of MMA operands per warp in the KernelIR`), before it is merged and before the v0.3 holdout is opened. Nothing below changes after that.

## What it corrects

The frontends counted the shared memory reads of a loop as the TTGIR operand tile sizes, each tile once per CTA, plus the PTX `ld.shared` bytes. The MMA operands are read with `ldmatrix`, which the PTX pattern did not match, and every warp reads its own operands, so the reads were undercounted. The PTX loop's `ldmatrix` bytes (128, 256 or 512 per `.x1`, `.x2`, `.x4` warp instruction, times the warps) now replace the tile sizes when the loop has any. This is a counting error, fixed whatever it does to the scores; it is not a new mechanism and has no parameter.

The profile's `smem_bw_per_sm_gbps` counts the bytes every warp actually reads (`smem_bw_kernel`, 16 warps of `ld.shared.v4`), the unit the corrected bytes use. That `ldmatrix` streams at the `ld.shared.v4` rate is assumed, not measured; a dedicated measurement is added only if the v0.3 holdout calls for one.

## Expected effect, stated before the v0.3 holdout

From the post-hoc v0.2 counterfactual ([h7-ldmatrix-bytes](h7-ldmatrix-bytes/v02-counterfactual.txt)), with no other change:

- Every cohort's predictions get slower. GEMM moves toward zero signed error; attention moves through it toward over-prediction, most at `num_stages = 2`.
- Attention ranking gets worse (lower mean tau, higher top-5 regret on GB10), because the over-predicted `num_stages = 2` round trip is no longer offset by the undercount. This is expected and does not count against the correction.

## Judged quantity on the v0.3 holdout

For each GPU and family, and within attention for each `num_stages`, the absolute signed median error with the correction against without it (both with the core that is current when the holdout is opened). The correction is confirmed where the absolute signed median error does not grow, except for attention `num_stages = 2`, where it is expected to grow for the reason above and is reported only. Ranking metrics are reported, not judged.
