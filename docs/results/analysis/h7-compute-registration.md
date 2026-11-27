# H7 (b) the compute base of attention and GEMM loops: registered test

Registered before any measurement of `microbench/attention_compute.py`, after review by 6 and the lead. Nothing below changes after the data are taken. The evaluator is `microbench/attention_compute_eval.py`, committed with this file; its output is the verdict.

## Question

On GB10 the attention ablation's `noload` loop (no memory traffic) runs 19% to 29% slower than the model's single-CTA chain, MMA plus the longest other pipe plus synchronization, and MMA-bound v0.2 holdout configs are predicted fast (GB10 attention median -9%, RTX 4070 GEMM -5.6%). How do the softmax ALU, SFU and shuffle work and the ldmatrix reads add to the MMA time in a real Triton instruction stream?

## Kernels

- Attention: the `noload` variant of `microbench/attention_ablation.py` (K and V resident, the QK product kept in the loop) with three compile-time knobs on the softmax's dependent chain: `EX2` more `exp2(w 0.5 - 0.5)` layers on the weights, `ALU` more `where(x > 1e30, 0.5 x, x) + 1e-4` layers on the logits, `RED` more row reductions with distinct scales. Shapes `m128n32-w8`, `m128n64-w8` and `m64n64-w4` (the 4-warp shape uses 64 query rows so nothing spills; `m64n32-w4` would run two CTAs per SM and is left out); knobs `(EX2, ALU, RED)` = (0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 2, 0), (0, 4, 0), (0, 0, 2), (0, 0, 4); 21 configs, all one CTA per SM and no spill on `sm_121`.
- GEMM: `resident_gemm`, where every program multiplies the same two small L2-resident tiles 8192 times (at least 1 ms per launch) with `num_stages = 3`, so the loop is copies, ldmatrix, MMA and address arithmetic; `128x128x32`, `128x64x32` and `64x64x32` with 4 and 8 warps; grid one program per SM. No spill.
- The unchanged ablation variants compile to the same resources as before the knobs were added.
- Timing as in the ablation's `run()` (CUDA graph, warm), on the development attention shape; `--samples 15`, three independent processes, nothing dropped; temperature, clock and throttle logged every second; the acquisition rules of 2026-11-27 are checked on `kernel_ns_samples` before anything is judged (lists with IQR over median above 10% at most 5%, lists with max over min above 1.5 at most 15%, points whose process medians span more than 10% at most 5%); a failing run is archived as invalid and repeated.

## SASS check (passed before registering)

Cubins from Triton's bundled `ptxas` for `sm_121a`, disassembled with `cuobjdump`. Per warp in the loop:

| Loop | HMMA | LDSM | MUFU | SHFL | BAR |
|---|---|---|---|---|---|
| real `fused_attention` `m128n32-w8-s2` | 64 | 40 | 18 | 8 | 2 |
| knobs (0, 0, 0) `m128n32-w8` | 64 | 40 | 18 | 8 | 1 |
| real `m128n64-w8-s2` | 128 | 72 | 34 | 8 | 2 |
| knobs (0, 0, 0) `m128n64-w8` | 128 | 72 | 34 | 8 | 1 |
| real `m64n64-w4-s2` | 128 | 64 | 40 | 64 | 10 |
| knobs (0, 0, 0) `m64n64-w4` | 128 | 64 | 40 | 64 | 10 |

The base loops differ from the real kernel only by the copies, their address arithmetic and one barrier. The knobs add work linearly: `EX2` 2 and 4 add 32 and 64 MUFU (8 warps, 32 query columns); `ALU` adds about 47 other instructions per layer; `RED` adds 4 SHFL per reduction with 8 warps, and 32 SHFL and 4 barriers with 4 warps, whose reductions cross warps through shared memory. A first `RED` form was deduplicated by the compiler and was replaced before registering.

## Forms (no free parameters)

Counts are per warp per loop iteration from the SASS (a body unrolled by the compiler is scaled by its HMMA count against the tile's MMAs per iteration); rates are those the model uses, from `profiles/gb10.json`: `t_mma = HMMA W 4096 / mma_tflops_per_sm_by_warps(W)`, `t_alu = OTHER W / non_mma_issue_rate`, `t_sfu = MUFU W / sfu_issue_rate(W)`, `t_shfl = SHFL W / shfl_issue_rate(W)`, `t_smem = LDSM bytes W / smem_bw` (128, 256 or 512 bytes per `.x1`, `.x2`, `.x4` warp instruction, written `LDSM.16.M88`, `LDSM.16.M88.2` and `LDSM.16.M88.4` in SASS, so per-CTA bytes in the unit of the model's `smem_bytes`; note that every warp's ldmatrix reads the whole K and V tile, 294912 B per CTA per iteration for `m128n64-w8`, while the KernelIR counts the tiles once, 65536 B), `sync = fixed_sync_ns` at `W` warps and one buffer.

- F-max, the current core: `t_mma + max((OTHER + MUFU + SHFL) W / issue_rate, t_smem, t_sfu, t_shfl) + sync`.
- F-sum: `t_mma + t_alu + t_sfu + t_shfl + t_smem + sync`.
- F-dep, from the dependences: the softmax is one chain from the QK accumulators through the row maximum, `exp2`, the sum and the conversion to the PV operand, so its ALU, SFU and shuffle work is serial; the K and V ldmatrix reads do not depend on it: `t_mma + max(t_smem, t_alu + t_sfu + t_shfl) + sync`.
- F-issue: as F-max with the HMMA and LDSM instructions also in the issue term.

## Rule

- Measured: time per loop iteration per CTA, the median over processes of each process's median kernel time, over `ceil(CTAs / SMs)` waves and the loop's iterations.
- Attention and GEMM are judged separately, on absolute relative error. A form wins if, against each other form on the points where their predictions differ by more than 2%, its median error is below half of the other's and it is closer on at least two thirds of them. If every form's median error exceeds 15%, all are reported wrong. A result between F-max and F-sum that passes neither is reported as such; no overlap factor is added.
- Reported, not judged: the increments `T(knobs) - T(base)` per shape, measured and predicted; the development attention kernel itself.
- The RTX 4070 runs the same, in a quiet window, through `capture_posthoc.py`, registered with its own profile and no other change.
