# H7 F-dep in the core: registered change and its tests

Registered before the RTX 4070 measurement it depends on. Nothing below changes after that data is taken. It pairs with the frontend correction of the ldmatrix bytes (its own registration): the GB10 evidence for F-dep was obtained with SASS shared memory bytes, so the two enter together or not at all.

## Change

In `rates()` (`crates/vgpu-core/src/sim.rs`) the single-CTA dependent chain is today `alone[mma] + max(alone[issue], alone[smem], alone[local], alone[sfu], alone[shfl]) + sync`: the non-MMA work fully overlaps, the longest pipe stands for all of it. F-dep makes the softmax chain serial and lets the shared memory reads run beside it:

`chain = alone[mma] + max(alone[smem], alone[local], t_alu + alone[sfu] + alone[shfl]) + sync`, with `t_alu = (warp_insts - sfu_insts - shfl_insts) / issue_rate`, the issue time of the instructions that are neither SFU nor shuffle.

The shared path (`pipes` plus synchronization) and everything else are unchanged. No new profile field and no free parameter.

## Evidence so far (GB10, post-hoc)

The compute-only attention sweep ([h7-compute-registration.md](h7-compute-registration.md), [compute-base-gb10-posthoc.json](compute-base-gb10-posthoc.json)): F-dep won with a median error of 4.5% against 8.6% for the current rule, on 21 points with SASS instruction counts and bytes. That sweep is not evidence for this registration's held-out tests.

## Tests

1. RTX 4070, cross-GPU held out: the same `attention_compute` sweep, binary and evaluator, run in a quiet desktop window through `capture_posthoc.py` with its gates, with the RTX 4070 profile and its own SASS (`sm_89`). F-dep must win against F-max by the rule of the GB10 registration (on the points where they differ by more than 2%: median error below half and closer on at least two thirds). If it does not, the change is not made.
2. v0.3 holdout, after both the frontend correction and F-dep are in: reported per GPU and family with the frontend correction alone and with both, as signed median error by `num_stages`, mean top-5 regret and mean tau. Expected, and stated here before measuring: attention moves toward over-prediction once the ldmatrix bytes are corrected, because the over-predicted `num_stages = 2` round trip is no longer offset; F-dep adds to the chain and moves it further. The judged quantity for the pair is the absolute signed median error by `num_stages`, not the ranking.
3. GEMM: in loops with no SFU or shuffle work today's issue term is `t_alu`, so F-dep gives the same chain as today; it changes only loops with softmax-like work. The v0.2 counterfactual of the pair is still reported for GEMM, where any change comes from the frontend correction.
