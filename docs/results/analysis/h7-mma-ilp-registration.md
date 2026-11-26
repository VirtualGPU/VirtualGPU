# H7 (a) dependent MMA chains: registered confirmation test

Registered before any `mma-ilp` measurement, after review by 6 and the lead. Nothing below changes after the data are taken. The evaluator is `microbench/mma_ilp_eval.py`, committed with this file; its output is the verdict. This test is not on the v0.3 fix list: it confirms that the current MMA rate holds where real kernels run, and explains the `mma > 0` deviations of 8's register staging sweep.

## Background

The core times MMA work with `mma_tflops_per_sm_by_warps` only, measured with four independent accumulators per warp. The profile also carries `serial_stage_overhead_ns` and `mma_chain_latency`, which the core does not use: on GB10, dependent MMAs take 11.46 ns each at 4 warps against 6.72 ns independent, and the same at 8 warps. Every v0.2 holdout GEMM and attention config has at least 16 independent accumulator chains per SM.

## Measurement

- `vgpu_bench --mode mma-ilp`, from the commit that adds this file; GB10 first, the RTX 4070 in a quiet window. `--samples 15 --warmup 3`, three independent processes, nothing dropped; on GB10 temperature, clock and throttle are logged every second, on the RTX 4070 the run goes through `capture_posthoc.py` with the desktop GPU load and per-point dispersion checks.
- Each warp keeps `ILP` accumulators (`m16n8k16`, f16 to f32) and runs `N` dependent MMAs on each per step, rotating over them, with no memory traffic. `W` warps per CTA; `cps` CTAs per SM, set by dynamic shared memory and asserted with the occupancy API; grid `cps` times the SM count. Each config runs `N` = 16 and 64 with the same step count (the `N` = 64 launch about 2 ms); the measured quantity is the slope, ns per MMA per warp.
- Grid: `ILP` 1, 2, 4, 8; `W` 1, 2, 3, 4, 8; `cps` 1, 2, 3; `W cps` at most 16 (56 configs). Not held out, as the profile has measured them: `ILP` 1 at `W` 4 and 8, and `ILP` 4 at `W` 1, 2, 4 and 8, all with `cps` 1.

## Fields and forms

- `t_tp(W_SM) = 4096 W_SM / mma_tflops_per_sm_by_warps(W_SM)`, linear in warps per SM, `W_SM = W cps`.
- `L_dep` = `serial_stage_overhead_ns`, variant `mma_only_dependent`, 4 warps, `slope_ns_per_mma` (GB10 11.46 ns, RTX 4070 11.72 ns).
- M0, the current core: `t_tp(W_SM)`. M1: `max(t_tp(W_SM), L_dep / ILP)`. No free parameters.

## Rule

- GB10, held-out points where the forms differ by more than 2% (8, all with `ILP` 1 and `W_SM` at most 6): M1 wins if its median `|pred / meas - 1|` is below half of M0's and it is closer on at least two thirds of them. If both exceed 25%, both are reported wrong.
- GB10, the other held-out points (where the forms agree, the region of real kernels): M0 holds there if its median `|pred / meas - 1|` is at most 10%.
- RTX 4070: by its profile M1 and M0 agree within 5% everywhere (largest ratio 1.041), so the same evaluation should find M0 within 10% and no dependent penalty above 10% at `ILP` 1. A penalty there would mean the profile's RTX 4070 dependent slope is wrong, and is reported as such.
- Reported, not judged: whether the penalty at `ILP` 1 depends only on the chains per SM, `ILP W_SM`.
