# H1 natural offsets: registered test on GB10

Registered before any `sync-loop-free` measurement, after review by 6 and the lead. It asks whether the offsets that a group of CTAs reading the same cold stage settle into can be a profile field, so the model could charge the merged wait as `E[max(0, X - d)]` without per-iteration events. `X = Ld - L2l`. Nothing below changes after the data are taken. The evaluator is `microbench/sync_loop_free_eval.py`, committed with this file; its output is the verdict.

## Measurement

- `vgpu_bench` for `sm_121` from the commit that adds this file, mode `sync-loop-free`, `--samples 15 --warmup 3`, three independent processes, nothing dropped except by the rules below. Temperature, SM clock and throttle reasons logged every second with `nvidia-smi`.
- Free-running single-buffer loop of 1024 steps, no schedule and no step barrier. Groups of `G` CTAs read the same cold stage each step (cold region advanced across launches on a 256 MiB ring); three groups at every `G` (12 CTAs at `G` = 4, 48 at `G` = 16), one CTA per SM.
- 24 configs: stage `SB` 16 and 32 KiB, warps `W` 4 and 8, MMA groups per step `MI` 2, 8 and 32, `G` 4 and 16. Each config runs three ways: warm L2 without stamps, cold without stamps, cold with a `%globaltimer` stamp before each step's copies.

## Quantities

- Judged: the group-mean stage latency excess `E`, cold without stamps minus warm without stamps, per config (median over processes). It does not depend on the stamps.
- From the stamped run: each follower's offset from the first issuer of the step, pooled over steps, groups and samples; its median `m` per config.
- Distorted stamps: if the stamped step time differs from the unstamped one by more than 5%, or the spread of finish times within a group by more than 25%, that config's offsets are not used.
- Saturated: if `3 SB / dram_bw_read_gbps` is at least half the unstamped step time, the config is reported but not judged.

## Forms

`X` from `profiles/gb10.json`: `dram_latency_ns - l2_latency_ns` (264.6 ns). Prediction `E = [X + (G - 1) E_d[max(0, X - d)]] / G`; the current model is `E = X`.

- Shapes of the follower offset, each with one parameter set from the median `m`: S1 uniform on `[0, 2m]`; S2 exponential with mean `m / ln 2`; S3 half-normal with scale `m / 0.6745`.
- Rules for the parameter across configs: R0, the calibration point's `m` everywhere (a property of the GPU); R1, `m` proportional to the step time, the ratio set at the calibration point (a property of the kernel).

## Calibration and rule

- One calibration config: `SB` 16 KiB, `W` 8, `MI` 8, `G` 16. If its stamps are distorted, the next of `W` 4 (same otherwise), then `MI` 32 with `W` 8; if all three are distorted, nothing is calibrated and only the distributions are reported. Every other unsaturated config is held out.
- A combination (rule, shape) passes if its median absolute error on the held-out configs is at most 40 ns and it beats the current model: median error below half of the model's and closer on at least two thirds of the configs.
- Among passing combinations, those not beaten by another passing one by the same pairwise test are equally best; the one used is the first in the order R0 before R1, then S1, S2, S3 within a rule.
- The first question (GPU or kernel property) is answered by whether some R1 combination beats every R0 combination, or the reverse. If none passes, the result is a known limitation, and no new form is fitted.
- The RTX 4070 is the cross-GPU held-out test, registered separately before it is measured.
