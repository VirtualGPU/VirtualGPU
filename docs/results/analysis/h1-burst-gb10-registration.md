# Single-buffer bursts against the L2 bandwidth: registered test on GB10

Registered before any `sync-loop-burst` measurement, after review by 6 and the lead. Nothing below changes after the data are taken. The evaluator is `microbench/sync_loop_burst_eval.py`, committed with this file; its output is the verdict.

## Question

The model (`rates()` in `crates/vgpu-core/src/sim.rs`) lets the L2 and DRAM shares bound only the average rate of a phase, for synchronous loads too, and takes the bytes in flight from the time-averaged MVA. When identical CTAs run a single-buffer loop in step, each step puts every CTA's whole stage in flight at once, so the stage may wait for the GPU's L2 bandwidth: `E = c0 + max(L2l + SB / P1, n SB / BW(n))`. That form was read off the 48-CTA points of `sync-loop-bytes` ([sync-loop-bytes-gb10-posthoc.json](sync-loop-bytes-gb10-posthoc.json)), so those configurations are not evidence for it.

## Measurement

- `vgpu_bench` for `sm_121` from the commit that adds this file, mode `sync-loop-burst`, `--samples 15 --warmup 3`, three independent processes, nothing dropped except by the rules below. Temperature, SM clock and throttle reasons logged every second with `nvidia-smi`.
- The `gemm_s2` shaped single-buffer loop on a warm 4 MiB L2 region, 8 warps, 8 MMA groups per step, one CTA per SM, at least 1024 steps. `free` and `noload` variants; the exposed time per step is `E = turn(free) - turn(noload)`, with `turn` the mean over CTAs of each CTA's cycles per step after its start offset, converted with block 0's measured SM clock.
- Two starts: `lock`, every CTA at once; `stag`, CTA `c` held back `c T / n` on `%globaltimer`, with `T` the step of the same `lock` configuration, so the starts are spread evenly over one step.
- Grid: `n` 12, 36, 40, 44 and 48 CTAs; stage `SB` 16, 32, 48 and 64 KiB; both starts.
- Phase coherence: each CTA stamps the start of steps 0, 16, 64, 256, `S / 2` and `S - 1`; the coherence at a mark is the mean resultant length of the CTAs' phases (time modulo the step), 1 in step and near 0 when spread evenly.

## Fields

From `profiles/gb10.json`: `L2l` = `measured.memory.l2_latency_ns`; `P1` = `l2_bw_by_active_sms` at 1 SM; `BW(n)` = `l2_bw_by_active_sms` interpolated linearly at `n`, as the model's `interpolate` does. This is the `ld.global.cg` curve, used as a proxy for the `cp.async` path because it is the only per-SM-count L2 curve in the profile (no GB10 `cp.async` curve is being taken).

## Forms

- B-avg: the current model, run on these points by the ignored test `sync_loop_burst` in `crates/vgpu-core/tests/posthoc_sync_loop.rs`; it does not see the start offsets. No free parameter.
- B-burst: `E = c0 + max(L2l + SB / P1, n SB / BW(n))`.
- `c0` from one point only: `n` 12, `SB` 16 KiB, `lock`, where the port term is the larger one (the evaluator checks this).

## Rule

- Not judged: the calibration point; the `lock` points at `n` = 48 (the configurations the form was read from); `stag` points whose median coherence at `S / 2` exceeds 0.5 (they fell back into step).
- `lock` and `stag` are judged separately. Within each, the two forms are compared only on the points where their predictions differ by more than 2%: a form wins if its median `|E_pred / E - 1|` there is below half of the other's and it is closer on at least two thirds of them. Before measuring, the forms must differ on at least six judged `lock` points (`--check-grid`, run on the model's predictions for this grid: 9 points, at `n` 36, 40 and 44 for 32, 48 and 64 KiB). If both forms' median errors exceed 25%, both are reported wrong and no new form is fitted.
- The mechanism is shown if B-burst wins `lock` and does not win `stag`. If B-burst also wins `stag`, the cause is a steady share, not a burst. Otherwise the mechanism is not shown.
- The coherence of the `lock` points over the steps is reported as the answer to whether simultaneous starts stay in step on their own; it does not enter the verdict.

## How the model would decide a phase is in step (open)

B-burst can only enter the model if `rates()` can tell which CTAs run in step. Candidate rule: CTAs of the same wave with the same tile shape and trip counts, one per SM, that entered the phase together. What the rule cannot settle yet:

- Two CTAs per SM: whether the pair interleaves (out of step with each other) or bursts together is not measured here.
- Real kernels: the attention ablation's stamps (`m128n32-w8-s2`, GB10) show the CTAs that share a head about 200 ns apart in a step of about 1 us, partly in step. Whether a GEMM's CTAs stay in step is not measured.
- How long simultaneous starts stay in step: answered only for this loop, by the `lock` coherence above.

Until the rule is settled, a positive result here shows the mechanism in a synthetic loop only, and does not by itself justify a change to `rates()`.
