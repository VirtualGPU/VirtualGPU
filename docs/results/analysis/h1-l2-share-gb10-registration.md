# Single-buffer L2 sharing with a cp.async curve: registered test on GB10

Registered before the measurements it describes. Nothing below changes after the data are taken. The evaluator is `microbench/l2_share_eval.py`, committed with this file; its output is the verdict.

## Question

`sync-loop-burst` ([sync-loop-burst-gb10-posthoc.json](sync-loop-burst-gb10-posthoc.json)) showed that a single-buffer loop on warm L2 slows down steadily as more CTAs run it with large stages, that the CTAs fall out of step within 16 steps, and that neither the current model nor a burst form fits. One CTA there also streamed faster than the profile's 1 SM point. The profile's `l2_bw_by_active_sms` is measured with `ld.global.cg`; VirtualGPU issue #8 found that `cp.async`, which these loops use, can reach more. This test measures a `cp.async` curve and asks whether the loop's exposed time follows it. The `sync-loop-burst` data shaped this question, so its points are not evidence here; the check grid below is new.

## Measurements (GB10, post-hoc, `vgpu_bench` for `sm_121` from the commit that adds this file)

- `l2-paths`: L2 read bandwidth on a warm 4 MiB region at 1, 12, 24, 36, 40, 44 and 48 active SMs, one CTA per SM, through two paths in the same session: `ld_global_cg` (the `run_l2` ILP-8 kernel, 16 warps) and `cp_async` (the 4-stage ring of issue #8, 16 warps, two 16 B copies per thread per stage); plus `cp_async` at three CTAs per SM on the full GPU (8 warps per CTA), as in #8, described only. The names follow #8; the profile schema for the second curve is left to the #8 PR.
- `sync-loop-burst-check`: the `sync-loop-burst` loop (8 warps, 8 MMA groups per step, one CTA per SM, warm L2, simultaneous starts) at `n` = 1, 16, 28, 34, 42 and 46 CTAs and stages of 24, 40 and 56 KiB, `free` and `noload`; `E = turn(free) - turn(noload)` from per-CTA cycles.
- Each run: `--samples 15 --warmup 3`, three independent processes, nothing dropped; temperature, SM clock and throttle reasons logged every second.

## Fields

`L2l` = `profiles/gb10.json` `measured.memory.l2_latency_ns`. From the `cp_async` points of `l2-paths` (median over processes of each process's median): `P1` at 1 SM, and `BW(n)` interpolated linearly at `n`.

## Forms

- model: the current core, from the ignored test `sync_loop_burst` on the check grid.
- burst: `E = c0 + max(L2l + SB / P1, n SB / BW(n))`.
- share: `E = c0 + L2l + SB / min(P1, BW(n) / n)`, each CTA streaming its stage at its share of the GPU's L2 rate.
- The same two shapes with the `ld_global_cg` curve are printed for comparison and not judged.
- `c0` from one point: `n` = 1, 24 KiB, where both shapes reduce to `L2l + SB / P1`. The other 17 points are held out.

## Rule

- Against each other form, a form is compared on the held-out points where their predictions differ by more than 2%: there its median `|E_pred / E - 1|` must be below half of the other's and it must be closer on at least two thirds of them. A form that does this against both others wins; otherwise none wins. If every form's median error exceeds 25%, all are reported wrong and no new form is fitted.
- Before measuring, with the profile's `ld.global.cg` curve standing in for the `cp.async` one, every pair of forms differs on at least six held-out points (`--check-grid`: model and burst 8 of 17, model and share 15, burst and share 15).
- No change to the model follows from this test alone; a profile field for the `cp.async` curve waits for the #8 PR or agreement with 5.
