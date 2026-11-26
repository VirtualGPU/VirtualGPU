# H1 merged wait: registered test on the RTX 4070

Registered before any RTX 4070 measurement of `sync-loop-delta`. This version replaces the first registration (`542e943`), whose GB10 delay grid left H-step and H-remain different on only 4 of 10 RTX 4070 delays, so no form could have won. The GB10 run ([sync-loop-delta-gb10-posthoc.json](sync-loop-delta-gb10-posthoc.json)) was seen first and shaped this test, so it is not evidence for it; this RTX 4070 run is the held-out test. Nothing below changes after the RTX 4070 data are taken.

## Measurement

- Binary: `vgpu_bench` built for `sm_89` from the commit that registers this version, mode `sync-loop-delta`, `--samples 15 --warmup 3`, `VGPU_DELTA_NS` set to the grid below, three independent processes, no point dropped except by the rule below. Temperature, SM clock and throttle reasons logged every second with `nvidia-smi` during each process.
- Kernel as on GB10: groups of 16 CTAs, one CTA per SM (32 CTAs on 46 SMs), 8 warps, the lead issues at a start time scheduled on `%globaltimer`, followers issue the same cold stage `delay_ns` later; warm L2 baselines at delay 0; cold region advanced across launches on a 256 MiB ring.
- Grid: stage 16 and 32 KiB; the delays follow from the device's `Ld` and its `%globaltimer` step `t`, which is measured first (`sync-loop-delta-flag` prints it as `globaltimer_step_ns`). With `s = max(t, Ld / 7)` rounded to a whole ns, the transition points are `k s` for `k` = 0 to 7, and the two plateau points are `ceil(1.4 Ld / s) s` and `ceil(3 Ld / 100) 100`. For the RTX 4070 (`Ld` = 227.12 ns) with `t` = 32 ns this gives 0, 32, 64, 96, 128, 160, 192, 224, 320 and 700 ns. If `t` is not 32 ns, the grid is recomputed by the same rule and recorded with the data.
- Before measuring, `python -m microbench.sync_loop_delta_eval --check-grid profiles/rtx4070.json <grid>` must pass: H-step and H-remain differ on at least two thirds of the delays and every pair of forms on at least two. If it fails, the test is not run.

## Judged quantity

For each stage size and delay, the excess `E` is the follower stage latency on the cold region minus the follower stage latency of the warm baseline of the same stage size (medians over the three processes; cycles converted with each point's measured SM clock). The offset `d` is the measured mean follower issue time minus the mean lead issue time. No calibration constant: the baseline subtraction removes the fixed cost.

## Forms

Fields from `profiles/rtx4070.json`: `L2l` = `measured.memory.l2_latency_ns` (93.44 ns), `Ld` = `measured.memory.dram_latency_ns` (227.12 ns).

- H-model (the current core): `E = Ld - L2l`, whatever `d`.
- H-step: `E = Ld - L2l` if `d < Ld`, else 0.
- H-remain: `E = max(0, Ld - L2l - d)`.

## Rule

- A point is excluded if its late steps exceed 1% of its follower steps in any process. If the warm baseline of a stage size is excluded, that stage size is excluded.
- Error is the absolute error in ns, `|E_pred - E|`, over all remaining points of both stage sizes. Against each other form, a form is compared only on the points where the two predict different excesses (by more than 0.5 ns): there its median error must be below half of the other's and it must be closer on at least two thirds of them. A form that does this against every other form wins; otherwise no form wins. On the RTX 4070 grid H-model and H-step differ at only the two plateau delays, so the evidence between those two is limited. If every form's median error over all points exceeds 50 ns, all are reported wrong and no new form is fitted.
- Each stage size is also reported on its own, without a verdict.
- The verdict is what `python -m microbench.sync_loop_delta_eval profiles/rtx4070.json <three run files>` prints.
- No change to the model's structure follows from this test alone.
