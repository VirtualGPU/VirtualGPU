# H1 merged wait: registered test on the RTX 4070

Registered before any RTX 4070 measurement of `sync-loop-delta`. The GB10 run ([sync-loop-delta-gb10-posthoc.json](sync-loop-delta-gb10-posthoc.json)) was seen first and shaped this test, so it is not evidence for it; this RTX 4070 run is the held-out test. Nothing below changes after the RTX 4070 data are taken.

## Measurement

- Binary: `vgpu_bench` built for `sm_89` from the commit that adds this file, mode `sync-loop-delta`, `--samples 15 --warmup 3`, three independent processes, no point dropped except by the rule below. Temperature, SM clock and throttle reasons logged every second with `nvidia-smi` during each process.
- Kernel as on GB10: groups of 16 CTAs, one CTA per SM (32 CTAs on 46 SMs), 8 warps, the lead issues at a start time scheduled on `%globaltimer`, followers issue the same cold stage `delay_ns` later; warm L2 baselines at delay 0; cold region advanced across launches on a 256 MiB ring.
- Grid: stage 16 and 32 KiB; delay 0, 50, 100, 150, 200, 250, 300, 400, 700 and 1500 ns.

## Judged quantity

For each stage size and delay, the excess `E` is the follower stage latency on the cold region minus the follower stage latency of the warm baseline of the same stage size (medians over the three processes; cycles converted with each point's measured SM clock). The offset `d` is the measured mean follower issue time minus the mean lead issue time. No calibration constant: the baseline subtraction removes the fixed cost.

## Forms

Fields from `profiles/rtx4070.json`: `L2l` = `measured.memory.l2_latency_ns` (93.44 ns), `Ld` = `measured.memory.dram_latency_ns` (227.12 ns).

- H-model (the current core): `E = Ld - L2l`, whatever `d`.
- H-step: `E = Ld - L2l` if `d < Ld`, else 0.
- H-remain: `E = max(0, Ld - L2l - d)`.

## Rule

- A point is excluded if its late steps exceed 1% of its follower steps in any process. If the warm baseline of a stage size is excluded, that stage size is excluded.
- Error is the absolute error in ns, `|E_pred - E|`, over all remaining points of both stage sizes. A form wins if its median error is below half of each other form's and it is closer than each other form on at least two thirds of the points. Otherwise no form wins. If every form's median error exceeds 50 ns, all are reported wrong and no new form is fitted.
- Each stage size is also reported on its own, without a verdict.
- The verdict is what `python -m microbench.sync_loop_delta_eval profiles/rtx4070.json <three run files>` prints.
- No change to the model's structure follows from this test alone.
