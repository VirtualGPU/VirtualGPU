# GB10 spill-latency at 1 KiB frames: explicit maximum-L1 carveout (plan, diagnostic)

Status: frozen before capture. Diagnostic only; no profile or rule changes.

## Question

In the first GB10 0.23.0 capture (main `42e039d`), `spill-latency` failed rule 1 in 6 of 7 rounds. Every failing list was a 1 KiB frame (`l1_boundary`, 128 KiB per SM, the L1 size): the serial-chain points at 8, 32, 68 and 128 reloads take one of two states per launch (a fast cluster near the median and a slow one 1.7 to 2.5 times it, at random sample positions), and the batched points at 68 and 128 reloads are wide but single-peaked. The spill kernels set no shared memory carveout. If the driver's carveout, and so the L1 left to the frame, differs from launch to launch, a frame exactly at the L1 size would switch between fitting in L1 and spilling to L2. Does an explicit maximum-L1 carveout (`cudaFuncAttributePreferredSharedMemoryCarveout` = 0) remove the two states?

## Design

- Binary built from `s8/diag-spill-carveout` (the `--spill-carveout` flag, which sets the attribute on all 12 spill kernel instances before the sweep). Same binary for both arms.
- `vgpu_bench --mode spill-latency` under the frozen schedule of the `42e039d` capture (`spill-latency.schedule`, all 60 points; the schedule binds them all, so no point is dropped), 15 samples, 3 warmups, `--clock-probe`.
- Arms: `default` (no attribute) and `max_l1` (`--spill-carveout 0`), alternating, 3 processes each, one session, after a 10 s quiet-host gate, each process traced and host-checked (`host_quiet.run_traced`), under the GPU lock (`microbench/spill_carveout.py`).

## Frozen reading

- A sample is slow when above 1.4 times its point's lower quartile in that process.
- The control reproduces the two states when at least 3 of the 4 serial 1 KiB points have a slow sample in at least 2 of the 3 `default` processes. If it does not, the run is recorded only.
- The two states are gone when, in all 3 `max_l1` processes, none of the 4 serial 1 KiB points has a slow sample and each has an interquartile range of at most 10% of its median.
- Otherwise they remain. A failed host check voids the run.
- Also reported, not judged: the batched 1 KiB points' interquartile ranges, and the serial medians per arm (a different carveout may move the medians).
