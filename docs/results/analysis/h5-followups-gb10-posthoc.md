# H5 follow-ups on GB10: DRAM-side narrow request form and register staging

**Post-hoc.** Both follow-ups come from observations on the first narrow-load points ([h5-narrow-microbench-posthoc.md](h5-narrow-microbench-posthoc.md)), so those points are not held out for them. The DRAM-side form was frozen before the new points below were measured; the register staging points are measurement only. No model code changed.

## Capture

DGX Spark GB10, 2026-11-27, one session under the shared GPU lock running three benchmarks in turn, each behind its own thermal gate, 5 repeats of 15 samples.

| Benchmark | Thermal gate (two 30 s windows) | During capture |
|---|---|---|
| `gmem-narrow-validate` (289 points) | 52.1 / 52.4 C, 2506 / 2522 MHz | 52 to 60 C, 2437 to 2535 MHz |
| `sync-loop-narrow-validate` (36 points) | 55.1 / 54.9 C, 2502 / 2502 MHz | 54 to 57 C, 2450 to 2489 MHz |
| `stage-narrow` (108 points) | 55.4 / 55.3 C, 2489 / 2502 MHz | 55 to 62 C, 2502 to 2535 MHz |

No throttle reason was active and about 115 GiB of memory was available. The raw captures are archived as `h5-followup-captures.tar.gz` (SHA-256 `87142a0cf52129a65f3325b12acee8cb3464a037eda13b2715b73a49f7f056f9`; `gmem-narrow-validate-1.json` `9277ba00...`, `sync-loop-narrow-validate-1.json` `9a110693...`, `stage-narrow-1.json` `109270ee...`), kept on the Bifrost workstation in `~/scratch/vgpu8/` and on the Spark in `~/virtualgpu/s8-narrow/runs/`.

## DRAM-side narrow request form: passes on GB10

From DRAM, the first points ran at one of two limits: about 7.3 sectors per ns over the GPU for 4 and 16 B loads, the plateau of the profile's strided DRAM curve with 128 B segments (236 GB/s), and about 3.3 warp requests per ns for 1 and 2 B loads. The form is

T = max(T_core, S x 32 B / BW, N / R)

with T_core the current core's prediction, S the sectors that DRAM-sourced warp loads touch (row by row, from each row's first byte), BW the largest point of every strided DRAM curve with 128 B segments in the profile (236.08 GB/s, not fitted), N the number of DRAM-sourced warp loads, and R the only constant. Hot-L2 loads are not DRAM-sourced, so their prediction stays T_core. An additive variant (S x 32 B / BW + N x tau) fitted the first points worse and was not frozen.

- **Frozen** in [validation/h5-dram-form-frozen.json](../../../validation/h5-dram-form-frozen.json) (SHA-256 `6bfb42158504465cac19b754864a871c27ddde3a230dcff2133bd1bcba05756d`) by [validation/h5_dram_form.py](../../../validation/h5_dram_form.py) before measurement, with the pass rule written in the script: MAPE under 15% on the 192 pure-load DRAM points, no width and alignment group with a median error beyond 10%, and unchanged hot-L2 predictions. The 36 `sync-loop-narrow` DRAM points are reported separately, since their shared memory stores carry a cost this form does not model.
- **New points.** 1 to 16 B, aligned and offset rows, 4, 12 and 24 loads in flight per thread, 1 and 3 CTAs per SM, 4 and 8 warps, 128 and 512 MiB per sample (the first points used 8, 16 and 32 in flight, 1 and 2 CTAs, 256 MiB); `sync-loop-narrow` with 2 and 12 KiB stages at 1, 2 and 4 CTAs per SM. All `.cg`.
- **Calibration** (1 B, aligned, DRAM, 2 CTAs per SM, 8 warps, 16 in flight, remeasured in this session): 2468.7 us against a floor of 1135.9 us from T_core and the sector limit, so R is identifiable: R = 3.39 warp requests per ns.

Scores ([h5-dram-form-gb10-posthoc.json](h5-dram-form-gb10-posthoc.json)):

| Group | Points | Form MAPE | Form p90 | Current model MAPE |
|---|---:|---:|---:|---:|
| Pure loads from DRAM (pass rule) | 192 | 10.4% | 15.6% | 27.3% |
| Register-staged loads from DRAM (cross-check) | 36 | 11.5% | 27.5% | 24.2% |
| Hot L2 (regression) | 96 | unchanged | | 24.5% |

Median error by width and alignment, pure loads from DRAM:

| | 1 B | 2 B | 4 B | 16 B |
|---|---:|---:|---:|---:|
| Aligned, form | +3.1% | -8.3% | -4.5% | -0.3% |
| Offset, form | +1.6% | -6.9% | -4.8% | -3.2% |
| Aligned, current model | -48.4% | -21.8% | -11.0% | -7.7% |
| Offset, current model | -48.5% | -38.3% | -27.2% | -12.5% |

The form passes its rule on GB10. The cross-GPU test on the RTX 4070, with R calibrated at the same point there, waits for that GPU to be free.

The hot-L2 regression points stay where the current model puts them, but that is not accurate everywhere: the median error is within 13% in every width and alignment group, while 1 B loads with 4 in flight are predicted 1.2 to 3.7 times as slow as measured, 3.2 to 3.7 times at 1 CTA per SM (MAPE 53% at 4 in flight against 9% to 11% at 12 and 24). This is a separate issue of the current model at hot L2 with few loads in flight, in the opposite direction to H5, and is left out of H5 (see the open item below).

## Register staging, split into its parts (measurement only)

`stage-narrow` runs one stage per iteration at hot L2 (12 MiB): global loads only, shared memory stores only, or both, then a barrier, `ldmatrix` and an MMA chain; 2, 4 and 16 B; 4 and 16 KiB stages; 1 to 3 CTAs per SM; 0 and 32 MMAs per warp. Measurements: [h5-stage-narrow-gb10-posthoc.json](h5-stage-narrow-gb10-posthoc.json), written by [validation/h5_stage_narrow.py](../../../validation/h5_stage_narrow.py). Each sample does the same total work, so a cost on each CTA's own chain shrinks as CTAs per SM grow and a cost on a shared pipe does not.

Observations, with the current core's predictions computed after the fact for orientation:

1. **Stores alone are a shared-pipe cost that grows with the instruction count.** With no MMA, stores-only time hardly changes from 1 to 3 CTAs per SM (ratios 0.76 to 0.90). For the same bytes, 2 B stores take 1.7 to 2.1 times as long as 16 B stores (16 KiB: 99 us against 51 us at 1 CTA). The model charges shared memory by bytes and predicts 43 us for every width there; a 2 B warp store moves 64 B per shared memory wavefront where a 16 B store moves 128 B.
2. **Loads alone, all issued before the barrier, cost more per byte when narrow, on each CTA's chain.** With 4 KiB stages, where every thread issues all its loads before its first use, 2 and 4 B loads are 19% to 37% slower than predicted and 16 B loads 4% to 18% faster; the gap shrinks from 1 to 3 CTAs per SM.
3. **The 16 KiB, 2 and 4 B kernels are not representative.** Their SASS interleaves each load with its shared memory store after the first seven loads (`LDG7 STS3 LDG2 STS1 ...`), so a thread keeps only a few loads in flight; `sync-loop-narrow` with 16 KiB stages compiles the same way. 4 KiB stages keep all 16 loads ahead of the stores. The Triton `mask-mnk` kernels issue most of their loads first (`m16n64k32-w4`: 18 of 20, `m128n128k64-w8`: 64 of 64; offline `ptxas-blackwell` from the archived PTX). The 16 KiB stages of the first `sync-loop-narrow` points, part of the 45% to 54% hot-L2 gap reported there, carry the same scheduling.

## Register staging with every load first (measurement only)

`stage-narrow-batched` repeats `stage-narrow` with two changes: every thread issues all its loads before its first shared memory store (each store address carries the XOR of every loaded value, masked by a run-time zero), and the stores go to Triton's swizzled layout for 128 B rows (16 B chunks XORed with the row index, 8 phases, as `swizzled_shared<vec = 8, perPhase = 1, maxPhase = 8>`). SASS on GB10: every instance issues all its loads and then all its stores (2 B, 16 KiB: 64 `LDG.E.U16`, then 64 `STS.U16`), store widths are `STS.U16`, `STS` and `STS.128`, and no kernel uses local memory. Stages of 4, 8 and 16 KiB, otherwise as `stage-narrow`; 162 points.

Captured on the same day under the lock (thermal gate 51.7 / 52.2 C, 2515 / 2522 MHz; 51 to 60 C and 2496 to 2515 MHz during capture, no throttle reason). Two points are excluded: 2 B loads and stores with 16 KiB stages fit only 2 CTAs per SM, not 3, since each thread holds 64 loaded values. Raw capture `stage-narrow-batched-1.json` (SHA-256 `559d279f...`) in `h5-stage-batched-captures.tar.gz` (`d933ee9e69196a73ee9cd6bc714fa636085aced521ba3bc29303b559fa8f257c`), in the same two places; summary [h5-stage-narrow-batched-gb10-posthoc.json](h5-stage-narrow-batched-gb10-posthoc.json).

1. **The interleaving was most of the 16 KiB gap.** 2 B loads and stores with 16 KiB stages at 1 CTA per SM take 306 us batched against 603 us interleaved.
2. **Narrow loads still cost more per byte, and narrow stores more again.** Time relative to 16 B at the same stage, 1 CTA per SM, no MMA:

   | Variant | 2 B, 4 / 8 / 16 KiB | 4 B, 4 / 8 / 16 KiB |
   |---|---|---|
   | Loads only | 1.58 / 1.30 / 1.42 | 1.31 / 1.02 / 1.09 |
   | Stores only | 2.33 / 2.47 / 2.31 | 1.37 / 1.45 / 1.54 |
   | Loads and stores | 1.52 / 1.57 / 1.59 | 1.21 / 1.14 / 1.26 |

   The load gap shrinks as CTAs per SM grow (a cost on each CTA's chain); stores-only time hardly changes with CTAs per SM (a shared pipe). The current model predicts the same time for every width; its errors for 2 B loads and stores are -39% to -49% at 1 CTA per SM, -0% to -19% for 16 B.
3. **A width-independent miss.** With 32 dependent MMAs per warp and 1 CTA per SM, every width is predicted 28% to 48% fast, loads only or stores only. This is the MMA chain, not narrow loads, and is left out of H5.

## Open item outside H5

**Hot L2 with few loads in flight is predicted too slow.** `gmem-narrow-validate`, hot L2, `.cg`: 1 B loads with 4 in flight per thread are predicted 3.2 to 3.7 times as slow as measured at 1 CTA per SM and 1.2 to 2.8 times at 3 CTAs, aligned or not; other widths at 4 in flight up to 1.3 times; at 12 and 24 in flight the MAPE is 9% to 11%. The point values are in [h5-dram-form-gb10-posthoc.json](h5-dram-form-gb10-posthoc.json) (`gmem_l2_regression`). Not investigated.

## Next steps

- Register staging: propose forms from the `stage-narrow` and `stage-narrow-batched` points (a per-instruction shared memory store cost on the shared pipe and a per-request cost on the CTA's chain for loads from L2), freeze them with one calibration point each, and validate on another set of points.
- DRAM-side form: the RTX 4070 cross-GPU test first; only after it passes, wiring the form into the core for `mask-mnk` with its DRAM-sourced fraction from the L2 model, agreed beforehand with the owner of the round-trip path in `sim.rs`.
