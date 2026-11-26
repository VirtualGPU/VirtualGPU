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

## DRAM-side narrow request form: first GB10 session

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

The form passed its rule in this session; the repeat and the RTX 4070 test are below.

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

## Second session: register staging forms and a repeat of the DRAM-side form

Captured under the lock on the same day: `stage-narrow-validate` (thermal gate 51.2 / 51.7 C, 2509 / 2515 MHz; 51 to 60 C and 2411 to 2522 MHz during capture) and a repeat of `gmem-narrow-validate` (52.9 / 53.3 C, 2515 / 2515 MHz; 53 to 62 C, 2405 to 2541 MHz), no throttle reason. Raw captures in `h5-followup2-captures.tar.gz` (SHA-256 `666492765d113467c9d2b66a7f73af25113a1124e8cf213001fa6c74f4acf851`; `stage-narrow-validate-1.json` `74f9e32d...`, `gmem-narrow-validate-2.json` `7ffc646d...`), in the same two places.

**The DRAM-side form misses its group rule on the repeat.** Scored with the same frozen file ([h5-dram-form-gb10-repeat-posthoc.json](h5-dram-form-gb10-repeat-posthoc.json)): the calibration point measured 2397.0 us against 2468.7 us in the first session, so R = 3.50 instead of 3.39 warp requests per ns. The 192 pure-load DRAM points give MAPE 10.8% (current model 27.5%), but the median error of aligned 2 B loads is -10.9%, beyond the 10% group limit (first session -8.3%); the other seven groups stay within -6.6% to -0.2%. A 2.9% drift of the single calibration point between sessions is enough to cross the limit. On GB10 the form passed one session of two (aligned 2 B: -8.3% and -10.9% against a 10% limit): marginal, not a robust pass. It is not judged a third time on these points; the RTX 4070 test below decides.

**The register staging forms do not pass.** Frozen in [validation/h5-staging-forms-frozen.json](../../../validation/h5-staging-forms-frozen.json) (SHA-256 `952cc5edb72eb49acaf448c3f221814b5490751f0335d1bc6d83bb1cec77b0c3`) by [validation/h5_staging_forms.py](../../../validation/h5_staging_forms.py) before measurement:

- b1: shared memory stores charged per warp instruction on the shared memory pipe, Q warp stores per ns per SM, calibrated where stores alone fill the pipe (stores only, 2 B, 16 KiB, 3 CTAs per SM): Q = 1.39.
- b2: a fixed time per load served from L2 on each CTA's chain, calibrated where loads alone bound one CTA (loads only, 2 B, 4 KiB, 1 CTA per SM): tau = 6.72 ns per load per thread. Loads from DRAM get nothing.
- b1 + b2: both constants as calibrated.

The new points are 8 warps, 8 and 12 KiB stages, 1 to 16 B, chains of 0 and 8 MMAs, 1 and 3 CTAs per SM, for all three variants. The pass rule (MAPE under 15%, every width's median error within 10%) used the 8-MMA points only if the current model's median error on the 16 B 8-MMA points was within 10%; it was -9.7%, so they count. Points whose kernel did not fit 3 CTAs per SM were to be excluded. The list frozen with the predictions came from the workstation's `nvcc` 13.5; the Spark's CUDA 13.0 allocates more registers, so seven points ran at fewer CTAs than set (three of them not on the frozen list) and one frozen-excluded point did fit. The scoring drops every point that did not fit, as measured, and every frozen exclusion: 88 points are judged ([h5-staging-forms-gb10-posthoc.json](h5-staging-forms-gb10-posthoc.json)).

| Form | MAPE | Median error 1 / 2 / 4 / 16 B | Loads only / stores only / both |
|---|---:|---|---|
| Current model | 28.7% | -58% / -42% / -17% / -6% | -19% / -29% / -23% |
| b1 | 18.6% | -32% / -26% / -9% / -6% | -19% / -8% / -18% |
| b2 | 20.9% | -42% / -29% / -8% / -4% | -8% / -29% / -13% |
| b1 + b2 | 10.7% | -15% / -15% / -3% / -4% | -8% / -8% / -8% |

b1 + b2 meets the MAPE limit, but 1 and 2 B stay 15% fast, so no form passes. Each part removes about half of the narrow gap where it acts; the 1 and 2 B points need more than a per-instruction store cost and a per-load chain cost calibrated at 2 B.

**Where the remaining 1 and 2 B error sits (post-hoc, after this verdict).** Split by the MMA chain, the b1 + b2 median errors on the judged points are 1 B -8%, 2 B -10%, 4 B +2%, 16 B -0% without MMA, and 1 B -20%, 2 B -17%, 4 B -6%, 16 B -10% with 8 MMAs per warp. The 4-warp `stage-narrow-batched` points, predicted after the fact with the same Q and tau, show the same split more cleanly: without MMA 2 B -1%, 4 B +2%, 16 B +4%; with 32 MMAs every width is 15% to 17% fast. Stage size, CTAs per SM and the variant move the narrow residual by a few points only (8 and 12 KiB: -15% and -16%; 1 and 3 CTAs: -14% and -17%). So b1 + b2 accounts for the width dependence, and most of what remains is the width-independent underestimate of dependent MMA chains (the same kind as H7 in attention), which reached the verdict because the 16 B 8-MMA points sat just inside the 10% rule (-9.7%).

## RTX 4070 cross-GPU test of the DRAM-side form: capture void

The decisive test, on a GPU none of the form's choices saw. Before the capture, [validation/h5_dram_form.py](../../../validation/h5_dram_form.py) was changed to calibrate R from 5 independent processes of the calibration point alone (`gmem-narrow-calibrate`), one before each of the 5 repeats of the validation sweep, taking their median; everything else, the points and the pass rule, stayed as on GB10. Predictions were frozen with the RTX 4070 profile (`0.21.0-posthoc`; BW = 457.10 GB/s, the largest point of its 128 B strided curves) in [validation/h5-dram-form-rtx4070-frozen.json](../../../validation/h5-dram-form-rtx4070-frozen.json) (SHA-256 `170d2f9c23ad677caed0e843a87c00c03167db0a9f6838ea7197e5abf41a1b3d`).

Captured on the workstation's RTX 4070 under WSL2, 2026-11-26 22:29 to 22:31 UTC, after the thermal gate (37.6 / 37.5 C, 2865 / 2865 MHz; 36 to 45 C and a constant 2865 MHz during capture, no throttle reason). The Windows desktop was running (28 graphics processes: browsers, chat clients, the Steam client, no game, no other compute process; 5.3 GiB of memory and 5% utilization before the capture), recorded in the capture's host snapshot. Raw capture `rtx4070-gmem-narrow-validate-1.json` (SHA-256 `56f328b9...`) in `h5-rtx4070-captures.tar.gz` (`7009de7626d10533467eb0fe4af7a0a00a9c1d06589da4c9cc9e164c0caeaaf7`), on the workstation in `~/scratch/vgpu8/`.

**The capture is void.** It ran right after another capture that was found bimodal because the desktop was decoding video on the GPU, which the quiet check did not see. Checked afterwards with the rule adopted for RTX 4070 captures (a sample list whose largest sample exceeds 1.3 times its smallest counts as dispersed; more than 10% dispersed lists void the capture): 913 of 1450 lists (63.0%) are dispersed, 843 of the 965 DRAM lists, with samples in two groups (for example 2.35 ms and 3.6 ms at one point). The scores below are kept as a record only; the verdict below is withdrawn and the test waits for a quiet desktop.

As captured, the five calibration runs agree within 0.2% (2351.1 to 2355.2 us, median 2354.2 us, R = 3.56 warp requests per ns), and the form scores as follows ([h5-dram-form-rtx4070-posthoc.json](h5-dram-form-rtx4070-posthoc.json)):

| | 1 B | 2 B | 4 B | 16 B |
|---|---:|---:|---:|---:|
| Aligned, form | +82.9% | -0.5% | -3.9% | +2.5% |
| Offset, form | +61.1% | -11.5% | -10.8% | +5.4% |
| Aligned, current model | -41.7% | -48.4% | -5.3% | +0.5% |
| Offset, current model | -49.2% | -53.7% | -26.2% | +0.0% |

MAPE over the 192 pure-load DRAM points is 27.1% (current model 33.8%, p90 72.7%). The 2, 4 and 16 B groups move close to the measurements, but 1 B is now predicted far too slow, and offset 2 and 4 B fall just outside 10%.

Post-hoc and from the void capture only, the 1 B points suggest why: on the RTX 4070 their request rate is not capped. It falls as each thread keeps more rows in flight, from up to 12.4 warp requests per ns at 4 loads per thread to 3.3 to 3.7 at 24, and the calibration point (16 per thread) sits on the slow side. Rows in flight per CTA set how many DRAM pages are open at once, so on this GPU the narrow DRAM cost follows access locality rather than a GPU-wide request rate. If a clean capture confirms this, the form's request limit is a GB10 observation that does not carry over.

## Last register staging round: K1 does not pass

K1 is b1 + b2 unchanged, with no new constant. Frozen in [validation/h5-staging-k1-frozen.json](../../../validation/h5-staging-k1-frozen.json) (SHA-256 `908c0d4302abac1423884134d3063e51184fb809c7b3ac1ee194609bb63b2475`) by [validation/h5_staging_k1.py](../../../validation/h5_staging_k1.py) before measurement, with the same pass rule and 8-MMA rule as before, and with the MAPE and width medians over the points without MMA registered as a descriptive result. Q and tau come from the same two calibration points, each the median of 5 independent processes run before the 5 repeats. The 96 new points are 4 warps, 6 and 10 KiB stages, 2 and 4 CTAs per SM, 1 to 16 B, the three variants, chains of 0 and 8 MMAs; four 10 KiB points with loads and stores were excluded before measurement because, with the Spark's own `ptxas`, they need more than 128 registers per thread, and exactly those four ran at 3 CTAs per SM instead of 4.

Captured on GB10 under the lock (thermal gate 51.4 / 51.7 C, 2496 / 2509 MHz; 51 to 60 C, 2424 to 2528 MHz, no throttle reason). Raw capture `stage-narrow-k1-1.json` (SHA-256 `57495a47...`) in `h5-k1-captures.tar.gz` (`5661dc1539a276c48736777cfb30a6bcab42af64870952450ddd77fee16ca9de`), in both places. Scores: [h5-staging-k1-gb10-posthoc.json](h5-staging-k1-gb10-posthoc.json).

- Calibration: 86.3 to 86.5 us and 446.8 to 449.0 us over the 5 runs; Q = 1.39 warp stores per ns per SM, tau = 6.76 ns per load.
- The current model's median error on the 16 B 8-MMA points is -11.1%, so the 8-MMA points are reported and not judged. 46 points without MMA are judged.
- K1: MAPE 16.9% (current model 25.7%); median error 1 B +3.6%, 2 B -3.8%, 4 B +8.8%, 16 B -1.4%. Every width is within 10%, but the MAPE is above 15%, so K1 does not pass.
- Post-hoc: stores alone are now predicted 17% to 18% slow at both 2 and 4 CTAs per SM (Q was calibrated with 16 KiB stages and 3 CTAs), and loads alone 8% to 10% slow at 2 CTAs; the 8-MMA points, not judged, add the MMA chain miss (2 B -17%).

H5 register staging stops here: b1 + b2 points the right way and lowers the MAPE from 28.7% to 10.7% (second round) and from 25.7% to 16.9% (this round), but it did not meet the pass rule and does not go into the core ([design.md section 9.5](../../design.md#95-narrow-register-staged-loads-h5)).

**Sample dispersion of the GB10 captures.** The dispersion rule adopted for RTX 4070 captures (more than 10% of sample lists with largest over smallest above 1.3) is not applied to GB10, but by it the GB10 captures of H5 would read 15.2% (`gmem-narrow-validate-1`), 14.6% (`-2`), 11.4% (`stage-narrow-validate-1`) and 12.0% (`stage-narrow-k1-1`). On GB10 these are single high samples in otherwise tight lists, at random positions, not two groups; every point uses the median over repeats of per-repeat medians, which these samples do not move.

## Open item outside H5

**Hot L2 with few loads in flight is predicted too slow.** `gmem-narrow-validate`, hot L2, `.cg`: 1 B loads with 4 in flight per thread are predicted 3.2 to 3.7 times as slow as measured at 1 CTA per SM and 1.2 to 2.8 times at 3 CTAs, aligned or not; other widths at 4 in flight up to 1.3 times; at 12 and 24 in flight the MAPE is 9% to 11%. The point values are in [h5-dram-form-gb10-posthoc.json](h5-dram-form-gb10-posthoc.json) (`gmem_l2_regression`). Not investigated.

## Next steps

- DRAM-side form: repeat the RTX 4070 test with the same frozen file in a quiet desktop window (GPU engine utilization under 5% for 60 s before the capture, dispersion check after it). It does not go into the core in any case before that test passes.
- Register staging: closed as a known limit (section above).
