# Post-hoc analysis: model core after phases 1 and 2

**Post-hoc.** This analysis was done after the phase 1 holdout and the phase 2 transfer results were opened. It uses their data for diagnosis only. It does not change the frozen conclusions of phase 1 or phase 2, which stand as scored with `phase1-final` (`129a6f5`). Any fix it leads to belongs to a later version, with its own freeze and holdout.

Predictions are the archived phase 1 and phase 2 predictions; measurements are the archived hardware reports. No model parameter was changed or fitted.

## 1. Phase 1 holdout: `mask-mnk` shortlist failure

`mask-mnk` is `(M, N, K) = (513, 4103, 4093)` on the RTX 4070. Its top-5 regret was 24.47% warm and 22.19% cold, the only large regret in the holdout.

### What the shortlist picked

Warm, predicted shortlist: `m128n128k64-w8-s1/s2/s4` and `m128n64k64-w4-s1/s2`. The fastest measured config is `m64n128k32-w8-s1-g1` at 485.4 us. The fastest shortlisted one measures 604.2 us, so the regret is 604.2 / 485.4 - 1 = 24.5%. Two groups put wrong configs in the shortlist:

| Configs | Measured (us) | Predicted (us) | Error | Spill (bytes per thread per iteration) |
|---|---|---|---|---|
| `m128n64k64-w4-s1/s2/s4` | 1715 to 1732 | 443.6 | -74% | 540 (stack frame 272) |
| `m128n128k64-w8-s1/s2/s4` | 604 to 605 | 425.0 | -30% | 0 |
| `m128n64k64-w8-s1/s2/s4` | 713 to 715 | 480.8 | -33% | 0 |
| `m128n128k64-w4-s1/s2/s4` | 1304 to 1310 | 1473.7 | +13% | 2052 (stack frame 728) |
| other 24 configs | 485 to 1048 | | -1% to -13% | 0 |

### Cause 1: spill cost is modeled as bandwidth, the hardware pays latency

`m128n64k64-w4` spills 540 bytes per thread per iteration (`spill_scope = inner_loop_assumed`, `ptxas_only_spill`). Its stack frame fits L1 at the predicted residency, so the model charges the spill traffic at the L1 local-memory bandwidth, which is cheap: the prediction ranks it among the fastest. On hardware it is the slowest group: 7.6 us per CTA iteration against the predicted 2.0 us (325 CTAs, two per SM, 64 iterations). Its 272 bytes of reloads are 68 four-byte reloads per thread per iteration; serialized at the profile's local-memory latencies they cost 1.7 us (L1 hit, 25 ns) to 10.9 us (L1 miss, 160 ns), which brackets the observed 5.6 us. A bandwidth charge does not reach it. The larger spill of `m128n128k64-w4` overflows L1 in the model and is charged L2 bandwidth, which overestimates it by 13%. Both point to spill reloads on the critical path being latency bound, which the model does not represent.

### Cause 2: unaligned shapes use 2-byte loads; the model sees 16-byte loads

Compiling the same `dense_gemm` configs offline (Triton 3.9, `sm_89`) shows that for `mask-mnk` every global load is `ld.global.b16`, one 2-byte element per thread, while the aligned `prefill-256` uses `ld.global.v4.b32`, 16 bytes per thread:

| Shape | `128x128x64 w8` loads per loop body | `64x128x32 w8` loads per loop body |
|---|---|---|
| `mask-mnk` (513, 4103, 4093) | 64 `ld.global.b16` | 24 `ld.global.b16` |
| `prefill-256` (256, 4096, 4096) | 8 `ld.global.v4.b32` | 3 `ld.global.v4.b32` |

With `K = 4093`, a row of `A` is 8186 bytes, not a multiple of 16, so Triton cannot vectorize. Moving the same bytes takes eight times the load instructions. The KernelIR still records `transfer_width_bytes = 16` for these loads, and the model charges no per-instruction cost for memory operations, so this cost is invisible to it. It affects every `mask-mnk` config, and the 128-row, `BLOCK_K = 64` tiles, which issue the most loads per stage, lose the most (-30% and -33%).

### What a fix needs (next version)

- Frontend: take `transfer_width_bytes` from the load instructions in the PTX instead of a constant.
- Core: charge memory instruction issue per SM. Needs a microbenchmark of global load issue throughput by width (`b16`, `b32`, `v2`, `v4`), L2-resident, by warp count.
- Core: model spill reloads as latency on the iteration's critical path. Needs a microbenchmark of a dependent compute loop with a controlled number of local-memory reloads per iteration, L1-resident and overflowing.

## 2. Phase 2 GB10: `decode-16`

`decode-16` had Kendall tau-b 0.232 warm and 0.235 cold on GB10, against 0.65 on the RTX 4070.

### Most of the loss is hardware noise

Pairs classed by the hardware verdict (warm; cold is similar):

| Hardware verdict | Concordant | Discordant | Tie |
|---|---|---|---|
| Indistinguishable (within the 5% band) | 131 | 219 | 2 |
| Ordered | 200 | 5 | 0 |
| Unresolved | 56 | 17 | 0 |

219 of the 241 discordant pairs (91%) are pairs the hardware itself cannot order. Only 5 hardware-ordered pairs are reversed.

### What remains: the prediction is too compressed

The measured times fall into three groups, while the model puts all 36 configs between 262 and 280 us and labels almost all of them DRAM latency bound:

| Group | Measured (us) | Predicted (us) |
|---|---|---|
| small tiles, `num_stages = 4` (prefetch distance 2) | 270 to 275 | 262 |
| `num_stages = 1/2` (prefetch distance 0), and 128-row tiles with one CTA per SM | 290 to 301 | 262 to 280 |
| `m128n64k64-w8-s1/s2` | 325 to 330 | 275 to 277 |

### Ruled out

- **Closed-loop latency hiding.** GB10's own closed-loop microbenchmark (`dram_closed_loop`, profile `0.11.0-m3`) shows no gap between one and four stage buffers at the grid sizes of `decode-16` (96 to 128 CTAs, 4 to 8 KB per stage: 228 to 256 GB/s for every buffer count), and the model's closed-loop formula predicts that microbenchmark with median error 3.2%. The GEMM's 8% gap between prefetch distance 0 and 2 is not in that measurement.
- **L2 bandwidth.** The working set (64 MB of `B`) exceeds GB10's 24 MB L2, and the measured times sit at the DRAM bandwidth.

### Observations, not yet explained

- GB10's memory microbenchmarks are noisier than the RTX 4070's. In `dram_closed_loop`, adjacent points differ by 5% to 10% without a trend (for example 128 CTAs: 252 GB/s with one buffer, 239 with two), and the DRAM MLP curve is not monotone (230 GB/s at 224 KB in flight, 222 at 384 KB). On the RTX 4070 the same points agree within 2%.
- The microbenchmarks stream contiguous addresses per CTA. A `decode-16` stage reads 32 to 64 rows of `B` in chunks of 128 to 256 bytes at a stride of 8 to 16 KB. No measurement covers that access granularity; whether GB10's LPDDR5X serves it differently from contiguous streams is untested.

### What a fix needs (next version)

- A strided-chunk variant of the MLP microbenchmark (chunk 128 B to 4 KB, stride up to 16 KB) on both GPUs, to test whether access granularity changes the delivered bandwidth or effective latency.
- Repeated GB10 memory microbenchmarks with intervals, to separate noise from structure in the MLP and closed-loop curves.

## 3. The synchronous-load residual of phase 1

Phase 1 Known limits (`docs/design.md` 9.1) left one residual unexplained: configs with synchronous loads (`num_stages = 1`, `ld.global` into registers, then `st.shared`) have two to four times the error of the others. A post-hoc microbenchmark (`measured.pipeline.sync_stage_loop`, profile `0.12.0-posthoc`) compares that path with single-buffer `cp.async` stage by stage. With at most two 16-byte loads per thread per stage the two paths take the same time; from four on, the synchronous path costs 96 to 160 ns more per stage with L2-resident data, and 184 to 448 ns more with DRAM data and MMA work when one CTA runs per SM, because each thread stages its loads through registers and stores them to shared memory in sequence.

The core's shortfall on the development set (prediction below measurement, core `18af028`, all 168 synchronous-load config and cache points across seven cases) follows the same threshold:

| 16-byte loads per thread per stage | Points | Median shortfall (ns per iteration) | Range |
|---|---|---|---|
| up to 2 | 42 | 0 | -566 to 95 |
| 2.5 to 3 | 42 | 13 | -320 to 130 |
| 4 | 14 | 26 | -42 to 98 |
| 6 | 28 | 147 | -498 to 277 |
| 8 to 16 | 42 | 287 | -128 to 837 |

The loads per thread per stage are the stage's global load bytes over `num_warps * 32 * 16`. This is a candidate mechanism for the next version, not a fitted correction: modeling it needs the per-stage cost as a function of loads per thread, warps, and memory level from that microbenchmark, then a check against `sync_stage_loop` itself before any development-set comparison.
