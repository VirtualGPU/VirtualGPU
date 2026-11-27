# Long-launch profile capture (design, v0.3)

Status: accepted (2026-11-27), being implemented. It changes how `microbench/collect_profile.py` captures every timed field of a GPU profile and which captures it accepts. The model does not change; the profiles it reads are captured again under it.

## 1. Why

Most timed fields of the v0.2 profiles were captured with single launches far under 1 ms, some under 50 us. `microbench/launch_audit.py` derives each launch's duration from the work one timed launch does (the `run_*` functions as they were at capture time) and the value the profile stores. Summary (points under 1 ms / points; the factor that brings the shortest launch to 1 ms):

| Field | GB10 (`0.22.0-posthoc`) | RTX 4070 (`0.21.0-posthoc`) |
|---|---|---|
| `dram_bw_by_bytes_in_flight` | 38/48, x11.9 | 47/48, x21.7 |
| `l2_bw_by_bytes_in_flight` | 44/48, x23 | 44/48, x20.8 |
| `dram_bw_by_bytes_in_flight_strided` | 40/40, x13.2 | 40/40, x23.3 |
| `dram_closed_loop` | 279/360, x3.5 | 345/360, x6.8 |
| `l2_closed_loop` | 360/360, x16.1 | 360/360, x18.1 |
| `dram_closed_loop_mma` | 118/150, x2.6 | 134/150, x4.9 |
| `l2_merge_by_sharing` | 89/90, x9.1 | 72/72, x15 |
| `gmem_load_issue_controlled` | 33/36, x6.2 | 34/36, x7.2 |
| `gmem_load_issue_per_sm` | 17/24, x4.5 | 18/24, x5.5 |
| `mma_tflops_per_sm_by_warps` | 6/6 per type, x4.9 | 4/6 per type, x2.8 |
| `mma_chain_latency` | 128/128, x9.9 | 128/128, x8.9 |
| `sync_stage_loop`, DRAM stages under 12 KiB | 150/162, x12.5 | 162/162, x19.9 |
| `sync_stage_loop_ablation` | 64/64, x4.9 | 64/64, x9.4 |
| `spill_reload_latency_by_count` | 51/60, x15.7 | 52/60, x15.8 |
| `serial_stage_fill` | 33/64, x1.5 | 41/64, x1.9 |
| SFU and shuffle issue rates | 1 to 4 of 6, x1.1 to x1.4 | 1 to 4 of 6, x1.1 to x1.7 |
| `dram_bw_by_active_sms`, `l2_bw_by_active_sms` | 0/6, 0/8 | 4/6, 0/8 |

Several modes aim at 0.35 to 0.5 ms by design (`closed-loop`, `mma-chain`, `spill-latency`); `sync-loop-reconcile` already adapts to about 1 ms. Short launches failed the capture dispersion rules on GB10 (rule 1, interquartile range over 10% of the median, in 7.7% to 15.1% of lists), and on GB10 CPU memory traffic slows DRAM-bound launches without spreading their samples ([gb10-dram-cpu-contention-posthoc.md](../results/analysis/gb10-dram-cpu-contention-posthoc.md)).

Consequence: the microbenchmark errors in [design.md section 6.4](../design.md#64-mechanisms-in-the-core) were measured against these short-launch fields. They are checked again on the new profiles.

## 2. Rules

1. Every timed window (one launch, or the group or batch section 4 names) takes at least 1.25 ms in the pilot and at least 1.0 ms in the capture.
2. A pilot chooses each point's length (steps, iterations, batches or bytes); the lengths are frozen, with their digest, before the capture, and the capture fails if a point's length differs.
3. The definition of what a field measures does not change. Its value may change because a longer launch amortizes each launch's fixed cost and start-up transient; every mode reports its long over short shift and explains it by 1/steps scaling or equivalent evidence. A residual of at most 1% after that explanation is recorded as within the clock variation of a device without locked clocks (GB10's SM clock alone spans 2412 to 2566 MHz); a larger residual is investigated before the schedule is used. Where more work per launch would change the definition, the fix is a repeat of the whole measured unit (section 4).
4. Each point records its length and its launch durations.
5. A capture that fails a gate (section 6) writes no profile.
6. Old profiles stay valid; new fields are optional.

## 3. One schedule for every timed mode

`microbench/src/long_read.cuh` (`LongReadSchedule`, written for `mlp`, `strided-mlp`, `dram-bw` and `l2`) already implements rules 1, 2 and 4: in pilot mode it scales a point's length until the fastest of three samples reaches 1.25 ms (at most eight trials, within address bounds) and prints every trial; with a schedule file it runs each point at its frozen length; either way it prints `steps`, `transfer_bytes`, `source_buffer_bytes` and `launch_ns_samples`. `microbench/long_mlp.py` freezes the schedule and checks it at capture.

The design generalizes it rather than writing one protocol per mode:

- Rename `LongReadSchedule` to `LaunchSchedule` (keep the old name as an alias) and drop the read-sweep restriction in `vgpu_bench.cu`; the flag becomes `--launch-schedule pilot|FILE` (`--long-read-schedule` stays accepted).
- Each `run_*` routes its timed loop through `schedule->measure(initial, minimum, maximum, ...)` with a `launch(length)` callback. `length` is the mode's existing work knob (table below). Without a schedule a mode behaves exactly as now, so old commands reproduce old captures.
- Modes whose length is bounded by a buffer (cold DRAM reads must not wrap) size their buffers from free device memory, 1 GiB at least and up to 4 GiB (1 GiB lasts only 0.5 to 0.7 ms at 1.5 to 2 TB/s); `LaunchSchedule` refuses a length that would wrap. Steady-state streams (`closed-loop`, `sync-loop`) may wrap as long as the buffer is a power of two of at least max(256 MiB, 8 x L2) and, for every geometry of the mode, the bytes touched between two reads of the same address (its actual reuse distance, which depends on the CTA count and stride, not only on the buffer) exceed the L2, so a re-read still comes from DRAM. Without a schedule these modes keep their v0.2 buffers (256 MiB for `closed-loop`), so old commands reproduce old captures.
- The two changes of #12 stay: `run_dram_bw` sizes its buffer as max(buffer, 16 x L2), and `run_l2` keeps `read_curve_cp_async`.
- `long_mlp.py` becomes `microbench/long_launch.py` with the same pilot, freeze, capture and audit steps for any mode; `collect_profile.py` calls it.
- A frozen point's identity is its place in the output plus the mode's explicit design fields (for `closed-loop`: `issue_mode`, `tokens`, `active_ctas`, `stage_bytes`, `compute_ns`); a mode without registered design fields cannot be frozen. Run-time diagnostics such as `observed_sms` are not part of the identity; `long_launch.py` checks them separately (`observed_sms` must equal `target_sms`). Reproducibility audits match points by identity, never by position, so outputs with different geometries are not compared.
- Before a capture, `long_launch.py check-schedule` compares the schedule file's SHA-256 with the freeze record, so the binary only runs the frozen file; the `steps` check on the output is the second check. After it, `verify` rejects a point whose CUDA event list is incomplete (not the requested sample count) or holds a NaN, an infinity or a value of zero or less.
- The full-suite freeze grid includes the optional `cp.async` active-SM curve where the mode emits it.

Lengthening keeps each field's definition; values shift where short launches carried a fixed cost or a start-up transient (rule 3). Each mode's commit compares long and short launches before its schedule is used (without the host gate: a mechanism check, not profile data). The standard comparison runs the scheduled path in one session at the legacy lengths (`x1`) and at the frozen lengths (`long`), alternating, two processes each, so that only the length differs; comparing against an earlier legacy capture mixes in process-to-process and point-history differences of up to a few percent. The `closed-loop` and `l2-merge` rows predate this and compare one scheduled process with one legacy process. Shift is the median over points of the ratio of per-point medians, long over short, minus one; F is the fixed cost per launch implied by the 1/steps model.

| Mode | Points, shortest launch | Rule 1 / rule 2 lists flagged, short to long | Shift | Explanation |
|---|---|---|---|---|
| `closed-loop` | 720, 1.23 ms | 2.4% to 0.0% / 1.5% to 0.0% | turnaround +0.2% (5th to 95th percentile: DRAM -1.0% to +2.8%, L2 -2.7% to +1.2%) | none needed |
| `l2-merge` | 90, 1.24 ms | 3.3% to 0.0% / 8.9% to 0.0% | `l2_to_sm_bw_gbps` +2.8% (up to +6.4%); turnaround -2.9% with `stagger_stages` 0 (-5.2% to -1.4%), +0.1% with 16 | bandwidth: about 3 us of fixed cost per launch (launch and stagger wait), the shift falls to zero for 256-step launches of 0.5 ms or more; turnaround: scales with 1/steps like a start-up transient of about 8 steps, absent when followers start after the leader; a 1 GiB buffer at 256 steps moves neither value (within 0.1%), so the shifts come from the length |
| `mma` | 13, 1.28 ms | 0% / 0% both | 1 to 16 warps +0.1% to +1.7%, largest where the legacy launch took about 0.2 ms; 32 warps, bf16 at 1 warp and `fp32_fma` -0.6% to -0.2% | 1 to 16 warps: F = 1.9 to 4.0 us, and the model predicts a run at twice the legacy length within 0.3 points; the rest is within 1% and in a run with the SM clock measured before and after every point (`--clock-probe`) tracks the clock (`fp32_fma` -0.2% at a 0.75% lower clock, +0.5% per cycle) |
| `gmem-issue` | 24, 1.28 ms | 4.2% / 0% to 0% to 4.2% / 0% | median +0.3%, 95th percentile +4.3% | F about 2.4 us for the 48-SM points; the 1-SM, 8 B, 8-warp point is unstable at any length (interquartile range 9.7% with long launches, repeats 10% apart) and is left to the low in-flight / few-CTA study; points whose frozen length equals the legacy one moved +1.4% to +1.8% in the first control but within 0.4% in a three-repeat run with the clock probe |
| `gmem-issue-controlled` | 36, 1.22 ms | 0% / 0 to 2.8% to 0% / 0% | median +0.7% (-0.5% to +2.3%), up to +2.6% for legacy launches of 160 to 210 us | F = 3.5 us |
| `sync-loop` | 288, 1.23 ms | 5.9% / 5.2% and 5.6% / 4.2% to 3.1% / 0.3% and 1.4% / 0% | median -1.5% (-8.5% to +1.1%). By legacy launch: L2 under 100 us -5.4%, 100 to 300 us -2.4%, 300 us and more -0.8%; DRAM 100 to 300 us -1.7%, 300 us and more +0.2% to +0.5% | F = 3.7 us (2.1 to 12) for L2 points; DRAM points repeat within 8%, so their F is resolved only in the median (4.4 us). The legacy `sync_stage_loop` turnarounds of short launches carry this fixed cost. The remaining rule 1 lists are DRAM points of 4 to 8 KiB stages at 1 to 2 CTAs per SM, the low in-flight region of the GB10 study, not a `sync-loop` exception |
| `closed-loop-mma` | 150, 1.24 ms | 4.0% / 2.7% and 2.7% / 3.3% to 2.0% / 0% and 2.7% / 0% | turnaround median +0.1% (-0.9% to +1.3%) | within 1%, the size of the repeat-to-repeat spread |
| `spill-latency` | 60, 1.09 ms | 6.7% / 5.0% and 8.3% / 6.7% to 8.3% / 6.7% and 6.7% / 6.7% | `reload_count` 0: per-step time falls (4 KiB frames 46 to 16.6 ns, 2 KiB 24.7 to 15.5 ns, smaller frames 16 to 17 to 15 ns); by legacy launch, under 200 us -9.3%, 200 to 1000 us -1.0% to -0.5% | F grows with the frame (about 126 us at 4 KiB, 40 us at 2 KiB, 5 us for small frames): the local frame is initialized once per launch, and the legacy `reload_count` 0 values carry it. Not from the length: the `l1_boundary` serial points (1 KiB frames, at the L1 size) are unstable at any length (interquartile range up to 94% short and 122% long), and are most of the rule 1 and 2 lists; `l2_spill` 4 KiB points with reloads differed 13% to 20% between processes at identical lengths, with the same clock, which rule 3 judges in the capture |
| `mma-chain` | 128, 1.26 ms | 0% / 0% both | stage time median -1.1% (-2.1% to -0.2%), largest for legacy launches under 200 us | F = 2.9 to 3.3 us; the SM clock was 0.3% to 0.4% lower during long launches (up to 1.1%), the droop the legacy 0.35 ms target avoided, which offsets part of the amortization |

**`spill-latency` and its derived values.** The local frame is initialized once per launch, a fixed cost F that grows with the frame and enters every point's per-step time as F / steps. `extra_ns_vs_zero` subtracts the `reload_count` 0 point, which runs many more steps (4096 in the legacy capture) than the points with reloads (128 to 2105), so the subtraction does not cancel F: the legacy `extra_ns_vs_zero` (and `ns_per_reload`) are overestimated by (F / steps at that count - F / steps at count 0) over the extra. The schedule does not remove it where 128 steps already reach 1.25 ms, so the long-launch values keep most of it at 4 KiB frames. From the `spill-latency` comparison (Spark 1, x1 and long, two processes each, legacy GB10 profile for the extras):

| Frame per thread | F (us) | `reload_count` 0 per step, legacy length / frozen length (ns) | Legacy `extra_ns_vs_zero` overestimate at 8 / 32 / 68 / 128 reloads, serial; batch | Left with frozen lengths, serial; batch |
|---|---|---|---|---|
| 128 B | 2.3 | 15.6 / 15.1 | +0.3 / +0.5 / +0.6 / +0.6%; +0.0 / +0.1 / +0.2 / +0.3% | +0.2%; +0.2% |
| 256 B | 3.8 to 4.3 | 16.1 / 15.2 | +0.5 / +0.9 / +1.0 / +1.0%; +0.0 / +0.1 / +0.3 / +0.5% | +0.3%; +0.3% |
| 512 B | 4.9 to 5.4 | 16.3 / 15.2 | +0.6 / +1.1 / +1.2 / +1.3%; +0.0 / +0.2 / +0.4 / +0.6% | +0.4%; +0.4% |
| 1 KiB | 8.1 to 8.9 | 17.1 / 15.1 | +2.0 / +2.0 / +1.4 / +0.8%; +0.9 / +2.2 / +2.3 / +2.6% | +0.3 to +0.6%; +0.5 to +0.7% |
| 2 KiB | 39 to 40 | 24.6 / 15.5 | +8.0 / +6.7 / +3.3 / +1.8%; +3.8 / +5.6 / +6.0 / +6.4% | +1.8 to +3.1%; +2.9 to +3.0% |
| 4 KiB | 125 to 127 | 45.7 / 16.6 | +11.4 / +9.5 / +5.5 / +3.1%; +6.8 / +9.8 / +10.3 / +11.2% | +3.2 to +10.9%; +11.0 to +11.4% |

The model checks where the length changed: at 2 KiB batch frames, 128 reloads, it predicts a 3.8% lower extra with the frozen lengths and the capture shows 4.1%. Values derived from these extras (the reload latency lookup in the core, issue #15) inherit the overestimate. Removing it needs either the extra computed with F subtracted (F from two lengths of the count 0 point) or per-step times from two lengths of each point; both change how the field is derived and are left to the core owner.

Points known to make a `spill-latency` capture fail, not from the length and not exempted: the `l1_boundary` serial points with 8 or more reloads (1 KiB frames, 128 KiB per SM at the L1 size), unstable at any length, possibly the same kind of effect as GB10's L2 bistability near its capacity (not verified); and the `l2_spill` 4 KiB points with reloads, which differed 13% to 20% between processes at identical lengths and clocks. If `spill-latency` fails a capture, its fields keep their previous values, marked provisional.


| Mode | Length knob (now) | Now | Plan |
|---|---|---|---|
| `mlp`, `strided-mlp`, `dram-bw`, `l2` | bytes per launch | 0.04 to 2.6 ms | done (`long_mlp`) |
| `closed-loop` | steps, clamp(5e5 / estimate, 256, 1024) | about 0.5 ms target | schedule over steps |
| `closed-loop-mma` | 256 steps | 0.2 to 1.6 ms | schedule over steps |
| `l2-merge` | 256 steps | 0.07 to 1.6 ms | schedule over steps; turnaround stays per step (clock64) |
| `gmem-issue`, `gmem-issue-controlled` | iterations, batches | 0.14 to 3.7 ms | schedule over iterations |
| `mma` (and `fp32_fma`) | iterations, max(4000, 32000 / warps) | 0.2 to 1.4 ms | schedule over iterations |
| `sfu-shfl` | iterations by warp count | 0.6 to 1.8 ms | schedule over iterations |
| `mma-chain`, `mma-softmax-chain` | steps, about 0.35 ms target | 0.1 to 0.8 ms | schedule over steps |
| `sync-loop`, `sync-loop-multicta` | 128 or 512 steps | 0.05 to 1.7 ms | schedule over steps |
| `sync-loop-reconcile` | adaptive to about 1 ms, unfrozen | about 1.0 ms | same knob, frozen through the schedule |
| `sync-loop-ablation` | 128 steps | 0.1 to 0.5 ms | schedule over steps (diagnostic, later) |
| `serial-fill` | 4096 steps | 0.5 to 2.1 ms | schedule over steps |
| `spill-latency` | steps, about 0.4 ms target | 0.06 to 4 ms | schedule over steps |
| `cp-async` | fixed | 1.5 to 1.9 ms | already over 1 ms; record lengths only |

## 4. Fields where a longer kernel is not the fix

- **Launch overheads** (`launch_overhead_us`, `per_cta_launch_latency_ns`, `host_launch_overhead_us`). They measure microsecond costs. The timed unit is a batch of 500 launches (a graph or a stream); the schedule raises the batch count instead (about 2500 brings it over 1 ms) and the per-launch value stays the batch time over the count. The per-CTA value stays a difference of two graphs of the same count.
- **In-kernel timed probes** (`latency` for the smem, L2 and DRAM latencies, `l1-carveout`, `stage-overhead` for `stage_sync_overhead_ns`, `fixed_sync_ns`, `min_async_roundtrip_ns`). They time inside the kernel with `%globaltimer` or `clock64`, so launch cost never enters the result. Their knob is the chase or loop length; the schedule raises it so the timed window is at least 1 ms, for timer resolution and to average over refresh and clock changes, and the per-step value stays the window over the steps.
- **`latency-curve`** (`dram_latency_by_bw`, `l2_latency_by_bw`). Same as above, and longer windows also stabilize the background bandwidth, which is measured over the chase (the RTX 4070 points under about 140 GB/s are under 1 ms now).
- **`l2-rescan`**. `pass2` means re-reading a working set right after one cold pass; more passes per launch would measure resident bandwidth instead. One sample becomes a group of K repeats of flush, pass 1, pass 2, each pass still one launch, with the per-pass values the medians over the group; K is chosen by the pilot so that the summed pass-2 time of a group is at least 1 ms.
- **GB10's L2 bistable band.** On GB10, working sets of 0.75 to 1.25 times the L2 size switch between two states from one process to the next, independent of the CPU ([v0.3-gb10-l2-bistability-observation.md](../results/analysis/v0.3-gb10-l2-bistability-observation.md)), and `l2-rescan` and the capacity sweep cross that band. Fixed before any capture, for these two modes and this band only: their points with working sets of 0.75 to 1.25 times `l2_size_bytes` do not count towards rule 3's decision for the capture (the points outside the band do, at 5% as usual); each in-band point stores the minimum and maximum of its per-process medians, takes the measured spread across processes as its `rel_uncertainty` (with the usual floor), and carries `"bistable_band": true`. No other field may use this exception. It applies only to devices with a documented bistability observation, currently GB10. On any other device (the RTX 4070 included) the capture reports rule 3 separately for the points inside and outside the band, and the band does not get the exception; a capture that fails rule 3 only because of in-band points is judged failed (suspected bistable), and the exception applies on that device only after a documented observation of at least 3 rounds on a quiet host and desktop, followed by a new capture.
- **GB10's low in-flight DRAM points.** On GB10, cold DRAM points with little in flight split into two clusters about 1.3 to 1.5 times apart (latency states of about 410 and 590 ns), in slow segments of fixed shape that do not follow the host CPU. In the five host-gated rounds of the full sweep they recurred at or near the same points (14 contiguous geometries had a slow segment at the exact point in at least 3 of 5 rounds); in the forward, forward, reverse replay with wall clock no geometry or wall-clock position was shared, so whether they follow the sweep order or the elapsed time is unresolved ([gb10-dram-lowbif-bimodality-posthoc.md](../results/analysis/gb10-dram-lowbif-bimodality-posthoc.md), [v0.3-gb10-mlp-wall-order-posthoc.md](../results/analysis/v0.3-gb10-mlp-wall-order-posthoc.md)). Fixed before any capture, for GB10 only: `dram_bw_by_bytes_in_flight` points with at most 256 KiB in flight, `dram_bw_by_bytes_in_flight_strided` points with at most 384 KiB (a future 512 KiB point is not covered), and `dram_bw_by_active_sms` points at 1 and 2 target SMs leave rules 1 and 3: they are removed from those rules' lists or points, and from their counts, before the check. Rule 2 still judges every list. Each such point stores the minimum and maximum of its per-round medians, takes a `rel_uncertainty` that covers both clusters, and carries `"low_in_flight_unstable": true`. No other field or device may use this exception. Both GB10 exceptions are registered per device and per field, with their evidence, in `microbench/capture_exceptions.py`, and apply only in `microbench/long_capture.py`, which writes their metadata; the earlier reductions (`microbench/repeat_mlp.py`) keep the rules without exceptions. These are the points decode-like kernels (few CTAs, little in flight, latency bound) land on; the core lists the two latency states as a known limitation.
- **`l2_effective_capacity_bytes`** (capacity sweep). The sweep's bandwidths are not stored in the profile, so its launch length cannot be audited; the capture records them and the schedule applies to the passes per launch.
- **`cta-dispatch`** is a dispatch trace, not a timed measurement; unchanged.

## 5. Profile format

- `MeasuredValue` gains an optional `launch` object: `{"length": integer, "min_ns": number, "median_ns": number}`, the frozen length and the shortest and median timed window of the point. A sample's timed window is one launch for most modes, a whole group of K flush, pass 1, pass 2 repeats for `l2-rescan`, and a whole batch for the launch overheads, so `min_ns` is the shortest such window. The JSON schema allows it, and an optional `bistable_band` flag on `l2-rescan` and capacity-sweep points; the Rust reader already ignores unknown fields.
- Every timed point is captured with `--clock-probe`: `launch` also carries optional `sm_mhz_before` and `sm_mhz_after`, the SM clock measured with `calibrate_clock_kernel` just before the point's first launch and just after its last (about 8 ms per point). Compute fields (`mma_tflops_per_sm_by_warps`, `fp32_fma_tflops`, the SFU and shuffle issue rates, the `gmem-issue` rates) also store an optional per-cycle value, the rate over the mean of the two clocks; memory fields are not converted, since DRAM and L2 do not follow the SM clock. On devices without locked clocks this separates clock differences between sessions (GB10: 2412 to 2566 MHz) from the hardware. The core does not read these values in this version; they are recorded only, and how to use them is designed once the new profiles exist.
- `scripts/check_profiles.py` checks, in a profile that claims the long-launch protocol, that every timed value has `launch.min_ns` of at least 1e6; it skips values whose `source` is `estimated` and derived values that are not timed themselves (`l2_effective_capacity_bytes`, the `l2_rescan` hit rates, `serial_stage_overhead_ns`).
- `provenance` gains an optional `capture_protocol` object: the minimum launch, the schedule digest per mode, the number of independent processes per mode, and each gate's result.
- `PROFILE_REVISION` moves to `0.23.0` for profiles captured this way. Profiles without `capture_protocol` keep reading as before.
- Implementation: `collect_profile.py --long-launch RUN_DIR` builds the profile from a passed `long_capture.py` run with the usual `build_profile` (whose sample lists are then the round medians, so a value is the median of round medians and its interval comes from them), and refuses unless every timed mode was captured that way. `microbench/profile_launch.py` then attaches to each value built from a sample list its `launch`, the range of its round medians (`round_median_range`) and, for compute rates, `per_cycle`; flags and widens the entries a device exception covers; and fails on a point without its profile entry or a value that is not the median of its list. SM clocks come from the probes. `scripts/check_profiles.py` applies the launch check to profiles with `capture_protocol`, skipping the derived values in `UNTIMED_LEAVES`.
- A mode added to the schedule registers, besides its `DESIGN_FIELDS` in `long_launch.py`: its name in `long_capture.MODES`, its raw-to-profile path in `profile_launch.PROFILE_PATH` (and any renamed sample lists in `METRIC`, compute rates in `COMPUTE`, a point that is one scalar in `SCALARS`, a timed point with no profile value in `NOT_IN_PROFILE`, a sample list that is not a profile value in `NOT_A_VALUE`, and its place in `OVERRIDE_ORDER` if `build_profile` merges it with other modes into one field, later overriding earlier), and any derived values in `check_profiles.UNTIMED_LEAVES`.

## 6. Capture gates in `collect_profile.py`

`collect_profile.py` runs the suite as 5 independent rounds. Each round passes the thermal gate and, where they apply, the host and desktop gates once, then runs every mode in turn, each as its own process. A field's value is the median over the 5 rounds of each round's median.

Per mode and round, at the end of its process: every launch at least 1.0 ms (checked at once by `long_launch.py`, not only by `check_profiles.py`), rules 1 and 2, and the host check.

- An interruption of the protocol before the mode's first timed launch (a gate that does not pass, a process that cannot start) does not count: the round restarts, and the restart does not count towards the limit below.
- A failure after timing has begun (a statistical or host check of that mode's process) makes that mode's round void. The void round is kept, its medians go into the audit with the void mark, and the capture adds a round; at most 2 extra rounds per capture across all modes, after which the capture fails.
- A launch found under 1.0 ms after the capture is a statistical failure after timing, not an interruption before it.
- Rounds are numbered consecutively, void ones included, and `capture_protocol` records every round, its result and the extra rounds used, with each mode's void rounds and their reasons. Every accepted round counts; no round is chosen or dropped by its values. After all rounds, rule 3 per mode across its 5 accepted rounds (with the L2 band exception of section 4). The gates:

- the three dispersion rules (`microbench/capture_posthoc.py`): at most 5% of sample lists with an interquartile range above 10% of the median, at most 15% with largest over smallest above 1.5, at most 5% of points whose per-process medians spread more than 10%. Each rule tolerates max(floor(limit x N), 1) failing lists or points, N being the mode's lists (rules 1 and 2) or points (rule 3) after exempted points are removed, so a single failing list or point never voids a small mode; for N of at least 20 (rules 1 and 3) or 7 (rule 2) this equals the share limit;
- on GPUs that share memory with the CPU (`integrated` in the device query, GB10): `microbench/host_quiet.py`, 10 quiet seconds before each round and no second during a mode's process with more than 10% of all CPUs busy or any foreign process above 5% of one CPU for two seconds;
- under WSL: the quiet Windows desktop gate (GPU utilization under 5% for 60 s, no game or streaming process) before each round.

`microbench/long_capture.py` implements this: `pilot` freezes every scheduled mode and writes a manifest binding the schedules, the freeze records, the binary and the protocol; `capture` runs the rounds (rerunning it resumes an interrupted round, gates first) and judges rule 3; `reduce` writes, per mode, a raw of the usual shape whose sample lists are the 5 round medians, with each timed point's `launch` (frozen length, shortest launch over the valid rounds, median of round medians, each round's SM clock). Until every timed mode runs under a schedule it produces these merged raws for diagnosis only. Still missing before the first `0.23.0` profile, besides the modes of section 7.2: `dram-bw` under a schedule emits only `read_curve` (no `write_full`), and `l2` stops after `read_curve` (no `read_curve_cp_async`, write bandwidth or capacity sweep). A profile at revision `0.23.0` comes only from frozen schedules; without one, `collect_profile.py` produces the old revision as now. A failed gate prints the report and exits with an error; the raw captures are kept. On machines with several GPUs, the compute-process and telemetry checks look at the target device only (by UUID). Where `nvidia-smi` or the `/proc` reads are not permitted, the capture stops with the permission it needs instead of skipping a gate. On a plain Linux machine with a discrete GPU only the dispersion rules and the thermal gate apply. Community profiles pass the same gates, so `docs/guides/contributing-a-profile.md` gains a section on them.

## 7. Order and cost

1. Fields the model reads directly: MLP and active-SM (done), `closed-loop`, `l2-merge`, `mma`, `gmem-issue` and `gmem-issue-controlled`, `sync-loop`, `closed-loop-mma`, `spill-latency`, `mma-chain`.
2. Then diagnostics: `sync-loop-ablation`, the spill diagnostics, `l1-carveout`, `stage-overhead`, the launch overheads, `l2-rescan`, the capacity sweep.
3. GB10 first, under the host gate; the RTX 4070 in a quiet desktop window.

About 2100 timed points per GPU at 15 samples, 3 warmups and about 1.5 ms per launch is about 1 minute of kernel time per round, plus about 15 s of DRAM flushes and the group and batch overheads; with 5 rounds, each with one thermal and quiet gate, and the pilot, about 30 to 45 minutes per GPU. Gating every mode's process separately would add about two hours of waiting, which the round structure avoids.

Implementation goes in small commits, one per mode, each with a test, in the order of section 7.1.

## 8. Not in this design

- Re-checking the section 6.4 errors on the new profiles (core owner, after the profiles exist).
- Changing any measured definition; a field that cannot be captured at 1 ms or more without changing it is listed in section 4 instead.
