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
3. What a field measures does not change. Where more work per launch would change it, the fix is a repeat of the whole measured unit (section 4).
4. Each point records its length and its launch durations.
5. A capture that fails a gate (section 6) writes no profile.
6. Old profiles stay valid; new fields are optional.

## 3. One schedule for every timed mode

`microbench/src/long_read.cuh` (`LongReadSchedule`, written for `mlp`, `strided-mlp`, `dram-bw` and `l2`) already implements rules 1, 2 and 4: in pilot mode it scales a point's length until the fastest of three samples reaches 1.25 ms (at most eight trials, within address bounds) and prints every trial; with a schedule file it runs each point at its frozen length; either way it prints `steps`, `transfer_bytes`, `source_buffer_bytes` and `launch_ns_samples`. `microbench/long_mlp.py` freezes the schedule and checks it at capture.

The design generalizes it rather than writing one protocol per mode:

- Rename `LongReadSchedule` to `LaunchSchedule` (keep the old name as an alias) and drop the read-sweep restriction in `vgpu_bench.cu`; the flag becomes `--launch-schedule pilot|FILE` (`--long-read-schedule` stays accepted).
- Each `run_*` routes its timed loop through `schedule->measure(initial, minimum, maximum, ...)` with a `launch(length)` callback. `length` is the mode's existing work knob (table below). Without a schedule a mode behaves exactly as now, so old commands reproduce old captures.
- Modes whose length is bounded by a buffer (cold DRAM reads must not wrap) size their buffers from free device memory, 1 GiB at least and up to 4 GiB (1 GiB lasts only 0.5 to 0.7 ms at 1.5 to 2 TB/s); `LaunchSchedule` refuses a length that would wrap. Steady-state streams (`closed-loop`, `sync-loop`) may wrap as long as the buffer is at least eight times the L2, so a re-read still comes from DRAM.
- The two changes of #12 stay: `run_dram_bw` sizes its buffer as max(buffer, 16 x L2), and `run_l2` keeps `read_curve_cp_async`.
- `long_mlp.py` becomes `microbench/long_launch.py` with the same pilot, freeze, capture and audit steps for any mode; `collect_profile.py` calls it.

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
- **GB10's L2 bistable band.** On GB10, working sets of 0.75 to 1.25 times the L2 size switch between two states from one process to the next, independent of the CPU ([v0.3-gb10-l2-bistability-observation.md](../results/analysis/v0.3-gb10-l2-bistability-observation.md)), and `l2-rescan` and the capacity sweep cross that band. Fixed before any capture, for these two modes and this band only: their points with working sets of 0.75 to 1.25 times `l2_size_bytes` do not count towards rule 3's decision for the capture (the points outside the band do, at 5% as usual); each in-band point stores the minimum and maximum of its per-process medians, takes the measured spread across processes as its `rel_uncertainty` (with the usual floor), and carries `"bistable_band": true`. No other field may use this exception. Whether the RTX 4070 has such a band is unknown; the same rule applies and the capture reports what it finds.
- **`l2_effective_capacity_bytes`** (capacity sweep). The sweep's bandwidths are not stored in the profile, so its launch length cannot be audited; the capture records them and the schedule applies to the passes per launch.
- **`cta-dispatch`** is a dispatch trace, not a timed measurement; unchanged.

## 5. Profile format

- `MeasuredValue` gains an optional `launch` object: `{"length": integer, "min_ns": number, "median_ns": number}`, the frozen length and the shortest and median timed window of the point. A sample's timed window is one launch for most modes, a whole group of K flush, pass 1, pass 2 repeats for `l2-rescan`, and a whole batch for the launch overheads, so `min_ns` is the shortest such window. The JSON schema allows it, and an optional `bistable_band` flag on `l2-rescan` and capacity-sweep points; the Rust reader already ignores unknown fields.
- `scripts/check_profiles.py` checks, in a profile that claims the long-launch protocol, that every timed value has `launch.min_ns` of at least 1e6; it skips values whose `source` is `estimated` and derived values that are not timed themselves (`l2_effective_capacity_bytes`, the `l2_rescan` hit rates, `serial_stage_overhead_ns`).
- `provenance` gains an optional `capture_protocol` object: the minimum launch, the schedule digest per mode, the number of independent processes per mode, and each gate's result.
- `PROFILE_REVISION` moves to `0.23.0` for profiles captured this way. Profiles without `capture_protocol` keep reading as before.

## 6. Capture gates in `collect_profile.py`

`collect_profile.py` runs the suite as 5 independent rounds. Each round passes the thermal gate and, where they apply, the host and desktop gates once, then runs every mode in turn, each as its own process. A field's value is the median over the 5 rounds of each round's median.

Per mode and round, at the end of its process: every launch at least 1.0 ms (checked at once by `long_launch.py`, not only by `check_profiles.py`), rules 1 and 2, and the host check. A mode that fails in a round is void for that round only and runs again in an extra round; at most 2 extra rounds per capture, after which the capture fails. After all rounds, rule 3 per mode across its 5 accepted rounds (with the L2 band exception of section 4). The gates:

- the three dispersion rules (`microbench/capture_posthoc.py`): at most 5% of sample lists with an interquartile range above 10% of the median, at most 15% with largest over smallest above 1.5, at most 5% of points whose per-process medians spread more than 10%;
- on GPUs that share memory with the CPU (`integrated` in the device query, GB10): `microbench/host_quiet.py`, 10 quiet seconds before each round and no second during a mode's process with more than 10% of all CPUs busy or any foreign process above 5% of one CPU for two seconds;
- under WSL: the quiet Windows desktop gate (GPU utilization under 5% for 60 s, no game or streaming process) before each round.

A profile at revision `0.23.0` comes only from frozen schedules; without one, `collect_profile.py` produces the old revision as now. A failed gate prints the report and exits with an error; the raw captures are kept. On machines with several GPUs, the compute-process and telemetry checks look at the target device only (by UUID). Where `nvidia-smi` or the `/proc` reads are not permitted, the capture stops with the permission it needs instead of skipping a gate. On a plain Linux machine with a discrete GPU only the dispersion rules and the thermal gate apply. Community profiles pass the same gates, so `docs/guides/contributing-a-profile.md` gains a section on them.

## 7. Order and cost

1. Fields the model reads directly: MLP and active-SM (done), `closed-loop`, `l2-merge`, `mma`, `gmem-issue` and `gmem-issue-controlled`, `sync-loop`, `closed-loop-mma`, `spill-latency`, `mma-chain`.
2. Then diagnostics: `sync-loop-ablation`, the spill diagnostics, `l1-carveout`, `stage-overhead`, the launch overheads, `l2-rescan`, the capacity sweep.
3. GB10 first, under the host gate; the RTX 4070 in a quiet desktop window.

About 2100 timed points per GPU at 15 samples, 3 warmups and about 1.5 ms per launch is about 1 minute of kernel time per round, plus about 15 s of DRAM flushes and the group and batch overheads; with 5 rounds, each with one thermal and quiet gate, and the pilot, about 30 to 45 minutes per GPU. Gating every mode's process separately would add about two hours of waiting, which the round structure avoids.

Implementation goes in small commits, one per mode, each with a test, in the order of section 7.1.

## 8. Not in this design

- Re-checking the section 6.4 errors on the new profiles (core owner, after the profiles exist).
- Changing any measured definition; a field that cannot be captured at 1 ms or more without changing it is listed in section 4 instead.
