# VirtualGPU design

What VirtualGPU is for, the ideas it rests on and where it stops. How each part works today is in [`docs/components/`](components/):

| Component | Doc |
| --- | --- |
| Triton frontend | [frontend.md](components/frontend.md) |
| Kernel IR and L2 reuse model | [kernel-ir.md](components/kernel-ir.md) |
| Performance model core | [model-core.md](components/model-core.md) |
| Microbenchmarks and GPU profiles | [profiles.md](components/profiles.md) |
| GPU-free autotuning | [autotune.md](components/autotune.md) |
| Validation | [validation.md](components/validation.md) |

What comes next is in the [roadmap](../ROADMAP.md).

## Decisions

- **Scope:** an open-source performance oracle for GPU kernels: which variant is faster on a given GPU, by roughly how much, and why. The first application is GPU-free autotuning for Triton; the others are in section 4.
- **Relative, not absolute:** the product is a ranking with confidence, not a predicted time.
- **One neutral Kernel IR:** the IR is independent of the source language and of the GPU vendor. Triton is the first frontend; a PTX frontend follows and covers CUDA, CUTLASS, TileLang and other languages that compile to PTX. AMD follows through its public ISA.
- **Implementation:** the model core (scheduler, L2 model, perturbation) is written in Rust and exposed to Python through PyO3. Frontends are written in Python. The microbenchmark suite is written in CUDA C++ to control the exact instructions it measures.
- **Test GPUs:** RTX 4070 (Ada, `sm_89`) for development; DGX Spark (GB10, `sm_121`) as a held-out GPU for the cross-GPU experiment.
- **Verdict rule:** A is faster than B only if A is at least 5% faster in every perturbation draw (|log(t_A/t_B)| >= log(1.05) with the same sign in all draws). Anything else is "indistinguishable". The 5% band is the same practical band used for hardware pairs; it is fixed, not tuned. Profile perturbation cannot cover structural model error, so a draw-only rule orders pairs that differ only through modeled structure.

## 1. Problem

Inference optimization is bottlenecked on GPUs. Kernel developers need hardware to tune kernels, the hardware they target is often not the hardware they own, and every hour spent benchmarking competes with training and serving for the same scarce compute.

VirtualGPU models a GPU architecture on the CPU so that kernel variants can be compared without the target GPU.

## 2. Core idea: relative, not absolute

VirtualGPU does not try to reproduce the absolute speed of a GPU. It answers one question:

> Given kernel variants A and B and GPU X, is A faster than B, by roughly how much, and why?

This is much easier than predicting absolute time:

- Systematic errors cancel. If the model underestimates memory bandwidth by 20%, every variant gets slower by the same factor, and the ranking is unchanged.
- No cycle-exact microarchitecture is needed, so no reverse engineering is needed. The model uses public specifications, generic architectural structure, and measured parameters.

The project works if this holds: **when VirtualGPU reports a difference between A and B, the same difference exists on the real GPU.**

## 3. Goals and non-goals

Goals:

- Rank kernel variants on a target GPU in seconds on a CPU.
- Report when two variants are indistinguishable instead of forcing an order.
- Explain each prediction: which resource bounds the kernel and why.
- Support GPUs the user does not own, through community-contributed GPU profiles.
- One Kernel IR for every language and vendor: Triton and PTX for NVIDIA, Triton for AMD and Ascend.
- Fast and deterministic enough to sit inside automated loops: CI checks, kernel search and AI agents.

Non-goals:

- Cycle-accurate timing.
- Functional simulation (computing kernel outputs). Inference kernel runtimes rarely depend on data values; see section 9 for exceptions.
- Replacing real hardware for final validation.
- Reverse engineering vendor drivers, firmware, or toolchains.

## 4. Use cases

Ordered by difficulty.

1. **Config ranking.** Rank the autotuning configs of one kernel (tile sizes, `num_warps`, `num_stages`). Systematic errors cancel almost entirely. This is the first target.
2. **Algorithm comparison.** Compare different implementations of the same operator, such as split-K versus stream-K GEMM. Variants may be bound by different resources, so per-resource scales must be roughly right.
3. **Bottleneck finding in a full model.** Find which operators in an end-to-end model are far from their achievable speed. This compares unrelated operators (GEMM versus normalization versus attention), so each operator class needs coarse absolute calibration.
4. **Regression checks.** Predict whether a code change makes a kernel slower on a set of GPUs the CI machine does not have. This is use case 1 or 2 applied to the old and new version.
5. **Scoring for kernel search.** Rank many candidate kernels from a search or an AI agent, so only the best few are measured. Needs high throughput, deterministic output and a reason per verdict.
6. **Hardware choice.** Rank GPUs for a set of kernels. Like use case 3 it compares across profiles, so it needs coarse absolute calibration.

The most direct application of use case 1 is **GPU-free autotuning**: a drop-in replacement for the measurement step of `@triton.autotune` that ranks configs against a chosen GPU profile. When each use case ships is in the [roadmap](../ROADMAP.md).

## 5. Input: one neutral IR, Triton first

Kernels are written in many languages: CUDA C++, CUTLASS and CuTe, Triton, TileLang, HIP, and more. VirtualGPU does not support languages one by one. Every frontend lowers into one Kernel IR, which is independent of language and vendor, and the performance model only reads that IR.

```
Triton   CUDA C++   CUTLASS   TileLang   ...        HIP
   |          \         |        /                   |
   |               PTX (public)                AMD GPU ISA (public)
   |                     \                        /
   +----------------->  Kernel IR (neutral)  <---+
                               |
                       Performance model
```

| Input | Pros | Cons |
|---|---|---|
| Triton IR (TTGIR) | Tiles, layouts, `num_warps`, `num_stages` are explicit; same frontend for NVIDIA and AMD; native autotuning workflow | Only covers Triton kernels |
| PTX | Public specification, covers any CUDA source | Tile structure must be recovered; misses `ptxas` decisions |
| Vendor machine code | Closest to what runs on hardware | NVIDIA machine code is undocumented |

VirtualGPU starts with a Triton frontend that reads TTGIR after the software pipelining pass, because the tile structure the performance model needs is explicit there, and Triton is where autotuning already happens. A PTX frontend comes later and recovers the same structure from public PTX, which covers every language that compiles to it.

Closed-source libraries such as cuBLAS and cuDNN often ship machine code without PTX. VirtualGPU does not model their internals; where they matter, they are treated as measured black boxes.

Register usage and spills are decided by the backend assembler, not by TTGIR. Triton can compile ahead of time for a given target without a GPU, and the vendor assembler reports per-kernel resource usage. VirtualGPU uses that report as an input.

## 6. Performance model

The model is tile-level and event-driven. The input is the kernel launch grid and per-CTA (thread block) behavior. The output is predicted time with a breakdown by resource.

### 6.1 Mechanisms

Autotuning configs usually have similar arithmetic intensity. What separates them is the list below, and each item must be modeled:

| Mechanism | Driven by | Effect |
|---|---|---|
| Arithmetic intensity / roofline | tile shape, dtype | Compute-bound or memory-bound baseline |
| Wave quantization | grid size, SM count, occupancy | Tail waves leave SMs idle |
| Occupancy | registers, shared memory, warps per CTA | CTAs resident per SM, latency hiding |
| Pipeline overlap | `num_stages`, async copy / TMA | Whether loads hide behind compute |
| L2 reuse | tile traversal order (swizzle / grouping), L2 size | DRAM traffic |
| Tensor core shape fit | tile shape versus MMA instruction shape | Wasted MMA throughput |
| Launch and prologue/epilogue cost | kernel size | Matters for small shapes |
| Register spills | assembler spill report, spill location | Extra local memory traffic per iteration or once |
| Persistent kernels and warp specialization | outer tile loop, producer/consumer warps | One CTA handles many tiles; loads and compute run in separate warps |

Items such as shared memory bank conflicts and instruction scheduling are out of scope for the first model. Variants that differ only in those will be reported as indistinguishable, which is the correct answer for a model that does not contain them.

### 6.2 Structure

- Per CTA: phases (prologue, steady-state loop, drain, epilogue). Stage counts come from the IR after pipelining, never from the requested `num_stages`. How a phase's time is computed is in 6.4.
- Per SM: resident CTAs share the SM's pipes (tensor core, issue, shared memory, local memory, special-function, shuffle, and memory instruction issue; 6.4).
- Per GPU: an event-driven scheduler dispatches CTAs to SMs using the dispatch policy measured on the real GPU, and produces waves and tails naturally. L2 and DRAM bandwidth are shared among active phases by max-min fair sharing, using bandwidth curves measured against the number of active SMs.
- L2: an online model, called by the scheduler at tile boundaries and once per phase segment, splits traffic into L2 hits and DRAM bytes from the actual set of resident tiles.

Events are per CTA phase, not per instruction or per loop iteration. The target is about 1 ms per simulation of a 4096-tile GEMM, so that a full autotuning space with perturbation sampling finishes in seconds.

### 6.2.1 Resolution through perturbation

When variants are bound by different resources, profile errors do not cancel. Every measured profile field therefore carries an uncertainty. The model runs each variant against draw 0 (the nominal profile) and 32 perturbed draws (Latin hypercube within each field's uncertainty, correlated fields drawn together). The dispatch policy is not drawn: the core dispatches round robin and flags `dispatch_ambiguous` when the profile lists more than one candidate policy or less than high confidence. All variants share the same draws, so comparisons are paired. A pair is ordered only if every draw agrees on the direction and on a difference of at least 5%; otherwise it is indistinguishable.

### 6.3 Output

- Predicted time per variant.
- Pairwise verdicts (faster, slower, indistinguishable). Verdicts are kept per pair; there is no hard grouping, because "indistinguishable" is not transitive.
- `unsupported` with a reason, when a kernel uses a feature the model does not handle.
- Per variant: the share of time each cause bounds the kernel (`bound_fraction`, 6.4), occupancy and its limiter, wave count and tail efficiency, and load, L2 hit, merged and DRAM bytes.

### 6.4 Mechanisms in the core

What the core computes in v0.2. Each mechanism was checked on its own microbenchmark before any development-set comparison; the field names are those of the GPU profile. Errors are over the microbenchmark's points unless stated, and none of those points set a parameter except where a calibration point is named.

**Phase time.** Each phase of a CTA runs at the slowest of three paths: the round-trip path (launch, L2 and DRAM round trips, and the compute that cannot overlap them), the SM's shared compute path (each pipe's time with the SM's resident CTAs sharing it, plus synchronization), and the CTA's L2 and DRAM bandwidth shares. Rates change only at events: CTA phase boundaries. A phase with less than 1e-3 of its work left after an event finishes in that event, and at one timestamp all tile ends are applied before all tile starts, then the demands.

**Bandwidth.** L2 and DRAM bandwidth are shared max-min fairly among active phases, against `l2_bw_by_active_sms` and `dram_bw_by_active_sms`. The L2 curve was remeasured with eight loads in flight per thread and 16 warps (`cuda_l2_cg_read_warm_ilp8_16w`, profile `0.16.0`) because the earlier one was latency bound at few SMs.

**Latency from bytes in flight.** A level's load latency is Little's law on `l2_bw_by_bytes_in_flight` and `dram_bw_by_bytes_in_flight`: the bytes all CTAs keep in flight there over the bandwidth the curve gives for that many, with the curve of the kernel's issue mode (`cp_async` or `ld_global_v4`). Bytes in flight are delivered bandwidth times latency, so the DRAM latency that reproduces itself is found by bisection (12 steps). Without the curves the latency moves linearly from idle to loaded with utilization.

**Pipelined loop as a closed loop.** A CTA with prefetch distance `d` (from the PTX `cp.async.wait_group`) cycles `d + 1` stage buffers: a stage arrives a round trip after its load, is computed, and only then is its buffer reloaded. Where loads and compute both exist, the stage buffers of all CTAs on an SM are a closed queueing network, one server (the SM's pipes, a stage's compute alone plus its synchronization) and one delay (the round trip), solved by exact mean value analysis. Basis `dram_closed_loop`, `l2_closed_loop` (720 points, predicted from the open-loop MLP curves alone): median error 2.6% DRAM and 6.8% L2, p90 7.3% and 16.6%. `dram_closed_loop_mma` (MMA contention, 150 points): median 1.0% to 1.9%, at most 16%. Synchronous loads are `d = 0`; while one CTA waits, the others on its SM use the pipes it leaves idle.

**Cross-wave reuse.** A chunk read again after its last tile ended hits per K slice under LRU with capacity `l2_effective_capacity_bytes`: chunks ending in one event streamed side by side, so a later wave that walks K the same way starts on the slices evicted first. The evidence is GB10 microbenchmarks whose direction agrees but which do not pass the capture rules (9.3): in all seven runs of three versions, forward re-reads at 1.5 times L2 or more hit -0.01 to 0.09, where the earlier per-chunk clock charged 0.5 to 0.9. The per-slice detail for reversed K walks is provisional and no supported kernel walks K backwards. Whether the change counts is decided by the sealed v0.3 holdout, not by these microbenchmarks.

**Miss merging.** The L2 model splits loads into resident hits (L2 latency and bandwidth), requests merged into a fill another CTA has in flight (DRAM latency, L2 bandwidth only), and fetches (DRAM latency and bandwidth). Basis `l2_merge_by_sharing` (42 points): stage turnaround median error 5.0% (p90 16.8%).

**Round-trip port floor.** A round trip takes at least the level's idle latency plus its own bytes over the L2 ceiling at one active SM, and at least the bytes of all the SM's resident CTAs over that ceiling, since their round trips run out of phase. The MLP curves are indexed by bytes in flight over the whole GPU and do not see one SM receiving them all. Basis `serial_stage_fill` (one CTA, warm L2, 2 to 16 KiB per stage): MAPE 17.4% to 9.2% on the RTX 4070 and 15.8% to 8.6% on GB10 (64 points each). On `sync_stage_loop` as remeasured in profile `0.21.0` (564 points each, 1 to 8 CTAs per SM, 2 to 32 KiB): MAPE 4.4% (RTX 4070) and 7.1% (GB10), against 4.7% and 7.8% when every round trip waits for all the SM's bytes after the latency.

**Per-stage synchronization.** Each steady-state iteration adds `fixed_sync_ns` (commit, `wait_group` and `bar.sync` with no data, by warps and stage buffers: 12.5 to 31.5 ns on the RTX 4070) to the SM's shared compute once per resident CTA, and to a CTA's dependent chain. Not to the round trip, whose MLP latency already includes it. Basis: `dram_closed_loop_mma`, where two or three CTAs per SM with 2 us of compute per stage exceed the compute by 14 to 18 ns per stage.

**A CTA's dependent chain.** An iteration's non-MMA work (softmax between attention's two MMAs, `ldmatrix` before the MMAs) depends on its MMAs or they on it, so one CTA alone takes its MMA time plus its longest other pipe. This chain replaces the longest single pipe on the round-trip path and in the queueing network's service time, and is a floor under the shared compute path, which still takes the maximum over pipes shared by the SM's CTAs. Basis `mma_softmax_chain` (72 points, 1 to 3 CTAs per SM): with the measured solo times the form's median error is -0.7%, -7.9% and +3.8% at 1, 2 and 3 CTAs per SM, against -30%, -8% and +4% for the maximum alone. The `ldmatrix` and MMA chain intercepts (`serial_stage_overhead_ns`, 4 to 60 ns per stage) are within about 10 ns of the chain's shared memory time.

**Pipes.** Per SM, with the SM's CTAs sharing each:

| Pipe | Charged | Rate from | Microbenchmark error |
|---|---|---|---|
| `mma` | MMA FLOP | `mma_tflops_per_sm_by_warps`, by MMA-issuing warps on the SM | measured curve, interpolated between points |
| `issue` | every non-MMA instruction | `non_mma_issue_rate_per_sm_gips`, the highest non-MMA rate measured on the card: pure FFMA (`mma`, `fp32_fma`) or the peak over warps of a 1:1 FFMA and LOP3 mix (`sfu-shfl`), whichever is higher. It is a lower bound on the issue ceiling. Bundled profiles still carry the 4 per clock estimate until re-collected. On the RTX 3070 pure FFMA wins, about 3.6 per clock against 3.2 for the mix ([post-hoc](results/analysis/non-mma-issue-rtx3070-posthoc.md)) | not checked on kernels |
| `smem` | shared memory bytes, loads and stores | `smem_bw_per_sm_gbps` | not checked on its own |
| `local` | all spill bytes | `local_mem_l1_hit_bw_per_sm_gbps` (estimated) | not measured |
| `sfu` | `ex2`-class instructions | `sfu_issue_rate_per_sm_gips`, by warps | measured points exact; leave-one-out -1% to -12% at 8 and 16 warps, -24% to -28% at the 4-warp knee |
| `shfl` | butterfly shuffles | `shfl_issue_rate_per_sm_gips`, by warps | within 0.4% at 8 and 16 warps leave-one-out, -24% to -28% at the knee |
| `lsu` | global load and store warp instructions, bytes over 32 times the load width | peak of `gmem_load_issue_controlled` at its narrowest width (0.952 / 0.859 Gwarp-inst/ns) | one calibration point; held-out MAPE 18.0% / 13.1% (35 points each), the rest of it from the benchmark's unsynchronized warps |

`sfu` and `shfl` instructions also take issue slots. `lsu` bounds the shared path only: issuing loads overlaps the wait for those in flight. Without a measured curve `sfu` falls back to `sfu_tflops` and `shfl` to the issue rate, and kernels that use them are flagged `unvalidated_feature`.

**Register spills.** Spill bytes (loads plus stores per thread, from the assembler report) are charged once or per iteration by the IR's spill scope; `inner_loop_assumed` scales by `assumed_spill_scale` (1.0). The share of spill traffic that leaves L1 is the measured L1 miss rate of the stack frames of the threads an SM actually holds (the CTAs of the busiest SM in the first wave, so a grid under one wave is not charged for full residency) (`l1_to_l2_spill_threshold_by_carveout`), taken from the curve of the smallest measured carveout at or above the kernel's carveout (the largest if none is). Each curve is a step: it rises from 0 to 1 within about 40 KB of the L1 capacity left by that carveout, which on both GPUs is about 128 KB at zero carveout, 105 KB at 16 KB, 74 KB at 32 and 48 KB, and 28 KB at 64 KB or more. A profile without the curves gets a hard step at `l1_to_l2_spill_threshold_bytes_per_sm`. The part that leaves L1 is split between L2 and DRAM by how much of the working set fits the effective L2. All spill traffic takes the `local` pipe, the part that leaves L1 too, since local accesses pass the SM's L1 and LSU wherever they are served. Reloads also cost latency on the CTA's dependent chain: the extra time per iteration measured for batched reloads (`spill_reload_latency_by_count`, `batch_before_dep`, since spill reloads use independent addresses), interpolated over the reload count and the frames' footprint per SM, mixed between the L1 hit and L2 points by the frames' L1 miss rate, and past the largest measured count at its average cost per reload. It adds to the chain, to the compute that cannot overlap a round trip, and to the delay between stages in the queueing network, not to a shared pipe, so other CTAs on the SM can still issue meanwhile; a phase whose chain is mostly reloads is bound by `local`. Since the measured latency already includes issuing the reloads, only the spill stores' share of the local bytes stays on the chain as `local` pipe time when it is charged; the shared `local` pipe still carries all of them. The reload count is the assembler's spill load bytes over 4 B, a static figure (9.2). The reload latency is provisional until executed reload counts are available and `spill_reload_latency_by_count` is recaptured with long launches: 51 to 52 of its 60 points come from launches under 1 ms. The rule for that recapture is registered in `docs/results/analysis/v0.3-spill-decision-registration.md`. Basis: the 4 `l1_boundary` points (128 KiB of frames, left out of the model) are predicted with MAPE 12.6% on the RTX 4070 and 13.1% on GB10, at -19% to +17%; leaving out 32 reloads and interpolating from 8 and 68 gives -2.7% to +0.4% on the RTX 4070 and -0.7% to +18.6% on GB10. The measured points use 128 threads per SM; how the latency changes with more resident warps is not measured. Through the whole core on the current short-launch profiles (`scripts/recheck_mechanisms.py`, not evidence), counting only the CTAs an SM holds moves the signed median error of the extra time per step on these points from +210% to +68% on GB10 and from +71% to +38% on the RTX 4070; the rest is mostly the estimated `local` pipe.

**Per-tile iterations.** Pipeline trip counts can vary per program and tile (causal attention); each tile's phases use its own count.

**Not supported.** Warp-specialized kernels (producer and consumer warp groups), thread block clusters, and epilogues that overlap the next tile are reported `unsupported`, though 6.1 lists warp specialization among what must be modeled.

**Bottleneck attribution.** `bound_fraction` gives the share of CTA time during which each of 13 causes bound the CTA's current phase: `mma`, `issue`, `smem`, `local`, `sfu`, `shfl`, `lsu`, `sync`, `l2_latency`, `dram_latency`, `l2_bandwidth`, `dram_bandwidth`, `launch`. The cause is the largest component of the slowest path; bandwidth counts only when it binds, since an unconstrained share equals the demand the other paths set.

## 7. GPU profiles

A GPU is described by a JSON profile: one architecture model, one profile per SKU.

A profile has two kinds of fields:

- **Specification fields** from public vendor documentation: SM/CU count, clocks, register file and shared memory per SM, L2 size, memory bandwidth, peak throughput per dtype.
- **Measured fields** from the VirtualGPU microbenchmark suite: achievable DRAM and L2 bandwidth, shared memory bandwidth, achievable tensor core throughput per dtype and instruction shape, load latencies, launch overhead.

Every profile records its provenance: GPU model, driver version, clocks, and the version of the microbenchmark suite.

### 7.1 Community calibration

"No GPU needed" holds for users, not for the project. Each architecture must be measured on real hardware at least once. This is where the community scales the project: anyone with a GPU runs the microbenchmark suite and submits the resulting profile. Everyone else can then optimize for that GPU.

The microbenchmark suite uses only public APIs and our own kernels.

GPUs that are announced but not shipping can only get specification-derived profiles. Their predictions should be read as trends and rankings, not numbers.

## 8. Validation

### 8.1 First experiment

This experiment decides whether the approach works before anything else is built:

1. Take a Triton GEMM with a realistic autotuning space (tens of configs).
2. Run all configs over a set of LLM-relevant shapes on a real GPU and record times.
3. Run the same configs through VirtualGPU with a profile for that GPU.
4. Compare.

Metrics:

- **Kendall τ** between predicted and measured order, per shape.
- **Top-k hit rate**: whether the truly fastest config is in VirtualGPU's top k.
- **Top-k regret**: how much slower the best config in VirtualGPU's top k is than the true best.
- **Pair accuracy**: on hardware, a pair is equivalent if its 95% interval lies within ±5%, ordered if it lies wholly outside, unresolved otherwise. Model verdicts are scored against resolved pairs.
- Absolute error (MAPE) is recorded but is not a pass criterion.

VirtualGPU must beat two independent baselines built by the validation work, not by the model authors: roofline with wave quantization, and a NeuSight-style learned tile predictor. If a simple formula ranks as well, the event-driven model has no reason to exist.

Pass criteria, revised and fixed on 2026-11-25, after the development baseline scores and before any model score or holdout timing was seen:

- **Ranking:** median Kendall τ-b at least 0.05 higher than each baseline.
- **Shortlist:** mean top-5 regret at most 1% absolute, and no more than 0.5 percentage points worse than any baseline (non-inferiority).

The original shortlist criterion (1 percentage point lower regret than each baseline) was infeasible: the NeuSight-style baseline already reaches 0.46% (warm) and 1.09% (cold) mean top-5 regret on the development set, measured on the same GPU it was trained on.

The same criteria apply to the cross-GPU experiment (8.2), where the learned baseline is trained only on RTX 4070 labels and VirtualGPU only receives the target GPU's profile. That experiment is the main test: a learned baseline needs labels from some GPU, while VirtualGPU needs none from any GPU.

Hardware measurements use locked clocks where possible, explicit cold or warm L2 preparation, randomized repeated blocks across sessions, and 95% intervals. Calibration data (microbenchmarks), development data, and holdout data are kept strictly separate.

### 8.2 Cross-GPU experiment

The real value is predicting GPUs the user does not own. The second experiment repeats 8.1 on a GPU whose measurements were not used in any model decisions, using only its profile. The model and baselines are frozen on the RTX 4070 before any GB10 GEMM timing is opened.

### 8.3 Dataset

The measured data is a deliverable in its own right: kernel source, config, shape, GPU, driver, clocks, and time. Few public datasets of this kind exist.

## 9. Known limits

- **Data-dependent kernels.** MoE routing, sparse attention, and paged KV cache depend on runtime data. They need a workload description (expert distribution, sparsity pattern, page table) as an extra input.
- **Non-Triton kernels.** Not covered until the PTX frontend exists. Closed-source libraries without PTX are treated as measured black boxes.
- **Hardware coverage.** Sealed holdouts cover the RTX 4070 (`sm_89`), GB10 (`sm_121`), and Huawei Ascend 910B3 (`ascend910b3`). Neither NVIDIA GPU has Hopper `wgmma` or data-center Blackwell `tcgen05`. On Ascend 910B3, the first formal sealed holdout ([`docs/results/ascend910b3-holdout.md`](results/ascend910b3-holdout.md)) passed both Attention cohorts (`warm` and `cold`, `0.00%` top-5 regret) but failed GEMM, so every Ascend prediction continues to carry `unvalidated_feature` and `unvalidated_target_ascend`. Kernels using uncalibrated paths carry an "unvalidated feature" flag until such hardware is validated.
- **Fine-grained effects.** Bank conflicts, instruction scheduling, power and clock throttling are not modeled.
- **Model resolution.** Differences smaller than the model's error cannot be resolved; real hardware decides those.

### 9.1 Phase 1 residuals on the RTX 4070

Measured with core `18af028`, profile `0.11.0-m3`, development set only; the holdout set was not opened.

**Serial large tiles.** Configs with `BLOCK_K = 64`, a 128-row tile, and no load overlap in the loop (prefetch distance 0, which includes `num_stages = 2`) have errors whose sign depends on the warp count. Warm, `decode-16`:

| Config | Prefetch distance | CTAs per SM | Measured (us) | Predicted (us) | Error |
|---|---|---|---|---|---|
| `m128n128k64-w4-s1-g1` | 0 | 2 | 213.5 | 197.2 | -7.6% |
| `m128n128k64-w4-s2-g1` | 0 | 2 | 211.5 | 196.7 | -7.0% |
| `m128n128k64-w8-s1-g1` | 0 | 2 | 242.7 | 197.2 | -18.7% |
| `m128n128k64-w8-s2-g1` | 0 | 1 | 226.3 | 233.7 | +3.3% |
| `m128n64k64-w4-s1-g1` | 0 | 3 | 156.7 | 154.3 | -1.5% |
| `m128n64k64-w4-s2-g1` | 0 | 3 | 165.9 | 154.3 | -7.0% |
| `m128n64k64-w8-s1-g1` | 0 | 2 | 191.0 | 213.8 | +11.9% |
| `m128n64k64-w8-s2-g1` | 0 | 2 | 176.1 | 212.5 | +20.6% |

No constant per-stage correction fits both signs. Explanations ruled out, each by a microbenchmark that does not use GEMM timings:

- **MMA chain fill and drain.** Dependent `mma.sync` chains have an intercept of 3 to 5 ns per stage (`mma_chain_latency`).
- **Shared memory to register latency.** The `ldmatrix` intercept is 7 to 45 ns per stage (`serial_stage_overhead_ns`), under 10% of the 180 to 830 ns per iteration these configs missed before the queueing model.
- **Serial shared memory fill.** One CTA fills shared memory at about 80 B/ns (`serial_stage_fill`, linear within 7% on held-out points), but adding stage bytes over that rate to serial stages overshoots the remaining error by 2 to 6 times, because the MLP curves already include part of it.
- **Tensor, issue, shared memory port, and scheduler contention between co-resident CTAs.** In the MMA-contention closed loop (`dram_closed_loop_mma`, 150 points, 4 and 8 warps, 0 to 32 KB of `ldmatrix` per stage), two CTAs per SM are never slower than one, and the model's queueing network predicts delivered bandwidth with median error 1.0% to 1.9% and at most 16%.
- **Register spills.** `ptxas` reports no spills for any of these configs.

Not yet explained: configs with synchronous loads (`num_stages = 1`, `ld.global` into registers) have two to four times the error of the others in most cases. Median absolute error, core `18af028`:

| Case | Warm: synchronous / other | Cold: synchronous / other |
|---|---|---|
| decode-1 | 13.1% / 4.2% | 4.2% / 2.4% |
| decode-16 | 1.7% / 3.0% | 1.0% / 2.0% |
| decode-32 | 6.0% / 0.9% | 5.3% / 1.5% |
| mask-m | 12.2% / 6.8% | 10.6% / 8.6% |
| sm-round-below | 16.2% / 5.1% | 14.5% / 5.0% |
| sm-round-exact | 17.6% / 4.2% | 15.0% / 5.0% |
| sm-round-above | 13.0% / 2.7% | 10.8% / 4.4% |

These were also the configs whose KernelIR carried the `linear_access_fallback` diagnostic. The frontend now recovers their access map (`18af028`), but no prediction changed: for these `GROUP_M = 1` configs the fallback map equaled the recovered one. The fallback did not cause the error; the synchronous-load path is the common factor, and no microbenchmark covers it in a GEMM loop yet. The `num_stages = 2` rows of the first table use asynchronous copies and are not explained by it either.

**Effect on users.** For these configs, rely on the indistinguishable verdict rather than the point prediction: autotuning keeps them in the shortlist for measurement on the real GPU instead of ruling them out.

### 9.2 v0.2 residuals and exclusions

These are residuals of the v0.2 core on the development set, not of `phase1-final`.

**Spill reload latency and the L1 transition were not in v0.2.** v0.2 charged spills as local-memory bandwidth with a linear overflow past one L1 threshold. The measurements show this is wrong in two ways: reloads cost latency on an iteration's critical path, so that model is 17% to 93% fast on `spill_reload_latency_by_count`, and the move from L1 to L2 is a step at the L1 capacity that follows the shared memory carveout (`l1_to_l2_spill_threshold_by_carveout`), not a linear overflow. Both are in the core since v0.3 (6.4). What is still missing is the number of local loads a kernel executes per iteration: the core uses the assembler's static spill load bytes over 4 B, which counts code on cold paths too and does not know wider reloads. For causal attention it splits that figure evenly between the main and the diagonal loop (`spill_loop_count`), which assumes the two loops hold the same spill code. Checking it needs the GPU performance counters (`microbench/spill_counters.py`, compared by `scripts/posthoc_spill_split.py`). A post-hoc RTX 3070 capture (`docs/results/analysis/spill-counters-rtx3070-posthoc.md`) shows the split still overcounts (1.4 and 4.1 times the measured reloads) and that the two loops do not hold the same spill code. One coupling is not settled (issue #15): spill frames that leave L1 enter the working set that scales cross-wave hits in `l2::split`, so a spilling kernel's global-memory hit split moves too. Under v0.2, spilling configs were predicted too fast: on the opened phase 1 holdout (post-hoc, `docs/results/analysis/posthoc-core-phase1-phase2.md`), `m128n128k64-w4` and `m128n64k64-w4` of `mask-mnk` are 42% and 60% fast.

**Synchronous loads in GEMM.** GEMM configs with `num_stages = 1`, which stage their loads through registers (`ld.global`, then `st.shared`), are predicted slow: median error +9.1% warm and +2.2% cold on the development set (core `7b1be46`, profile `0.21.0`), against +3.6% and +0.2% before the round-trip port floor (`e51f8ef`). The floor first made them +11% and +5%, mostly because it let every CTA on an SM wait for all the SM's bytes after the latency, as if their round trips were in phase; `7b1be46` shares the port out of phase, as `sync_stage_loop` with 1 to 8 CTAs per SM shows. What remains is not explained. It is not a register-staging cost the model misses: on `sync_stage_loop` (profile `0.21.0`, 12 to 32 KiB stages, two or three CTAs per SM) register-staged loops are predicted with MAPE 4.2% (RTX 4070) and 8.3% (GB10), median error -0.2% and -7.1%, so no separate staging mechanism (next-version mechanism (c)) is needed there; with one CTA per SM the staging shows, and the model charges it by keeping the shared memory stores on a CTA's dependent chain, which taking them off makes worse.

The serial-fill explanation ruled out in 9.1 was ruled out as an addition to the round trip. `e51f8ef` uses it as a floor under the round trip instead, which the MLP curves, indexed by bytes in flight over the whole GPU, do not provide.

### 9.3 Cross-wave L2 reuse near capacity

- **Status.** The change of 6.4 is kept on GB10 microbenchmark evidence that agrees in direction but does not pass the capture rules. Seven runs of three versions (`wave-rescan`, `wave-rescan-long`, `wave-rescan-v3`) each pass rules 1 and 2; none of the three versions passes rule 3 across its runs. Its standing is set by the sealed v0.3 holdout.
- **Stable part.** Forward re-reads at 1.5 times L2 or more hit -0.01 to 0.09 in every run, against 0.5 to 0.9 under the earlier per-chunk clock, which is wrong in every run; reverting would bring back the form that is further off in all of them. Restricted after the fact to these points, all three versions pass all three rules (post-hoc, [stable-subset-rules-posthoc.json](results/analysis/v0.3-gb10-h3-h4/stable-subset-rules-posthoc.json)).
- **Near capacity (known limit).** From about 0.75 to 1.25 times the effective capacity, the GB10 steady state differs between runs while each run is internally consistent and the host CPU is quiet: at 0.91 times L2 one run re-reads in 1701 us and another in 3528 us, and at 1.25 times L2 one run of `wave-rescan-v3` hits 0.31 to 0.40 where the others hit 0.01 to 0.09. This is hardware behavior that repeated captures do not remove. The model gives a step at `l2_effective_capacity_bytes` there, with no smoothing parameter, so predictions for working sets near the L2 capacity carry a large uncertainty. See [the observation](results/analysis/v0.3-gb10-l2-bistability-observation.md).
- **Repeated scans beyond capacity (known limit).** On GB10 the L2 capacity sweep, which re-reads one working set many times in a row, still hits 0.79 to 0.94 at 1.5 times L2 and 0.01 to 0.24 at 2.0 times, varying between processes and between buffers of one process. The cross-wave form, the rescan's single re-read and the forward `wave-rescan` re-reads all hit about 0 to 0.09 at 1.5 times L2, and the model follows them. A kernel that re-reads a working set of 1.25 to 2 times L2 many times in a row is therefore likely to be predicted with too few L2 hits, by an amount that also varies between processes. The mechanism is unknown; page placement, a per-process state and compression were excluded ([the near-capacity exception](results/analysis/v0.3-gb10-l2-near-capacity-exception-registration.md)). The model is unchanged.
- **Reversed walks.** The per-slice LRU form fits reversed re-reads best in the captures that have them, but this is provisional, and no supported kernel walks K backwards.
- The form is checked on GB10 only; on the RTX 4070 it is unvalidated.

### 9.4 Ascend 910B3 mapping and known limits

Ascend 910B3 (`profiles/ascend910b3.json`, `0.3.1-ascend`) brings a second vendor into the existing schema and core, with both Triton-Ascend GEMM and forward GQA Attention wired end-to-end and a Cube/Vector split execution path gated on `specs.capabilities.cube_vector_split`:

- **Occupancy placeholders and Cube/Vector split.** The profile maps 20 physical AI Cores (20 AIC + 40 AIV) onto `sm_count = 20` with occupancy slot placeholders (`max_warps_per_sm = 1`, `max_blocks_per_sm = 1`, `registers_per_sm = 1024`, `num_warps = 1`, `registers_per_thread = 1`). Per-core on-chip SRAM limits (`specs.ascend.{ub_bytes, l0c_bytes}`) are checked against `resources.vendor_extensions.ascend` and reject overflows with `OccupancyError::TooMuchOnChipBuffer`. Under `cube_vector_split`, inter-core GM workspace traffic (`workspace_loop_bytes`) stays L2-resident while total active workspace fits in `l2_effective_capacity_bytes` (spilling only overflow to DRAM), multi-buffered rings (`workspace_multibuffer >= 2`) overlap Cube (MMA) and Vector (non-MMA) steady-state compute, and `cv_sync_hops_per_iter` charges measured cross-core FFTS+ sync latency (`measured.pipeline.cv_sync_hop_ns_by_active_sms`). All NVIDIA/AMD profiles have `cube_vector_split = false` and remain bit-identical.
- **MLP issue-mode naming.** The AscendC M2 MLP curves (`dram_bw_by_bytes_in_flight`, `l2_bw_by_bytes_in_flight`) reuse the existing schema enum names `cp_async` and `ld_global_v4` to represent 2-buffer asynchronous and 1-buffer synchronous MTE2 DMA transfers.
- **Sealed holdout results and post-hoc residuals.** The preregistered multi-shape sealed holdout ([`docs/results/ascend910b3-holdout.md`](results/ascend910b3-holdout.md), `0.3.1-ascend` plus `hivm.py` fractal NZ / `AIC`-only `vbrc` parser fixes, 16 fresh unseen shapes across 48 sessions, L2 probe ratio `1.51-1.65`) passed both Attention cohorts (`ascend910b3-attention-warm`: median tau-b `0.756` vs `0.545` roofline and `0.578` NeuSight, `0.00%` top-5 regret, `23.6%` median latency error; `ascend910b3-attention-cold`: `0.756` vs `0.571` and `0.622`, `0.00%` regret, `23.9%` error). On GEMM, VirtualGPU passed top-5 regret (`0.00%` warm, `0.09%` cold), ordered agreement (`89.8%` warm, `86.7%` cold), and NeuSight ranking (`+0.131` warm, `+0.453` cold), but trailed `roofline_wave` in median tau-b (`0.773` vs `0.803` warm; `0.720` vs `0.840` cold), so `unvalidated_feature` and `unvalidated_target_ascend` remain in place. Post-hoc inspection shows why: on Ascend 910B3 every GEMM config runs at `ctas_per_sm = 1` (giving `roofline_wave` `0.80-0.84` tau-b), while the core's closed-loop round-trip model predicts a `~1.5x` penalty for `multibuffer = False` vs `True` against a `1%-8%` measured hardware gap on `AIC_ONLY` GEMM and under-predicts short `BK=32` / decode iterations lacking an explicit per-K-step MTE/Cube setup term. Informal dev sanity results remain archived in [`docs/results/ascend910b3-gemm-sanity.md`](results/ascend910b3-gemm-sanity.md).

### 9.5 Narrow register-staged loads (H5)

GEMM configs whose global loads are 1 to 4 B wide and staged through registers into shared memory are predicted too fast; on GB10 `mask-mnk-1153` (odd K, every load 2 B) all 36 configs are 27% to 82% fast (post-hoc, `docs/results/analysis/h5-two-byte-posthoc.md`). The core lets the load width reach only the `lsu` pipe on the shared path, and the round trip uses MLP curves measured with 16 B loads. The narrow-load microbenchmarks (`docs/results/analysis/h5-narrow-microbench-posthoc.md`, `h5-followups-gb10-posthoc.md`) point to two costs the core misses, and the current model is systematically fast on narrow widths through both:

- **From DRAM.** On GB10 narrow loads appeared to run at a GPU-wide request limit, or at the sector limit of the strided DRAM curve, which offset rows reach sooner. A form with these two limits was frozen and scored (MAPE 10.4% and 10.8% in two GB10 sessions, 27.1% on the RTX 4070), but every one of these captures fails the capture dispersion rule, so the numbers are for the record only and the form is not judged.
- **Register staging.** Narrow shared memory stores appeared to cost per instruction on the shared memory pipe, and narrow loads from L2 to add time per load on each CTA's chain. Forms for both (b1 + b2) were scored on captures that fail the dispersion rule (MAPE from 28.7% to 10.7% in one round and from 25.7% to 16.9% in another; for the record only, not judged). The one qualified capture of H5, the last round repeated with every launch over 1 ms, scores b1 + b2 at MAPE 15.7% on 92 new GB10 points (current model 26.8%), every width's median error within 10%, which misses the 15% limit.

Neither form is in the core. Redoing both with launches over 1 ms on GB10 and the RTX 4070, with captures that pass the dispersion rule, is a candidate for after v0.3.

**Effect on users.** Configs with narrow register-staged loads, which Triton emits when rows are not 16-byte aligned, rank faster than they are; measure them on the target GPU.

## 10. Related work

- [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) / [Accel-Sim](https://accel-sim.github.io/): cycle-level NVIDIA simulation; detailed but hours per large kernel and behind on recent architectures.
- [Timeloop](https://github.com/NVlabs/timeloop): loop-nest mapping analysis for accelerator architects; its data reuse analysis is relevant to tile-level GEMM modeling.
- [Triton-CPU](https://github.com/triton-lang/triton-cpu): runs Triton kernels on CPUs for correctness; no GPU performance model.
- [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262) (MICRO 2026): tile-graph simulation from TileLang IR, 1.22% to 8.71% MAPE on A100 and H100. Closest in spirit; no public code found.
- [NeuSight](https://arxiv.org/abs/2407.13853) (ASPLOS 2025): tile-based performance forecasting for unseen GPUs, bounded by hardware peaks.
- [KernelSight-LM](https://arxiv.org/abs/2606.28565): kernel-level prediction; 12.1% per-kernel error on an unseen GPU generation versus 22.0% for roofline.

These works establish that tile-level modeling of LLM kernels is accurate. What does not exist is an open, usable tool aimed at kernel developers: Triton in, ranking and explanation out, in seconds, on a CPU, against any GPU profile.

## 11. Open questions

1. How to obtain register usage and spills for AMD targets.
2. How to feed workload descriptions (expert routing, page tables) into data-dependent kernels.
3. Profile format versioning, and how to review community-submitted profiles.
4. Whether a softer verdict rule than the 5% unanimity rule is worth calibrating.
5. Which clock controls are available on each test system (WSL2 for the RTX 4070, GB10's unified memory).

When one is answered, record the answer in the section it belongs to and delete it here. Work that follows from it goes in the [roadmap](../ROADMAP.md).
