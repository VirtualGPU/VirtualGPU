# Performance model core

How the core turns a `KernelIR` and a GPU profile into a predicted time, a resource breakdown and pairwise verdicts. Code: `crates/vgpu-core/src/`. The mechanisms the core runs today are listed in [design.md section 6.4](../design.md#64-mechanisms-in-the-core). If this file and the code disagree, the code is right and this file gets fixed.

## 1. Scope

The model core turns one kernel description (from the frontend, via the kernel IR) and one GPU profile into a predicted execution time with a resource breakdown, and turns a set of variants into pairwise comparisons with a confidence attached.

In scope:

- Occupancy (CTAs resident per SM and the limiting resource).
- Event-driven CTA dispatch onto SMs: waves and tails emerge from the simulation, they are not computed by formula.
- Per-CTA phase model: prologue, steady-state main loop with `num_stages` overlap, drain, epilogue.
- Sharing of SM-level (tensor pipe, issue, shared memory) and GPU-level (L2, DRAM) bandwidth among resident CTAs.
- Persistent kernels and warp-specialized kernels.
- Register spill penalty.
- Launch overhead.
- Perturbation-based resolution and pairwise verdicts.

Out of scope (owned elsewhere):

- Extracting kernel structure from Triton ([frontend.md](frontend.md)).
- The kernel IR format and the L2 reuse model ([kernel-ir.md](kernel-ir.md)). The core calls the L2 model through its API.
- Measuring profile fields ([profiles.md](profiles.md)).
- Baselines and the evaluation protocol ([validation.md](validation.md)). The core must not contain baseline code.


## 2. Interfaces

### 2.1 Input: `KernelIR`

The core consumes `KernelIR` v0.1 as defined in [kernel-ir.md](kernel-ir.md). How the core uses each part:

- `launch.grid`, `launch.num_warps`, `resources.*`: occupancy (3.3).
- `body.outer_loop.trip_counts`: tiles per program. One for ordinary kernels, more for persistent kernels (3.6). No separate code path.
- `launch.warp_groups`: one `unified` group, or `producer` and `consumer` groups (3.6).
- `body.prologue`, `body.pipeline_loop`, `body.epilogue`: the phase list per tile. From `pipeline_loop` the core uses the post-pipelining `num_stages`, `async_copy`, and per-op `stage_offset`. It never reads the decorator's `num_stages`.
- `mma` ops: `op_class`, `inst_mnk`, dtypes, and `count` select the throughput entry in the profile.
- `gmem_load` / `gmem_store` ops: bytes per iteration, split into L2 and DRAM bytes by the L2 model.
- `alu` / `sfu` ops: issue work.

Other `KernelIR` fields the core reads:

- `resources.spill_scope`: `none` and `prologue_epilogue` are one-time costs, `inner_loop_confirmed` and `inner_loop_assumed` are per-iteration costs, and `inner_loop_assumed` gets a widened uncertainty (3.7).
- `pipeline_loop.prefetch_distance`: the overlap depth in 3.5, instead of assuming `num_stages - 1`.
- `epilogue.overlaps_next_tile`: epilogue and next-tile overlap for persistent kernels (3.6).
- `launch.cluster_shape`.
- `access_map` entries with `tile_coords` and `is_masked`: masked edge tiles are charged their valid bytes, not the full tile.
- Top-level `unsupported_features`: any entry gives status `unsupported` with that entry as the reason.
- Top-level `diagnostics`: the variant is still simulated, but any entry sets the `unvalidated_feature` flag (lower confidence, section 4) and is copied into the output diagnostics. Targets without a sealed holdout (currently `vendor == "ascend"`) automatically receive `unvalidated_target_ascend` in `diagnostics` and `unvalidated_feature` in `flags` even if the input IR carries no diagnostics.
- `resources.vendor_extensions.ascend` (`ub_bytes`, `l0c_bytes`, `workspace_bytes`, `workspace_loop_bytes`, `workspace_multibuffer`, `cv_sync_hops_per_iter`): `ub_bytes` and `l0c_bytes` are checked against `specs.ascend.{ub_bytes, l0c_bytes}` in occupancy (3.3). When `specs.capabilities.cube_vector_split` is true, inter-core GM workspace traffic stays L2-resident while total active workspace fits in `l2_effective_capacity_bytes` (spilling only overflow to DRAM), a multi-buffered workspace ring (`workspace_multibuffer >= 2`) overlaps Cube (MMA) and Vector (non-MMA) execution in steady state (`max` instead of `sum`), and `cv_sync_hops_per_iter` charges measured cross-core sync latency from `measured.pipeline.cv_sync_hop_ns_by_active_sms`. On non-`cube_vector_split` architectures (NVIDIA, AMD), these paths are inactive and predictions are bit-identical.
- `alu` / `sfu` counts in `insts_per_warp_per_iter`.
- `gmem_load.nominal_bytes_per_cta` is the full-tile size; actual bytes for masked edge tiles come from `access_map`.

### 2.2 Input: GPU profile

The core consumes profile v0.1 as defined in [profiles.md](profiles.md).

- `specs`: used exactly, never perturbed. Occupancy uses `max_warps_per_sm`, `max_blocks_per_sm`, `registers_per_sm`, `register_alloc_unit_regs`, `shared_mem_per_sm_bytes`, `max_shared_mem_per_block_bytes`, and `shared_mem_alloc_granularity_bytes`.
- `measured.*`: every `MeasuredValue` is perturbed within `rel_uncertainty` (section 4).
- `measured.cta_dispatch`: selects the dispatch policy (3.2).

Profile fields the core reads beyond the basic specs:

1. Shared memory carveout: the allowed carveout sizes and the per-CTA reserved shared memory. Without them occupancy is wrong whenever shared memory is the limiter.
2. MMA throughput as a function of MMA-issuing warps per SM (a short table per key, for example 4, 8, 16, 32 warps), not one peak number. This is how `num_warps` enters the model (3.5).
3. The same active-SM curve for L2 bandwidth that `dram_bw_by_active_sms` gives for DRAM.
4. An optional `perturb_group` tag on `MeasuredValue`: fields with the same tag share one draw. Default: each field is drawn independently, and each curve (`*_by_active_sms`, `*_by_warps`) is scaled by one draw as a whole, so it keeps its shape.
5. In `cta_dispatch`: a list of `candidate_policies` whenever `confidence` is not high, and the raw first-wave `(cta_id -> sm_id, start_ns)` table for each probe whenever `observed_policy` is `unknown`. The fitted enum is enough otherwise.
6. Optional: load latency under load (latency versus achieved bandwidth). Without it the core derives a loaded latency from the bandwidth curve.

### 2.3 L2 model calls

The core uses the online API in [kernel-ir.md](kernel-ir.md):

- `begin_tile` / `end_tile` when a tile (one outer iteration of a CTA) starts and ends, so the L2 model always knows the resident set.
- `split(sm_id, cta_id, outer_idx, op_id, k_begin, k_end, is_write, cache_policy, t_ns)` once per (CTA, op, phase segment). The segment covers the half-open inner-step range `[k_begin, k_end)` between two changes of the resident set; prologue and epilogue pass `[0, 1)`. The result is the aggregate over the range. Calls stay per segment, never per pipeline iteration, which keeps a 4096-tile GEMM at a few calls per tile and operand.
- `account_spill` once per run at full residency, with the resident threads per SM and the shared memory carveout from occupancy. The share that leaves L1 is the measured L1 miss rate of the frames at that carveout (design.md 6.4).

Other points:

- One `L2Model` instance per simulation run, built from the perturbed profile of that draw. No shared state, so draws can run in parallel.
- `initial_state` comes from the validation case (cold or warm), and is echoed in the output.
- The L2 model's discrete options, `eviction_mode` (`soft_lru` or `strict_lru`) and `cross_wave_reuse` (`true` or `false`), are sampled as extra dimensions in section 4.
- `near_cliff = true` on any call sets the `near_l2_cliff` diagnostic on the variant. The draws already move `l2_effective_capacity_bytes`, so a ranking that depends on the cliff should flip on its own. The flag is for diagnosis, not a second verdict rule.

### 2.4 Output (consumed by validation)

One record per (case ID, config ID, profile revision):

- `status`: `ok`, `unsupported`, or `failed`, with a `reason` string. Unsupported configs keep their row, so coverage is counted correctly.
- `nominal_latency_ns`: the prediction at draw 0, which is the unperturbed profile.
- `draws`: latency per draw ID. Draw IDs and the values drawn are the same for every config on the same device and profile revision, so validation can compute pairwise direction counts itself.
- `assumptions`: initial L2 state, and the dispatch policy used in each draw.
- `diagnostics` (nominal draw): occupancy and its limiting resource, wave count, tail efficiency, per-resource utilization and fraction of time as the bound, L2 and DRAM bytes, and flags (`spill`, `near_l2_cliff`, `dispatch_ambiguous`, `unvalidated_feature`).

The core also emits its own pairwise verdicts (section 4) for users. Validation recomputes them from `draws` and does not trust the core's verdicts.

## 3. Key decisions

### 3.1 Event granularity: per CTA phase, not per instruction or per loop iteration

Options:

- Per instruction (Accel-Sim style): accurate, hours per large kernel, needs undocumented microarchitecture. Rejected.
- Per main-loop iteration: a large GEMM has hundreds of thousands of iterations; with perturbation sampling this multiplies by tens. Too slow, and per-iteration detail adds little the steady-state model does not already capture.
- Per CTA phase (chosen): events at CTA start, phase boundaries (prologue, steady state, drain, epilogue), and CTA end. Inside a phase, progress is a rate that is recomputed whenever the set of active phases on a shared resource changes.

Cost is O(tiles x phases) events per simulation. Requirement: a 4096-tile GEMM simulates in about 1 ms, so that 50 configs x 20 shapes x 32 samples finishes in under a minute. This requirement constrains the implementation language (section 7).

### 3.2 CTA dispatch

The scheduler keeps a queue of programs in launch order and a slot count per SM from the occupancy computation. When a slot frees, the next program is assigned according to the profile's dispatch policy. Initial policy set:

- `round_robin`: program i goes to SM (i mod SM count) in the first wave, then to whichever SM frees first.
- `grouped`: the same, but SMs are filled in groups (for example by GPC) with a measured group order.
- `measured_table`: the first-wave mapping is taken directly from the measured trace; later waves use first-free.

Later waves always use "first SM to free a slot", because that is the behavior the trace measurement can observe directly, and it does not depend on undocumented arbitration. When the profile lists several candidate policies, the model runs each and treats the policy as one more perturbation dimension (section 4). Comparisons that flip with the policy become indistinguishable.

### 3.3 Occupancy

Computed analytically from the profile's allocation rules: CTAs per SM is the minimum over registers (rounded to allocation granularity), shared memory (rounded, for the carveout the runtime would pick), warps, and the CTA slot limit. The carveout choice is a known source of error: the model uses the smallest carveout that fits the maximum resident CTAs, and this rule must be checked against measurements on both test GPUs.

When `specs.ascend` is present, the occupancy check also enforces the per-core Unified Buffer (`ub_bytes`) and L0C accumulator (`l0c_bytes`) limits from `resources.vendor_extensions.ascend`, rejecting overflows with `OccupancyError::TooMuchOnChipBuffer { buffer, bytes, max }` rather than `TooMuchSharedMemory`. The Ascend profile maps 20 physical AI Cores (20 AIC + 40 AIV) onto `sm_count = 20` with occupancy slot placeholders (`max_warps_per_sm = 1`, `registers_per_sm = 1024`, `num_warps = 1`, `registers_per_thread = 1`), while Cube/Vector steady-state overlap, L2-resident inter-core workspace traffic, and cross-core sync handoff latency are gated on `specs.capabilities.cube_vector_split` (see [design.md section 9.4](../design.md#94-ascend-910b3-mapping-and-known-limits)).

### 3.4 Resource sharing: fluid processor sharing

Each active phase demands work on one or more resources (tensor pipe, issue, shared memory bandwidth, L2 bandwidth, DRAM bandwidth). At each event the core recomputes the rate of every active phase:

- SM-local resources are shared among the CTAs resident on that SM.
- L2 and DRAM bandwidth are shared among all active phases on the GPU, with the achievable bandwidth taken from the profile table for the current number of active SMs.
- A phase progresses at the rate of its most constrained resource (max-min fair share).

Rejected alternative: a fixed per-CTA time computed in isolation and then packed into waves. It cannot represent the case where a memory-bound tail wave runs faster because fewer SMs compete for DRAM, which is exactly what distinguishes some configs.

### 3.5 Steady-state pipeline overlap

For one CTA in steady state with `S` stages and prefetch distance `S - 1`, the time per iteration is the maximum of:

- compute time: MMA work divided by the CTA's share of tensor throughput, plus non-MMA work divided by its share of issue rate;
- transfer time: bytes per iteration divided by the CTA's share of the bandwidth of the level that serves them (from the L2 split);
- latency bound: load latency divided by the number of iterations in flight (`S - 1`), with the latency taken from the profile and increased under load.

With several CTAs resident on one SM, latency is additionally hidden across CTAs: the latency bound applies to the SM's total iterations in flight, not to one CTA. Synchronous loads (no async copy) are modeled as `S = 1`: no overlap within the CTA, hiding only across CTAs.

Prologue time is the latency plus the transfer time of the first `S - 1` stages. Drain is the last `S - 1` iterations of compute without new loads.

The MMA throughput share depends on how many warps issue MMA on the SM, using the profile's measured curve. This is how `num_warps` enters the model.

### 3.6 Persistent and warp-specialized kernels

Persistent kernels need no special scheduler: the grid is the number of programs, and each program walks its tile list from the kernel IR. Each tile contributes its phases in sequence. If the kernel IR marks epilogue-to-next-tile overlap, the epilogue of tile i runs as a concurrent phase with the main loop of tile i+1 on the same CTA, sharing its resources.

Warp specialization is modeled as two concurrent phases in one CTA, a producer (loads only) and a consumer (compute only), coupled by an `S`-deep buffer: the consumer cannot run ahead of the producer by more than `S` iterations and vice versa. In steady state this reduces to the same maximum as 3.5, without the latency term competing with compute warps for issue slots.

Neither test GPU (sm_89, sm_121) is known to us to exercise Triton's warp specialization or Hopper-style TMA paths; what sm_121 supports must be read from official documentation before we rely on it. Until a test GPU exercises these paths, kernels that use them get the "unvalidated feature" flag.

### 3.7 Register spills

The assembler reports static spill bytes, not dynamic counts. The core uses:

- `inner_loop_confirmed`: spill bytes are added as local memory traffic to every iteration, and `account_spill` decides how much of it reaches L2 and DRAM from the L1 miss curve of the chosen carveout.
- `inner_loop_assumed`: the same at draw 0. In draws 1 to `N` the per-iteration spill traffic is scaled by a factor sampled uniformly in `[0, 1]`, covering everything from a one-time cost to a per-iteration cost. A ranking that depends on where the spill code sits then becomes indistinguishable.
- `prologue_epilogue`: added once to the prologue or epilogue.
- With either loop scope, `spill_loop_count` (default 1) divides the per-iteration spill bytes and reloads. Causal attention has two sequential loops that share one assembler figure, and each iteration runs one of them, so each pays half. The frame is not divided.
- `none`: no spill cost.

Reloads also add latency to the CTA's dependent chain, from the measured reload curves at the spill load bytes over 4 B (design.md 6.4). That latency includes issuing the reloads, so the chain then keeps only the stores' share of the local bytes as `local` time; the shared `local` pipe keeps all of them. Both overestimate dynamic traffic when spill code is on a cold path. That is acceptable for ranking: a spilling config should rank below an otherwise similar non-spilling one, and the sign matters more than the magnitude. Every spilling variant carries the spill flag, and the validation must check spilling configs separately.

### 3.8 Launch overhead

A constant from the profile, plus per-CTA launch latency at dispatch. Matters only for small shapes, where it can dominate.

## 4. Perturbation-based resolution

When variants are bound by different resources, profile errors do not cancel. The core therefore treats the profile as uncertain and makes every comparison paired.

- Draw 0 is the unperturbed profile. Draws 1 to `N` (default `N = 32`, Latin hypercube, fixed seed) sample each measured field uniformly within `+/- rel_uncertainty` of its value. `rel_uncertainty` already includes the 10% floor from [profiles.md](profiles.md). Grouping and curve rules are in 2.2. Specification fields are fixed. Derived quantities are recomputed from the drawn values so that they stay consistent.
- Discrete dimensions are sampled alongside: the dispatch policy when the profile lists `candidate_policies` (3.2), and the L2 model's `eviction_mode` and `cross_wave_reuse` (2.3).
- The same draws are used for every variant (common random numbers). An error shared by all variants moves them together, so only errors that affect the variants differently change their order. This is what makes the comparison paired.
- Verdict for a pair (A, B): "A faster" if A is at least 5% faster in every draw including draw 0; "B faster" likewise; otherwise "indistinguishable". The 5% band is fixed, not tuned (see the verdict rule in [design.md](../design.md#decisions)). A softer rule can only be introduced if it is calibrated on calibration data, never on holdout data.
- If either variant has status `unsupported`, no verdict is given. If either has `unvalidated_feature`, the verdict is still computed, and the flag goes with it.
- Verdicts are kept per pair, with no hard grouping, so non-transitive results are reported as they are.

Cost: `N + 1` simulations per variant. This is where the 1 ms per simulation requirement in 3.1 comes from.
