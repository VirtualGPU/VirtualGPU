# Kernel IR and L2 model

The `KernelIR` format that frontends emit and the L2 reuse model that reads it. Schema: `schemas/kernel_ir.schema.json`. Code: `crates/vgpu-core/src/ir/` and `crates/vgpu-core/src/l2/`. The mechanisms the core runs today are listed in [design.md section 6.4](../design.md#64-mechanisms-in-the-core). If this file and the code disagree, the code is right and this file gets fixed.

## 1. Scope

This document defines two components that sit between any kernel frontend (Triton first, followed by PTX for CUDA/CUTLASS/TileLang and AMD frontends) and the Rust event-driven performance model core (the model core):

1. **Kernel IR (`KernelIR` v0.1)**: A **language- and vendor-agnostic** intermediate representation emitted by frontends and consumed by the L2 reuse model and the performance model core. It captures post-pipelining tile structure, persistent loops, warp/wavefront specialization, assembler resource/spill reports, diagnostic/unsupported flags, and evaluated per-CTA memory footprints.
2. **L2 Reuse Model**: A Rust-implemented, online, phase-level, microsecond-scale cache reuse model invoked by the model core's event-driven scheduler to split global memory phase traffic into L2 and DRAM bytes, account for sector overfetch, and inject register-spill L2 pressure.

Out of scope: Frontend compilation and scalar evaluation (the frontend); CTA/SM scheduling, occupancy, fluid resource sharing, and perturbation sampling (the model core); microbenchmarking and GPU profile collection (the microbenchmark suite); ground-truth hardware measurement, baselines, and validation metrics (validation).

---

## 2. Input and Output Interfaces

### 2.1 `KernelIR` Schema (`schema_version: "0.1"`, Frontends -> Rust Core)

In memory, Python frontends pass `KernelIR` directly into the Rust core (the L2 model and the model core) as typed structs via **PyO3**. For disk caching, debugging, and dataset archival, `KernelIR` serializes to JSON with an optional compact binary sidecar for `access_map` (see Decision D5).

Terminology note: `cta` refers to a thread block (NVIDIA) or workgroup (AMD); `warp` refers to a warp (NVIDIA, size 32) or wavefront (AMD, size 32/64); `sm` refers to an SM (NVIDIA) or CU (AMD).

```jsonc
{
  "schema_version": "0.1",
  "metadata": {
    "kernel_name": "matmul_kernel",
    "target_vendor": "nvidia",           // "nvidia" | "amd" | ...
    "target_arch": "sm_89",              // e.g., "sm_89", "sm_121", "gfx942"
    "frontend": "triton",                // e.g., "triton", "ptx", "hip"
    // Opaque frontend-defined dicts for provenance and caching ONLY.
    // The model core MUST NOT read or branch on toolchain or config fields.
    "toolchain": { "compiler": "triton", "compiler_version": "3.3.0", "assembler": "ptxas", "assembler_version": "12.9" },
    "config": { "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8 }
  },
  "launch": {
    "grid": [64, 1, 1],
    "cluster_shape": [1, 1, 1],          // thread-block / workgroup cluster dims (default [1, 1, 1])
    "num_warps": 4,                      // warps or wavefronts per CTA
    "warp_size": 32,                     // 32 (NVIDIA / RDNA) or 64 (AMD CDNA)
    "warp_groups": [
      // Standard kernels emit one "unified" group.
      // Specialized kernels emit separate "producer" and "consumer" groups.
      { "id": 0, "role": "unified", "num_warps": 4 }
    ],
    "dynamic_smem_bytes": 0
  },
  "resources": {
    "source": "assembler_report",        // "assembler_report" | "estimated" (specific assembler recorded in metadata.toolchain)
    "registers_per_thread": 96,          // scalar/vector regs per lane (frontends map vendor register classes per profile)
    "static_smem_bytes": 49152,          // static shared memory / LDS bytes per CTA
    "stack_frame_bytes": 0,              // per-thread local/scratch memory frame
    "spill_stores_bytes": 0,             // per-thread spill stores from assembler report
    "spill_loads_bytes": 0,              // per-thread spill loads from assembler report
    "spill_scope": "none",               // "none" | "prologue_epilogue" | "inner_loop_confirmed" | "inner_loop_assumed"
    "barriers": 1
  },
  "tensors": [
    {
      "id": "A",
      "dtype": "f16",
      "element_bits": 16,
      "shape": [4096, 4096],
      "strides": [4096, 1],              // in elements
      "base_addr": 0,                    // virtual aligned base address
      // Vendor-neutral cache hint; each frontend maps its native modifiers
      // (e.g., PTX ca->default, cg->bypass_l1, cs->streaming, cv->volatile; AMD glc/slc/nt similarly).
      "cache_policy": "default"          // "default" | "bypass_l1" | "streaming" | "evict_first" | "volatile"
    }
  ],
  "body": {
    // Unifies standard (trip_counts=1) and persistent kernels (trip_counts > 1 per CTA).
    "outer_loop": {
      "trip_count_mode": "uniform",      // "uniform" | "per_cta"
      "trip_counts": 1                   // int if uniform, else list[int] of length prod(grid)
    },
    "prologue": { "ops": [] },
    "pipeline_loop": {
      // Extracted strictly AFTER software pipelining passes (or from lowered IR/assembly).
      "trip_count_mode": "uniform",      // "uniform" | "per_outer_iter"
      "trip_counts": 128,
      "num_stages": 3,                   // actual materialized pipeline depth
      "prefetch_distance": 2,            // stages prefetched ahead (typically num_stages - 1)
      // Vendor-neutral async transfer class:
      // "none" (synchronous), "async_element" (e.g., cp.async, AMD buffer_load_lds), "async_tensor" (e.g., TMA)
      "async_copy": "async_element",
      "ops": [
        {
          "op_id": "load_A",
          "kind": "gmem_load",           // gmem_load | gmem_store | smem_load | smem_store | mma | alu | sfu | barrier
          "warp_group_id": 0,
          "stage_offset": 2,             // prefetch lead in stages (0 = current iter)
          "tensor_id": "A",
          "nominal_bytes_per_cta": 8192, // unmasked upper bound; actual per-tile bytes come from access_map (rows * row_bytes)
          "transfer_width_bytes": 16,    // per-thread vectorized width
          "access_ref": "acc_A"
        },
        {
          "op_id": "dot_0",
          "kind": "mma",
          "warp_group_id": 0,
          "stage_offset": 0,
          // Profile-keyed matrix instruction family matching the microbenchmark suite's throughput table
          // (e.g., "mma.sync", "wgmma", "tcgen05" on NVIDIA; "mfma", "wmma" on AMD).
          "op_class": "mma.sync",
          "tile_mnk": [128, 128, 32],
          "inst_mnk": [16, 8, 16],       // hardware matrix instruction shape
          "a_dtype": "f16",
          "b_dtype": "f16",
          "acc_dtype": "f32",
          "count": 1
        }
        // Note: for "alu" and "sfu" ops, "count" is instructions per warp per iteration (insts_per_warp_per_iter).
      ]
    },
    "epilogue": {
      "overlaps_next_tile": false,       // true if persistent epilogue overlaps next tile's main loop
      "ops": []
    }
  },
  "access_map": {
    // Produced by the frontend's scalar evaluator over CTA id and loop bounds.
    // Indexed densely by (cta_id, outer_idx). On disk, may reference a binary sidecar.
    "acc_A": {
      "encoding": "strided_2d_per_outer_tile", // or "explicit_blocks"
      "sidecar_ref": null,                     // optional { "file": "...", "offset": 0, "count": 64 } for JSON archival
      "entries": [
        // [tile_coord_0, tile_coord_1, base_byte_offset, rows, row_bytes, row_stride_bytes, inner_step_byte_stride, is_masked]
        // Note: rows and row_bytes are post-boundary-mask valid extents (clamped for unaligned shapes).
        [0, 0, 0, 128, 64, 8192, 64, false]
      ]
    }
  },
  // Soft warnings: model still runs, but the model core lowers confidence / widens uncertainty.
  "diagnostics": [],                     // e.g., ["assembler_only_spill", "unpipelined_fallback"]
  // Hard blockers: model cannot represent these; the model core returns status="unsupported" with reasons.
  "unsupported_features": []             // e.g., ["atomic_work_stealing", "distributed_smem", "data_dependent_address"]
}
```

### 2.2 GPU Profile Fields Consumed by L2 Model (from the microbenchmark suite)

Aligned to `profiles.md`:

- **Specification fields**: `specs.l2_size_bytes`, `specs.l2_line_bytes` (default 128), `specs.l2_sector_bytes` (default 32), `specs.l1_line_bytes` (default 128), `specs.l1_size_bytes`, `specs.l2_partitions` (default 1), `specs.shared_mem_per_sm_bytes`.
- **Measured fields** (`MeasuredValue` envelopes, perturbed by the model core per sample): `measured.memory.l2_effective_capacity_bytes`, `measured.memory.l2_bw_read_gbps`, `measured.memory.l2_bw_write_gbps`, `measured.memory.dram_bw_read_gbps`, `measured.memory.dram_bw_write_gbps`, `measured.memory.l2_latency_ns`, `measured.memory.dram_latency_ns`, `measured.spill.l1_to_l2_spill_threshold_bytes_per_sm`, `measured.spill.local_mem_l1_hit_bw_per_sm_gbps`, `measured.spill.local_mem_l1_miss_bw_per_sm_gbps`.

### 2.3 Online L2 Model API (Rust Crate Interface Called by the model core)

Because the model core's event loop runs at **per-CTA-phase segment** granularity (~1 ms for a 4096-tile GEMM), the L2 model exposes an online, O(1)-per-call, deterministic Rust API:

- `L2Model::new(l2_profile, initial_state: L2InitState /* Cold | Warm */, options: L2Options)`
  - Strictly instance-local state (no globals) so the model core can run parallel perturbation samples in Rust.
  - Exposes discrete perturbation options to the model core: `eviction_mode` (`SoftLru` | `StrictLru`) and `cross_wave_reuse` (`bool`).
- `L2Model::begin_tile(&mut self, sm_id, cta_id, outer_idx, t_ns)`
  - Registers the tile's active chunk footprint (derived from `access_map`) into the concurrent resident working set.
- `L2Model::end_tile(&mut self, sm_id, cta_id, outer_idx, t_ns)`
  - Moves the tile's chunks from the active resident set into the bounded recent-wave LRU history (sized by `l2_effective_capacity_bytes`).
- `L2Model::split(&mut self, sm_id, cta_id, outer_idx, op_id, k_begin, k_end, is_write, cache_policy, t_ns) -> L2SplitResult`
  - Evaluates an entire phase segment over the half-open inner-step interval `[k_begin, k_end)` (pass `[0, 1)` for prologue/epilogue), assuming concurrent resident CTAs advance at equal rate during the segment. Actual per-tile logical bytes are derived from `access_map` (`rows * row_bytes`, post-mask).
  - Returns `{ l2_bytes, dram_bytes, hit_ratio, sector_aligned_bytes, near_cliff }`, where `near_cliff` is a diagnostic flag only (perturbation sampling over `l2_effective_capacity_bytes` governs verdicts).
  - Guaranteed microsecond-level cost and deterministic output for fixed inputs and profile sample.
- `L2Model::account_spill(&mut self, sm_id, active_threads_in_sm, stack_frame_bytes, spill_loads_bytes, spill_stores_bytes) -> SpillTraffic`
  - Called once at CTA dispatch to estimate L1-resident vs. L1-overflow fraction of local/scratch-memory spill traffic and account for the overflow's L2 capacity pressure and L2/DRAM bytes.

---

## 3. Key Decisions

### D1. Unified Loop + WarpGroup IR vs. Kernel-Specific Flags

- **Decision**: Represent every CTA uniformly with `outer_loop` (`trip_counts` per CTA) and `launch.warp_groups` (`unified` | `producer` | `consumer`), plus `epilogue.overlaps_next_tile`.
- **Alternative**: Boolean flags (`is_persistent`, `is_warp_specialized`) with branched logic in the model core.
- **Trade-off**: Eliminates special-case branching in the scheduler and cleanly extends to persistent GEMM, split-K, stream-K, and FlashAttention across Triton, PTX, and AMD frontends.

### D2. `access_map` Encoding: 2D Strided Rectangles vs. Unrolled Trace vs. Symbolic IR

- **Decision**: Materialize dense per-`(cta_id, outer_idx)` entries as `strided_2d_per_outer_tile` (`[tile_coord_0, tile_coord_1, base_byte_offset, rows, row_bytes, row_stride_bytes, inner_step_byte_stride, is_masked]`), with an `explicit_blocks` fallback for irregular patterns.
- **Alternatives**: (A) Fully unrolled `(cta_id, inner_step, offset, len)` trace; (B) Symbolic affine expressions in the IR.
- **Trade-off**: (B) cannot express div/mod swizzling (`GROUP_SIZE_M`) without duplicating frontend evaluators inside the Rust core. (A) bloats to tens of MB for large GEMMs. The 2D strided encoding stays compact (~100 KB for 4096 CTAs), handles arbitrary nonlinear swizzles from any frontend, and preserves exact row/stride geometry for sector overfetch calculation.

### D3. Online Phase-Level Chunk-Sharing L2 Model vs. Analytical Wave Model vs. 128B Set-Associative Sim

- **Decision**: **Online Resident-Set + Bounded Recent-History Chunk Model**.
  - On `begin_tile` / `end_tile`, the model updates reference counts over normalized tensor tile chunks (preserving exact `sector_aligned_bytes` from the 2D stride descriptor).
  - On `split()` over `[k_begin, k_end)`, intra-wave reuse is derived from the chunk sharing factor `R` across currently resident CTAs (a chunk shared by `R` resident CTAs incurs `1/R` DRAM fetch when the active working set `W_active` fits in `C_eff`), combined with cross-wave hit credit from chunks still present in the recent-history LRU window.
  - Around `C_eff`, a soft eviction band (`[0.85 * C_eff, 1.15 * C_eff]`) smoothly degrades hit rate and sets `near_cliff = true`.
- **Alternatives**: (A) Static analytical wave footprint (`unique_bytes_per_wave <= L2_size`); (B) Full per-iteration 128B set-associative cache simulation.
- **Trade-off**: (A) ignores intra-wave K-step locality, cross-wave reuse, and persistent CTA skew. (B) violates the model core's 1 ms/sim budget by 100x-1000x and creates false precision since vendor L2 hash/replacement policies are undocumented (no reverse engineering). The chosen model runs in microseconds per `split()` call, aligns directly with the model core's phase-segment scheduler, and captures swizzle knees without brittle hard cutoffs.

### D4. Spill Traffic Attribution (L1-Resident vs. L2/DRAM Overflow)

- **Decision**: Combine `resources.spill_scope` with `(active_threads_per_sm * stack_frame_bytes)` vs. `measured.spill.l1_to_l2_spill_threshold_bytes_per_sm`. L1-resident spills pay `local_mem_l1_hit` cost (in the model core) with zero DRAM traffic; overflowing spill traffic pays `local_mem_l1_miss` cost and consumes L2 capacity/bandwidth via `account_spill()`.
- **Alternative**: Route all assembler-reported spills to DRAM, or ignore spill memory traffic.
- **Trade-off**: Captures both mild L1-cached spills and severe L2-thrashing spills without inspecting vendor machine code.

### D5. Python-to-Rust Boundary and Persistence Format

- **Decision**: Pass `KernelIR` structs directly from Python frontends to the Rust core via **PyO3** in memory (zero serialization overhead during autotuning). For on-disk caching and validation datasets, serialize `KernelIR` to **JSON** plus a compact **binary sidecar** (`.bin` fixed-layout records) for `access_map.entries`.
- **Alternatives**: (A) Always serialize full `KernelIR` + `access_map` as JSON across the boundary; (B) Binary-only format (Protobuf/FlatBuffers) with no human-readable JSON.
- **Trade-off**: PyO3 keeps the hot path fast and zero-copy for dense entry arrays; JSON + binary sidecar keeps cached IR human-inspectable without multi-megabyte JSON files for large grids.

---
