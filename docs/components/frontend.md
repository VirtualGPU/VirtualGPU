# Triton frontend

How a Triton kernel becomes a `KernelIR` on a CPU, with no GPU. Code: `python/virtualgpu/frontend/`. The mechanisms the core runs today are listed in [design.md section 6.4](../design.md#64-mechanisms-in-the-core). If this file and the code disagree, the code is right and this file gets fixed.

## 1. Scope

The Triton frontend transforms a Triton kernel, concrete launch parameters, and a target GPU specification into a structured **Kernel IR** and backend resource report, running entirely on CPU without a physical GPU.

Responsibilities:
1. **AOT compilation** for arbitrary GPU targets (`GPUTarget` + backend assembler).
2. **Post-pipelining TTGIR extraction** of loops, materialized pipeline stages, async memory ops, MMA shapes, and sync points.
3. **Minimal scalar evaluation** of `program_id` arithmetic to recover tile coordinates and byte footprints (L2 swizzle recovery).
4. **Structural detection** of persistent kernels and warp specialization.
5. **Backend resource and spill parsing** from `ptxas --verbose` (registers, shared memory, stack frame, spill stores/loads).

Out of scope: L2 reuse modeling (the L2 model), event-driven performance simulation (the model core), functional tensor execution, non-Triton kernels (cuBLAS/CUTLASS/CUDA), and reverse engineering of vendor binaries or SASS.

## 2. Inputs and Outputs

### Inputs
- **Kernel source**: A `@triton.jit` function (or a cached `(ttgir, ptx, ptxas_log)` bundle for reproducible offline runs).
- **Concrete invocation**: Tensor shapes, strides, dtypes, `constexpr`/autotune config (`BLOCK_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `maxnreg`), and grid dimensions (or grid lambda).
- **Target spec** (subset of GPU profile): `backend` (`cuda` | `hip`), `arch` (e.g., `sm_89`, `sm_121`, `sm_90`), `warp_size`, and required toolchain version.

### Outputs (aligned to the `KernelIR` v0.1)
Output conforms directly to `KernelIR` v0.1 in `kernel-ir.md`:

| `KernelIR` Section | Frontend Mapping |
|---|---|
| `metadata` | `kernel_name`, `target_arch`, `frontend = "triton"`, `config`, `triton_version`, `ptxas_version` |
| `launch` | `grid`, `cluster_shape`, `num_warps`, `warp_groups` (1 `unified` group for standard kernels; split `producer`/`consumer` groups when warp-specialized), `dynamic_smem_bytes` |
| `resources` | `source = "ptxas"`, `registers_per_thread`, `static_smem_bytes`, `stack_frame_bytes`, `spill_stores_bytes`, `spill_loads_bytes`, `spill_scope` (`none` \| `prologue_epilogue` \| `inner_loop_confirmed` \| `inner_loop_assumed`), `barriers` |
| `tensors` | `id`, `dtype`, `element_bits`, `shape`, `strides`, `base_addr`, `cache_policy` (`ca`/`cg`/`cs`/`cv` from TTGIR load/store cache modifiers) |
| `body` | `outer_loop` (`trip_count_mode`: `uniform` \| `per_cta`, unifying standard `=1` and persistent `>1`); `prologue` / `pipeline_loop` (post-pipelining `num_stages`, `prefetch_distance`, `async_copy`: `none` \| `cp_async` \| `tma`, `ops` with `op_id`, `kind`, `warp_group_id`, `stage_offset`, `bytes_per_cta`, `transfer_width_bytes`, `tile_mnk`, `inst_mnk`, `op_class`, dtypes) / `epilogue` (with `overlaps_next_tile`) |
| `access_map` | Default `strided_2d_per_outer_tile` entries `[tile_coord_0, tile_coord_1, base_byte_offset, rows, row_bytes, row_stride_bytes, inner_step_byte_stride, is_masked]` evaluated per `(cta_id, outer_iter)` with boundary masks clamped into effective `rows`/`row_bytes`; degrades to `explicit_blocks` for irregular access |
| `diagnostics` | Soft confidence-lowering flags where simulation still runs (e.g., `ptxas_only_spill`, `unpipelined_fallback`) |
| `unsupported_features` | Hard-stop flags causing the simulator to return `unsupported` rather than guess (e.g., `atomic_work_stealing`, `cluster_smem`, `data_dependent_address`) |

## 3. Key Decisions and Trade-offs

### D1. AOT Compilation Pipeline: `triton.compile` + Direct `ptxas --verbose`
- **Decision**: Invoke Triton's public CPU-only compilation path (`triton.compile(ASTSource(...), target=GPUTarget(...), options=...)`) to produce post-pass `ttgir` and `ptx`, then run `ptxas --gpu-name <arch> --verbose` directly on the emitted PTX. Cache artifacts by `(kernel_ast_hash, signature, config, target, triton_ver, ptxas_ver)`.
- **Alternatives**:
  1. *Use `CompiledKernel.metadata` only*: Rejected. Triton's metadata exposes `n_regs`, `shared`, and a coarse `n_spills`, but omits separate spill store/load byte counts and stack frame size needed for spill modeling.
  2. *Parse Python AST directly*: Rejected. Misses Triton layout inference, MMA lowering, SMEM allocation, and whether the compiler actually pipelined the loop.
  3. *Disassemble cubin (`nvdisasm`)*: Rejected by the no-reverse-engineering principle and unnecessary for tile-level modeling.
- **Trade-off**: Requires a host `ptxas` binary that supports the target architecture (e.g., CUDA 12.9+/13.0+ for `sm_121`). Compilation caching keeps repeated autotune sweeps fast.

### D2. Extraction Stage: Post-Pipelining TTGIR (Not TTIR, Not LLIR/PTX)
- **Decision**: Parse the final `ttgir` emitted *after* `TritonGPUPipeline` and layout/MMA passes.
- **Alternatives**:
  1. *Pre-pass TTIR*: Rejected. `num_stages` is only a hint before pipelining; the compiler may silently clamp or disable pipelining when shared memory or loop structure prevents it.
  2. *LLIR or PTX*: Rejected. Tile shapes, tensor layouts, and multi-buffered stage descriptors are lowered into scalarized pointer arithmetic, making tile-level recovery fragile.
- **Trade-off**: TTGIR op names (`ttg.async_copy_global_to_local`, `ttng.async_tma_copy_global_to_local`, `ttg.local_alloc`, `ttg.warp_specialize`) drift across Triton releases. Isolate dialect matching behind a thin versioned adapter and parse MLIR text with a small dedicated parser rather than binding to LLVM/MLIR C++ internals.

### D3. Swizzle & Tile-Order Recovery: Minimal Scalar Evaluator on TTGIR
- **Decision**: Build a small deterministic scalar interpreter (~200-300 LOC) over the backward slice of global load/store addresses and masks (`tt.get_program_id`, `tt.get_num_programs`, `arith.*`, `scf.for` induction variables). Given concrete `(shapes, strides, grid, config)`, evaluate per-`pid` tile coordinates and byte ranges directly.
- **Alternatives**:
  1. *Regex/pattern matching for `GROUP_SIZE_M`*: Rejected. Breaks on diagonal swizzles, split-K, grouped GEMM, or non-standard index math.
  2. *Symbolic/polyhedral analysis*: Rejected as over-engineered; concrete launch dimensions are always known at simulation time.
  3. *CPU execution (`triton-cpu`)*: Rejected. Too slow and executes tensor math we do not need.
- **Trade-off**: Handles arbitrary integer swizzle logic cleanly. If an address slice depends on a runtime memory load (e.g., MoE routing or paged KV indices), evaluation stops on that operand and emits `data_dependent_address` in `unsupported_features` instead of guessing.

### D4. Persistent Kernel and Warp Specialization Recognition
- **Decision**: Detect persistent kernels structurally in TTGIR: an outer `scf.for` loop whose bounds/step depend on `tt.get_program_id` / `tt.get_num_programs` (or a `NUM_SMS` constant) and whose body wraps both tile-coordinate computation, an inner reduction `scf.for`, and an output tile store. Detect warp specialization from explicit TTGIR warp-partition ops (e.g., `ttg.warp_specialize` / producer-consumer barrier pairs) and record the producer/consumer warp split.
- **Alternatives**:
  1. *Assume 1-CTA-1-tile everywhere*: Rejected (fails on persistent GEMM and multi-tile kernels).
  2. *Require user annotations*: Rejected as primary mechanism because it breaks drop-in autotuning (optional hints may be supported as a fallback).
- **Trade-off**: Covers standard static-strided persistent loops cleanly; dynamic atomic work-stealing loops cannot be statically resolved and emit `atomic_work_stealing` in `unsupported_features`.

### D5. Register Spill Extraction and Attribution
- **Decision**: Parse `ptxas --verbose` for `registers`, `smem`, `stack frame`, `spill stores` (bytes), and `spill loads` (bytes). When spills are non-zero, scan the emitted PTX (public ISA) for `ld.local`/`st.local` placement relative to the inner loop:
  - If visible in PTX, tag `spill_scope` as `inner_loop_confirmed` or `prologue_epilogue`.
  - If `ptxas` reports spills not present in PTX (introduced during `ptxas` register allocation), conservatively tag `inner_loop_assumed` so the model core models the worst-case per-iteration penalty with widened uncertainty.
- **Alternatives**:
  1. *Report total spill bytes only*: Rejected. A small prologue spill has negligible cost, whereas the same bytes spilled inside a K-loop scale with `K / BLOCK_K` and can flip config rankings.
- **Trade-off**: Without SASS inspection we cannot know the exact instruction placement of `ptxas`-only spills; explicit `inner_loop_assumed` tagging avoids false precision.

### D6. Initial Target Hardware Coverage (`sm_89` and `sm_121`)
- **Decision**: Prioritize op-table validation for the two available physical GPUs:
  - **RTX 4070 (`sm_89`, Ada)**: `cp.async` (`ttg.async_copy_global_to_local`) + warp-level `mma.sync`.
  - **DGX Spark GB10 (`sm_121`, client/edge Blackwell)**: TMA (`cp.async.bulk.tensor` / `ttng.async_tma_copy_global_to_local`) + warp-level `mma.sync` (including Blackwell block-scaled formats where exposed by the toolchain). Per NVIDIA public architecture/PTX documentation, `sm_121` does **not** expose datacenter `sm_100` `tcgen05`/TMEM/DSMEM pipelines or Hopper `wgmma`; exact supported MMA shapes and TMA ops must follow official CUDA 12.9+/13.0+ docs and be verified on GB10.
- Keep `sm_90` (`wgmma`) and AMD `gfx942` op mappings as extensible adapter tables, marked unvalidated until hardware is available.

## Fused attention

x

### Scope

This document analyzes how Triton's forward fused-attention kernel lowers on `sm_89` (RTX 4070) after software pipelining, how its tile structure maps to `KernelIR` v0.1 (`schemas/kernel_ir.schema.json`), and which minimal schema extensions are needed from the L2 model before implementing the extractor.

Not covered: backward pass, paged KV cache, FlashDecoding split-KV, and `sm_121` TMA validation (not validated yet).

### Post-pipelining structure (`sm_89`, Triton 3.9.0)

We compiled the Triton 3.9 fused-attention forward kernel offline (`CUDA_VISIBLE_DEVICES=""`, `ptxas 12.9 V12.9.86`, `Z=4, H=32, N_CTX=2048`, `fp16` I/O, `fp32` accumulator) using `tl.make_tensor_descriptor` (`tl.make_block_ptr` was removed in Triton 3.9; on `sm_89`, tensor descriptors lower to `tt.addptr` + `cp.async`).

| Variant (`D`, `BMxBN`, `w`, `req_s`) | `scf.for` | SMEM (B) | Regs | Spill S/L (B) | SMEM bufs (`K`, `V`) | Loop `wait_group` / `commit` | PTX `mma` / `ex2` / `shfl` (per warp/iter) |
|---|---:|---:|---:|---:|---|---:|---:|
| Non-causal `128`, `128x64`, `w=4`, `s=1` | 1 | 49152 | 255 | 1132 / 828 | none (`ld.global`) | none / 0 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=2` | 1 | 65536 | 255 | 248 / 228 | `1x64x128`, `1x64x128` | `0` / 2 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=3` | 1 | 98304 | 255 | 416 / 356 | `2x64x128`, `2x64x128` | `2` / 2 | 256 / 68 / 16 |
| Non-causal `128`, `128x64`, `w=4`, `s=4` | 1 | 131072 | 255 | 416 / 356 | `3x64x128`, `3x64x128` | `4` / 2 | 256 / 68 / 16 |
| Causal `128`, `128x64`, `w=4`, `s=3` | 2 | 98304 | 255 | 720 / 684 | 2 loops x (`2x64x128` x 2)* | `2` / 2 | 256 / 68 / 16 |
| Non-causal `64`, `64x64`, `w=4`, `s=3` | 1 | 40960 | 144 | 0 / 0 | `2x64x64`, `2x64x64` | `2` / 2 | 64 / 34 / 8 |
| Causal `128`, `64x64`, `w=4`, `s=3` | 2 | 90112 | 255 | 0 / 0 | 2 loops x (`2x64x128` x 2)* | `2` / 2 | 128 / 40 / 64 |

*\*In causal mode, Triton emits two sequential `scf.for` loops with separate `ttg.local_alloc` ops, but reuses the same SMEM allocation across them (`98304 B`, not doubled).*

#### Key structural findings vs. GEMM

1. **Q is staged in SMEM and reloaded every KV iteration.**  
   In the prologue, Q is loaded synchronously (`tt.load`) and immediately written to SMEM via `ttg.local_alloc %q` (`128x128xf16 = 32768 B`, accounting for the extra 32 KB in `98304 = 32768(Q) + 32768(2xK) + 32768(2xV)`). Inside the KV loop, `ttg.local_load %q_smem` reloads Q from SMEM on every iteration before `dot(Q, K^T)`. Thus per-iteration SMEM read volume is `Q + K + V` (`65536 B`), not just `K + V`.
2. **K and V are co-pipelined into one stage on `sm_89`.**  
   Even though `desc_v.load()` appears after online softmax in Python, `TritonGPUPipeline` hoists both K and V loads into the same pipeline stage:
   - Prologue issues `async_copy(K) + commit`, `async_copy(V) + commit` per prefilled buffer.
   - Loop body opens with a **single** `ttg.async_wait %k_tok, %v_tok {num = N}` (`cp.async.wait_group N`) waiting on both K and V together, then executes `dot(Q, K^T)` -> online softmax -> `dot(P, V)`, and finishes by issuing the next K and V async copies (2 commit groups per iteration).
   - Therefore, our existing PTX ISA rule `prefetch_distance = wait_group_N // commits_per_stage` yields `0 / 0 / 1 / 2` for `req_stages = 1 / 2 / 3 / 4` with exact PTX/TTGIR agreement, identical to GEMM.
   - **Caveat for future variants:** kernels with split K/V waits (one `wait_group` before `dot(Q, K^T)` and a second before `dot(P, V)`) will have two distinct `wait_group` counts inside one loop body; the extractor must detect multiple distinct waits rather than blindly taking `min()`.
3. **Two distinct MMA ops + heavy SFU/shuffle softmax per iteration.**  
   Each KV iteration executes:
   - `mma_qk`: `tile_mnk = [BM, BN, D]`, `inst_mnk = [16, 8, 16]`, `count = (BM/16)*(BN/8)*(D/16)`.
   - Online softmax: row-max `tt.reduce`, scale/sub, `math.exp2` (`64` `ex2.approx`/warp for `P`), row-sum `tt.reduce`, `math.exp2` (`4` `ex2.approx`/warp for `alpha`), `acc * alpha` rescaling, and `f32 -> f16` cast. At `BM=128, BN=64, w=4`, SFU (`68 ex2.approx`/warp) and warp shuffles (`16 shfl.sync`/warp; `64` at `64x64`) are first-order latency contributors alongside MMA.
   - `mma_pv`: `tile_mnk = [BM, D, BN]`, `inst_mnk = [16, 8, 16]`, `count = (BM/16)*(D/8)*(BN/16)`.
4. **Spill pressure is much sharper than in GEMM.**  
   `BM=128, BN=64, D=128, w=4` hits 255 regs with `248 to 720 B` of `ptxas_only_spill` across all `num_stages`, whereas `BM=64, BN=64` has `0 B` spill. Accurate spill modeling directly governs config ranking.
5. **Causal attention compiles to two sequential `scf.for` loops with per-CTA trip counts.**  
   - Loop 1 (unmasked off-diagonal): `start_n in [0, start_m * BM)`, trip count `start_m * (BM // BN)` (varies linearly from `0` to `N_CTX/BN - BM/BN` across CTAs).
   - Inter-loop drain (`ttg.async_wait {num = 0}`) and refill.
   - Loop 2 (masked diagonal): `start_n in [start_m * BM, (start_m + 1) * BM)`, constant trip count `BM // BN` (e.g., `2`), adding causal comparison + `arith.select`.

### Mapping attention to `KernelIR` v0.1

Most of fused attention maps cleanly onto existing `KernelIR` v0.1 primitives:

| Attention Component | `KernelIR` v0.1 Mapping | Notes |
|---|---|---|
| Grid `(cdiv(N_CTX, BM), Z*H, 1)` | `launch.grid = [grid_m, Z*H, 1]` | 2D grid already supported |
| Tensors `Q, K, V, Out` | `tensors`: 4D `[Z, H, N_CTX, D]` | Arbitrary rank already supported in schema |
| `Q` load + SMEM staging | `body.prologue.ops`: `gmem_load(Q)` + `smem_store(Q)` | Uses the non-pipelined prologue slot reserved by the L2 model and the model core |
| KV loop `K, V` loads | `body.pipeline_loop.ops`: two `gmem_load` (`acc_K`, `acc_V`) | `stage_offset = prefetch_distance` |
| `Q, K, V` SMEM reads | `body.pipeline_loop.ops`: `smem_load` (`Q + K + V` bytes/iter) | Must include Q reload bytes from `ttg.local_load` |
| `Q*K^T` and `P*V` | Two `mma` ops (`mma_qk`, `mma_pv`) in `pipeline_loop.ops` | Schema `ops` array already allows multiple `mma` ops |
| Online softmax `exp2` | `scalar_op` (`kind: "sfu"`, `dtype: "f32"`, `count: ex2_per_warp`) | Schema already defines `sfu` with `insts_per_warp_per_iter` |
| Softmax ALU & reductions | `scalar_op` (`kind: "alu"`, `dtype: "f32"`, `count: ...`) | See Section 4 for warp-shuffle serialization gap |
| Final `acc / l_i` + `Out` store | `body.epilogue.ops`: `alu`/`sfu` + `gmem_store(Out)` | Fits existing epilogue block |
| Cross-Q-block K/V L2 reuse | `access_map`: `strided_2d_per_outer_tile` for `acc_Q/K/V/O` | CTAs with the same `off_hz` (or same KV head in GQA) emit identical `base_byte_offset` for `K`/`V`, so the L2 model's L2 model captures inter-CTA KV reuse without schema changes (`tile_coord_0 = start_m`, `tile_coord_1 = off_hz`) |
