# Changelog

## 0.3.0

### Added

- NVIDIA L4 (`sm_89`) profile
- NVIDIA L40S (`sm_89`) profile
- NVIDIA GeForce RTX 3070 (`sm_86`) profile

### Fixed

- The L2 miss merging model no longer fails on GPUs with more than 128 SMs

## 0.2.0

### Added

- Fused attention for Triton: forward attention kernels, causal and non-causal, with grouped query attention
- MMA, barrier and shared memory loop counts read straight from the PTX, for GEMM and attention
- Load and store widths read from the PTX
- SFU and shuffle pipes in the model core, driven by measured per-SM throughput curves
- Per-CTA trip counts, shuffle ops and shared KV prefixes for causal GQA attention in the Kernel IR
- Profile for the NVIDIA DGX Spark GB10 (`sm_121`) and a `--nvcc` build option for the microbenchmarks
- Microbenchmarks for SFU and shuffle throughput, shuffle latency, the MMA, softmax and MMA chain, controlled global load issue, synchronous and `cp.async` stage loops, L1 capacity by shared memory carveout and L1 miss rate curves
- Linux wheels for `x86_64` and `aarch64` and a source archive, with the GPU profiles bundled
- `list_profiles()` and `profile_path()`, and `--profile` takes a bundled profile name
- Release workflow that attaches the wheels and SHA-256 sums to a GitHub Release
- Sealed v0.2 holdouts for GEMM and attention on the RTX 4070 and GB10, a preregistration, and a scorer with a freeze gate and per-report digests
- Audited scoring of published holdouts with an access log

### Changed

- Each iteration's MMAs and the work that depends on them run in series within a CTA
- A memory round trip takes at least its idle latency plus its bytes through the SM port, and the port is shared out of phase between the CTAs on an SM
- Global memory warp instruction issue per SM is capped by a measured rate
- Register spill bytes served by L2 or DRAM are also charged on the SM's local port
- The L2 bandwidth ceiling was measured again with 8 loads in flight and 16 warps, since the old kernel could not reach it
- The `sync_stage_loop` benchmark uses about 1 ms kernel windows and compile time loop counts, and adds large stage points

### Fixed

- Cold GEMM labels flushed L2 with writes, which left dirty writebacks inside the timed window
- Microbenchmark L2 flushes used writes, and streaming bandwidth was counted the wrong way
- The spill reload benchmark used scattered per-lane offsets instead of coalesced accesses
- The serial spill reload benchmark only walked half of the stack frame
- Mixed dtype attention compiles could change between runs after garbage collection

## 0.1.0

### Added

- Tile-level, event-driven performance model in Rust with an L2 reuse and miss merging model, occupancy and carveout rules, and perturbation draws
- Closed queueing network solver for pipelined loops
- Triton GEMM frontend that compiles ahead of time for any target with Triton 3.9.0
- GPU-free autotuning from Python and the `virtualgpu autotune` CLI, with parallel compiles and a Kernel IR cache
- CUDA microbenchmark suite `vgpu_bench` and the RTX 4070 (`sm_89`) profile
- Profile library with schema and provenance checks in CI
- Sealed workload holdout capture and scoring, used for the phase 1 and phase 2 evaluations
