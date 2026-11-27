# Changelog

## 0.3.0

### Added

- Huawei Ascend 910B3 preregistered sealed multi-shape holdout (`validation/ascend-910b3-*`, `docs/results/ascend910b3-holdout.md`): 16 unseen GEMM and Attention shapes across 48 sessions on `0.3.1-ascend`; both Attention cohorts passed (`0.00%` top-5 regret, `+0.089` to `+0.122` tau-b above baselines) while GEMM failed, keeping `unvalidated_target_ascend` gating in place and recording post-hoc frontend and harness findings
- Post-hoc GB10 microbenchmarks for single-buffer bursts and L2 sharing (`sync-loop-burst`, `l2-paths`), dependent MMA chains (`mma-ilp`) and the attention compute base (`attention_compute`), with their registrations and data
- Design for long-launch profile capture and a launch-duration audit of the reference profiles
- Huawei Ascend 910B3 Cube/Vector split modeling and Triton-Ascend forward GQA Attention frontend: L2-resident inter-core GM workspace accounting, multi-buffered AIC/AIV steady-state overlap, isolated FFTS+ cross-core sync microbenchmark (`cv_sync_hop_ns_by_active_sms`, profile `0.3.1-ascend`), `equal_to_1` stride specialization in Ascend AOT (fixing GEMM `multibuffer=False` overestimate), and CPU-only `extract_ascend_attention_kernel_ir` routed via `extract_attention_kernel_ir`
- Long-launch sizing, capture and audit tools for NVIDIA MLP and active-SM read curves
- Huawei Ascend 910B3 (`ascend910b3`) support: AscendC M1/M2 microbenchmark suite (`microbench/ascend/`), measured `profiles/ascend910b3.json` profile, optional `specs.ascend` and `resources.vendor_extensions.ascend` on-chip SRAM checks (`OccupancyError::TooMuchOnChipBuffer`), automatic `unvalidated_feature` / `unvalidated_target_ascend` flagging, and CPU-only Triton-Ascend HIVM GEMM frontend
- Post-hoc microbenchmarks for the merged wait of single-buffer loops (delayed followers, free-running groups) and issue stamps for the attention ablation, with GB10 data
- Post-hoc GB10 GEMM analysis of H3 and H4 with the `wave-rescan`, `b-stream` and `staged-mma` microbenchmarks and their frozen predictions
- Post-hoc single-buffer loop microbenchmarks and a fused attention ablation for the `num_stages = 2` round trip, with RTX 4070 and GB10 data
- Post-hoc analysis of GB10's 2-byte register-staged GEMM loads (H5), narrow-load microbenchmarks and frozen predictions for their candidate model forms
- Candidate v0.3 GEMM and attention holdouts for RTX 4070 and GB10, with a historical shape audit and CPU AOT resource checks
- NVIDIA RTX PRO 5000 72GB Blackwell (`sm_120`) profile
- Draft the v0.3 workload holdout protocol and audit the two reference profiles for MLP repeat coverage
- Post-hoc Nsight Compute capture for dynamic local loads and stores in the spill microbenchmark and causal attention
- Optional `l2_bw_by_active_sms_cp_async` curve in the GPU profile schema and microbenchmarks, and physical sanity checks in `scripts/check_profiles.py`
- NVIDIA A100-SXM4-80GB (`sm_80`) profile
- NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`) profile
- NVIDIA L4 (`sm_89`) profile
- NVIDIA L40S (`sm_89`) profile
- NVIDIA GeForce RTX 3070 (`sm_86`) profile

### Fixed

- The KernelIR's shared memory reads count the MMA operands' `ldmatrix` bytes per warp from the PTX, instead of each operand tile once per CTA (2 to 6 times fewer bytes on the v0.2 holdout configs)
- Cross-wave L2 reuse: chunks that end together share one stream clock, and re-reads hit per K slice under LRU with the effective L2 capacity. Kept on GB10 microbenchmarks that agree in direction but fail the cross-run capture rule: forward re-reads at 1.5 times L2 or more do not hit in any run (the old clock charged 50% to 90%); from about 0.75 to 1.25 times L2 the hardware steady state is bistable across runs, a known limit; the reversed-walk detail is provisional; the sealed v0.3 holdout decides
- Mark the GB10 MLP medians from five independent sweeps as provisional after the dispersion audit and propose long-launch recapture on both GPUs
- Scale the DRAM streaming bandwidth buffer to at least 16x L2 capacity, map per-architecture unified L1 capacity in `collect_profile.py`, re-collect `rtxpro6000.json` with locked SM clocks and `rtx3070.json` on an idle GPU
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
