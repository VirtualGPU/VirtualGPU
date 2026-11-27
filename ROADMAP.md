# Roadmap

The one plan for VirtualGPU, from now to 1.0.0 and after. 1.0.0 is the first release we announce. Each item is a checkbox. Check it off in the same change that finishes it, and add new work as a new box under the right version. A version is done when every box in it is checked and its exit report is published.

## What VirtualGPU is for

VirtualGPU is a performance oracle for GPU kernels. Give it two or more versions of a kernel and a GPU, and it tells you which is faster, by roughly how much, why, and when it can't tell. It runs on a CPU in seconds, from a profile of the GPU, with no access to the card.

That one answer powers several tools:

- Autotuning without the GPU: rank Triton autotune configs and only measure the shortlist.
- Comparing implementations: split-K against stream-K, Triton against CUTLASS, two attention designs.
- Performance checks in CI: flag a change that makes a kernel slower on GPUs the CI machine doesn't have.
- A scorer for kernel search and AI agents: thousands of candidate kernels ranked per minute, deterministic, with a reason for each verdict, so only the best few reach a real GPU.
- Later: which GPU suits a set of kernels, and which kernels in a whole model are furthest from their best speed.

What it will not do: predict exact run times, compute kernel outputs, or replace a final check on real hardware.

## How it fits together

- Frontends turn a kernel into one Kernel IR: Triton for NVIDIA, AMD and Ascend, and PTX for everything that compiles to it (CUDA, CUTLASS, CuTe, TileLang, ThunderKittens).
- The Rust core plays the kernel out on a GPU profile and returns ranked verdicts with the resource that bounds each variant.
- GPU profiles are JSON files of public specs and our own microbenchmark results, one per SKU, measured once by anyone who has the card.
- Every tool (CLI, Python API, Triton plugin, local server, GitHub Action) reads the same versioned report format.

## Rules for every milestone

A milestone that changes the model ends with a sealed test:

1. Commit the preregistration (cases, cache regimes, baselines, pass criteria) before any label is captured.
2. Capture and seal a fresh holdout. An opened holdout is never reused.
3. Freeze the model and baselines with a tag, archive the predictions, then open the labels and score.
4. Publish the report in `validation/` and the summary in `docs/results/`, pass or fail.

Pass criteria for a kernel family on a GPU, unchanged from v0.2: median Kendall tau-b at least 0.05 above each baseline, mean top-5 regret at most 1% and no more than 0.5 points worse than any baseline, in warm and cold L2. A GPU passes when every family run on it passes.

A milestone starts only when the one before it has published its exit report. Work that needs no hardware (frontends, docs, tooling) can start early.

## GPUs to profile by 1.0.0

Required means the profile ships at 1.0 and has a sealed holdout. Profile only means it ships flagged as not validated.

| Architecture | GPU | Target | Status |
| --- | --- | --- | --- |
| Ampere `sm_80` | A100 80GB SXM | Required | Profile, holdout open |
| Ampere `sm_86` | A10G | Required | To do |
| Ampere `sm_86` | RTX 3090 | Profile only | To do |
| Ampere `sm_86` | RTX 3070 | Profile only | Profile |
| Turing `sm_75` | T4 | Profile only | To do |
| Ada `sm_89` | RTX 4070 | Required | Validated (dev GPU) |
| Ada `sm_89` | RTX 4090 | Required | Profile |
| Ada `sm_89` | L4 | Required | Profile |
| Ada `sm_89` | L40S | Required | Profile |
| Hopper `sm_90` | H100 SXM | Required | To do |
| Hopper `sm_90` | H200 | Required | To do |
| Hopper `sm_90` | H100 PCIe | Profile only | To do |
| Hopper `sm_90` | H20 | Profile only | To do |
| Hopper `sm_90` | GH200 | Profile only | To do |
| Blackwell `sm_100` | B200 | Required | To do |
| Blackwell `sm_100` | GB200 | Profile only | To do |
| Blackwell `sm_103` | B300 | Profile only | To do |
| Blackwell `sm_120` | RTX 5090 | Required | To do |
| Blackwell `sm_120` | RTX 5080 | Profile only | To do |
| Blackwell `sm_120` | RTX PRO 6000 | Required | Profile |
| Blackwell `sm_120` | RTX PRO 5000 72GB | Profile only | Profile |
| Blackwell `sm_121` | GB10 | Required | Validated (dev GPU) |
| AMD CDNA3 | MI300X | Beta, holdout if possible | To do |
| AMD CDNA3 | MI325X | Beta | To do |
| AMD CDNA4 | MI355X | Beta | To do |
| AMD RDNA3 | RX 7900 XTX | Beta | To do |
| Ascend | 910B3 | Beta, holdout if possible | Validated |

## 0.3.0: close the known model gaps

Fix what made v0.2 miss on the RTX 4070 and GB10, and fix the profiles we already have.

- [x] Spill reloads cost latency on the critical path, using `spill_reload_latency_by_count`
- [x] Spill traffic moves from L1 to L2 as a step at the L1 capacity for the chosen carveout (`l1_to_l2_spill_threshold_by_carveout`)
- [ ] Dynamic count of local loads per iteration, so the two causal attention loops are not summed into one spill figure
- [x] Split the static spill figure of causal attention between its two loops (`spill_loop_count`)
- [ ] Capture `microbench.spill_counters` on the RTX 4070 and check the loop split with `scripts/posthoc_spill_split.py`
- [x] Do not charge local memory bandwidth on a phase's chain on top of the measured reload latency (#15)
- [x] Take the per-launch frame cost out of `extra_ns_vs_zero` in `spill-latency` (#15)
- [ ] Recollect `spill_reload_latency_by_count` on every profile with the launch cost taken out
- [x] Register a decision rule and freeze predictions for the spill mechanisms before `spill-latency` is recaptured with long launches (#15, [v0.3-spill-decision-registration.md](docs/results/analysis/v0.3-spill-decision-registration.md))
- [ ] Score the spill decision on the GB10 and RTX 4070 long-launch profiles with `scripts/spill_decision.py` and report it
- [x] Size the spill frames' L1 footprint by the CTAs an SM actually holds, not full residency (`simulate` in `sim.rs`), which overcharges grids under one wave such as the one-CTA-per-SM `spill-latency` kernel
- [ ] Re-check the spill frame footprint on the qualified 0.23.0 profiles with `scripts/recheck_mechanisms.py` ([v0.3-mechanism-recheck-baseline.md](docs/results/analysis/v0.3-mechanism-recheck-baseline.md))
- [ ] Find out why `spill-latency` frames stay in L1 past the pointer-chase L1 cliff on GB10 (512 B frames at 96 to 160 KiB per SM, [carveout probe](docs/results/analysis/v0.3-gb10-carveout-probe-reregistration.md)), before the L1 to L2 step for spill frames is changed; after the spill decision is scored (#21)
- [ ] Re-run miss merging under the round-trip port floor: since `e51f8ef` the `l2_merge_by_sharing` error grew from 4.8% to 12.2% (RTX 4070) and 4.6% to 8.8% (GB10), most at fewer CTAs than SMs; confirm on the 0.23.0 GB10 profile with `scripts/recheck_mechanisms.py` before registering a fix ([v0.3-mechanism-recheck-baseline.md](docs/results/analysis/v0.3-mechanism-recheck-baseline.md))
- [x] Capture `microbench.spill_counters` on the RTX 3070 as a first post-hoc look at static against dynamic local loads
- [ ] Narrow register-staged loads on unaligned shapes (H5): redo both forms with launches over 1 ms and captures that pass the dispersion rule, then decide
- [ ] Synchronous load GEMM (`num_stages = 1`, +9.1% warm): find the missing mechanism with a microbenchmark
- [ ] GB10 decode shapes are predicted too compressed: find out why
- [x] Microbenchmark for `non_mma_issue_rate_per_sm_gips` (the higher of pure FFMA and an FFMA and LOP3 mix)
- [ ] Measure `non_mma_issue_rate_per_sm_gips` and the local memory rates that are still estimated, on every profile
- [ ] Run the registered same-device check of the measured non-MMA issue rate on the qualified GB10 and RTX 4070 profiles ([v0.3-non-mma-same-device-check-registration.md](docs/results/analysis/v0.3-non-mma-same-device-check-registration.md))
- [ ] The core charges a stage synchronization (`sync_ns`) on every pipeline step, also on a loop with no barrier: decide whether a barrier-free loop should pay it (no v0.2 kernel has such a loop)
- [ ] The core charges one `sync_ns` per step whatever the number of barriers in the loop: find out whether the barrier count should enter it
- [ ] Fix the measurement problems in the rtxpro6000, a100-80gb and rtx3070 profiles (issue #8)
- [ ] Sealed v0.3 holdout decides the cross-wave L2 reuse change

- [ ] Exit test: pass on fresh GEMM and attention holdouts on both the RTX 4070 and GB10.

## 0.4.0: data center NVIDIA (Ampere and Hopper)

Most inference kernels are tuned for data center cards, so the model has to work there first.

- [ ] H100 SXM and H200 profiles, plus A10G, measured with the same suite
- [ ] Hopper mechanisms in the frontend, Kernel IR and core: TMA loads and stores, `wgmma` with its async wait, warp specialization with producer and consumer warp groups, register reallocation (`setmaxnreg`), persistent kernels with an outer tile loop, epilogues that overlap the next tile
- [ ] Thread block clusters and distributed shared memory, for the kernels Triton emits them in
- [ ] New microbenchmarks for each mechanism, under the same no fitting rule
- [ ] Publish wheels to PyPI from the release workflow with build attestations, so testers install with one command (no announcement yet)

- [ ] Exit test: sealed holdouts for GEMM and attention on H100 and A100 with a model frozen before their labels exist, passing.

## 0.5.0: any Triton kernel

Move from two kernel families to the kernels that make up an inference engine.

- [ ] A general Triton frontend: read any `@triton.jit` kernel and its real call arguments, not only GEMM and attention shapes
- [ ] Kernel families, each with its own holdout cases: split-K, stream-K and grouped GEMM; reductions and norms (softmax, LayerNorm, RMSNorm); fused elementwise kernels (SwiGLU, rotary embedding, quantize and dequantize); decode attention with split KV
- [ ] Data dependent kernels with a workload description as input: MoE grouped GEMM with an expert routing histogram, paged attention with a page table
- [ ] Low precision: FP8 GEMM on Ada and Hopper, measured per dtype
- [ ] Autotuning for any kernel from the CLI and `@virtualgpu.autotune`, driven by real call arguments
- [ ] Warn when a run targets a profile for a different GPU than the one in the machine

- [ ] Exit test: every new family passes on H100 and on one Ada GPU.

## 0.6.0: PTX frontend and implementation comparison

Cover kernels that are not written in Triton, and compare unrelated implementations of one operator.

- [ ] PTX frontend: recover tiles, loops, pipelining and memory traffic from public PTX, with registers and spills from `ptxas --verbose`. Inputs: a `.ptx` or `.cubin` with embedded PTX, or a CUDA source compiled with `nvcc`
- [ ] First libraries through PTX: CUTLASS and CuTe GEMM, TileLang, ThunderKittens
- [ ] `virtualgpu compare`: rank different implementations of one operator (Triton against CUTLASS, split-K against stream-K), with the bound resource for each
- [ ] Core wheels for macOS and Windows that take PTX or Kernel IR directly, so the oracle runs on any laptop; Triton input stays Linux and WSL2
- [ ] Pass criteria for comparison: pair accuracy on resolved pairs, written into a preregistration before any label

- [ ] Exit test: a sealed holdout of CUTLASS GEMM configs and a sealed comparison holdout (several implementations per operator) on H100, passing.

## 0.7.0: Blackwell

- [ ] B200 profile and data center Blackwell mechanisms: `tcgen05` MMA, Tensor Memory, two CTA MMA, cluster launch control
- [ ] Block scaled low precision (FP8, MXFP8, NVFP4) on `sm_100` and `sm_120`
- [ ] RTX 5090 profile
- [ ] Validate the consumer and workstation Blackwell profiles (RTX 5090, RTX PRO 6000) and the Ada profiles (RTX 4090, L4, L40S) with a smaller sealed GEMM and attention holdout each

- [ ] Exit test: full holdouts pass on B200; every profile marked Required in the GPU table that exists so far has passed its holdout.

## 0.8.0: tools people use every day

Turn the oracle into tools that fit how people already work.

- [ ] Stable formats: report schema v1, Kernel IR schema v1 and profile format v1, each versioned, with loaders for older versions and a written rule for what counts as breaking after 1.0
- [ ] Stable Python API: `rank`, `compare`, `explain`, documented and typed
- [ ] `virtualgpu serve`: a local batch server with a JSON API for search loops and AI agents, with a cache of compiled kernels and a throughput target written into the docs
- [ ] A GitHub Action that predicts a kernel change on a list of GPU profiles and comments when it gets slower
- [ ] Profile hub: `virtualgpu profile collect` measures a GPU in one command, an issue form and CI check validate a submitted profile and its provenance, and a review checklist covers the rest
- [ ] Public dataset: every sealed holdout so far (kernel, config, shape, GPU, driver, clocks, time) published with a loader, as a benchmark for performance models
- [ ] Docs site at virtualgpu.site built from `docs/`: install, quick start, reading a report, comparing kernels, using it from an agent, contributing a profile
- [ ] Runnable examples: tune a GEMM, an attention and a norm kernel for a GPU you don't own, compare a Triton and a CUTLASS GEMM, gate a PR in CI

- [ ] Exit test: at least five people outside the project install from PyPI, finish each example on a GPU they don't own, and the results hold when checked on that GPU. Their problems are fixed before 0.9.0.

## 0.9.0: AMD and Ascend beta, release candidate

- [ ] AMD through Triton's AMD backend and the public CDNA and RDNA ISA docs: MFMA and WMMA, LDS, Infinity Cache, registers and spills from the compiler's public metadata
- [ ] MI300X, MI325X, MI355X and RX 7900 XTX profiles, with a ROCm port of the microbenchmark suite
- [x] Ascend: a sealed holdout on the 910B3 if hardware time allows
- [ ] AMD and Ascend predictions carry `unvalidated_feature` unless a sealed holdout passed
- [ ] 1.0.0 release candidates through the full release workflow
- [ ] Launch evaluation: one preregistered sealed holdout across every Required GPU and every kernel family, scored with the frozen release candidate. This is the number the launch reports

- [ ] Exit test: the launch evaluation is published, pass or fail per GPU, and every failure is either fixed in a new release candidate with a fresh holdout or listed as a known limit.

## 1.0.0: launch

Everything here must be true before we announce.

- [ ] 0.3.0 to 0.9.0 exit reports published
- [ ] Every Required GPU in the table has a profile and a passing sealed holdout; every other profile is marked not validated
- [ ] Triton any kernel and PTX input both work on the documented kernel families
- [ ] PyPI install and the quick start work on a clean Linux machine, in WSL2, and (PTX input) on macOS and Windows
- [ ] Report, Kernel IR and profile formats v1 frozen and documented
- [ ] `serve`, the GitHub Action and the profile hub are live and documented
- [ ] Public dataset released
- [ ] Docs site, README and examples match the release
- [ ] Security policy, issue forms and the profile review flow are live
- [ ] CI green on `main`, release workflow tested end to end with a release candidate
- [ ] Technical report with the method, the launch evaluation and the limits, plus a launch post on how to contribute a profile

## After 1.0.0

Rough order:

- [ ] 1.1: AMD validated: sealed holdouts on MI300X and MI355X, AMD out of beta
- [ ] 1.2: Hardware what-if: rank GPUs for a set of kernels. Needs coarse absolute calibration per kernel family, checked on sealed holdouts across GPUs
- [ ] 1.3: Whole model view: take a `torch.compile` graph or a serving trace, list the kernels furthest from their achievable speed
- [ ] 1.4: Multi GPU kernels: collectives and communication overlap over NVLink and PCIe
- [ ] Profiles from specs only for announced GPUs, reported as trends
- [ ] Effects the core leaves out today: shared memory bank conflicts, instruction scheduling, clock and power limits
- [ ] More vendors through the same Kernel IR, each one beta until a sealed holdout passes
