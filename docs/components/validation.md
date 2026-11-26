# Validation

How accuracy is measured: candidate matrix, hardware labels, baselines and scoring. Code and sealed records: `validation/`, indexed in [validation/README.md](../../validation/README.md). Results per release: [docs/results/](../results/).

## Scope

Measure whether VirtualGPU ranks compiled Triton GEMM configurations well enough to shortlist candidates, and whether its pairwise calls survive measurement noise. Use dense GEMM with one output tile per CTA for the shared test. Start with FP16 input and output and FP32 accumulation. Treat persistent scheduling, warp specialization, and target-specific copy paths as separate diagnostic cases. The evaluation code and baselines belong to validation and are independent of the event model.

Use the RTX 4070 (Ada, `sm_89`) as the proposed development GPU and DGX Spark (GB10, `sm_121`) as the proposed profile-only transfer target. Freeze the model and baselines before opening GB10 GEMM labels. A profile-only transfer test allows independently measured GB10 microbenchmarks. It estimates transfer between this pair of devices.

Both targets can use dense `mma.sync.m16n8k16` with FP16 or BF16 operands and FP32 accumulation. Record the actual MMA opcode, instruction shape, types, and copy path from compiler-emitted PTX. The PTX ISA lists basic `cp.async.bulk.tensor` for `sm_90` and later, including GB10. Gate a GB10 CTA-local TMA diagnostic on compiler output and a successful device run. Record the exact PTX target suffix and consult Target ISA Notes for each optional qualifier or additional MMA type. Hopper `wgmma`, data-center Blackwell `tcgen05`, and data-center TMA behavior have no hardware coverage in this experiment. [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus), [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).

## Interfaces

| Producer | Required input to validation |
| --- | --- |
| Suite manifest | Stable workload and config IDs, kernel source hash, shape, dtype, strides, input seed, launch grid, config knobs, cache protocol, and split. Run identity also includes the GPU, compiler, and profile revisions. |
| Frontend and kernel IR | Target, compile status, artifact hashes, launch grid and CTA resources, spill report with units, tile work and masked operand bytes, traversal and CTA-to-tile mapping, post-pipeline stage structure, and MMA/copy path. Keep unsupported features explicit. |
| GPU profile | Versioned device and environment provenance, SM count and resource limits, allocation rules for residency, measured compute/bandwidth/launch parameters with units, memory hierarchy sizes, and dispatch observations or uncertainty. Record the profile revision and the calibration run ID or manifest digest. The calibration manifest can hold clock observations, host/OS/WSL context, power settings, and protocol details. |
| Model | One record per case ID, config ID, and profile revision. Consume `status` (`ok`, `unsupported`, or `failed`), `reason`, `nominal_latency_ns` (draw 0), and per-draw latencies in `draws`. Read initial L2 state and per-draw dispatch policy from `assumptions`. Preserve nominal occupancy, wave, utilization, bottleneck, traffic, and feature flags from `diagnostics`. |
| Hardware runner | Per-config correctness and status, raw per-block timings, profiling sidecars, clock and thermal telemetry, and protocol metadata. |

Join by case ID, config ID, and profile revision. Export per-case baseline and model scores, shortlist and regret, pair decisions with measured intervals, coverage and failure counts, aggregate deltas with uncertainty, and enough versioned inputs to replay the scoring. Keep unsupported and failed rows for coverage, and report compiler rejections and invalid outputs in the attempted matrix. The ranking oracle consists of correctly measured configs. A model or baseline that declines to score one of those configs loses coverage, and the config remains eligible for the oracle.

## Candidate matrix

Use one shared kernel with fixed layout and epilogue. Cross six tile shapes `(BM, BN, BK)` with `num_warps in {4, 8}` and `num_stages in {1, 2, 4}` for 36 attempted configs per shape:

- Tile shapes `(16,64,32)`, `(32,64,32)`, `(64,64,32)`, `(64,128,32)`, `(128,64,64)`, and `(128,128,64)`.
- For swizzle diagnostics, compare linear traversal with grouped-M traversal (group size 8) on identical legal variants. Record the realized order and compare only cases where the tile grid yields different orders.
- Record compile and launch resource failures separately. Retain legal spilling candidates and identify them by backend report. If the core sweep yields no spills, add a separate pressure sweep with validated outputs and report a matched low-pressure reference. Compiler-reported spill bytes are metadata. Use separate local-memory traffic counters where available to check the modeled penalty.

| Shape family | Example cases and purpose |
| --- | --- |
| Decode and small | `(M,N,K) = (1,4096,4096), (16,8192,4096), (32,4096,11008)` for small grids and low effective compute intensity. |
| Prefill and steady state | `(256,4096,4096), (1024,4096,11008), (2048,8192,4096)` for deeper pipelines, larger grids, and bandwidth/compute transitions. |
| Tail and cache boundaries | With a frozen reference config, generate grids just below, at, and above one SM-wide round and one full-residency wave, and working sets below and above the documented or measured L2 size. Derive the cases from profile and compiler resources before seeing GEMM timings. |
| Masking and feature diagnostics | Include a nonmultiple case such as `(513,4103,4093)`. Test valid tuned pairs for post-pipeline stage changes, spills, swizzle, multiple output tiles per persistent CTA, and producer/consumer warp roles where supported. |

These cases seed a manifest. Assign entire shape families or generation rules to development and holdout, with distinct boundary and masking cases for each. Freeze the split, case weights, and candidate eligibility rules before reading holdout timings. Keep the same shared manifest on both GPUs and enumerate device-specific diagnostics separately.

## Hardware measurements

1. Pin kernel source, Triton, CUDA/assembler, driver, OS, GPU identity, and power settings. Allocate and touch device buffers before sampling. Compile, load, and verify each candidate against a reference outside timed runs. Record a failed correctness check with its case and config.
2. On a quiet device, stabilize temperature and attempt fixed graphics and memory clocks where supported. Verify the achieved clocks under load. Where locking or telemetry is unavailable, use repeatable steady-state sessions, record observable clocks and throttle events, and report the run separately when stability cannot be established. Record display activity on the 4070 and avoid concurrent CPU memory workloads on GB10, which has shared system memory. Restore any changed clock settings after collection. [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/), [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
3. Measure GPU elapsed time on one stream with CUDA events. Exclude compilation, allocation, host transfers, correctness, and cache setup. Aim for at least 30 randomized complete blocks across at least three sessions, with each eligible config sampled in each block. For short hot kernels, batch repeated launches using a fixed graph replay protocol if needed for timer resolution and label it separately. For cold kernels, prepare the cache before each individually timed launch. Report the event timing floor and any unresolved short cases. Measure host dispatch or end-to-end latency separately if it becomes a project goal.
4. For warm trials, prime with a fixed untimed sequence using the same addresses, held consistent across configs. For cold trials, scan a disjoint, once-initialized eviction buffer with L2-caching, L1-bypassing loads before the start event. Use at least three times the device's reported L2 capacity and at least 128 MiB. Read-only eviction avoids leaving dirty lines that compete with the timed kernel for DRAM bandwidth. Verify eviction with a cache-sensitive control and, where supported, a separate counter run. Warm and cold describe preparation protocols and have separate scores. Score a cache regime only when the predictors implement its initial-state assumptions. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).
5. Measure L2/DRAM and local-memory traffic, when available, in separate profiler runs with recorded replay/cache settings. Use the profile suite's `%smid` and `%globaltimer` observations across grid/residency changes to check plausible CTA placement and ordering. Repeat representative comparisons across plausible dispatch policies. A ranking sensitive to dispatch stays low confidence. Keep instrumentation out of ranking runs.

Keep profile microbenchmarks, baseline training data, model-development GEMMs, and sealed validation labels in separate datasets. The evaluator gets all labels after predictions are frozen. The model and baseline get identical profile revisions and no held-out GEMM timings or counters for fitting. Use development repeats to freeze stability exclusions and timing budgets. Keep rejected blocks and reasons auditable. Report median latency and 95% intervals by resampling matched blocks within sessions, and report between-session drift separately.

## Independent baselines

- **Roofline with wave quantization.** Use fixed measurements or specified ceilings for dense MMA throughput, off-chip bandwidth, and device-side launch overhead. For `G` CTAs with uniform tile work `F` and requested bytes `B` per CTA, set `C = SM_count * resident_CTAs_per_SM` from compiled resources and allocation rules. Each wave costs `max(C*F/P, C*B/BW)`, where `P` and `BW` are device-wide rates. Multiply by `ceil(G/C)` and add launch overhead. For masked/nonuniform tiles use mean padded compute and mean valid requested bytes, with writes counted once per tile. Equal full-wave duration for the partial wave is the deliberate quantization approximation. The baseline uses no event scheduling or inter-CTA reuse model.
- **NeuSight-style tile predictor.** Independently fit a small roofline-bounded wave-efficiency regressor from source-GPU development GEMMs and profile data. Use normalized tile work, requested bytes, working-set/L2 ratio, occupancy, and compile-time tile, warp, and stage features. Predict a bounded efficiency against the full-wave roofline time, then aggregate the waves analytically. Each training label comes from launch-adjusted kernel time divided by its analytical wave count. Tune only on the development split. Freeze weights and feature transforms before predicting GB10 with its profile. This adapts NeuSight's tile-learning idea to fixed, externally supplied Triton configs. An unmodified reproduction may be reported separately if it can score the same configs. [Lee, Phanishayee, and Mahajan, NeuSight](https://arxiv.org/abs/2407.13853).

Both baselines predict from the same sealed candidates and allowed compiler/profile inputs and are versioned separately from the simulator. Keep their portable-GEMM coverage explicit. Use rank and regret for baseline comparisons even if a baseline has no abstention rule.

## Scoring and decision

- **Shortlist.** Sort by nominal prediction with frozen config-ID tie-breaking and take up to `k` scored candidates. Report top-k hit rate and `min(measured time in shortlist) / min(measured time over all valid configs) - 1` as top-k regret. A hit requires a measured best point estimate in the shortlist. Report regret and intervals so small timing differences remain visible. An empty shortlist is a failure. Report full-case and config coverage alongside the metrics.
- **Rank.** Report Kendall tau-b per case on shared scored candidates. Aggregate by case with stratification by device, cache regime, and shape family. Include paired baseline deltas and confidence intervals across cases, with matched timing-block resampling for measurement noise. Track absolute error as a diagnostic.
- **Pairs.** On hardware, use the 95% interval of the paired log-time ratio and the practical band `[-log(1.05), log(1.05)]`. The pair is equivalent when the interval lies inside the band, ordered when wholly outside on one side, and unresolved otherwise. Report a confusion table on resolved pairs, including equivalence precision/recall, false confident orderings, and directional coverage. Report unresolved counts separately. Keep pair decisions as edges. Equivalence can fail to carry across a chain of pairs.
- **Sensitivity.** Recompute pair decisions independently from the model's shared draw IDs. Draw 0 uses the nominal profile. The remaining draws use a fixed-seed Latin hypercube with uniform marginals within each measured parameter's `rel_uncertainty` (at least 10%). Each draw shares values across configs for the same device and profile revision and can include a candidate dispatch policy. Compare pairs with two `ok` records and a complete matching draw set. Predict a direction only when every draw, including draw 0, agrees by at least the fixed practical log margin `log(1.05)`. Any smaller gap, tie, or reversal yields indistinguishable. Count unavailable comparisons separately. Report the seed, draw budget, and dispatch assumptions alongside pair scores.

The fixed criteria are a median per-case tau-b at least 0.05 higher than each
baseline, mean top-5 regret at most 1%, and mean top-5 regret no more than
0.5 percentage points above either baseline. Report paired case-level deltas
with intervals and disclose the coverage denominator. Apply the same criteria
to the cross-GPU experiment, where the NeuSight-style baseline trains only on
RTX 4070 labels.
