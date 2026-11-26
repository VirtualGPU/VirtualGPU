# Phase 1 summary: RTX 4070

Phase 1 asked one question: **when VirtualGPU says kernel variant A is faster than variant B, does the real GPU agree?** It was answered on an RTX 4070 (Ada, `sm_89`) with Triton GEMM autotuning configs, using a preregistered protocol and a sealed holdout.

## Protocol

- The model, frontend, Triton 3.9.0, and GPU profile `0.11.0-m3` were frozen as the git tag `phase1-final` (commit `129a6f5`).
- Nine workload shapes (36 configs each, warm and cold L2) were measured on the GPU and sealed before the freeze. Nobody read them until all predictions were archived.
- VirtualGPU used **no GEMM timings**. Its only hardware input is the GPU profile, measured by the microbenchmark suite.
- Two baselines were scored on the same data: roofline with wave quantization, and a NeuSight-style learned predictor trained on all RTX 4070 development GEMM timings.
- Pass criteria, fixed before any model score was seen, are in [design.md](../design.md) section 8. The full report with digests and timestamps is [validation/phase1-holdout-report.md](../../validation/phase1-holdout-report.md).

## Result

| Holdout | VirtualGPU | NeuSight-style | Roofline + wave |
|---|---:|---:|---:|
| Kendall tau-b, warm / cold | **0.787 / 0.750** | 0.387 / 0.476 | 0.512 / 0.514 |
| Mean top-5 regret, warm / cold | 3.11% / 3.02% | **1.19% / 1.24%** | 9.30% / 9.25% |

**Formal decision: Fail.** The ranking criterion passed in both cache regimes by a wide margin. Both shortlist criteria (absolute regret of at most 1%, and non-inferiority to each baseline) failed. Six of the nine shapes have 0% top-5 regret; the mean is dominated by one shape with M, N and K all unaligned (24.5% warm, 22.2% cold).

## Do pairwise verdicts agree with the hardware?

This is a descriptive analysis of the committed scores, not a preregistered criterion. Reproduce it with `python -m validation.pair_precision validation/phase1-holdout-scores.json`.

A verdict is "ordered" when a method says one config is faster than the other. For VirtualGPU, that requires the difference to be at least 5% in every one of 33 perturbation draws; otherwise it reports the pair as indistinguishable. The baselines order a pair whenever their nominal predictions differ by at least 5%.

| Cache | Method | Ordered verdicts | Agree with hardware | Reversed | Hardware says equivalent | Hardware-ordered pairs ordered correctly |
|---|---|---:|---:|---:|---:|---:|
| warm | VirtualGPU | 3047 | **94.5%** | **2.1%** | 2.3% | 59.7% |
| warm | NeuSight-style | 4930 | 67.3% | 20.3% | 8.9% | 68.8% |
| warm | Roofline + wave | 4875 | 71.1% | 17.6% | 8.1% | 71.8% |
| cold | VirtualGPU | 3035 | **94.8%** | **2.1%** | 1.9% | 59.1% |
| cold | NeuSight-style | 4799 | 67.8% | 21.4% | 7.8% | 66.9% |
| cold | Roofline + wave | 4875 | 71.3% | 18.0% | 7.8% | 71.4% |

The remaining percentage of each method's ordered verdicts falls on pairs that the hardware measurement left unresolved.

When VirtualGPU says A is faster than B, the RTX 4070 agrees about 95% of the time and disagrees about 2% of the time. The baselines order more pairs, but about one in five of their ordered verdicts is reversed. VirtualGPU trades coverage for reliability: it declines to order about 40% of the pairs that the hardware does order.

## What Phase 1 shows

- On an RTX 4070, for Triton GEMM configs, a CPU-side model built only from microbenchmarked hardware parameters can tell which variant is faster, and its confident verdicts are right about 95% of the time.
- This holds on shapes the model never saw. The learned baseline, trained on GEMM timings from the same GPU, did not generalize to the new shapes.

## What Phase 1 does not show

- Transfer to a GPU whose data was not used to build the model. That is Phase 2, preregistered in [validation/phase2-preregistration.md](../../validation/phase2-preregistration.md).
- Other kernel types such as attention or mixture of experts.
- Reliable shortlists for shapes with unaligned dimensions. The cause is under post-hoc investigation and cannot change the Phase 1 decision.

## Lessons

Several apparent model errors turned out to be measurement or frontend errors. Each was found by checking the data before changing the model:

| Problem | Effect | How it was found |
|---|---|---|
| Write-based L2 flush before cold runs | Cold timings about 25% slow | Two working sets of different size showed the same speedup |
| Stream benchmark did not saturate DRAM | Peak bandwidth 9% low | A GEMM delivered more than the "peak" |
| Memory throughput not limited by bytes in flight, then pointer-chasing latency used as the Little's law latency | Memory-bound kernels predicted about 22% fast | Effective latency (bytes in flight / delivered bandwidth) agreed across two shapes but not with pointer chasing; a bytes-in-flight microbenchmark replaced it |
| `num_stages=2` assumed to overlap loads | Prefetch depth off by one | `cp.async.wait_group` semantics in the PTX |
| Processing order within a timestamp | Results depended on iteration order | A 4% warm-versus-cold anomaly |

Every model change is justified by a microbenchmark or a hardware principle and recorded in its commit message. No model parameter was fitted to GEMM timings. Residuals that could not be explained this way are listed in [design.md](../design.md) section 9.
