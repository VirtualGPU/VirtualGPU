# Docs

Every doc in the repo, and what belongs in each. Each fact lives in one place. Link to it instead of copying it.

| Doc | What it holds |
| --- | --- |
| [README](../README.md) | What VirtualGPU is, how to install and use it, and what works now |
| [ROADMAP](../ROADMAP.md) | The only plan. Everything still to do before 1.0.0 and after |
| [CHANGELOG](../CHANGELOG.md) | What changed in each release |
| [CONTRIBUTING](../CONTRIBUTING.md) | How to set up, test and send a change |
| [design.md](design.md) | Goals, the core idea, use cases, the verdict rule, known limits and related work |

## Components

How each part works today. When the code changes, the doc for that part changes in the same commit.

| Doc | Part |
| --- | --- |
| [components/frontend.md](components/frontend.md) | Triton frontend, GEMM and fused attention |
| [components/kernel-ir.md](components/kernel-ir.md) | `KernelIR` format and the L2 reuse model |
| [components/model-core.md](components/model-core.md) | Event-driven performance model and perturbation draws |
| [components/profiles.md](components/profiles.md) | GPU profile format and the microbenchmark suite |
| [components/autotune.md](components/autotune.md) | GPU-free autotuning, online and offline |
| [components/validation.md](components/validation.md) | Candidate matrix, labels, baselines and scoring |

## Guides

| Doc | For |
| --- | --- |
| [guides/contributing-a-profile.md](guides/contributing-a-profile.md) | Measuring your GPU and sending its profile |
| [profiles/README.md](../profiles/README.md) | Reading a profile and its provenance |
| [microbench/README.md](../microbench/README.md) | Building and running the microbenchmark suite |

## Results

One summary per evaluation, newest first. Each links to its sealed report in `validation/`.

| Doc | Evaluation |
| --- | --- |
| [results/v0.2.md](results/v0.2.md) | v0.2 on fresh GEMM and attention holdouts, RTX 4070 and GB10 |
| [results/phase2.md](results/phase2.md) | Phase 2, transfer from the RTX 4070 to the GB10 |
| [results/phase1.md](results/phase1.md) | Phase 1, GEMM on the RTX 4070 |
| [results/analysis/](results/analysis/) | Post-hoc and development analyses behind those results |

## Validation records

[validation/README.md](../validation/README.md) indexes the preregistrations, seals, scores and reports. Those files are evidence: they are never moved or edited after they are sealed.
