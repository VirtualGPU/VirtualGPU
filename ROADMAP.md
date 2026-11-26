# Roadmap

This is the one plan for VirtualGPU, from where it is now to 1.0.0, the first release we announce widely. Each milestone ends with a test that has to pass before the next one starts. When something here is done, move it into the CHANGELOG and delete it from this file.

## Where we are

- 0.2.0 ranks Triton GEMM and fused attention configs on a CPU, using only a GPU profile. On fresh sealed holdouts on the RTX 4070 and GB10, the GPU agrees with 87% to 94% of the pairs VirtualGPU orders, against 54% to 76% for the baselines.
- Both GPUs formally failed the v0.2 criteria, because the shortlist still misses the 1% top-5 regret ceiling on RTX 4070 attention and GB10 GEMM. Details: [docs/results/v0.2.md](docs/results/v0.2.md).
- Profiles exist for four GPUs (RTX 4070, GB10, L40S, RTX 3070), all consumer or workstation cards. Nothing data center class has been tested.
- Wheels are on GitHub Releases only. Autotuning takes GEMM shapes only.

## How a milestone is tested

Every milestone that changes the model ends the same way the earlier phases did:

1. Write the preregistration (cases, cache regimes, baselines, pass criteria) and commit it before any label is captured.
2. Capture and seal a fresh holdout. Old holdouts that were opened can't be used again.
3. Freeze the model and the baselines, archive the predictions, then open the labels and score them.
4. Publish the report in `validation/` and the summary in `docs/results/`, pass or fail.

The pass criteria stay the ones from v0.2: rank correlation above both baselines by the set margin, mean top-5 regret at most 1%, in both warm and cold L2, for every kernel family on the device.

## 0.3.0: close the known model gaps

Fix what made v0.2 miss on the GPUs we already have.

- Spill reloads cost latency on the critical path, not only bandwidth. Use the measured `spill_reload_latency_by_count` curves.
- Spill traffic moves from L1 to L2 as a step at the L1 capacity for the chosen shared memory carveout, using `l1_to_l2_spill_threshold_by_carveout`.
- Get the dynamic count of local loads per iteration, so the two causal attention loops are not summed into one spill figure. Needs a counter based measurement path that stays out of the model inputs.
- Narrow loads on unaligned shapes: use the 2-byte load width the compiler emits instead of 16-byte loads.
- Synchronous load GEMM (`num_stages = 1`) is still predicted slow, +9.1% warm. Find the missing mechanism with a microbenchmark, not with GEMM timings.
- GB10 decode shapes: predictions are too compressed. Find out why.
- Measure `non_mma_issue_rate_per_sm_gips` and the local memory rates that are still estimated, on every profile.
- Validate the L40S and RTX 3070 profiles: one sealed GEMM holdout each, so every bundled profile has been checked against hardware.

Exit: pass the v0.2 criteria on fresh GEMM and attention holdouts on both the RTX 4070 and the GB10.

## 0.4.0: data center GPUs

Most people who tune kernels for inference target data center cards. The model has to work there before launch.

- Profiles for Ampere `sm_80` (A100) and Hopper `sm_90` (H100), measured with the same suite. This needs access to both cards for a few hours each.
- Hopper mechanisms in the frontend, Kernel IR and core: TMA, `wgmma`, warp specialization with producer and consumer warp groups, and persistent kernels.
- Thread block clusters, if Triton emits them for the kernels we cover.
- Extend the microbenchmark suite for the new mechanisms, with the same no fitting rule.

Exit: a sealed transfer holdout on one Hopper or Ampere card, scored with a model frozen before its labels were captured, passing the same criteria.

## 0.5.0: make it easy to use

Turn the research tool into something people can install and use in a few minutes.

- Publish wheels to PyPI, from the same release workflow, with build provenance attestations.
- Autotuning for any kernel, not only GEMM shapes: take real call arguments, so attention and other kernels can be tuned from the CLI and `@virtualgpu.autotune`.
- Warn when an online run targets a profile for a different GPU than the one in the machine.
- A profile submission flow: an issue form, a CI check that validates the profile and its provenance, and a review checklist. Version the profile format so old profiles keep loading.
- Docs site at virtualgpu.site built from `docs/`: install, quick start, reading a report, contributing a profile.
- Runnable examples: a GEMM and an attention kernel tuned for a GPU you don't own, with the commands and the expected output.
- Stable output: freeze the `AutotuneReport` and CLI formats, and document what counts as a breaking change after 1.0.0.

Exit: someone outside the project installs from PyPI, tunes a kernel for a GPU they don't own, and the result holds on that GPU.

## 1.0.0: launch

Everything here must be true before we tell people about VirtualGPU.

- [ ] 0.3.0, 0.4.0 and 0.5.0 exit tests passed, with their reports published
- [ ] At least 6 bundled profiles across at least 3 architectures, every one validated on a sealed holdout
- [ ] Install from PyPI and the quick start work on a clean Linux machine and in WSL2
- [ ] README, docs site and examples match what the release does
- [ ] Report and CLI formats are frozen and documented
- [ ] Security policy, issue forms and the profile submission flow are live
- [ ] CI green on `main`, release workflow tested end to end with a release candidate
- [ ] Launch post with the results, the limits and how to contribute a profile

## After 1.0.0

Not needed for launch, in rough order:

- Mixture of experts kernels, with a workload description of the expert routing as an extra input
- Paged and sparse attention, with the page table or sparsity pattern as input
- Whole model view: which kernels in an end to end model are furthest from their achievable speed
- PTX frontend, so CUDA, CUTLASS, TileLang and other kernels that compile to PTX work without Triton
- Data center Blackwell (`sm_100`, `tcgen05`, Tensor Memory) once the public docs and a test card allow it
- AMD CDNA3 and CDNA4 through the public ISA, including register and spill data for AMD targets
