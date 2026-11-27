# VirtualGPU v0.3 preregistration draft

This draft defines the evaluation and freeze procedure and records the
current mechanism status. The release lead will finalize the scope before
the freeze. Publish a reviewed amendment naming the included changes and
the frozen implementation before opening any v0.3 holdout labels.
Earlier release decisions remain frozen.

## Mechanism scope (draft)

This inventory compares `v0.2-final` (`e70824b`) with main at
`7c205d5`. It covers changes to core execution, frontend IR and
profile inputs. Verified refers to the named evidence and device.
Provisional evidence has a qualification or coverage gap. A
convention correction changes how a defined quantity is represented
or charged. The lead will approve the exact release scope before
freeze. Only the RTX 4070 and GB10 enter the four cohorts below.

### Prediction-path inventory

| Change | Commits | Status and device evidence |
| --- | --- | --- |
| Cross-wave L2 reuse, form c | `a5c01c6`, `f48763f` | **Provisional.** K-slice LRU uses a shared end-event clock and counts hit slices by position. On GB10 all seven runs passed rules 1 and 2 individually, while each of three versions failed rule 3 across runs. Forward re-reads at or above 1.5 times L2 were about -0.01 to 0.09 hits versus 0.5 to 0.9 predicted by the old form. That subset passes in a post-hoc check. Near capacity the hardware is bistable. The RTX 4070 attempt was void and remains unvalidated. The sealed v0.3 holdout decides the standing of this change. [Evidence and limits](../docs/design.md#93-cross-wave-l2-reuse-near-capacity) |
| NVIDIA `ldmatrix` shared-memory read bytes | `4c70e03`, registration `6914195` | **Convention correction.** Per-warp PTX operand traffic replaces the one-per-CTA TTGIR tile count where `ldmatrix` is present. GEMM IR cache version 5 invalidates old extractions. Offline recounts cover RTX 4070 and GB10 GEMM and attention. The registered check compares the absolute signed median latency error per GPU and kernel family and by attention stage count. Error should not grow, with attention `num_stages = 2` reported separately because growth is expected. Ranking changes are descriptive. The `ld.shared.v4` service-rate proxy for `ldmatrix` remains an assumption. [Registered check](../docs/results/analysis/h7-ldmatrix-registration.md) |
| Spill L1 transition and fallback | `d050c15` | **Provisional.** The L1 miss curves are keyed by shared-memory carveout and replace linear overflow. Profiles without a curve use a hard threshold. Legacy carveout measurements are from RTX 4070 and GB10. Field and end-to-end rechecks await qualified profiles. [Spill model and evidence](../docs/design.md#64-mechanisms-in-the-core) |
| Spill reload latency on the dependent chain | `b605efa` | **Provisional.** Batched reload curves mix the L1 and L2 costs and use static assembler load bytes divided by four as the reload count. Legacy points cover RTX 4070 and GB10. Both executed counts and qualified long launches remain open. The large-frame derivation retains some launch-time frame initialization cost. [Spill limits](../docs/design.md#92-v02-residuals-and-exclusions) [Long-launch evidence](../docs/components/long-launch-capture.md) |
| Spill loop split and duplicate reload charge | `ae02278`, `c2e920c` (PR #16) | **Convention corrections.** The static causal attention total is divided equally between the main and diagonal loops. RTX 3070 post-hoc counters show those loops differ and the estimates still overcount by 1.4 and 4.1 times. The chain counts only stores on its local pipe when reload latency already charges the loads. The shared local pipe still carries all bytes, with the charge split covered by a deterministic core test. There is no executed-count validation on the RTX 4070 or GB10. [Counter evidence](../docs/results/analysis/spill-counters-rtx3070-posthoc.md) [Issue #15](https://github.com/VirtualGPU/VirtualGPU/issues/15) |
| GB10 reference profile MLP refresh | `71fb55a`, `36f1259`, quality notes `1262553` and `7c3921f` | **Provisional profile inputs.** Updated contiguous DRAM and L2 MLP, strided DRAM MLP, and the DRAM and L2 read scalars. The partial refresh preserves the original global clock policy. All 10 short-launch repeats failed rule 1 under the adopted rule. GB10's first full long-launch attempt passed the MLP modes in isolation and failed suite qualification. The bundled RTX 4070 numeric profile is unchanged over this interval. [Capture audit](../docs/results/analysis/v0.3-mlp-repeat-plan.md) [Full capture](../docs/results/analysis/v0.3-gb10-first-full-capture.md) |
| Optional cp.async active-SM L2 curve and read ceiling | `117db65` (PR #12) | **Convention correction in collection, provisional as a reference input.** The curve has a backward-compatible reader and shares the L2 perturbation group. The core still uses the original port-floor curve and does not select the new curve by instruction. In a re-collected profile the full-SM cp.async point can supply `l2_bw_read_gbps`, subject to the existing MLP peak comparison. Evidence includes the RTX PRO 6000 refresh and a GB10 same-session mechanism check. The matching RTX 4070 reference input awaits recapture. [Profile format](../docs/components/profiles.md) [GB10 mechanism check](../docs/components/long-launch-capture.md) |
| Physical unified L1 size and DRAM buffer floor | `117db65` (PR #12) | **Convention correction in collection.** Use architecture-specific unified L1 capacity and a DRAM buffer of at least 16 times L2. Both can change measured or derived prediction inputs when profiles are rebuilt. This commit refreshes RTX PRO 6000 only. GB10's scheduled write kept the 384 MiB legacy minimum and moved -0.02% in a mechanism check. [Profile contributor guide](../docs/guides/contributing-a-profile.md) [GB10 mechanism check](../docs/components/long-launch-capture.md) |
| 0.23.0 collection and profile construction | `75bdfec`, `663e618`, `de3d217`, `c0a46fa`, mode registration series `071fcc6`, `257f53f`, `33a03a9`, `ce38c5b`, `f983272`, `9c915a3`, `b4cc2f9` | **Provisional on both reference devices.** Use the median of five valid process medians from frozen lengths. Long-launch inputs and their uncertainties can change predictions. Launch records and per-cycle fields are provenance at this checkpoint. The core still uses the native rate. GB10 mode comparisons and a full failed attempt exist. There is no qualified RTX 4070 campaign. [Protocol](../docs/components/long-launch-capture.md) [First GB10 capture](../docs/results/analysis/v0.3-gb10-first-full-capture.md) |
| Measured non-MMA issue rate | `7c205d5` (PR #20) | **Provisional on the reference devices.** The collector takes the greater of the pure FFMA rate and the peak FFMA/LOP3 mix. The core already reads that field. Only a re-collection changes it, and both bundled reference profiles still use the four-per-clock estimate. RTX 3070 had five repeats that passed the dispersion checks, without a full 0.23.0 gate report. The archived RTX 4070 and GB10 IR replay with the RTX 3070 value is post-hoc sensitivity. The registered same-device rate check rejects a miss above 10%. [RTX 3070 evidence](../docs/results/analysis/non-mma-issue-rtx3070-posthoc.md) |
| Ascend occupancy, frontend and Cube/Vector execution | `737bcfb`, `0c3fec4`, `3ce23b1` | **Verified for Ascend attention holdout, provisional for the target overall.** Adds Ascend-only SRAM resource checks, GEMM and attention IR extraction, workspace residency and overlap, and HIVM fractal layout and AIC parser corrections. The Ascend 910B3 sealed result passed attention and failed the GEMM roofline ranking margin. The capability gates preserve the existing NVIDIA paths. The separate Ascend evaluation stays outside this freeze. [Model and limits](../docs/design.md#94-ascend-910b3-mapping-and-known-limits) [Sealed Ascend result](../docs/results/ascend910b3-holdout.md) |
| Other bundled NVIDIA profiles and refreshes | `5d03053`, `1f499a7`, `df6bae8`, `36f321a`, `a3d603c`, `07de592`, `17fd096`, `6ff4a91`, `117db65` | **Provisional.** Device-specific calibration exists for RTX 3070, L40S, L4, RTX PRO 6000, A100 80GB, RTX PRO 5000 72GB, and RTX 4090. These commits add or refresh target inputs and do not supply evidence for either v0.3 reference cohort. The L40S commit also extends the `l2-merge` harness to GPUs above 128 SMs. [Profile evidence and provenance](../profiles/README.md) |

The mainline non-MMA change in `3736b0a` was reverted by `a239702` and
landed in revised form as `7c205d5`. Mechanical reformats, test moves
and profile-provenance-only changes do not introduce new prediction
mechanisms. Post-hoc effects on opened workloads do not enter v0.3
validation. No v0.3 labels may select model parameters.

The spill step and reload lookup landed without preregistration and
remain under issue #15 review. Its four findings are the static reload
proxy, the initial duplicate causal loop charge, the initial duplicate
chain bandwidth charge, and the fallback change for profiles without
carveout curves. The table records their dispositions. Spill frames
leaving L1 also change the working set used to split global traffic
between L2 and DRAM. This coupling remains open. The zero-step
frame-initialization control in PR #19 is pending at this snapshot.

### Registered work outside the core

- **H7 F-dep** (`6d2aa19`) awaits a qualified RTX 4070 cross-device test
  against F-max. On points where the forms differ by more than 2%,
  F-dep must bring median error below half and be closer on at least
  two thirds. Keep it outside the freeze if that test is unavailable
  or fails. H7(a) is a separate MMA-chain confirmation test. It does
  not authorize a new dependency-latency rule for the supported
  workloads. [F-dep registration](../docs/results/analysis/h7-fdep-registration.md) [MMA-chain registration](../docs/results/analysis/h7-mma-ilp-registration.md)
- **H5 narrow loads.** Keep both forms outside the core. DRAM
  request/sector-limit captures failed qualification and leave that
  form unjudged. The qualified GB10 register-staging form scored
  15.7% MAPE against a 15% limit. [H5 evidence](../docs/design.md#95-narrow-register-staged-loads-h5)
- **H1 merged-wait discount and lockstep bursts.** Leave both out.
  Free-running groups did not support discounting by arrival offset.
  The positive fixed-lead result was provisional. CTAs drifted out
  of step, and the burst form did not win its registered test.
  A slowest-copy partial-miss rule still needs its own registration
  and test. [H1 evidence](../docs/results/analysis/v0.2-posthoc.md)
- **H3 staged MMA and H4 closed-loop retuning.** The GB10 staged-MMA
  negative control rejected the proposed multi-buffer penalty. The
  b-stream study did not locate H4 in the closed-loop mechanism.
  Keep DRAM reference normalization outside the predictor freeze
  unless a later reviewed amendment establishes its scope on
  qualifying data. [GB10 H3 and H4 evidence](../docs/results/analysis/v0.3-gb10-h3-h4-posthoc.md)

### Open core items and freeze recommendation

| Item | Evidence | Draft recommendation |
| --- | --- | --- |
| Round-trip port floor with miss merging | Old-profile recheck traces the regression to `e51f8ef`, before `v0.2-final`. On the same benchmark inputs the RTX 4070 error rose from 4.8% to 12.2% and GB10 from 4.6% to 8.8%, most at sparse CTA counts. | Keep the existing coupling as an explicit known limit if it persists but no registered alternative wins on a qualified profile. Require the new-profile recheck before deciding. The current floor has separate evidence on `serial_stage_fill` and `sync_stage_loop`. |
| Spill frame footprint at full occupancy | The phase-entry per-SM correction and its sparse-grid, full-wave and tail regressions landed in `e8f241a` (PR #23). The legacy whole-core spill recheck has signed median errors of +67.6% on GB10 and +38.0% on RTX 4070. [Post-hoc evidence](../docs/results/analysis/v0.3-spill-per-sm-posthoc.md) | Recheck the spill mechanism on a qualified profile before freeze. Executed reload counts and local bandwidth still need separate evidence. |

[Core recheck baseline](../docs/results/analysis/v0.3-mechanism-recheck-baseline.md)
[Open core work](../ROADMAP.md#030-close-the-known-model-gaps)

### Profile and holdout freeze prerequisites

Each reference device needs one complete, independently qualified
`0.23.0` profile with a frozen schedule and binary, five valid
independent process rounds, and the applicable thermal, host and
desktop gates. The profile stores per-point SM clocks and per-cycle
compute rates. Keep the min and max of the per-round medians for
registered exception points and cover the observed spread. GB10 has
the following capture exceptions:

| Region | Points and handling |
| --- | --- |
| L2 rescan bistability | In `l2-rescan`, working sets from 0.75 to 1.25 times physical L2 are excluded only from rule 3 and tagged `bistable_band`. |
| L2 capacity sweep near capacity | Working sets from 0.75 to 2.0 times physical L2 are excluded only from rule 3. Rules 1 and 2 examine every sample list. The sweep records the `l2_near_capacity_unstable` exception separately from rescan. [Registration](../docs/results/analysis/v0.3-gb10-l2-near-capacity-exception-registration.md) |
| Low-in-flight cold DRAM | Contiguous reads at or below 256 KiB, strided reads at or below 384 KiB and active-SM read targets 1 and 2 are excluded from rules 1 and 3 and tagged `low_in_flight_unstable`. Rule 2 examines every sample list and uncertainty covers both observed clusters. |
| Serial 1 KiB spill reloads | **Adopted in the final redo with cross-allocation sampling.** The four GB10 `serial_chain` boundary points at 8, 32, 68 and 128 reloads are excluded only from rules 1 and 3 and tagged `l1_boundary_two_state`. Rule 2 judges all lists. [Final redo result](../docs/results/analysis/v0.3-gb10-spill-allocation-redo-result.md) |

| Prerequisite | Status before the next full capture |
| --- | --- |
| GB10 profile | A qualifying 0.23.0 profile is still pending. The [first full campaign](../docs/results/analysis/v0.3-gb10-first-full-capture.md) failed. The second campaign at `8450ba9` included the closed-loop MMA freeze fix, grouped L2 merge, the revised L2 capacity exception, P3a DRAM references and the [adopted spill redo](../docs/results/analysis/v0.3-gb10-spill-allocation-redo-result.md). All 26 modes reached five valid rounds. Only `spill-latency` failed rule 3, with four non-exempt 4 KiB points above 10% cross-round spread and three allowed among 68 evaluated points. The run stopped before reduction and produced no candidate profile. [Second full campaign](../docs/results/analysis/v0.3-gb10-second-full-capture.md) The registered same-device non-MMA rate check still needs a qualifying profile. |
| RTX 4070 profile | Pending a quiet Windows desktop slot for the full gated sweep. Capture and check the measured non-MMA issue rate. The same quiet slot dependency applies to the RTX 4070 F-dep microbenchmark and any attempt to qualify form c on this card. An executed-spill counter comparison also needs working profiling permissions. |
| Four fresh holdout cohorts | The candidate manifests and CPU AOT/resource audit include the opened Ascend shapes in the novelty inventory. The [manifest release](v03-manifest-release.json) is published. Measurement is pending device-specific capture approval. The GB10 approval follows the full GB10 profile recapture. Capture both GB10 cohorts on Spark after that approval. Both RTX 4070 cohorts also require a quiet desktop slot. Seal the reports with receipts and keep them unopened until the frozen predictions are archived and release is approved. |
| Freeze evidence | Recheck the model mechanisms on the qualified profiles, resolve the core-item decisions above, and pin the source, toolchain, profile digests, scoring code and historical baseline fits. Publish all four prediction archives before any label release. |

A pre-timing interruption can resume the same frozen capture. Modes
from independent sessions or different binaries cannot be combined
into one qualified profile. The lead records any scope change in
the amendment and leaves the registered manifests and decision
criteria in effect.
[Long-launch protocol](../docs/components/long-launch-capture.md)
[GB10 cold DRAM evidence](../docs/results/analysis/v0.3-gb10-mlp-wall-order-posthoc.md)

## Evaluation design

| Device | GEMM | Attention |
| --- | --- | --- |
| RTX 4070 | Fresh workload holdout | Fresh workload holdout |
| DGX Spark GB10 | Separate fresh workload holdout | Separate fresh workload holdout |

Report four cohorts and two independent device decisions. Both families
and both cache regimes must pass for a device to pass. The evaluation
measures generalization to new workloads on two previously studied
devices. VirtualGPU uses each target's calibration profile. The
learned comparator can use historical labels from both devices.

Publish a versioned inventory of previously opened development,
diagnostic and holdout workloads through the v0.3 manifest approval.
Use disjoint workload tuples on the two devices and publish the
novelty audit. Determine eligibility and case mix from the workload
specification, compiler resource reports and profile metadata before
any timing or ranking preview.

- **GEMM.** For each new tuple, require that M, N, and K are each
  absent from the corresponding historical axis across both GPUs.
  The inventory must include RTX 4070 M2 development, Phase 1 workload
  holdout, GB10 Phase 2 primary, secondary and TMA diagnostic, and both
  v0.2 GEMM holdouts. Include other opened GEMM datasets from the
  inventory. Cover decode-like and aligned prefill-like workloads,
  shapes with non-aligned M, N and K, and an adjacent triplet below,
  at and above a resident-wave boundary per device. Declare the
  reference configuration and verify its occupancy and CTA counts
  offline from compiled resources. Use the reviewed 36-config space
  as the starting point.
- **Attention.** Require every new `seq_len` on either GPU to be
  absent from the entire opened attention inventory, including Phase 4
  development, the Phase 4/v0.2 RTX 4070 holdout and the v0.2 GB10
  holdout. Use disjoint `seq_len` sets across the two device cohorts.
  Cover forward prefill with equal query and KV lengths, causal and
  non-causal, MHA and GQA, head dimensions 64 and 128, aligned lengths,
  and distinct odd sequence tails. Include both short and longer
  sequence regimes on both GPUs. Use the existing versioned Triton
  configuration space as the starting point.
- Freeze exact case counts, configuration filters and manifests in a
  reviewed amendment before any v0.3 workload labels are collected.
  Record every attempted configuration and why any attempt is invalid.

## Candidate manifests for review

The [GEMM candidates](v03-gemm-holdout.json) cover nine shapes and
324 planned configurations per device. Each wave triplet uses one
output tile in M and moves N across three adjacent CTA counts while
holding M and K fixed. All three shapes in each triplet have the same
64-column tail for the reference configuration. Aligned prefill cases
are divisible by the largest candidate tiles on each axis.

The [attention candidates](v03-attention-holdout.json) cover eight
shapes and 228 planned configurations per device, one for each
causality, grouping, and head dimension stratum. Each device has
four sequence lengths aligned to both query tile sizes and four odd
sequence lengths. All 16 sequence lengths are distinct.

The [historical inventory](v03-holdout-history.json) records each
source manifest's SHA-256 and excluded axes or sequence lengths.
It includes the original M1 suite, the materialized Phase 1 and
Phase 2 manifests, all v0.2 workload groups, and the opened Ascend
shapes and documented development exclusions. The historical checks
use manifest metadata and published seal/index digests.

The [CPU AOT audit](v03-holdout-aot-audit.json) records resource status
by case, including every configuration exceeding a profile limit,
and identifies the full resource reports by SHA-256. Resource
eligibility is a static check. Capture still checks the approved
toolchain, compiled resources and JIT artifacts. Review the candidate
manifests and release them in a separate amendment before collection.

| Cohort | Cases | Compiled | Fits profile | Exceeds a resource limit | Wave boundary CTAs |
| --- | ---: | ---: | ---: | ---: | --- |
| RTX 4070 GEMM | 9 | 324 | 324 | 0 | 91 / 92 / 93 |
| GB10 GEMM | 9 | 324 | 324 | 0 | 95 / 96 / 97 |
| RTX 4070 attention | 8 | 228 | 188 | 40 | n/a |
| GB10 attention | 8 | 228 | 188 | 40 | n/a |

The GEMM wave reference is `m64n128k32-w4-s4-g1`. AOT reported
128 registers per thread and 36,864 bytes of shared memory for all six
boundary cases. Each admits two resident CTAs per SM. Keep all
planned configurations in the manifests, with resource-limit flags
available for review and later capture.

The audit used Triton 3.9.0, `ptxas` 12.9 (V12.9.86) for `sm_89`
and `ptxas-blackwell` 13.4 (V13.4.59) for the GB10 PTX target
`sm_121a`. The profiles are RTX 4070 `0.21.0-posthoc` and GB10
`0.22.0-posthoc`. The audit did not collect timing or run either GPU.
The complete CPU review bundle is
`results/v03-holdout-release-aot.tar.gz` (SHA-256
`f73d6ec8b94897fd2c38f3e33342c7aad58621072ee45d3ee3ed7bbe14873bcf`).
The [draft release amendment](v03-holdout-release-draft.md) records
the replacement rule, changed geometries and publication gates.

Run the [candidate generator](v03_candidates.py) on a checkout with
the archived historical manifests and AOT index:

```sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m validation.v03_candidates results/v03-candidate-audit
```

## Hardware and baselines

Reuse the reviewed v0.2 capture protocol. Measure warm and clean-cold
with single-launch CUDA Graph events, three sessions of ten randomized
complete blocks, seed schedule 101 to 103, clean read-only L2 eviction
and a valid cold read probe. Check AOT/JIT artifact identity.
Use the GPU lock, telemetry, process checks and the existing GB10
thermal/clock gate. Record any documented protocol failure, including
its immutable receipt and any authorized retry.

Compare with roofline plus wave quantization and the independently
implemented NeuSight-style tile baseline, each for both kernel
families. Train one frozen learned fit per family on all eligible
published historical labels from both devices. Use the same fitted
coefficients on both devices with the target's hardware features.
The currently published sources are:

| Family | RTX 4070 | GB10 |
| --- | --- | --- |
| GEMM | M2 development (original warm and clean-cold replacement), Phase 1 workload holdout, v0.2 GEMM holdout | Phase 2 primary and secondary cohorts, v0.2 GEMM holdout |
| Attention | Phase 4 development, Phase 4/v0.2 holdout | v0.2 attention holdout |

Preserve the original split, device and source receipt for every
training observation. Keep observations of the same workload on
different devices distinct. Derive one row per eligible case,
configuration and cache regime for each source. Use effective valid
sessions and deduplicate reused reports, including the Phase 4/v0.2
RTX 4070 attention holdout. Exclude the Phase 2 TMA diagnostic,
invalid or failed captures, and all v0.3 holdout labels. Freeze the
exact training inventory, data digests, profile features, feature
builders and fitted coefficients before opening the new labels.
Use hardware specifications and isolated microbenchmarks for
model/profile calibration.

## Freeze and release

1. Publish the manifests, split audit, candidate space, procedure and
   scoring implementation. Capture labels only after the plan is
   approved. Store mode-zero reports and publish SHA-256 and UTC
   receipts without reading the timings.
2. At the clean release tag `v0.3-final`, freeze the model, frontend,
   toolchain, profiles, scoring implementation and both baseline fits.
   Record the build ID and the hashes of the profiles, fits, source
   inputs, manifests, seals and capture artifacts.
3. Generate and archive core, roofline and NeuSight-style predictions
   for all four cohorts, including unsupported and failed rows.
   Publish a freeze amendment with digests and UTC timestamps.
4. After the amendment is on the remote branch and release is
   approved, verify the archive and every sealed report digest before
   interpreting timing data. Score once with the frozen implementation.

## Decisions

Apply [design section 8](../docs/design.md#8-validation) independently
to each kernel family and cache regime:

- Median Kendall tau-b reaches at least 0.05 above each baseline.
- Mean top-5 regret is at most 1% and at most 0.5 percentage points
  worse than either baseline.
- Any eligible case missing a core shortlist fails both shortlist
  criteria. The oracle includes all hardware-eligible candidates.
- Publish per-case and aggregate scores, top-5 hit rate, paired
  judgments using the 5% resolution band, directional agreement,
  latency error, read probe results, coverage, and config status counts.
  The first two bullets define the numerical pass criteria.

## Known limits

On GB10, repeated runs of the same microbenchmark geometry near the
L2 capacity (roughly 0.75 to 1.25 times the 24 MiB physical L2 in the
reported sweep) can settle in either a mostly hit or a mostly miss
state. At 0.91 times physical capacity, one geometry took 1701 us and
3528 us in separate runs. The model represents the transition as a single
step. A single scan of `l2_effective_capacity_bytes` or
`l2_rescan_hit_rate_by_working_set` therefore understates the uncertainty
near capacity. Flag GB10 working-set and resident-wave cases that
approach that range before label release, using predicted working-set
geometry and the target profile. Keep the registered cases, configs,
pass thresholds and scoring procedure. [GB10 L2 observation](../docs/results/analysis/v0.3-gb10-l2-bistability-observation.md)

The next review fixes the mechanism scope, manifests, exact toolchain
and training inventory. Record changes in an amendment before label
release.
