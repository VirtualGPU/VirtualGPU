# VirtualGPU v0.3 preregistration draft

This draft defines the evaluation and freeze procedure and records the
current mechanism status. The release lead will finalize the scope before
the freeze. Publish a reviewed amendment naming the included changes and
the frozen implementation before opening any v0.3 holdout labels.
Earlier release decisions remain frozen.

## Mechanism scope (draft)

### On main

- **Cross-wave L2 reuse.** The core models reuse by K slice under LRU
  (`a5c01c6`, `f48763f`). All seven GB10 runs passed capture rules 1 and
  2 individually. Each of the three benchmark versions failed rule 3
  across runs. At working sets of at least 1.5 times L2, forward re-read
  hit fractions were about -0.01 to 0.09 in all seven runs, versus 0.5 to
  0.9 predicted by the previous form. That subset passed all three rules
  in a post-hoc check. Near the capacity boundary, the GB10 observations
  are bistable. Reversed walks are provisional and no supported kernel
  uses them. The form remains unvalidated on RTX 4070. Its standing is
  decided by the sealed holdout. [Evidence and limits](../docs/design.md#93-cross-wave-l2-reuse-near-capacity)
- **Frontend `ldmatrix` byte correction.** The extractor counts the
  per-warp bytes of MMA shared-memory operands from PTX (`4c70e03`).
  This is a unit correction. Its registered check compares the
  absolute signed median latency error with and without the correction
  per GPU and kernel family, and by `num_stages` in attention. Error
  should not grow, with attention `num_stages = 2` reported separately
  because growth is expected. Report ranking changes descriptively.
  [Registered check](../docs/results/analysis/h7-ldmatrix-registration.md)

### Spill changes under review

The carveout-dependent L1 spill transition (`d050c15`) and critical-path
reload latency (`b605efa`) went directly to main without preregistration.
They remain under review in [issue #15](https://github.com/VirtualGPU/VirtualGPU/issues/15).
The four findings and current status are:

| Finding | Current status |
| --- | --- |
| Static `ptxas` spill load bytes divided by four stand in for executed reloads. Cold paths and wider loads can inflate the count. | Executed per-iteration counts are still missing for the model. |
| Causal attention originally charged the combined static count in both loops. | `ae02278` splits the count between the main and diagonal loops. Post-hoc RTX 3070 counters still show overcounts of 1.4 and 4.1 times, and the loops have different spill code. |
| The dependent chain originally charged local reload bandwidth alongside measured reload latency. | PR #16 (`c2e920c`) removes the reload bytes from the chain's local pipe charge when reload latency is charged. Both loads and stores still use the shared local pipe. |
| Profiles without a carveout curve now use a hard threshold in place of linear overflow. | The behavior change remains and is documented in the changelog. |

Reload latency is provisional pending executed counts and qualified
long-launch recapture. The profile derivation still carries large-frame
initialization bias. Spill frames leaving L1 can also change the working
set used to split global traffic between L2 and DRAM. Impacts already
reported on opened workloads are post-hoc. [Spill limitations](../docs/design.md#92-v02-residuals-and-exclusions)

### Registered candidates and rejected forms

- **H7 F-dep.** The dependent softmax chain is registered and has not
  entered the core. Its decision awaits the RTX 4070 cross-device
  microbenchmark against F-max. On points where the forms differ by
  more than 2%, F-dep must bring median error below half and be closer on at
  least two thirds. [Registered test](../docs/results/analysis/h7-fdep-registration.md)
- **H5 narrow loads.** Neither the DRAM request/sector limit nor the
  combined register-staging form enters the core. The DRAM captures
  failed dispersion and leave that form unjudged. The qualified GB10
  staging run had 15.7% MAPE against a 15% limit.
  [Evidence](../docs/design.md#95-narrow-register-staged-loads-h5)
- **H1 merged-wait discount and lockstep bursts.** Neither enters the
  core. Free-running CTA groups did not support discounting the merged
  wait by arrival offset. The fixed-lead result was provisional. CTAs
  fell out of step, and the burst form did not win its registered test.
  [Evidence](../docs/results/analysis/v0.2-posthoc.md)

### Profiles planned for the freeze

The planned RTX 4070 and GB10 revision is `0.23.0` under the long-launch
protocol. Size timed points with a pilot, freeze their schedules, use at
least 1 ms per timed window, and reduce five valid independent sweeps
under the applicable thermal, host and desktop gates and the three
capture rules. The protocol records per-point SM clocks and per-cycle
rates for compute measurements. The final amendment will name the
actual qualified profile revisions and digests.

Two registered capture exceptions apply to GB10:

| Region | Points and handling |
| --- | --- |
| L2 bistability | In `l2-rescan` and the capacity sweep, working sets within 0.75 to 1.25 times physical L2 are excluded from the rule 3 decision and tagged `bistable_band`. |
| Low-in-flight cold DRAM | Contiguous reads at or below 256 KiB, strided reads at or below 384 KiB, and the DRAM active-SM read curve at targets 1 and 2 are excluded from the rule 1 and 3 decisions and tagged `low_in_flight_unstable`. Rule 2 still examines every sample list. |

Keep the min and max of the per-round medians and report uncertainty from
the observed spread, covering both clusters for low-in-flight points.
The exceptions are fixed before recapture.
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
Phase 2 manifests, and all v0.2 workload groups. The historical
checks use manifest metadata and published seal/index digests.

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
`results/v03-holdout-review.tar.gz` (SHA-256
`0063e98648cadd13f7773fc442527ec273929765c65e841226ab2834e88e28dd`).

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
