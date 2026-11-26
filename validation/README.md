# Validation

Code and records that measure how well VirtualGPU ranks kernel variants on real GPUs. How the protocol works is in [docs/components/validation.md](../docs/components/validation.md), and the results are summarized in [docs/results/](../docs/results/).

The JSON and Markdown records here are evidence. Seals and fits store the paths and SHA-256 digests of the sources they used, so nothing in this folder is moved, renamed or reformatted once it is sealed. A new evaluation adds new files.

## Records

### v0.2

| File | What it is |
| --- | --- |
| [v02-preregistration.md](v02-preregistration.md) | Protocol and pass criteria, fixed before any label was opened |
| [v02-gemm-holdout.json](v02-gemm-holdout.json), [v02-attention-holdout.json](v02-attention-holdout.json) | Holdout case manifests |
| [v02-rtx4070-gemm-seal.json](v02-rtx4070-gemm-seal.json), [v02-rtx4070-gemm-retry.json](v02-rtx4070-gemm-retry.json) | RTX 4070 GEMM seal and its recorded protocol retry |
| [v02-gb10-gemm-seal.json](v02-gb10-gemm-seal.json), [v02-gb10-attention-seal.json](v02-gb10-attention-seal.json) | GB10 seals |
| [attention-holdout-seal.json](attention-holdout-seal.json) | RTX 4070 attention seal |
| [v02-frozen-fit.json](v02-frozen-fit.json), [v02-freeze-amendment.json](v02-freeze-amendment.json) | Frozen baseline fits and the freeze amendment |
| [v02-holdout-scores.json](v02-holdout-scores.json), [v02-holdout-report.md](v02-holdout-report.md) | Scores and the report |

### Attention suite

| File | What it is |
| --- | --- |
| [attention-suite.md](attention-suite.md), [attention-suite.json](attention-suite.json) | Attention candidate space, splits and label protocol |

### Phase 2 (RTX 4070 to GB10)

| File | What it is |
| --- | --- |
| [phase2-preregistration.md](phase2-preregistration.md) | Protocol and pass criteria |
| [phase2-aot-seal.json](phase2-aot-seal.json), [phase2-gb10-seal.json](phase2-gb10-seal.json) | Compiled artifact index and label seal |
| [phase2-thermal-stop.md](phase2-thermal-stop.md), [phase2-thermal-audit.json](phase2-thermal-audit.json), [phase2-thermal-trend-audit.json](phase2-thermal-trend-audit.json), [phase2-thermal-prefill-1024.json](phase2-thermal-prefill-1024.json) | Thermal gate stop and its audits |
| [phase2-acquisition-audit.md](phase2-acquisition-audit.md) | Compute contention audit |
| [phase2-gb10-scores.json](phase2-gb10-scores.json), [phase2-gb10-report.md](phase2-gb10-report.md) | Scores and the report |

### Phase 1 (RTX 4070)

| File | What it is |
| --- | --- |
| [suite.json](suite.json) | GEMM candidate matrix and splits |
| [m2-audit.md](m2-audit.md) | Development scoring audit |
| [holdout-seal.json](holdout-seal.json) | Workload holdout seal |
| [phase1-holdout-scores.json](phase1-holdout-scores.json), [phase1-holdout-report.md](phase1-holdout-report.md) | Scores and the report |

### Post-hoc access

| File | What it is |
| --- | --- |
| [post-hoc-access-log.md](post-hoc-access-log.md) | Every read of a published holdout after its decision, written by `posthoc.py` |

## Running

Run commands from the repository root. The scorer uses the installed
VirtualGPU Python package to identify the current hash protocol. The runner
uses Python 3.12, PyTorch 2.8 with CUDA 12.8, Triton 3.9,
`cuda-bindings`, `nvidia-ml-py`, and the installed VirtualGPU Python package.
Use a worktree-local `.venv` and reuse cached CUDA wheels where available.

### Run a development case

```sh
flock /tmp/virtualgpu-gpu.lock .venv/bin/python -m validation.runner \
  validation/suite.json decode-16 results/decode-16.json --blocks 15 \
  --config m32n64k32-w4-s1-g1 --config m64n64k32-w4-s2-g1 \
  --config m64n128k32-w4-s2-g1
```

Repeat in separate processes with unique output paths and order seeds. M1
accepts explicit development cases on the RTX 4070. The `m1-cube` case uses the
shape and compile knobs in `schemas/examples/gemm_sm89.json`. The hand-written
IR has illustrative resource metadata, and the runner records the actual
compiled resources. Device-dependent boundary rules in the suite reserve
development and holdout cases for a later sweep.

The runner checks every candidate against an FP32 reference before timing.
Each randomized block has one warm and one cold trial for every valid
candidate. Each CUDA Graph primes the same buffers or scans a disjoint
read-only buffer with L2-caching, L1-bypassing loads, then brackets one GEMM
with CUDA events. The eviction buffer holds at least three times the reported
L2 capacity and at least 128 MiB. It is initialized once. Cache
preparation and host submission are outside that interval. A separate read
probe records the observed effect of cache preparation. Clock telemetry is
record-only and is sampled asynchronously. An unsupported telemetry query is
recorded as unavailable. Reports include every sample and its block, the
cache control, compiled artifact hashes, timing dispersion, and telemetry.
Candidate statuses distinguish compilation failures, launch failures, and
incorrect outputs. Only successful candidates contribute timing samples.
Each cache regime reports a bootstrap 95% median interval, its half-width
divided by the median, and each config's median gap to the fastest measured
candidate. The scorer also reports these statistics across input sessions,
resampling blocks within each session.

### Compare with a prediction

```sh
python3 -m validation.baseline schemas/examples/profile_example.json \
  results/decode-16.json > results/baseline.json
.venv/bin/python -m validation.score results/baseline.json results/decode-16.json \
  --profile-revision 0.1.0-example --cache cold --top-k 2 --unchecked --diagnostic
.venv/bin/python -m unittest validation.test_validation
```

The schema example is a substitute for a measured profile when checking the
interfaces. Use the measured profile and its revision for experiments. The
baseline uses a full-wave duration for a partial wave and the profile's DRAM
read bandwidth as one off-chip rate.

For the core comparison, pass its JSON array of prediction records in place of
`results/baseline.json`. Each record contains `case_id`, `config_id`,
`profile_revision`, `status`, and, for successful predictions,
`nominal_latency_ns`. The score report places the measured median beside the
nominal prediction. Provide the `draws` object keyed by draw ID, including
`"0"`, to enable pairwise scoring. Pass multiple raw files to score matched
blocks across sessions. Compare one cache regime at a time with a matching
initial-cache assumption.

For validated comparisons, supply `--aot CONFIG_ID=KERNEL_IR_JSON` once per
config in place of `--unchecked`. Generate the IR offline from the exact same
`dense_gemm` function, arguments, specializations, and compiler options used
by the runner. Pass explicit `call_args` containing three aligned FP16
`MockTensor` arguments followed by `M`, `N`, and `K` to the frontend
extractor. It takes the tile, traversal, warp, and stage knobs in `config`.
The scorer verifies the versioned PTX digest, toolchain, launch grid, and
resources. It reports TTGIR digest agreement as a diagnostic. Unverified or
mismatched samples remain in runner reports and are excluded from validated
scoring. The scorer requires the current hash version for both artifacts.
The exploratory flag skips the AOT comparison and still requires a
current-version JIT report.

For formal core scoring, pass `--core-revision` with the expected full Git
SHA or a file containing it. Every prediction record must identify that
same clean formal build. Use `--diagnostic` for baseline smoke checks or
historical predictions with missing, dirty, or stale build information.
The report records the result class. AOT IR toolchain metadata and JIT
session reports include the generation HEAD and the dirty state of
`python/` and `validation/`.

### Run the M2 development sweep

Generate the development-only manifest using the measured 4070 profile. This
also materializes the planned single-SM-round boundary cases from its SM
count. The frozen portable manifest retains its holdout declarations.

```sh
mkdir -p results/m2
.venv/bin/python -m validation.suite results/m2/suite.json \
  --development-profile profiles/rtx4070.json
TRITON_DEFAULT_FP_FUSION=1 .venv/bin/python -m validation.aot \
  results/m2/suite.json results/m2/aot
TRITON_DEFAULT_FP_FUSION=1 flock /tmp/virtualgpu-gpu.lock \
  .venv/bin/python -m validation.runner results/m2/suite.json \
  decode-1 results/m2/decode-1-session-1.json --blocks 10 --seed 101
```

Collect sessions 1, 2, and 3 per development case, using order seeds 101,
102, and 103. Each invocation takes and releases the lock independently.
The offline compiler records rejected candidates in each case directory.
The experiment verifies the artifacts and all blocks before using the timings.

```sh
.venv/bin/python -m validation.experiment results/m2/suite.json results/m2 \
  profiles/rtx4070.json results/m2/scores.json
```

The learned baseline needs NumPy. It learns bounded wave efficiency with a
fixed ridge penalty from the other development cases, using both cache regimes.
The report includes a fit trained on all development cases for future transfer
evaluation. The two baselines apply the same 5% practical band to nominal
predictions for pair scoring. Pass `--core-predictions` to include the model
comparison and the development checks from `docs/design.md`.

```sh
git rev-parse HEAD > results/m2/candidate-core-revision.txt
.venv/bin/python -m validation.predict results/m2/suite.json results/m2 \
  profiles/rtx4070.json results/m2/core-predictions.json \
  --core-revision results/m2/candidate-core-revision.txt
.venv/bin/python -m validation.experiment results/m2/suite.json results/m2 \
  profiles/rtx4070.json results/m2/scores.json \
  --core-predictions results/m2/core-predictions.json \
  --core-revision results/m2/candidate-core-revision.txt
```

The prediction JSON carries `profile_revision`, `manifest_sha256`, and
`perturbation_draws`. Its `cases` object maps case ID, cache regime, and config
ID to prediction records. Each successful record has draws keyed by ID,
including draw 0 equal to `nominal_latency_ns`. The core orders a pair when
all draws agree on a direction by at least the fixed 5% practical margin.
Formal prediction also requires that the loaded extension matches HEAD and
that the Python frontend and validation sources are clean. Build the extension
from the candidate checkout before running the command. Both the prediction
and scoring commands accept an explicit `--diagnostic` mode.

Use `validation.runner --cache cold` to replace a cold measurement without
rerunning warm trials. Keep the original sessions in place and pass the
replacement directory to both `validation.predict` and `validation.experiment`
through `--cold-results`. The experiment reads only warm samples from the
original sessions and requires read-only eviction for every cold sample.

### Sealed workload holdout

Generate the holdout manifest and AOT files before measurement. The
holdout collector checks the compiled artifacts before timing. It writes
raw reports with no read permissions and produces separate digest receipts.
Use the GPU lock and suppress the runtime output during sealed capture.

```sh
.venv/bin/python -m validation.aot results/holdout-sealed/manifest.json \
  results/holdout-sealed/aot --holdout
flock /tmp/virtualgpu-gpu.lock .venv/bin/python -m validation.holdout \
  capture results/holdout-sealed prefill-256 1
.venv/bin/python -m validation.holdout audit results/holdout-sealed \
  validation/holdout-seal.json
```

The audit command reads receipt files and report metadata only. Commit the
digest index and keep the raw reports in the ignored results directory.
To score a sealed case, supply `--sealed-audit validation/holdout-seal.json`,
`--core-revision` with the full `phase1-final` SHA, and
`--prediction-sha256` from the separately archived formal prediction file.
The scorer checks the tag, the archive digest and build provenance, and the
committed audit before it accesses the timing reports. It checks each
report digest before parsing it and restores the no-read permissions.

### GB10 transfer

Follow [the preregistration](phase2-preregistration.md). Run the AOT and
prediction entry points with the phase1-final checkout as the working
directory. Put its root and its `python/` directory first on `PYTHONPATH`,
followed by the validation harness checkout. Pass the frozen development
manifest and workload-holdout manifest to `python -m validation.phase2_aot`.
The artifact directory must be new. Record the resulting index digest in the
AOT seal before running `python -m validation.phase2_predict` with the frozen
fit, published GB10 profile commit, and published AOT seal commit.

Collection uses the committed harness checkout on Spark 1, with the tagged
`python/` directory on `PYTHONPATH`. Reserve the GPU, review uptime and GPU
usage, and use the lock for each case-session:

```sh
flock /tmp/virtualgpu-gpu.lock python -m validation.phase2_capture capture \
  "$artifact_dir" "$phase1_final_checkout" "$cohort" "$case_id" "$session" \
  "$published_amendment"
```

The collector checks the amendment against `origin/main` and verifies the
AOT index, three-method prediction archive, and measurement policy digests.
The policy binds the harness commit, GPU driver, L2 capacity, clock policy,
and power mode to calibration. Collection writes mode-zero reports and
separate receipts. Use `validation.phase2_capture audit` to check the full
receipt census without opening labels.

### Post-hoc access to a published holdout

`validation.score --post-hoc --purpose "..."` accepts the published
`validation/holdout-seal.json` and `validation/phase2-gb10-seal.json` seals.
Refresh `origin/main` before running it. The published report must reference
the seal digest. Supply the prediction build's full SHA with
`--core-revision`, the seal with `--sealed-audit`, and one matching
`--aot CONFIG_ID=KERNEL_IR_JSON` per scored config.

The scorer logs every access attempt before opening any bytes in
`results/post-hoc-access.jsonl`. After the access attempt, it appends the UTC
time, prediction revision, purpose, and requested entries from the seal to
[the tracked access log](post-hoc-access-log.md). Include this summary in the
next commit. On Linux, the scorer pins the mode-zero inode,
temporarily opens a read descriptor, then restores mode zero before reading
through that descriptor. It verifies every selected seal digest before
parsing any reports. No readable copy is retained. The output carries
`result_class=post-hoc` and is ineligible for formal decisions. The
published conclusions and the formal scoring path retain their original
release requirements.

### v0.2 CPU prediction archive

After the four capture receipts are published, check out the clean
`v0.2-final` tag and load a matching formal core build. Keep the archived
AOT metadata from each capture. The prediction command consumes those AOT
directories, the frozen GPU profiles, the committed receipt metadata,
and an already frozen NeuSight-style fit bundle.

```sh
python -m validation.v02_predict results/v02-formal \
  --fit results/v02-fit.json --fit-sha256 "$fit_sha256" \
  --rtx4070-gemm-aot results/v02-rtx4070/gemm/aot \
  --rtx4070-attention-aot results/attention-fde371d/aot \
  --gb10-gemm-aot results/v02-gb10/gemm/aot \
  --gb10-attention-aot results/v02-gb10/attention/aot
```

The fit bundle contains the full `core_revision`, a `profile_sha256` mapping
keyed by `rtx4070` and `gb10`, and a `fits` mapping keyed by `gemm` and
`attention`. Each fit uses the `EfficiencyFit.to_json()` fields plus
`training_inputs_sha256`. Train only with the inputs enumerated in
[the preregistration](v02-preregistration.md).

The command is CPU-only. It verifies the capture AOT digests and regenerates
KernelIR using the frozen frontend, with the KernelIR and AOT caches off,
from the `virtualgpu` package of this checkout (not an installed wheel).
Candidates with failed or mismatched artifacts remain in the archive with
their reasons. Such a candidate has no baseline predictions either: both
baselines read the MMA instruction shape, which only the KernelIR gives, so
they share the frozen frontend's failures. The output directory
contains `predictions.json`, the IR archive and AOT index, and
`prediction-evidence.json` with the digests for the freeze amendment.
The four AOT arguments identify directories of per-case compiler metadata.
Raw timing labels are outside the command's inputs.

#### v0.2 formal scoring and release

Publish the archive's `prediction-evidence.json` verbatim as
`validation/v02-freeze-amendment.json` after review. The command requires
the published `v0.2-final` tag, a clean checkout at that tag, and the
published amendment commit. Run it only after the release is approved.

```sh
python -m validation.v02_score results/v02-formal "$amendment_commit" \
  results/v02-scores --measurement-root "$capture_checkout"
```

The measurement root contains the original per-cohort capture directories
under `results/`, including their sealed reports and AOT archives. For
archives collected on Spark, preserve the `results/v02-gb10/` hierarchy
when staging them for release. The RTX 4070 GEMM retry's effective report
and failed predecessor are both referenced by the published receipt.
The scorer verifies all report digests together before interpreting any
timings and scores only effective captures. The original RTX 4070 attention
reports use the older receipt format. Their complete sample census verifies
ten blocks per session.

The command creates `scores.json` and `report.md`. The JSON records every
candidate's capture status by session, prediction coverage, and confusion
matrices. The Markdown reports both device decisions and the per-cohort
criteria together with coverage and missing candidates.
