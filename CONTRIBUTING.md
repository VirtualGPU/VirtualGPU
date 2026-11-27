# Contributing

VirtualGPU is a Rust performance model with a Python package around it. The Rust core is in `crates/vgpu-core/`, the Python package and Triton frontend in `python/virtualgpu/`, the CUDA microbenchmarks in `microbench/`, the GPU profiles in `profiles/` and the hardware validation in `validation/`. The [docs index](docs/README.md) lists what every doc covers.

## Prerequisites

- [Rust](https://rustup.rs/) stable
- [Python](https://www.python.org/downloads/) 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- [Git](https://git-scm.com/downloads)
- Linux or WSL2. Triton only runs on Linux
- An NVIDIA GPU and the CUDA toolkit only for the microbenchmarks and validation runs. Everything else runs on a CPU

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/VirtualGPU/VirtualGPU
   cd VirtualGPU
   ```
2. Create a virtual environment and install the tools:
   ```sh
   uv venv --python 3.12
   uv pip install --group dev
   ```
3. Build the Rust core into the environment:
   ```sh
   .venv/bin/maturin develop --uv
   ```
4. Set up the Git hooks. They format your staged files and check your commit message:
   ```sh
   .venv/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

## Useful commands

```sh
cargo test                                   # Rust tests
cargo clippy --all-targets -- -D warnings    # Rust lint
cargo fmt                                    # format Rust
.venv/bin/python -m pytest                   # Python tests
.venv/bin/python -m unittest validation.test_validation   # validation tests, CPU only
.venv/bin/ruff check . && .venv/bin/ruff format .         # lint and format Python
.venv/bin/python scripts/check_profiles.py   # validate every GPU profile
.venv/bin/python scripts/check_text.py       # English only, no em or en dashes
```

## Making a change

1. Fork the [repo](https://github.com/VirtualGPU/VirtualGPU).
2. Create a branch:
   ```sh
   git checkout -b feature/my-new-feature
   ```
3. Make your changes and add or update tests for them.
4. Add a line to `CHANGELOG.md` under the top version.
5. Commit your changes:
   ```sh
   git commit -m "feat: add my new feature"
   ```
6. Push to your fork and open a pull request against `main` with a clear description.

## Guidelines

- Everything is written in English: code, comments, docs, commits, issues and pull requests.
- No reverse engineering. Use public specs, documented tool output and our own measurements.
- Never fit model parameters to kernel timings. Calibration data (microbenchmarks), development data and holdout data stay apart.
- The schemas in `schemas/` are the contracts between parts. Change one only together with every part that reads it.
- When the code changes how a part works, update its doc in `docs/components/` in the same pull request.
- Files in `validation/` that a seal or fit points to are never moved, renamed or reformatted. A new evaluation adds new files.
- Tests for the Python package live in `python/tests/`, mirroring `python/virtualgpu/`. Rust tests live next to the code or in `crates/vgpu-core/tests/`. Every bug fix needs a test that fails without it.
- `ruff`, `rustfmt` and `clippy` set the code style. They run on every pull request and must pass.
- Keep comments short and only where they help.
- Use short commit messages that start with `feat:`, `fix:`, `docs:`, `ci:` or `chore:`, and link related issues.
- Security problems go through a private report, not a pull request. See [SECURITY.md](https://github.com/VirtualGPU/.github/blob/main/SECURITY.md).

## Changes to model mechanisms

Open a pull request for any change to prediction logic in `crates/vgpu-core/`,
frontend IR extraction, or profile field derivation. This applies to maintainers
too. Profile data, docs, and tooling changes follow the usual process when
they leave prediction logic unchanged.

Include three things in the PR:

- **Prediction diff.** Replay archived KernelIR on the base and candidate.
  Record both revisions, IR and profile digests, cache state, draws, and seed.
  Report which cohorts or cases remain bit-identical, how many predictions
  change, the size of the changes, and any status transitions. For an IR or
  profile derivation change, regenerate the affected input from the same source
  and keep the other input fixed.
- **Microbenchmark evidence.** Link the evidence and raw capture receipts.
  Report whether each of the three capture rules and applicable thermal,
  host or desktop gates passed. Use
  `python -m microbench.capture_posthoc check RUN.json [RUN.json ...]`
  to check the dispersion rules and include the recorded gate results.
- **Validation.** For changes that need a mechanism test, register the criterion
  and freeze the predictions before taking the validation measurements.
  Report effects on opened workloads as post-hoc. They do not change a
  frozen conclusion or supply formal validation for the PR.

For a minimal CPU comparison, run the command below in both checkouts with
their own built environments. Set `IR_DIR` to an absolute archived case
directory and `PROFILE` to the same absolute profile path. Use an actual
case ID for `CASE`, and set `OUT` to separate paths such as
`/tmp/vgpu-base.json` and `/tmp/vgpu-candidate.json`. Repeat for both cache
states and the affected cases.

```sh
CUDA_VISIBLE_DEVICES="" python -m virtualgpu.cli predict --ir-dir "$IR_DIR" \
  --case "$CASE" --profile "$PROFILE" --cache cold --draws 32 --seed 17 --out "$OUT"
```

Generate a per-config diff table from two such files:

```sh
python - /tmp/vgpu-base.json /tmp/vgpu-candidate.json <<'PYDIFF'
import json
import sys
from pathlib import Path
from statistics import median

def read(path):
    rows = json.loads(Path(path).read_text())
    return {(row["case_id"], row["config_id"]): row for row in rows}

def prediction(row):
    timings = row.get("draws", {}) | {"nominal": row.get("nominal_latency_ns")}
    return row["status"], {k: float(v).hex() for k, v in timings.items() if v is not None}

base, candidate = map(read, sys.argv[1:])
assert base and base.keys() == candidate.keys(), "Use the same nonempty case/config set"
deltas = []
unchanged = 0
print("| Case | Config | Base status | Candidate status | Nominal delta (%) |")
print("| --- | --- | --- | --- | ---: |")
for (case, config), old in sorted(base.items()):
    new = candidate[case, config]
    if prediction(old) == prediction(new):
        unchanged += 1
        continue
    delta = "n/a"
    if old["status"] == new["status"] == "ok":
        change = 100 * (new["nominal_latency_ns"] / old["nominal_latency_ns"] - 1)
        deltas.append(change)
        delta = f"{change:+.3f}"
    print(f"| {case} | {config} | {old['status']} | {new['status']} | {delta} |")
print(f"\nBit-identical predictions: {unchanged}/{len(base)}")
if deltas:
    print(f"Changed OK rows: median {median(deltas):+.3f}%, max absolute {max(map(abs, deltas)):.3f}%")
PYDIFF
```

## Contributing a GPU profile

If you have a GPU we don't have a profile for, measuring it is the most useful thing you can do. See [contributing a profile](docs/guides/contributing-a-profile.md).

## Support

Bugs and ideas go in [this repo](https://github.com/VirtualGPU/VirtualGPU/issues/new/choose). Add as much detail as you can, in English if possible.
