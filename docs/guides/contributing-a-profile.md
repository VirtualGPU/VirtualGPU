# Contributing a GPU profile

VirtualGPU lets developers compare kernel variants for GPUs they do not own, as long as someone with that GPU has run the isolated microbenchmark suite once and submitted the resulting profile to `profiles/`.

This guide walks through building the microbenchmark binary, collecting a profile with `microbench/collect_profile.py`, running self-checks, and opening a pull request.

## Principles

- **Public APIs and isolated microbenchmarks only.** Never reverse engineer vendor drivers, firmware, or machine code.
- **Never fit to workloads.** Do not hand-edit profile numbers to match GEMM, attention, or MoE timings.
- **Be honest about provenance.** Keep `provenance.source` (`measured`, `partial`, or `spec_derived`) and `clock_policy.locked` accurate; VirtualGPU relies on those flags and `rel_uncertainty` to avoid overconfident rankings.

## Prerequisites

- **Hardware:** An NVIDIA GPU with a working driver and `nvidia-smi` accessible. Native Linux is preferred so clocks can be locked; WSL2 is supported in record-only mode (the collector records clock drift and widens `rel_uncertainty` automatically).
- **Python:** Python `>= 3.10` and [`uv`](https://docs.astral.sh/uv/) (or standard `venv` + `pip`).
- **CUDA Toolkit (`nvcc` + runtime):** Either a system CUDA installation on `PATH`, or pip-installed `nvidia-cuda-nvcc` and `nvidia-cuda-runtime` inside the project virtual environment. Use a toolkit version that supports your target GPU architecture (for example, CUDA 12.8+ for `sm_120`/`sm_121`).

## Step 1: Set up the environment and build `vgpu_bench`

From the repository root:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv jsonschema nvidia-cuda-nvcc nvidia-cuda-runtime
```

If you already have a matching system `nvcc` on `PATH`, you only need `jsonschema` in `.venv`. `microbench/build.py` checks the venv first and falls back to `nvcc` on `PATH`.

Compile the CUDA C++ microbenchmark binary for your GPU's compute capability (`--arch`):

```bash
.venv/bin/python microbench/build.py --arch sm_89
```

Common `--arch` values:

| Architecture | Example GPUs | `--arch` |
|---|---|---|
| Ampere | A100, A800, RTX 3090 | `sm_80` / `sm_86` |
| Ada | L40S, L20, RTX 4090, RTX 4070 | `sm_89` |
| Hopper | H100, H800, H200, H20 | `sm_90` |
| Blackwell (datacenter) | B200, GB200 / B300, GB300 | `sm_100` / `sm_103` |
| Blackwell (client/edge) | RTX 5090, RTX PRO 6000 / GB10 | `sm_120` / `sm_121` |

This produces `microbench/build/vgpu_bench`.

## Step 2: Prepare the GPU (clock locking and isolation)

Ensure no other GPU workloads are running. On a shared machine, wrap the run with `flock /tmp/virtualgpu-gpu.lock ...`.

On **native Linux** (recommended when you have root privileges), lock SM and memory clocks to sustainable frequencies before collecting so measurements have minimal drift:

```bash
# Inspect supported clocks:
nvidia-smi --query-supported-clocks=gr,mem --format=csv -i 0

# Lock clocks (replace with sustainable values for your GPU):
sudo nvidia-smi -i 0 -lgc <SM_MHZ>,<SM_MHZ>
sudo nvidia-smi -i 0 -lmc <MEM_MHZ>,<MEM_MHZ>
```

After collection finishes, reset clocks back to default:

```bash
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
```

On **WSL2** or platforms where clock locking is unavailable, skip this step. `collect_profile.py` polls `nvidia-smi` under active load, sets `clock_policy.locked = false`, and widens `rel_uncertainty` automatically.

## Step 3: Collect the profile

Choose a lowercase filename slug for your GPU under `profiles/` (for example `profiles/rtx4090.json`, `profiles/h100_sxm5.json`, or `profiles/gb10.json`), then run:

```bash
.venv/bin/python microbench/collect_profile.py \
  --device 0 \
  --output profiles/<gpu_slug>.json
```

Useful CLI options:

- `--binary`: Path to `vgpu_bench` (default: `microbench/build/vgpu_bench`).
- `--device`: CUDA device index (default: `0`).
- `--samples`: Timed iterations per benchmark point (default: `15`).
- `--repeats`: Repeats per CTA dispatch probe (default: `5`).

What it runs: `collect_profile.py` runs every `vgpu_bench` mode (memory, L2, MMA, pipeline, sync loop, SFU and shuffle, spill, launch and CTA dispatch), polls `nvidia-smi` for clocks and thermal throttling, computes bootstrap 95% confidence intervals, validates the result against `schemas/gpu_profile.schema.json` and writes the profile. Runtime depends on the GPU's memory size and bandwidth.

## Step 4: Check it before opening a PR

Run all checks locally before committing:

1. **Schema and semantic validation:**
   ```bash
   .venv/bin/python -m virtualgpu.profile profiles/<gpu_slug>.json
   ```
   Confirm it prints `OK: <SKU> (<arch>, schema 0.1)`.

2. **Provenance and thermal hygiene:**
   Open `profiles/<gpu_slug>.json` and verify:
   - `provenance.driver_version`, `toolkit_version`, `suite_version`, and `host_os` are populated.
   - `provenance.source` matches the leaf `MeasuredValue.source` tags (`measured` if all leaves are `measured`, `partial` if any leaves are `estimated`, `spec_derived` if all leaves are `estimated`).
   - `provenance.clock_policy.throttle_events` is `0`. If throttling occurred, let the GPU cool down (or lock to a lower sustainable clock) and re-run.
   - If you locked clocks on native Linux, verify `observed_sm_clock_range_mhz` shows `[min, max]` equal to the locked clock.

3. **Physical sanity check:**
   - `specs.sm_count` and `specs.l2_size_bytes` match your GPU.
   - `measured.memory.dram_bw_read_gbps.value` is below the card's theoretical DRAM bandwidth.
   - `measured.memory.l2_bw_read_gbps.value` exceeds `dram_bw_read_gbps.value`.

4. **End-to-end Rust core check (optional, if Rust is installed):**
   ```bash
   cargo test
   ```

5. **Repo checks:**
   ```bash
   python scripts/check_profiles.py
   python scripts/check_text.py
   ```

## Step 5: Open a pull request

1. Add a row for your GPU to the table in [`profiles/README.md`](../../profiles/README.md).
2. Commit only your profile JSON and the `profiles/README.md` update:
   ```bash
   git add profiles/<gpu_slug>.json profiles/README.md
   git commit -m "feat: add <GPU SKU> (<arch>) profile" -- profiles/<gpu_slug>.json profiles/README.md
   ```
3. Open a pull request against `main` on `github.com/VirtualGPU/VirtualGPU`. In the PR description, note your GPU model, host OS (native Linux vs. WSL2), driver/CUDA versions, and whether clocks were locked.

CI will automatically validate your profile against the JSON schema, all semantic invariants in `python/virtualgpu/profile.py`, `provenance` completeness, `source` honesty, and the text check.
