<div align="center">

<img src="./assets/icon.svg" width="120" height="120" alt="VirtualGPU Logo" />

<h1>VirtualGPU</h1>

<p>Predict which GPU kernel variant is faster, without the GPU</p>

<p>
  <img src="https://img.shields.io/github/stars/VirtualGPU/VirtualGPU?style=flat&label=Stars&color=ffffff&labelColor=141513" />
  <img src="https://img.shields.io/github/forks/VirtualGPU/VirtualGPU?style=flat&label=Forks&color=ffffff&labelColor=141513" />
  <img src="https://img.shields.io/github/v/release/VirtualGPU/VirtualGPU?style=flat&label=Release&color=ffffff&labelColor=141513" />
  <img src="https://img.shields.io/github/license/VirtualGPU/VirtualGPU?style=flat&label=License&color=ffffff&labelColor=141513" />
</p>

</div>

<br />

## Overview

VirtualGPU is a free, open source performance model for GPU kernels that runs on a CPU. Give it a Triton kernel, its autotuning configs and a GPU profile, and it ranks the configs from fastest to slowest, says which ones it can't tell apart, and says what bounds each one.

It does not predict how long a kernel takes. It predicts which of two kernels is faster. Errors that slow every variant by the same amount cancel out, so a ranking is much easier to get right than an exact time. Real hardware stays the final judge: VirtualGPU narrows hundreds of configs down to a few worth measuring.

The model is built only from public specs and our own microbenchmarks. Nothing is reverse engineered and nothing is fitted to kernel timings.

<br />

## How it works

```
Triton kernel + autotune configs
        |
        |  compiled ahead of time for the target GPU, no GPU needed
        v
KernelIR: tiles, pipeline stages, memory footprints, registers and spills
        |
        |  + GPU profile (public specs and measured rates, one JSON file per GPU)
        v
Event-driven, tile-level performance model, run once per perturbed profile draw
        |
        v
Ranking, configs it can't tell apart, and the bound resource for each config
```

- Frontend: compiles each config with Triton for the target architecture and reads the tile structure from the compiler's output. Code in `python/virtualgpu/frontend/`.
- Kernel IR: one neutral format that any frontend can emit. Schema in `schemas/kernel_ir.schema.json`.
- Model core: schedules CTAs onto SMs, shares compute, L2 and DRAM between them, and plays the kernel out. Written in Rust in `crates/vgpu-core/`.
- GPU profiles: one JSON file per GPU, measured once by the [microbenchmark suite](microbench/README.md).

The model reruns every config on 32 slightly nudged copies of the profile. A pair is ordered only if one config is at least 5% faster in every copy. Otherwise it is reported as indistinguishable.

More detail is in the [design](docs/design.md) and the [docs index](docs/README.md).

<br />

## Installation

Linux `abi3` wheels for `x86_64` and `aarch64` (Python 3.10 to 3.13) and a source archive are on [GitHub Releases](https://github.com/VirtualGPU/VirtualGPU/releases). It is not on PyPI yet.

```bash
# A release wheel, with the Triton extra for reading Triton kernels
pip install "virtualgpu-0.2.0-cp310-abi3-manylinux_2_34_x86_64.whl[triton]"

# Or from source, with a Rust toolchain installed
pip install ".[triton]"
```

The base package has the Rust model core, the JSON schemas and the bundled GPU profiles. Add the `[triton]` extra to extract or autotune Triton kernels. It pins Triton 3.9.0.

<br />

## Usage

List the bundled GPU profiles. Any profile can be passed by name or by path to a JSON file:

```python
import virtualgpu

virtualgpu.list_profiles()  # ['gb10', 'l40s', 'rtx3070', 'rtx4070']
```

Rank the configs of a `@triton.autotune` GEMM kernel on a CPU:

```bash
virtualgpu autotune --kernel my_module:matmul --shape 2048 8192 4096 \
    --profile rtx4070 --cache cold --top-k 5
```

Or prune the search inside Triton, so only the top configs are benchmarked on the real GPU. Set `profile=` or `VIRTUALGPU_PROFILE`:

```python
@virtualgpu.autotune(configs=CONFIGS, key=["M", "N", "K"], profile="rtx4070", top_k=5)
@triton.jit
def matmul(...):
    ...
```

`virtualgpu autotune` prints a JSON report, or writes it with `--out report.json`:

- `ranking`: configs from fastest to slowest by predicted time, with the time in every profile draw. Configs the model can't handle come last with `status: unsupported` and a reason.
- `shortlist`: the top configs to measure on real hardware (`top_k`, 5 by default).
- `indistinguishable_pairs`: pairs the model can't order. Treat them as tied.
- `primary_bottleneck`: what bounds each config and for what share of its time, like `mma`, `l2_bandwidth` or `dram_latency`.

The full report format is in [docs/components/autotune.md](docs/components/autotune.md).

<br />

## What works now

| Area | Status |
| --- | --- |
| Kernels | Triton GEMM and fused attention (causal, non-causal, GQA) |
| Autotuning | GEMM shapes, offline from the CLI or Python, or online inside `@triton.autotune` |
| GPUs | NVIDIA GPUs with a profile: `sm_86`, `sm_89` and `sm_121` today |
| Platforms | Linux. Triton needs Linux, so on Windows use WSL2 |

Bundled profiles:

| Profile | GPU | Arch | Validated on a sealed holdout |
| --- | --- | --- | --- |
| `gb10` | NVIDIA DGX Spark GB10 | `sm_121` | Yes |
| `rtx4070` | NVIDIA GeForce RTX 4070 | `sm_89` | Yes |
| `a100-80gb` | NVIDIA A100-SXM4-80GB | `sm_80` | No |
| `l4` | NVIDIA L4 | `sm_89` | No |
| `l40s` | NVIDIA L40S | `sm_89` | No |
| `rtx3070` | NVIDIA GeForce RTX 3070 | `sm_86` | No |
| `rtxpro6000` | NVIDIA RTX PRO 6000 Blackwell Server Edition | `sm_120` | No |

Latest results (v0.2, fresh sealed holdouts, no kernel timings used by the model):

| Cohort | Kendall tau-b, warm / cold | Verdicts that agree with the GPU: VirtualGPU / baselines |
| --- | ---: | ---: |
| RTX 4070 GEMM | 0.748 / 0.693 | 93% to 94% / 54% to 73% |
| RTX 4070 attention | 0.710 / 0.711 | 89% to 90% / 60% to 72% |
| GB10 GEMM | 0.683 / 0.689 | 94% / 59% to 64% |
| GB10 attention | 0.670 / 0.703 | 87% to 91% / 63% to 76% |

When VirtualGPU orders two configs, the GPU agrees about 9 times out of 10. The shortlist still misses the 1% regret target on two cohorts, so both GPUs formally failed the v0.2 criteria. Full results are in [docs/results/v0.2.md](docs/results/v0.2.md) and known limits in [design.md section 9](docs/design.md#9-known-limits).

<br />

## Roadmap

Everything planned up to the 1.0.0 release is in [ROADMAP.md](ROADMAP.md).

<br />

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most useful thing you can send is a profile for a GPU we don't have yet: see [contributing a profile](docs/guides/contributing-a-profile.md).

<br />

## Support

Bugs and ideas go in this repo: [report a bug](https://github.com/VirtualGPU/VirtualGPU/issues/new?template=bug_report.yml) or [request a feature](https://github.com/VirtualGPU/VirtualGPU/issues/new?template=feature_request.yml). Add as much detail as you can, in English if possible. Security problems go through a private report, see [SECURITY.md](https://github.com/VirtualGPU/.github/blob/main/SECURITY.md).

<br />

## License

Apache License 2.0. See [LICENSE](LICENSE).
