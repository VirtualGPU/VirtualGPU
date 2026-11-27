<div align="center">

<img src="./assets/icon.svg" width="120" height="120" alt="VirtualGPU Logo" />

<h1>VirtualGPU</h1>

<p>Predict which GPU kernel variant is faster, without the GPU</p>

<p>
  <img src="https://img.shields.io/github/stars/VirtualGPU/VirtualGPU?style=flat&label=Stars&color=F39044&labelColor=1a1a1a" />
  <img src="https://img.shields.io/github/forks/VirtualGPU/VirtualGPU?style=flat&label=Forks&color=F39044&labelColor=1a1a1a" />
  <img src="https://img.shields.io/github/v/release/VirtualGPU/VirtualGPU?style=flat&label=Release&color=F39044&labelColor=1a1a1a" />
  <img src="https://img.shields.io/github/license/VirtualGPU/VirtualGPU?style=flat&label=License&color=F39044&labelColor=1a1a1a" />
</p>

</div>

<br />

## Overview

Tuning a GPU kernel usually means trying dozens of settings on the real card and keeping the fastest. That needs the GPU you are tuning for, and often you don't have it.

VirtualGPU does that search on a normal CPU. You give it a Triton kernel, its autotune configs and a profile of the GPU you care about. It tells you which configs should be fastest, which ones are too close to call, and what is holding each one back.

It doesn't try to guess exact run times. It only compares configs against each other, which is a much easier problem: if the model is off by 20% on memory speed, every config is off by about the same amount and the order stays the same. You still check the top few on real hardware, but you skip the rest.

Everything comes from public specs and small benchmarks we wrote ourselves. Nothing is reverse engineered and nothing is fitted to kernel timings.

<br />

## What it is for

A kernel like a matrix multiply (GEMM) or attention has settings that change its speed but not its result: tile size, number of warps and number of pipeline stages. Each combination is a config. The fastest config changes from shape to shape and from GPU to GPU, so today you time every candidate on the real card, which is what `@triton.autotune` does. That ties up the GPU, and you can't do it at all for a card you don't have.

VirtualGPU tries to rule out most candidates on a CPU first, so only a few need to run on the GPU.

We test this honestly: the pass rules are written down first, and the real GPU timings stay sealed until every prediction is archived. Each round is scored on how well the predicted order matches the real one (Kendall tau-b), how much slower the best of the top 5 picks is than the true fastest (top-5 regret), and how often the GPU agrees when VirtualGPU says one config is faster. It is compared against a roofline model and a learned NeuSight-style predictor.

In the latest round, [v0.2](docs/results/v0.2.md), the GPU agreed with 87% to 94% of VirtualGPU's "A is faster than B" calls, against 54% to 76% for the baselines. RTX 4070 GEMM passed, but the other kernel and GPU pairs missed the top-5 regret limit or the ranking margin, so the formal result is Fail. In short, when it orders two configs it is usually right, but you should still time the top few on the card. Earlier rounds are in [Phase 1](docs/results/phase1.md) and [Phase 2](docs/results/phase2.md).

Limits:

- It ranks configs of one kernel and shape against each other. It does not predict run times.
- It needs a profile of the GPU you are tuning for. Another card's profile says nothing about yours.
- Only the `rtx4070` and `gb10` profiles have been checked against sealed GPU timings. The Ascend 910B3 has its own test in [docs/results/ascend910b3-holdout.md](docs/results/ascend910b3-holdout.md). The other bundled profiles have not been checked.

<br />

## How it works

VirtualGPU compiles each config with Triton for the GPU you picked, without needing that GPU. It reads the compiler's output to see the tile sizes, how many pipeline stages the loop really got, how much memory each block touches, and how many registers it uses or spills.

That goes into a simulator written in Rust. It places thread blocks onto the GPU's SMs, splits compute, L2 and memory bandwidth between them, and plays the kernel out from start to finish.

The GPU itself is described by a profile: one JSON file with the card's public specs and the rates we measured on it with our [benchmark suite](microbench/README.md). Anyone with a GPU can measure one and send it in.

Measurements are never exact, so each config is run against 32 slightly changed copies of the profile. VirtualGPU only says one config is faster if it wins by at least 5% in every copy. Otherwise it calls them a tie.

The full design is in [docs/design.md](docs/design.md), and the [docs index](docs/README.md) lists everything else.

<br />

## Installation

Wheels for Linux on `x86_64` and `aarch64` (Python 3.10 to 3.13) are on [GitHub Releases](https://github.com/VirtualGPU/VirtualGPU/releases). It is not on PyPI yet.

```bash
pip install "virtualgpu-0.2.0-cp310-abi3-manylinux_2_34_x86_64.whl[triton]"
```

To build from source you need a Rust toolchain:

```bash
pip install ".[triton]"
```

The `[triton]` extra installs Triton 3.9.0, which you need to read Triton kernels. Triton only runs on Linux, so on Windows use WSL2.

<br />

## Usage

Rank the configs of a `@triton.autotune` GEMM kernel for an RTX 4070, on any machine:

```bash
virtualgpu autotune --kernel my_module:matmul --shape 2048 8192 4096 --profile rtx4070
```

Or let VirtualGPU cut down the search inside Triton, so only the 5 most promising configs get benchmarked on your GPU:

```python
@virtualgpu.autotune(configs=CONFIGS, key=["M", "N", "K"], profile="rtx4070", top_k=5)
@triton.jit
def matmul(...):
    ...
```

The report lists every config from fastest to slowest, a shortlist worth measuring, the pairs it can't tell apart, and what limits each config (like tensor cores, L2 bandwidth or memory latency). Configs it can't model are listed last with the reason. Add `--out report.json` to save it. The full format is in [docs/components/autotune.md](docs/components/autotune.md).

Profiles can be passed by name or as a path to a JSON file. `virtualgpu.list_profiles()` shows the bundled ones.

<br />

## What works now

- Triton GEMM and fused attention kernels, including causal attention and grouped query attention
- Autotuning for GEMM, from the command line, from Python, or inside `@triton.autotune`
- NVIDIA GPUs that have a profile

Bundled profiles:

| Profile | GPU | Checked against real results |
| --- | --- | --- |
| `rtx4070` | NVIDIA GeForce RTX 4070 | Yes |
| `gb10` | NVIDIA DGX Spark GB10 | Yes |
| `a100-80gb` | NVIDIA A100-SXM4-80GB | Not yet |
| `l4` | NVIDIA L4 | Not yet |
| `l40s` | NVIDIA L40S | Not yet |
| `rtx3070` | NVIDIA GeForce RTX 3070 | Not yet |
| `rtx4090` | NVIDIA GeForce RTX 4090 | Not yet |
| `rtxpro5000-72gb` | NVIDIA RTX PRO 5000 72GB Blackwell | Not yet |
| `rtxpro6000` | NVIDIA RTX PRO 6000 Blackwell Server Edition | Not yet |
| `ascend910b3` | Huawei Ascend 910B3 | Not yet |

How well it does right now: in the latest test (v0.2), when VirtualGPU said one config was faster than another, the real RTX 4070 and GB10 agreed 87% to 94% of the time. The simple baselines we compare against agreed 54% to 76% of the time. The shortlist isn't good enough yet to skip measuring altogether, so v0.2 didn't pass all of its own targets. The numbers are in [docs/results/v0.2.md](docs/results/v0.2.md), and the known gaps are in [design.md](docs/design.md#9-known-limits).

<br />

## Roadmap

What we are building, version by version up to 1.0.0 and after, is in [ROADMAP.md](ROADMAP.md).

<br />

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). If you have a GPU we don't have a profile for yet, measuring it is the most helpful thing you can do. See [contributing a profile](docs/guides/contributing-a-profile.md).

<br />

## Support

Bugs and ideas go in this repo: [report a bug](https://github.com/VirtualGPU/VirtualGPU/issues/new?template=bug_report.yml) or [request a feature](https://github.com/VirtualGPU/VirtualGPU/issues/new?template=feature_request.yml). Add as much detail as you can, in English if possible. Security problems go through a private report, see [SECURITY.md](https://github.com/VirtualGPU/.github/blob/main/SECURITY.md).

<br />

## License

Apache License 2.0. See [LICENSE](LICENSE).
