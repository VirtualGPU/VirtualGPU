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

**The task.** A GPU kernel such as a matrix multiply (GEMM) or attention has settings that do not change its result but change its speed: the tile size each thread block works on, the number of warps, and the number of pipeline stages that overlap loading data with computing on it. Each combination is a config. For one input shape there are usually dozens of candidate configs, and the fastest one differs from shape to shape and from GPU to GPU.

**How it is done today.** Each candidate is compiled and timed on the real GPU, and the fastest is kept. Triton's `@triton.autotune` does exactly this. It needs the target GPU to be free for the whole search, and it can't be done at all for a GPU you don't have.

**The question.** Can most of the candidates be ruled out on a CPU first, so that only a few need to run on the card?

**How we test it.** VirtualGPU predicts how the configs of one shape rank against each other, from a profile of the target GPU (its public specs plus rates measured by small benchmarks, never kernel timings). The real GPU timings of the test shapes stay sealed until all predictions are archived, and the rules for passing are written down before any result is opened. Three numbers are reported:

- **Kendall tau-b**: how well the predicted order matches the measured order. 1 is the same order, 0 is no relation.
- **Top-5 regret**: how much slower the best of the five predicted fastest configs is than the truly fastest config. 0% means the fastest config is in the five.
- **Agreement of ordered verdicts**: VirtualGPU only says "A is faster than B" when A wins by at least 5% in all of its perturbed copies of the profile. This is the share of those verdicts the real GPU agrees with.

Two baselines are scored on the same data: a roofline model with wave quantization, and a NeuSight-style predictor learned from earlier kernel timings.

**What the results say so far.** To pass, a round needs a ranking better than both baselines by a set margin, and a top-5 regret of at most 1% and no more than 0.5 points worse than the baselines'. None of these three rounds has passed in full:

- [Phase 1](docs/results/phase1.md), RTX 4070, GEMM: tau-b 0.787 warm and 0.750 cold, against at most 0.514 for the baselines, and about 95% of ordered verdicts agreed with the GPU. Top-5 regret was 3.11% and 3.02%: six of the nine shapes had 0%, and one shape with unaligned dimensions had 24.5% and 22.2%. Formal decision: Fail.
- [Phase 2](docs/results/phase2.md), the same frozen model on the DGX Spark GB10 with a GB10 profile: tau-b 0.446 warm and 0.597 cold, above both baselines by the required margin, and over 95% of ordered verdicts agreed. Top-5 regret was 1.095% warm, 0.095 points over the 1% limit, and 0.156% cold. Formal decision: Fail.
- [v0.2](docs/results/v0.2.md), fresh shapes, GEMM and attention on both GPUs: 87% to 94% of ordered verdicts agreed with the GPU, against 54% to 76% for the baselines. RTX 4070 GEMM passed. RTX 4070 attention (top-5 regret 1.30% warm, 1.43% cold) and GB10 GEMM (2.12% and 2.00%, and the learned baseline's regret was lower on cold) failed the regret limit, and GB10 attention warm missed the ranking margin over the learned baseline by 0.002 (0.048 against 0.05). A GPU passes only if both kernels pass warm and cold, so the formal decision for both GPUs is Fail.

So far: when VirtualGPU orders two configs it is usually right, but its shortlist is not yet reliable enough to skip measuring the top few on the card.

**Limits.**

- It ranks configs of the same kernel and shape against each other. It is not built to predict how long a kernel takes, and the times it computes along the way are not meant to be used as such.
- It needs a profile of the GPU you are tuning for. A ranking with another card's profile says nothing about yours.
- Only the `rtx4070` and `gb10` profiles have been checked against sealed GPU timings in the rounds above. The Ascend 910B3 has its own sealed test, reported in [docs/results/ascend910b3-holdout.md](docs/results/ascend910b3-holdout.md). The other bundled profiles have not been checked.

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
