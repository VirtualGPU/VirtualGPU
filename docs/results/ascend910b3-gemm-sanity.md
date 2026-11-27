# Ascend 910B3 FP16 GEMM and Attention Sanity Check (Historical Dev Archive)

> **Status: Informal dev sanity check archived from `0.3.1-ascend`, not a sealed holdout.** For the preregistered multi-shape sealed holdout evaluation and formal verdict (**Pass** on all 4 cohorts: GEMM warm/cold and Attention warm/cold, lifting `unvalidated_target_ascend` for `ascend910b3`), see [`ascend910b3-holdout.md`](ascend910b3-holdout.md).

## Setup

- **Hardware:** Huawei Ascend 910B3 (`npu:0`, 20 AI Cores = 20 AIC + 40 AIV, 64 GiB HBM, 192 MiB L2, 1800 MHz Cube clock).
- **Software:** Driver `26.1.1`, CANN `9.2.1`, Triton `3.5.0`, Triton-Ascend `3.2.2`, `torch_npu 2.10.0.post4`.
- **Profile:** `profiles/ascend910b3.json` (`0.3.1-ascend`, M1 + M2 microbenchmarks including isolated FFTS+ cross-core sync handoff latency `cv_sync_hop_ns_by_active_sms`, zero GEMM or Attention fitting).
- **Measurement:** Cold L2 (384 MiB flush before each timed kernel), 3 warmup runs + 10 timed runs via `torch.npu.Event`, median latency reported.
- **Simulation:** `l2_state = "cold"`, 64 Latin hypercube draws (`seed = 17`), 10 Triton-Ascend FP16 GEMM configs (7 `multibuffer = True`, 3 `multibuffer = False`) and 8 forward GQA Attention configs (`validation/attention.py:fused_attention`, `Z=2, H=16, H_kv=4, S=2048, D=64`, both `causal = False` and `causal = True`).

## Summary Metrics

| Workload | Profile / Core | Spearman rho | Median rel error | Pairwise verdict precision | Top-3 regret |
|---|---|---:|---:|---:|---:|
| GEMM `2048 x 2048 x 2048` | `0.3.0-ascend` (pre CV-split) | 0.624 | -24.0% | 33/41 (80.5%) | 0.00% |
| GEMM `2048 x 2048 x 2048` | `0.3.1-ascend` (CV-split + `equal_to_1`) | 0.891 | -23.6% | 37/42 (88.1%) | 0.00% |
| GEMM `4096 x 4096 x 4096` | `0.3.0-ascend` (pre CV-split) | 0.709 | -21.0% | 32/40 (80.0%) | 0.00% |
| GEMM `4096 x 4096 x 4096` | `0.3.1-ascend` (CV-split + `equal_to_1`) | 0.879 | -21.0% | 36/38 (94.7%) | 0.00% |
| Attention `Z=2 H=16/4 S=2048 D=64` (`causal=False`) | `0.3.1-ascend` (CV-split) | 0.929 | +40.0% | 23/23 (100.0%) | 0.00% |
| Attention `Z=2 H=16/4 S=2048 D=64` (`causal=True`) | `0.3.1-ascend` (CV-split) | 0.881 | +20.8% | 23/25 (92.0%) | 0.00% |

In all four cases, the predicted top-3 shortlist achieves 0.00% regret against the fastest measured hardware config.

## Per-Config Results (`0.3.1-ascend`)

### GEMM `2048 x 2048 x 2048`

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `m128n256k64_g8_mb1` | 141.0 | 115.2 | -18.3% | `dram_latency` (0.47) |
| `m128n256k64_g8_mb0` | 148.8 | 153.4 | +3.1% | `mma` (0.81) |
| `m128n128k64_g8_mb1` | 189.8 | 136.2 | -28.2% | `dram_latency` (0.76) |
| `m128n128k64_g8_mb0` | 200.8 | 192.9 | -3.9% | `dram_latency` (0.78) |
| `m128n128k32_g8_mb1` | 299.1 | 170.9 | -42.9% | `dram_latency` (0.78) |
| `m128n128k128_g8_mb1` | 156.3 | 121.1 | -22.5% | `dram_latency` (0.75) |
| `m64n128k64_g8_mb1` | 258.0 | 194.1 | -24.8% | `l2_latency` (0.73) |
| `m64n64k64_g8_mb1` | 428.6 | 301.8 | -29.6% | `dram_latency` (0.89) |
| `m64n64k32_g8_mb0` | 787.8 | 863.6 | +9.6% | `dram_latency` (0.91) |
| `m128n256k32_g1_mb1` | 184.2 | 132.3 | -28.2% | `l2_latency` (0.80) |

### GEMM `4096 x 4096 x 4096`

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `m128n256k64_g8_mb1` | 983.2 | 801.0 | -18.5% | `mma` (0.62) |
| `m128n256k64_g8_mb0` | 1011.7 | 1089.0 | +7.6% | `mma` (0.95) |
| `m128n128k64_g8_mb1` | 1343.3 | 1028.7 | -23.4% | `dram_latency` (0.88) |
| `m128n128k64_g8_mb0` | 1415.5 | 1501.4 | +6.1% | `dram_latency` (0.88) |
| `m128n128k32_g8_mb1` | 2184.1 | 1307.7 | -40.1% | `dram_latency` (0.88) |
| `m128n128k128_g8_mb1` | 1100.1 | 901.6 | -18.0% | `dram_latency` (0.87) |
| `m64n128k64_g8_mb1` | 1948.3 | 1451.3 | -25.5% | `l2_latency` (0.86) |
| `m64n64k64_g8_mb1` | 3221.2 | 2330.3 | -27.7% | `dram_latency` (0.94) |
| `m64n64k32_g8_mb0` | 6009.5 | 6908.4 | +15.0% | `dram_latency` (0.95) |
| `m128n256k32_g1_mb1` | 1231.4 | 925.8 | -24.8% | `l2_latency` (0.94) |

### Attention `Z=2, H=16, H_kv=4, S=2048, D=64` (`causal=False`)

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `bm64_bn64_mb1` | 5721.5 | 7758.4 | +35.6% | `issue` (1.00) |
| `bm64_bn64_mb0` | 5770.2 | 8365.5 | +45.0% | `issue` (1.00) |
| `bm64_bn32_mb1` | 11417.2 | 13865.8 | +21.4% | `issue` (1.00) |
| `bm64_bn128_mb1` | 2890.6 | 4708.3 | +62.9% | `issue` (1.00) |
| `bm32_bn64_mb1` | 11412.2 | 14107.0 | +23.6% | `issue` (1.00) |
| `bm32_bn32_mb1` | 22744.5 | 25845.0 | +13.6% | `issue` (1.00) |
| `bm128_bn64_mb1` | 2901.6 | 4533.8 | +56.2% | `issue` (1.00) |
| `bm32_bn128_mb1` | 5713.8 | 8247.5 | +44.3% | `issue` (1.00) |

### Attention `Z=2, H=16, H_kv=4, S=2048, D=64` (`causal=True`)

| Config | Hardware (us) | Predicted (us) | Rel error | Dominant bound |
|---|---:|---:|---:|---|
| `bm64_bn64_mb1` | 3488.8 | 4168.0 | +19.5% | `issue` (1.00) |
| `bm64_bn64_mb0` | 2976.3 | 4522.2 | +51.9% | `issue` (1.00) |
| `bm64_bn32_mb1` | 6393.8 | 7805.8 | +22.1% | `issue` (1.00) |
| `bm64_bn128_mb1` | 2141.9 | 2655.0 | +24.0% | `issue` (0.99) |
| `bm32_bn64_mb1` | 7075.8 | 7543.5 | +6.6% | `issue` (1.00) |
| `bm32_bn32_mb1` | 12572.7 | 13393.2 | +6.5% | `issue` (1.00) |
| `bm128_bn64_mb1` | 1806.3 | 2921.4 | +61.7% | `issue` (1.00) |
| `bm32_bn128_mb1` | 4332.8 | 4416.1 | +1.9% | `issue` (0.99) |

## Observed Residuals

1. **GEMM `multibuffer = False` overestimate resolved:** Specializing `equal_to_1` strides in `ascend_compiler.py` matches Triton-Ascend's runtime `AIC_ONLY` lowering (eliminating spurious AIV GM workspace traffic), and genuine `mix`-mode workspaces stay L2-resident when fitting in `l2_effective_capacity_bytes`. `multibuffer = False` errors dropped from +64%..+127% to -3.9%..+15.0%.
2. **GEMM `multibuffer = True` optimistic bias (-18% to -43%):** Cube `MTE1` (`L1 -> L0A/L0B`) and `Fixpipe` (`L0C -> GM` NZ-to-ND) formatting/transfer pipelines are not yet modeled separately from UB copy bandwidth.
3. **Attention conservative latency bias (+2% to +63%):** Isolated FFTS+ `CrossCoreSetFlag/WaitFlag` handoff latency (`570 ns/hop` at 20 active AI Cores x 6 hops/iter = `3421 ns/iter`) accounts for the hardware per-iteration floor (`3475-3543 ns/iter`). The core currently adds AIV issue time and cross-core sync latency serially in `Parts::compute()`, causing a +20% to +56% conservative offset on larger tiles while preserving strong config ranking (Spearman rho 0.88-0.93, 92%-100% pairwise precision, 0.00% top-3 regret).
