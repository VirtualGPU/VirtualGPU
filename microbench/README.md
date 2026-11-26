# Microbenchmark suite

CUDA C++ benchmarks that read a GPU's static limits and measure its real rates, latencies and dispatch behavior. `collect_profile.py` runs them all and writes one profile JSON (`schemas/gpu_profile.schema.json`). Nothing here is fitted to GEMM or attention timings.

| File | What it is |
| --- | --- |
| `src/vgpu_bench.cu` | The `vgpu_bench` binary and its subcommands (`dram-bw`, `l2`, `mma`, `cta-dispatch`, `spill-latency` and more) |
| `src/m2_kernels.cuh` | Kernels for the closed loop, pipeline, sync loop, SFU, shuffle and spill benchmarks |
| `src/narrow_kernels.cuh` | Post-hoc narrow-load benchmarks (`gmem-narrow`, `sync-loop-narrow`): 1 to 16 B register-staged loads, aligned and offset rows, from hot L2 and DRAM |
| `src/narrow_validate.cuh` | Follow-ups: new validation points for the DRAM-side narrow request form (`gmem-narrow-validate`, `sync-loop-narrow-validate`) and register staging split into loads and shared memory stores (`stage-narrow`, `stage-narrow-batched` with every load issued before the first store and Triton's swizzled layout, `stage-narrow-validate` and `stage-narrow-k1`, new points for the staging forms) |
| `build.py` | Builds `vgpu_bench` for one architecture |
| `collect_profile.py` | Runs every benchmark and writes the profile |
| `run_narrow.py` | Repeats a narrow-load benchmark behind the thermal gate; not part of the profile |

## Build

Install `nvidia-cuda-nvcc` and `nvidia-cuda-runtime` in the project virtual environment, then build for your GPU:

```bash
.venv/bin/python microbench/build.py --arch sm_89
```

Pass `--nvcc <path>` to use a system CUDA toolkit instead.

## Run

```bash
.venv/bin/python microbench/collect_profile.py --output profiles/<gpu>.json
```

Lock clocks first when you can. On WSL2 clocks can't be locked from Linux, so the collector records the observed clocks, sets `clock_policy.locked = false` and widens `rel_uncertainty` for clock sensitive fields.

The full steps, including the checks to run before you open a pull request, are in [contributing a profile](../docs/guides/contributing-a-profile.md).
