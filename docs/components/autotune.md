# GPU-free autotuning

How VirtualGPU plugs into Triton's autotuner, online and offline. Code: `python/virtualgpu/autotune.py` and `python/virtualgpu/cli.py`. Autotuning takes GEMM shapes today.

## 1. Scope

VirtualGPU's first end-user workflow is **GPU-free (and GPU-assisted) Triton autotuning**: ranking a kernel's `@triton.autotune` candidate configs on CPU against a chosen GPU profile, explaining bottlenecks, marking indistinguishable pairs, and narrowing tens or hundreds of configs down to a small shortlist for optional hardware verification.

Design goals:
- **Zero monkey-patching of Triton internals.** Use only Triton 3.9's public `Autotuner` extension points and public attributes.
- **Minimal user diff.** Opt-in via a single wrapper/argument or environment variable (`VIRTUALGPU_PROFILE`).
- **Two first-class modes:** *online pruning* (on a machine with a GPU, narrowing to top-k before hardware benchmarking) and *offline ranking* (no GPU or targeting a GPU you do not own).
- **Honest failure and full provenance.** Never silently drop configs or misreport unmodeled kernels as ranked.

## 2. Triton 3.9 `Autotuner` Public Extension Points

Direct inspection of `triton==3.9.0` (`triton.runtime.autotuner.Autotuner`) establishes what is possible without monkey-patching:

| Extension point | Public signature / behavior in Triton 3.9 | Fit for VirtualGPU |
|---|---|---|
| `prune_configs_by={"early_config_prune": fn}` | Called inside `Autotuner.run()` *before* any hardware benchmarking, receiving `(self.configs, self.nargs, **kwargs)` and returning the surviving `List[Config]`. If length is `1`, `Autotuner.run()` skips `benchmark()` completely. | **Primary hook for online mode.** Sees all configs and real runtime arguments at once, allowing batched AOT compilation, shared-draw perturbation, and pairwise verdicts. |
| `prune_configs_by={"perf_model": fn, "top_k": k}` | Called once per config, returning a single `float`. | **Secondary / insufficient alone.** Scalar return discards perturbation draws, indistinguishable pairs, and bottleneck diagnostics. |
| `do_bench` | Receives a zero-arg `kernel_call` closure that invokes `self.fn.run(...)` (triggering real GPU compilation and launch). | **Unusable for GPU-free tuning.** Does not expose `config` or `args`, and executing `kernel_call` requires an active CUDA driver. |
| `knobs.autotuning.listener` | Callback fired *after* hardware benchmarking finishes. | **Too late** for pruning; useful only for logging. |
| `Autotuner` attributes (`fn`, `configs`, `arg_names`, `keys`) | Public attributes on the decorated kernel object. | **Primary hook for offline mode.** Allows reading the `JITFunction` and `triton.Config` list without calling `Autotuner.run()`. |

### Hard boundary without a GPU

Even if `early_config_prune` selects a single config on CPU (skipping `benchmark()`), the final lines of `Autotuner.run()` unconditionally call:
```python
ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
```
`JITFunction.run()` immediately invokes `driver.active.get_current_device()` and `driver.active.get_current_stream(device)`, which fails when no GPU/CUDA driver is present. Moreover, callers on a CPU-only machine cannot construct `torch.cuda.Tensor` inputs to pass into `kernel[grid](...)`.

Therefore, **online mode** and **offline mode** must have distinct entry points that share the same ranking engine and report format.

## 3. User-Facing API and Mode Switching

### Mode A: Online Pruning (GPU present, shortlist top-k for hardware verification)

Two zero-monkey-patch opt-ins:

1. **Drop-in decorator wrapper** (recommended for new or lightly edited code):
   ```python
   @virtualgpu.autotune(configs=CONFIGS, key=["M", "N", "K"], top_k=5)
   @triton.jit
   def matmul_kernel(...): ...
   ```
   - Reads `VIRTUALGPU_PROFILE` (e.g., `profiles/rtx4070.json`), plus optional env overrides `VIRTUALGPU_TOP_K` (default `5`), `VIRTUALGPU_L2_STATE` (`cold` | `warm`, default `cold`), `VIRTUALGPU_DRAWS` (default `32`), `VIRTUALGPU_SEED` (default `17`), and `VIRTUALGPU_REPORT_DIR` (optional JSON dump path).
   - **When `VIRTUALGPU_PROFILE` is unset and no `profile=` argument is passed**, `@virtualgpu.autotune` returns a plain `@triton.autotune(...)` with zero overhead.
   - **When enabled**, it wires `prune_configs_by={"early_config_prune": ...}` on top of `@triton.autotune`. If the user already supplied an `early_config_prune`, VirtualGPU runs the user's pruner first and ranks the surviving configs.
   - `top_k > 1` (default `5`): VirtualGPU narrows tens/hundreds of configs down to the top-k shortlist; Triton benchmarks only those `k` configs on real hardware to pick the final winner.
   - `top_k == 1`: Triton skips hardware benchmarking altogether and launches the VirtualGPU #1 config directly (useful for fast iteration when hardware tuning time matters more than the final few percent).
   - The full `AutotuneReport` for the latest tuning key is attached to the autotuner (e.g., `matmul_kernel.vgpu_reports[key]`) and printed concisely when `VIRTUALGPU_VERBOSE=1` or `triton.knobs.autotuning.print` is enabled.

2. **Composable pruner for untouched `@triton.autotune`** (1-line change inside existing decorator):
   ```python
   @triton.autotune(
       configs=CONFIGS,
       key=["M", "N", "K"],
       prune_configs_by=virtualgpu.prune_configs_by(profile="profiles/rtx4070.json", top_k=5),
   )
   ```

### Mode B: Offline Ranking (No GPU, or targeting a GPU you do not own)

Works on **any** existing `@triton.autotune`-decorated kernel without modifying the kernel or touching a CUDA driver:

```python
report = virtualgpu.autotune_offline(
    matmul_kernel,  # existing @triton.autotune object (or JITFunction + configs)
    shape=(4096, 4096, 4096),  # or explicit MockTensor/int call_args
    profile="profiles/gb10.json",
    dtype="f16",
    l2_state="cold",
    draws=32,
    seed=17,
    top_k=5,
)
best_cfg = report.best_config
```

And the matching CLI subcommand:
```bash
virtualgpu autotune \
  --kernel my_module:matmul_kernel \
  --shape 4096 4096 4096 \
  --profile profiles/gb10.json \
  --cache cold --draws 32 --top-k 5 --out report.json
```

`autotune_offline` reads `autotuner.fn` and `autotuner.configs`, builds `MockTensor` arguments (16B-aligned by default, matching typical PyTorch allocations), runs the existing CPU-only AOT frontend + `vgpu-core` predictor with shared `(draws, seed)`, and returns `AutotuneReport`.

## 4. Return Contract (`AutotuneReport`)

Both online and offline modes produce the same structured, JSON-serializable `AutotuneReport`:

| Field | Contents |
|---|---|
| `status` | `"ok"` \| `"unsupported"` (if no candidate config could be modeled) |
| `best_config` | Top-ranked `triton.Config` (and serialized dict) by `nominal_latency_ns` |
| `shortlist` | Ordered list of `top_k` candidate config IDs recommended for real-hardware verification, plus human-readable advice string (e.g., `"Verify top 5 candidates on hardware; 3 configs are indistinguishable from #1 within 5%."`) |
| `ranking` | Per-config entries sorted by `nominal_latency_ns` (unsupported configs last): `config_id`, `config`, `status` (`ok` \| `unsupported`), `reason` (if unsupported), `nominal_latency_ns`, `draws` (`{"0": ..., "1": ..., ...}`), `primary_bottleneck` (dominant resource from `bound_fraction` and its share), `bound_fraction`, `occupancy` (`ctas_per_sm`, `limiter`, `waves`, `tail_efficiency`), `resources` (`registers_per_thread`, `static_smem_bytes`, `spill_stores_bytes`, `spill_loads_bytes`), `flags`, `diagnostics`, `unsupported_features` |
| `indistinguishable_pairs` | Pairs `(config_a, config_b)` where `_core.pair_verdict(draws_a, draws_b) == "indistinguishable"` (computed across all pairs in `shortlist` and between `#1` and every `ok` config, so users see the full tie band around the winner) |
| `provenance` | `profile_revision`, `profile_sha256`, `target_arch`, `l2_state`, `draws`, `seed`, `virtualgpu_version`, `core_version`, `triton_version`, `assembler`, `assembler_version`, `artifact_hash_version`, `kernel_source_sha256`, `timestamp_utc` |

## 5. Caching, Reproducibility, and Fallback Policy

### Two-layer caching
1. **AOT compilation cache** (already implemented in `virtualgpu.frontend.compiler`): keyed by `(source_sha256, signature, constexprs, attrs, codegen options, target_arch, warp_size, triton_version, assembler_version, artifact_hash_version)`. Avoids recompiling unchanged configs across shapes, profiles of the same SM, or model updates.
2. **Autotune ranking cache** (new, optional, default `~/.cache/virtualgpu/autotune/`): keyed by `(aot_cache_keys, grid, profile_revision, profile_sha256, l2_state, draws, seed, virtualgpu_version, core_version)`. Invalidates automatically whenever the GPU profile, L2 state, seed, or simulator version changes. Controlled by `VIRTUALGPU_CACHE_DIR` / `--no-cache`.

### Reproducibility
- Every `AutotuneReport` embeds full `provenance` (profile revision + SHA-256, toolchain versions, `artifact_hash_version`, `draws`, `seed`).
- Shared `(profile, draws, seed)` guarantees that all configs in one autotune run are evaluated against the exact same perturbed machine draws, matching `vgpu-core` and `validation/score.py`.

### Failure and fallback rules (fail-safe, never silent)
1. **Invalid user configuration** (missing `VIRTUALGPU_PROFILE` file, malformed profile JSON, invalid `top_k < 1` or `draws < 0`): raise immediately (`FileNotFoundError` / `ValueError`). Never swallow setup errors.
2. **Per-config failure** (single config hits `PTXASError`, exceeds SMEM/registers, or emits `unsupported_features`): record `status="unsupported"` with `reason` in `ranking` and exclude from `shortlist`; continue ranking remaining configs.
3. **Total modeling failure** (zero configs return `status="ok"`, e.g., persistent kernel with atomic work-stealing or unmodeled op):
   - *Online mode*: emit a visible `UserWarning` explaining why VirtualGPU could not rank the kernel, and return the full unpruned `configs` list to Triton so native hardware autotuning proceeds (note: Triton 3.9 raises `AutotunerError` if `early_config_prune` returns an empty list). If `strict=True`, raise instead of falling back.
   - *Offline mode*: return `AutotuneReport(status="unsupported", ...)` with all per-config reasons and `unsupported_features` (or raise if `strict=True`).
