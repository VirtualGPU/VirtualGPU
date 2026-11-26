"""GPU-free and GPU-assisted Triton autotuning powered by VirtualGPU."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from virtualgpu.api import _build_kernel_call_args, _load_profile, predict_kernel_ir
from virtualgpu.frontend.compiler import ARTIFACT_HASH_VERSION
from virtualgpu.frontend.triton_gemm import extract_gemm_kernel_ir_batch

DEFAULT_ONLINE_WARMUP = 50
DEFAULT_ONLINE_REP = 200


def _unwrap_kernel_and_configs(
    kernel: Any,
    configs: Sequence[Any] | None,
) -> tuple[Any, list[Any]]:
    """Extract the underlying JITFunction and candidate configs from an Autotuner or JITFunction."""
    from triton.runtime.autotuner import Autotuner

    if isinstance(kernel, Autotuner):
        base_fn = kernel.fn
        while isinstance(base_fn, Autotuner):
            base_fn = base_fn.fn
        resolved_configs = list(configs) if configs is not None else list(kernel.configs)
        return base_fn, resolved_configs

    if configs is None:
        raise ValueError(
            "configs must be provided when kernel is not wrapped with @triton.autotune"
        )
    return kernel, list(configs)


def _config_to_dict(cfg: Any, index: int) -> tuple[str, dict[str, int]]:
    """Convert a triton.Config or mapping into (config_id, flat_int_dict)."""
    if isinstance(cfg, Mapping):
        cid = str(cfg.get("config_id", f"cfg_{index:03d}"))
        flat: dict[str, int] = {}
        if "tile" in cfg:
            bm, bn, bk = cfg["tile"]
            flat["BM"], flat["BN"], flat["BK"] = int(bm), int(bn), int(bk)
        if "group_m" in cfg:
            flat["GROUP_M"] = int(cfg["group_m"])
        for k, v in cfg.items():
            if k in ("config_id", "tile", "group_m") or v is None:
                continue
            if isinstance(v, (int, bool)):
                flat[str(k)] = int(v)
        return cid, flat

    # triton.Config instance
    raw = cfg.all_kwargs()
    flat = {}
    for k, v in raw.items():
        if v is not None and isinstance(v, (int, bool)):
            flat[str(k)] = int(v)
    bm = flat.get("BLOCK_SIZE_M", flat.get("BLOCK_M", flat.get("BM")))
    bn = flat.get("BLOCK_SIZE_N", flat.get("BLOCK_N", flat.get("BN")))
    bk = flat.get("BLOCK_SIZE_K", flat.get("BLOCK_K", flat.get("BK")))
    nw = flat.get("num_warps")
    ns = flat.get("num_stages")
    if bm and bn and bk and nw and ns:
        cid = f"cfg_{index:03d}_m{bm}n{bn}k{bk}_w{nw}s{ns}"
    else:
        cid = f"cfg_{index:03d}"
    return cid, flat


def _primary_bottleneck(bound_fraction: Mapping[str, float] | None) -> dict[str, Any] | None:
    if not bound_fraction:
        return None
    resource, share = max(bound_fraction.items(), key=lambda kv: float(kv[1]))
    return {"resource": str(resource), "fraction": float(share)}


def _draws_in_order(draws_map: Mapping[str, float]) -> list[float]:
    ordered_keys = sorted(draws_map.keys(), key=lambda k: int(k))
    return [float(draws_map[k]) for k in ordered_keys]


@dataclass
class AutotuneReport:
    """Structured ranking, bottleneck, tie, and provenance report from VirtualGPU autotuning."""

    status: str  # "ok" | "unsupported"
    best_config: Any
    best_config_id: str | None
    shortlist: list[str]
    advice: str
    ranking: list[dict[str, Any]]
    indistinguishable_pairs: list[tuple[str, str]]
    provenance: dict[str, Any]
    _config_objects: dict[str, Any] = field(default_factory=dict, repr=False)

    def shortlist_configs(self) -> list[Any]:
        """Return the original config objects corresponding to `self.shortlist`."""
        return [self._config_objects[cid] for cid in self.shortlist if cid in self._config_objects]

    def to_dict(self) -> dict[str, Any]:
        best_dict = None
        if self.best_config_id is not None:
            for item in self.ranking:
                if item["config_id"] == self.best_config_id:
                    best_dict = item["config"]
                    break
        return {
            "status": self.status,
            "best_config_id": self.best_config_id,
            "best_config": best_dict,
            "shortlist": list(self.shortlist),
            "advice": self.advice,
            "ranking": list(self.ranking),
            "indistinguishable_pairs": [[a, b] for a, b in self.indistinguishable_pairs],
            "provenance": dict(self.provenance),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        ok_count = sum(1 for r in self.ranking if r["status"] == "ok")
        total = len(self.ranking)
        arch = self.provenance.get("target_arch", "unknown")
        rev = self.provenance.get("profile_revision", "unknown")
        lines = [
            f"VirtualGPU autotune ({arch}, profile {rev}): {ok_count}/{total} configs ranked.",
            f"Advice: {self.advice}",
        ]
        for idx, item in enumerate(self.ranking[: len(self.shortlist)], start=1):
            bn = item.get("primary_bottleneck") or {}
            bn_str = (
                f"{bn.get('resource', '?')} ({bn.get('fraction', 0.0) * 100:.0f}%)" if bn else "n/a"
            )
            lat_us = item["nominal_latency_ns"] / 1e3
            lines.append(f"  #{idx} {item['config_id']}: {lat_us:.1f} us | bottleneck={bn_str}")
        if self.indistinguishable_pairs:
            pair_str = ", ".join(f"({a}, {b})" for a, b in self.indistinguishable_pairs[:8])
            suffix = " ..." if len(self.indistinguishable_pairs) > 8 else ""
            lines.append(f"Indistinguishable pairs (5% band): {pair_str}{suffix}")
        return "\n".join(lines)


def _rank_configs(
    fn: Any,
    configs: Sequence[Any],
    *,
    M: int,
    N: int,
    K: int,
    call_args: tuple[Any, ...],
    profile_text: str,
    profile_data: Mapping[str, Any],
    dtype: str,
    l2_state: str,
    draws: int,
    seed: int,
    top_k: int,
    cache_dir: str | Path | None,
    use_cache: bool,
    core_module: Any,
    hw_bench_params: dict[str, int] | None = None,
) -> AutotuneReport:
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    if core_module is None:
        from virtualgpu import _core as core_module  # type: ignore[no-redef]

    target_arch = profile_data["identity"]["arch"]
    profile_revision = profile_data["provenance"]["profile_revision"]
    profile_sha256 = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()
    src_text = getattr(fn, "src", None) or getattr(fn, "__name__", repr(fn))
    kernel_source_sha256 = hashlib.sha256(src_text.encode("utf-8")).hexdigest()

    config_objects: dict[str, Any] = {}
    cids: list[str] = []
    flat_configs: list[dict[str, int]] = []
    for idx, orig_cfg in enumerate(configs):
        cid, flat_cfg = _config_to_dict(orig_cfg, idx)
        config_objects[cid] = orig_cfg
        cids.append(cid)
        flat_configs.append(flat_cfg)

    extracted_batch = extract_gemm_kernel_ir_batch(
        fn,
        M=M,
        N=N,
        K=K,
        configs=flat_configs,
        target_arch=target_arch,
        dtype=dtype,
        call_args=call_args,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    ok_entries: list[dict[str, Any]] = []
    bad_entries: list[dict[str, Any]] = []
    toolchain_sample: dict[str, Any] = {}

    ready_items: list[tuple[str, dict[str, int], dict[str, Any], str]] = []
    for cid, flat_cfg, item in zip(cids, flat_configs, extracted_batch):
        if isinstance(item, Exception):
            bad_entries.append(
                {
                    "config_id": cid,
                    "config": flat_cfg,
                    "status": "unsupported",
                    "reason": f"{type(item).__name__}: {item}",
                }
            )
            continue
        ir, ir_json = item
        if not toolchain_sample:
            toolchain_sample = dict(ir["metadata"].get("toolchain", {}))
        ready_items.append((cid, flat_cfg, ir, ir_json))

    # Use core.predict_batch when available so profile parsing and draw sampling happen once.
    raw_preds: list[dict[str, Any]] = []
    if ready_items:
        if hasattr(core_module, "predict_batch"):
            ir_jsons = [item[3] for item in ready_items]
            batch_raw = core_module.predict_batch(
                ir_jsons,
                profile_text,
                l2_state=l2_state,
                draws=draws,
                seed=seed,
            )
            for raw_item in batch_raw:
                d = dict(raw_item)
                status = d.pop("status")
                rec: dict[str, Any] = {
                    "status": status,
                    "assumptions": {
                        "l2_state": d.pop("l2_state", l2_state),
                        "dispatch_policy": d.pop("dispatch_policy", "round_robin"),
                    },
                }
                if status == "ok":
                    rec["nominal_latency_ns"] = float(d.pop("time_ns"))
                    raw_draws = d.pop("draws", None)
                    if draws > 0 and isinstance(raw_draws, Mapping) and "0" in raw_draws:
                        rec["draws"] = {str(k): float(v) for k, v in raw_draws.items()}
                    rec["diagnostics"] = d
                else:
                    rec["reason"] = d.pop("reason", "unsupported")
                raw_preds.append(rec)
        else:
            for cid, _flat_cfg, _ir, ir_json in ready_items:
                raw_preds.append(
                    predict_kernel_ir(
                        ir_json,
                        profile_text,
                        case_id="autotune",
                        config_id=cid,
                        l2_state=l2_state,
                        draws=draws,
                        seed=seed,
                        core_module=core_module,
                    )
                )

    for (cid, flat_cfg, ir, _ir_json), pred in zip(ready_items, raw_preds):
        if pred["status"] != "ok":
            bad_entries.append(
                {
                    "config_id": cid,
                    "config": flat_cfg,
                    "status": "unsupported",
                    "reason": pred.get("reason", "unsupported"),
                    "resources": ir["resources"],
                    "diagnostics": ir["diagnostics"],
                    "unsupported_features": ir["unsupported_features"],
                }
            )
            continue

        diag = pred.get("diagnostics", {})
        entry: dict[str, Any] = {
            "config_id": cid,
            "config": flat_cfg,
            "status": "ok",
            "nominal_latency_ns": pred["nominal_latency_ns"],
            "primary_bottleneck": _primary_bottleneck(diag.get("bound_fraction")),
            "bound_fraction": diag.get("bound_fraction", {}),
            "occupancy": {
                "ctas_per_sm": diag.get("ctas_per_sm"),
                "limiter": diag.get("occupancy_limiter"),
                "waves": diag.get("waves"),
                "tail_efficiency": diag.get("tail_efficiency"),
            },
            "resources": ir["resources"],
            "flags": diag.get("flags", []),
            "diagnostics": diag.get("diagnostics", ir["diagnostics"]),
            "unsupported_features": ir["unsupported_features"],
        }
        if "draws" in pred:
            entry["draws"] = pred["draws"]
        ok_entries.append(entry)

    ok_entries.sort(key=lambda e: (e["nominal_latency_ns"], e["config_id"]))
    ranking = ok_entries + bad_entries

    shortlist_entries = ok_entries[:top_k]
    shortlist = [e["config_id"] for e in shortlist_entries]
    best_id = shortlist[0] if shortlist else None
    best_cfg = config_objects[best_id] if best_id is not None else None

    # Compute pairwise indistinguishable pairs via core.pair_verdict across
    # shortlist pairs and between #1 and every remaining ok config.
    indistinguishable: list[tuple[str, str]] = []
    if draws > 0 and len(ok_entries) >= 2 and hasattr(core_module, "pair_verdict"):
        checked: set[tuple[str, str]] = set()
        candidate_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = list(
            itertools.combinations(shortlist_entries, 2)
        )
        top1 = ok_entries[0]
        for other in ok_entries[top_k:]:
            candidate_pairs.append((top1, other))
        for ea, eb in candidate_pairs:
            key = (ea["config_id"], eb["config_id"])
            if key in checked or "draws" not in ea or "draws" not in eb:
                continue
            checked.add(key)
            verdict = core_module.pair_verdict(
                _draws_in_order(ea["draws"]),
                _draws_in_order(eb["draws"]),
            )
            if verdict == "indistinguishable":
                indistinguishable.append(key)

    tied_with_best = sum(1 for a, _b in indistinguishable if a == best_id)
    if not ok_entries:
        status = "unsupported"
        advice = "No candidate configs could be modeled; inspect per-config reasons and benchmark on hardware."
    elif top_k == 1:
        status = "ok"
        advice = (
            f"Unverified on hardware (top_k=1 skips hardware benchmarking). "
            f"Selected {best_id}; {tied_with_best} other config(s) are indistinguishable from #1."
        )
    else:
        status = "ok"
        advice = (
            f"Recommend verifying the top {len(shortlist)} shortlisted config(s) on real hardware "
            f"({tied_with_best} other config(s) are indistinguishable from #1 within 5%)."
        )

    from virtualgpu import __version__ as vgpu_version

    build = core_module.build_info() if hasattr(core_module, "build_info") else None
    provenance = {
        "profile_revision": profile_revision,
        "profile_sha256": profile_sha256,
        "target_arch": target_arch,
        "l2_state": l2_state,
        "draws": draws,
        "seed": seed,
        "top_k": top_k,
        "virtualgpu_version": vgpu_version,
        "core_version": getattr(core_module, "__version__", "unknown"),
        "build": build,
        "triton_version": toolchain_sample.get("compiler_version"),
        "assembler": toolchain_sample.get("assembler"),
        "assembler_version": toolchain_sample.get("assembler_version"),
        "artifact_hash_version": ARTIFACT_HASH_VERSION,
        "kernel_source_sha256": kernel_source_sha256,
        "hardware_bench": hw_bench_params,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    return AutotuneReport(
        status=status,
        best_config=best_cfg,
        best_config_id=best_id,
        shortlist=shortlist,
        advice=advice,
        ranking=ranking,
        indistinguishable_pairs=indistinguishable,
        provenance=provenance,
        _config_objects=config_objects,
    )


def autotune_offline(
    kernel: Any,
    *,
    shape: tuple[int, int, int],
    profile: str | Path | Mapping[str, Any],
    configs: Sequence[Any] | None = None,
    call_args: tuple[Any, ...] | None = None,
    dtype: str = "f16",
    l2_state: str = "cold",
    draws: int = 32,
    seed: int = 17,
    top_k: int = 5,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    strict: bool = False,
    core_module: Any = None,
) -> AutotuneReport:
    """Rank a Triton kernel's autotune configs completely offline on CPU (no GPU/driver needed).

    Accepts either a kernel already decorated with `@triton.autotune` (reads its
    public `fn` and `configs` attributes) or a raw `@triton.jit` function plus
    an explicit `configs` sequence.
    """
    profile_text, profile_data = _load_profile(profile)
    fn, resolved_configs = _unwrap_kernel_and_configs(kernel, configs)
    if not resolved_configs:
        raise ValueError("At least one candidate config is required")

    M, N, K = int(shape[0]), int(shape[1]), int(shape[2])
    resolved_args = (
        call_args
        if call_args is not None
        else _build_kernel_call_args(fn, M=M, N=N, K=K, dtype=dtype)
    )

    report = _rank_configs(
        fn,
        resolved_configs,
        M=M,
        N=N,
        K=K,
        call_args=resolved_args,
        profile_text=profile_text,
        profile_data=profile_data,
        dtype=dtype,
        l2_state=l2_state,
        draws=draws,
        seed=seed,
        top_k=top_k,
        cache_dir=cache_dir,
        use_cache=use_cache,
        core_module=core_module,
    )
    if strict and report.status != "ok":
        raise RuntimeError(f"VirtualGPU could not model any candidate config: {report.advice}")
    return report


def _infer_gemm_shape_and_dtype(nargs: Mapping[str, Any]) -> tuple[tuple[int, int, int], str]:
    """Infer (M, N, K) and canonical dtype from Triton Autotuner nargs."""
    m = nargs.get("M", nargs.get("rows"))
    n = nargs.get("N", nargs.get("cols"))
    k = nargs.get("K", nargs.get("inner"))
    if m is None or n is None or k is None:
        raise ValueError(
            "Could not infer GEMM shape (M, N, K) from autotuner arguments; "
            f"available keys: {sorted(nargs.keys())}"
        )
    dtype = "f16"
    for val in nargs.values():
        if hasattr(val, "dtype"):
            s = str(val.dtype).replace("torch.", "")
            dtype = {"float16": "f16", "bfloat16": "bf16", "float32": "f32"}.get(s, s)
            break
    return (int(m), int(n), int(k)), dtype


def _resolve_env_options(
    *,
    profile: str | Path | Mapping[str, Any] | None,
    top_k: int | None,
    l2_state: str | None,
    draws: int | None,
    seed: int | None,
) -> tuple[str | Path | Mapping[str, Any] | None, int, str, int, int]:
    resolved_profile = profile if profile is not None else os.environ.get("VIRTUALGPU_PROFILE")
    resolved_top_k = top_k if top_k is not None else int(os.environ.get("VIRTUALGPU_TOP_K", "5"))
    resolved_l2 = (
        l2_state if l2_state is not None else os.environ.get("VIRTUALGPU_L2_STATE", "cold")
    )
    resolved_draws = draws if draws is not None else int(os.environ.get("VIRTUALGPU_DRAWS", "32"))
    resolved_seed = seed if seed is not None else int(os.environ.get("VIRTUALGPU_SEED", "17"))
    return resolved_profile, resolved_top_k, resolved_l2, resolved_draws, resolved_seed


def prune_configs_by(
    fn: Any,
    *,
    profile: str | Path | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    l2_state: str | None = None,
    draws: int | None = None,
    seed: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    strict: bool = False,
    user_prune: Callable[..., list[Any]] | None = None,
    report_sink: list[AutotuneReport] | None = None,
    core_module: Any = None,
    hw_bench_params: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a Triton `prune_configs_by` dict backed by VirtualGPU ranking.

    Uses only Triton 3.9's public `early_config_prune(configs, nargs, **kwargs)`
    hook. Defaults to `top_k=5` so real hardware verifies the shortlist; `top_k=1`
    must be requested explicitly and is marked unverified in the report.
    """
    resolved_profile, resolved_top_k, resolved_l2, resolved_draws, resolved_seed = (
        _resolve_env_options(
            profile=profile, top_k=top_k, l2_state=l2_state, draws=draws, seed=seed
        )
    )
    if resolved_profile is None:
        raise ValueError("profile must be provided or VIRTUALGPU_PROFILE must be set")
    # Validate profile eagerly so bad configuration fails loudly instead of inside a launch.
    profile_text, profile_data = _load_profile(resolved_profile)
    if resolved_top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {resolved_top_k}")

    def _early_config_prune(
        configs: list[Any], nargs: Mapping[str, Any], **kwargs: Any
    ) -> list[Any]:
        candidates = list(configs)
        if user_prune is not None:
            candidates = user_prune(candidates, nargs, **kwargs)
        if not candidates:
            return candidates

        try:
            shape, dtype = _infer_gemm_shape_and_dtype(nargs)
            call_args = tuple(nargs[name] for name in fn.arg_names if name in nargs)
            report = _rank_configs(
                fn,
                candidates,
                M=shape[0],
                N=shape[1],
                K=shape[2],
                call_args=call_args,
                profile_text=profile_text,
                profile_data=profile_data,
                dtype=dtype,
                l2_state=resolved_l2,
                draws=resolved_draws,
                seed=resolved_seed,
                top_k=resolved_top_k,
                cache_dir=cache_dir,
                use_cache=use_cache,
                core_module=core_module,
                hw_bench_params=hw_bench_params,
            )
        except Exception as exc:
            if strict:
                raise
            warnings.warn(
                f"VirtualGPU pruning failed ({type(exc).__name__}: {exc}); "
                "falling back to all candidate configs for hardware autotuning.",
                UserWarning,
                stacklevel=2,
            )
            return candidates

        if report_sink is not None:
            report_sink.append(report)

        report_dir = os.environ.get("VIRTUALGPU_REPORT_DIR")
        if report_dir:
            out_dir = Path(report_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{getattr(fn, '__name__', 'kernel')}_{shape[0]}x{shape[1]}x{shape[2]}.json"
            (out_dir / fname).write_text(report.to_json(indent=2), encoding="utf-8")

        if os.environ.get("VIRTUALGPU_VERBOSE") == "1":
            print(report.summary())

        if report.status != "ok" or not report.shortlist:
            if strict:
                raise RuntimeError(
                    f"VirtualGPU could not model any candidate config: {report.advice}"
                )
            warnings.warn(
                f"VirtualGPU could not model any candidate config ({report.advice}); "
                "falling back to all candidate configs for hardware autotuning.",
                UserWarning,
                stacklevel=2,
            )
            return candidates

        return report.shortlist_configs()

    return {"early_config_prune": _early_config_prune}


def autotune(
    configs: list[Any],
    key: list[str],
    *,
    profile: str | Path | Mapping[str, Any] | None = None,
    top_k: int | None = None,
    l2_state: str | None = None,
    draws: int | None = None,
    seed: int | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    strict: bool = False,
    prune_configs_by: dict[str, Any] | None = None,
    core_module: Any = None,
    **triton_kwargs: Any,
) -> Callable[[Any], Any]:
    """Thin drop-in wrapper over `@triton.autotune` that activates VirtualGPU pruning.

    When neither `profile` nor `VIRTUALGPU_PROFILE` is set, returns a plain
    `@triton.autotune` with zero VirtualGPU overhead.
    """
    import triton

    resolved_profile = profile if profile is not None else os.environ.get("VIRTUALGPU_PROFILE")

    def _decorator(fn: Any) -> Any:
        if resolved_profile is None:
            return triton.autotune(
                configs=configs,
                key=key,
                prune_configs_by=prune_configs_by,
                **triton_kwargs,
            )(fn)

        user_prune = None
        merged_prune: dict[str, Any] = dict(prune_configs_by) if prune_configs_by else {}
        if "early_config_prune" in merged_prune:
            user_prune = merged_prune.pop("early_config_prune")

        # With only top-k (default 5) candidates left for hardware benchmarking,
        # increase warmup/rep via Triton 3.9's public `do_bench` hook (warmup/rep
        # kwargs are deprecated in Triton 3.9) to suppress clock jitter.
        effective_kwargs = dict(triton_kwargs)
        hw_bench_params: dict[str, int] | None = None
        if (
            "do_bench" not in effective_kwargs
            and "warmup" not in effective_kwargs
            and "rep" not in effective_kwargs
            and not effective_kwargs.get("use_cuda_graph", False)
        ):
            hw_bench_params = {
                "warmup": DEFAULT_ONLINE_WARMUP,
                "rep": DEFAULT_ONLINE_REP,
            }
            effective_kwargs["do_bench"] = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=DEFAULT_ONLINE_WARMUP,
                rep=DEFAULT_ONLINE_REP,
                quantiles=quantiles,
            )
        elif "warmup" in effective_kwargs or "rep" in effective_kwargs:
            hw_bench_params = {
                "warmup": int(effective_kwargs.get("warmup", 25)),
                "rep": int(effective_kwargs.get("rep", 100)),
            }

        reports: list[AutotuneReport] = []
        vgpu_prune = _make_prune_dict(
            fn,
            profile=resolved_profile,
            top_k=top_k,
            l2_state=l2_state,
            draws=draws,
            seed=seed,
            cache_dir=cache_dir,
            use_cache=use_cache,
            strict=strict,
            user_prune=user_prune,
            report_sink=reports,
            core_module=core_module,
            hw_bench_params=hw_bench_params,
        )
        merged_prune.update(vgpu_prune)
        tuned = triton.autotune(
            configs=configs,
            key=key,
            prune_configs_by=merged_prune,
            **effective_kwargs,
        )(fn)
        tuned.vgpu_reports = reports
        return tuned

    return _decorator


_make_prune_dict = prune_configs_by
