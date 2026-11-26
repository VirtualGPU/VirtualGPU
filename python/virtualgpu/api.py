"""High-level Python API for batch KernelIR extraction and performance prediction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from virtualgpu.frontend.compiler import MockTensor
from virtualgpu.frontend.triton_gemm import extract_gemm_kernel_ir


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _load_profile(profile: str | Path | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(profile, Mapping):
        data = dict(profile)
        return json.dumps(data), data
    if isinstance(profile, str) and profile.lstrip().startswith("{"):
        return profile, json.loads(profile)
    path = Path(profile)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        return text, json.loads(text)
    if path.parent == Path("."):
        # A bare name such as "rtx4070": a bundled profile.
        from virtualgpu.profile import profile_path

        text = profile_path(str(path)).read_text(encoding="utf-8")
        return text, json.loads(text)
    raise FileNotFoundError(f"Profile file not found: {profile}")


def normalize_gemm_config(entry: Mapping[str, Any], index: int = 0) -> tuple[str, dict[str, int]]:
    """Normalize a suite config entry or raw Triton config dict into (config_id, config_dict)."""
    config_id = str(entry.get("config_id", f"cfg_{index:03d}"))
    cfg: dict[str, int] = {}
    if "tile" in entry:
        bm, bn, bk = entry["tile"]
        cfg["BM"] = int(bm)
        cfg["BN"] = int(bn)
        cfg["BK"] = int(bk)
    if "group_m" in entry:
        cfg["GROUP_M"] = int(entry["group_m"])
    for key in (
        "BM",
        "BN",
        "BK",
        "GROUP_M",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "num_warps",
        "num_stages",
        "maxnreg",
    ):
        if key in entry and entry[key] is not None:
            cfg[key] = int(entry[key])
    return config_id, cfg


def _build_kernel_call_args(
    fn: Any,
    *,
    M: int,
    N: int,
    K: int,
    dtype: str,
) -> tuple[Any, ...]:
    """Build positional runtime arguments matching a GEMM kernel's signature."""
    elem_bytes = 2 if dtype in ("f16", "fp16", "bf16") else (4 if dtype in ("f32", "fp32") else 1)
    ptr_a = 0x10000
    ptr_b = ptr_a + _align_up(M * K * elem_bytes)
    ptr_c = ptr_b + _align_up(K * N * elem_bytes)
    a = MockTensor(dtype, ptr=ptr_a)
    b = MockTensor(dtype, ptr=ptr_b)
    c = MockTensor(dtype, ptr=ptr_c)

    # Support kernels with (A, B, C, M, N, K, ...) as well as (a_ptr, b_ptr, c_ptr, M, N, K, stride_...)
    arg_map: dict[str, Any] = {
        "A": a,
        "B": b,
        "C": c,
        "a_ptr": a,
        "b_ptr": b,
        "c_ptr": c,
        "M": M,
        "N": N,
        "K": K,
        "stride_am": K,
        "stride_ak": 1,
        "stride_bk": N,
        "stride_bn": 1,
        "stride_cm": N,
        "stride_cn": 1,
    }
    positional: list[Any] = []
    for name in fn.arg_names:
        if name in arg_map:
            positional.append(arg_map[name])
    return tuple(positional)


def predict_kernel_ir(
    ir: Mapping[str, Any] | str,
    profile: str | Path | Mapping[str, Any],
    *,
    case_id: str,
    config_id: str,
    l2_state: str = "cold",
    draws: int = 32,
    seed: int = 17,
    core_module: Any = None,
) -> dict[str, Any]:
    """Run virtualgpu._core.predict on a single KernelIR and format a validation record."""
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    if core_module is None:
        from virtualgpu import _core as core_module  # type: ignore[no-redef]

    profile_text, profile_data = _load_profile(profile)
    revision = profile_data["provenance"]["profile_revision"]
    ir_text = ir if isinstance(ir, str) else json.dumps(ir)

    raw = dict(
        core_module.predict(
            ir_text,
            profile_text,
            l2_state=l2_state,
            draws=draws,
            seed=seed,
        )
    )
    status = raw.pop("status")
    rec: dict[str, Any] = {
        "case_id": case_id,
        "config_id": config_id,
        "profile_revision": revision,
        "status": status,
        "assumptions": {
            "l2_state": raw.pop("l2_state", l2_state),
            "dispatch_policy": raw.pop("dispatch_policy", "round_robin"),
        },
    }
    if status == "ok":
        rec["nominal_latency_ns"] = float(raw.pop("time_ns"))
        raw_draws = raw.pop("draws", None)
        if draws > 0:
            if not isinstance(raw_draws, Mapping) or "0" not in raw_draws:
                raise RuntimeError("core.predict returned ok with draws > 0 but omitted draw map")
            rec["draws"] = {str(k): float(v) for k, v in raw_draws.items()}
        rec["diagnostics"] = raw
    else:
        rec["reason"] = raw.pop("reason", "unsupported")
    return rec


def extract_gemm_configs(
    fn: Any,
    *,
    shape: tuple[int, int, int],
    configs: Sequence[Mapping[str, Any]],
    target_arch: str,
    dtype: str = "f16",
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    ir_out_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Batch-extract KernelIR v0.1 dicts for a list of GEMM configs using AOT caching."""
    M, N, K = (int(shape[0]), int(shape[1]), int(shape[2]))
    out_path = Path(ir_out_dir) if ir_out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    call_args = _build_kernel_call_args(fn, M=M, N=N, K=K, dtype=dtype)
    results: dict[str, dict[str, Any]] = {}
    for idx, entry in enumerate(configs):
        config_id, cfg = normalize_gemm_config(entry, idx)
        try:
            ir = extract_gemm_kernel_ir(
                fn,
                M=M,
                N=N,
                K=K,
                config=cfg,
                target_arch=target_arch,
                dtype=dtype,
                call_args=call_args,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            results[config_id] = {"status": "ok", "ir": ir}
            if out_path is not None:
                (out_path / f"{config_id}.json").write_text(json.dumps(ir), encoding="utf-8")
        except Exception as exc:
            results[config_id] = {
                "status": "compile_error",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return results


def predict_gemm_configs(
    fn: Any,
    *,
    case_id: str,
    shape: tuple[int, int, int],
    configs: Sequence[Mapping[str, Any]],
    profile: str | Path | Mapping[str, Any],
    l2_state: str = "cold",
    dtype: str = "f16",
    draws: int = 32,
    seed: int = 17,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
    ir_out_dir: str | Path | None = None,
    core_module: Any = None,
) -> list[dict[str, Any]]:
    """Compile a batch of GEMM configs offline and predict them against a GPU profile.

    Returns a list of prediction records matching validation/score.py's schema.
    """
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    profile_text, profile_data = _load_profile(profile)
    target_arch = profile_data["identity"]["arch"]
    revision = profile_data["provenance"]["profile_revision"]

    extracted = extract_gemm_configs(
        fn,
        shape=shape,
        configs=configs,
        target_arch=target_arch,
        dtype=dtype,
        cache_dir=cache_dir,
        use_cache=use_cache,
        ir_out_dir=ir_out_dir,
    )

    records: list[dict[str, Any]] = []
    for config_id, item in extracted.items():
        if item["status"] != "ok":
            records.append(
                {
                    "case_id": case_id,
                    "config_id": config_id,
                    "profile_revision": revision,
                    "status": "unsupported",
                    "assumptions": {"l2_state": l2_state, "dispatch_policy": "round_robin"},
                    "reason": item["reason"],
                }
            )
            continue
        records.append(
            predict_kernel_ir(
                item["ir"],
                profile_text,
                case_id=case_id,
                config_id=config_id,
                l2_state=l2_state,
                draws=draws,
                seed=seed,
                core_module=core_module,
            )
        )
    return records


def predict_ir_dir(
    ir_dir: str | Path,
    *,
    case_id: str,
    profile: str | Path | Mapping[str, Any],
    l2_state: str = "cold",
    draws: int = 32,
    seed: int = 17,
    core_module: Any = None,
) -> list[dict[str, Any]]:
    """Predict every KernelIR JSON file in a directory and return validation records."""
    if draws < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    profile_text, _ = _load_profile(profile)
    records: list[dict[str, Any]] = []
    for ir_file in sorted(Path(ir_dir).glob("*.json")):
        records.append(
            predict_kernel_ir(
                ir_file.read_text(encoding="utf-8"),
                profile_text,
                case_id=case_id,
                config_id=ir_file.stem,
                l2_state=l2_state,
                draws=draws,
                seed=seed,
                core_module=core_module,
            )
        )
    return records
