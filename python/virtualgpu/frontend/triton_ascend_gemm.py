"""Triton-Ascend GEMM extractor emitting KernelIR v0.1 from post-pass HIVM MLIR."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from virtualgpu.frontend.ascend_compiler import (
    AscendCompiledArtifacts,
    _mock_signature_for_fn,
    canonical_ascend_arch,
    compile_triton_ascend_aot,
)
from virtualgpu.frontend.compiler import (
    ARTIFACT_HASH_VERSION,
    MockTensor,
    _canonical_json,
    default_aot_cache_dir,
)
from virtualgpu.frontend.hivm import HivmSummary, parse_hivm_summary
from virtualgpu.frontend.scalar_eval import evaluate_ttgir_cta_offsets
from virtualgpu.frontend.ttgir import dtype_bits

ASCEND_KERNEL_IR_CACHE_VERSION = "5"


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _build_ascend_gemm_call_args(
    fn: Any,
    *,
    dtype: str,
    M: int,
    N: int,
    K: int,
    base_a: int,
    base_b: int,
    base_c: int,
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
    strides_c: tuple[int, int] | None = None,
) -> tuple[Any, ...]:
    sa = strides_a if strides_a is not None else (K, 1)
    sb = strides_b if strides_b is not None else (N, 1)
    sc = strides_c if strides_c is not None else (N, 1)
    value_by_name: dict[str, Any] = {
        "A": MockTensor(dtype, ptr=base_a),
        "B": MockTensor(dtype, ptr=base_b),
        "C": MockTensor(dtype, ptr=base_c),
        "a_ptr": MockTensor(dtype, ptr=base_a),
        "b_ptr": MockTensor(dtype, ptr=base_b),
        "c_ptr": MockTensor(dtype, ptr=base_c),
        "M": M,
        "N": N,
        "K": K,
        "stride_am": sa[0],
        "stride_ak": sa[1],
        "stride_bk": sb[0],
        "stride_bn": sb[1],
        "stride_cm": sc[0],
        "stride_cn": sc[1],
    }
    return tuple(value_by_name[name] for name in fn.arg_names if name in value_by_name)


def _recover_logical_tile_coords(
    ttir: str,
    logical_grid: tuple[int, int, int],
    *,
    fn: Any | None,
    call_args: tuple[Any, ...],
    signature: Mapping[str, str],
    M: int,
    N: int,
    block_m: int,
    block_n: int,
    stride_am: int,
    stride_bk: int,
    group_size_m: int | None = None,
) -> tuple[list[tuple[int, int]], bool, bool]:
    """Recover logical (pid_m, pid_n) tile order from TTIR scalar evaluation or swizzle fallback."""
    total_tiles = logical_grid[0] * logical_grid[1] * logical_grid[2]
    if fn is not None and ttir:
        param_int_values: list[int | None] = []
        runtime_int_args: dict[str, int] = {}
        pos_iter = iter(call_args)
        for arg_name in fn.arg_names:
            if arg_name not in signature:
                continue
            val = next(pos_iter, None)
            if isinstance(val, int) and not isinstance(val, bool):
                param_int_values.append(val)
                runtime_int_args[arg_name] = val
            else:
                param_int_values.append(None)
        ev = evaluate_ttgir_cta_offsets(
            ttgir=ttir,
            grid=logical_grid,
            runtime_int_args=runtime_int_args,
            param_int_values=param_int_values,
        )
        offs_a = ev.ptr_elem_offsets.get(0)
        offs_b = ev.ptr_elem_offsets.get(1)
        if (
            offs_a is not None
            and offs_b is not None
            and len(offs_a) == total_tiles
            and len(offs_b) == total_tiles
        ):
            coords = [
                (
                    (oa // stride_am if stride_am > 0 else 0) // block_m,
                    (ob % stride_bk if stride_bk > 0 else ob) // block_n,
                )
                for oa, ob in zip(offs_a, offs_b)
            ]
            return coords, False, ev.data_dependent

    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)
    if group_size_m is not None and group_size_m > 1:
        swizzled: list[tuple[int, int]] = []
        num_pid_in_group = group_size_m * num_pid_n
        for pid in range(num_pid_m * num_pid_n):
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * group_size_m
            gs = min(num_pid_m - first_pid_m, group_size_m)
            pm = first_pid_m + ((pid % num_pid_in_group) % gs)
            pn = (pid % num_pid_in_group) // gs
            swizzled.append((pm, pn))
        return swizzled, False, False

    linear = [(pm, pn) for pm in range(num_pid_m) for pn in range(num_pid_n)]
    return linear, True, False


def _reorder_for_autoblockify(
    logical_coords: list[tuple[int, int]],
    stride: int,
) -> tuple[int, list[int], list[tuple[int, int]]]:
    """Fold logical tiles into persistent per-AI-core outer iterations matching autoblockify.

    BiShengIR's `--enable-auto-blockify-loop` emits:
      `for pid = core_id; pid < total_tiles; pid += stride`
    Returning `(physical_grid, trip_counts_per_core, ordered_coords)` ordered densely by
    `(core_id, outer_idx)` as required by `KernelIR.access_map`.
    """
    total = len(logical_coords)
    phys_grid = min(total, max(1, stride))
    trip_counts: list[int] = []
    ordered: list[tuple[int, int]] = []
    for core_id in range(phys_grid):
        core_tiles = logical_coords[core_id::phys_grid]
        trip_counts.append(len(core_tiles))
        ordered.extend(core_tiles)
    return phys_grid, trip_counts, ordered


def assemble_ascend_gemm_kernel_ir(
    artifacts: AscendCompiledArtifacts,
    *,
    M: int,
    N: int,
    K: int,
    config: Mapping[str, Any],
    dtype: str = "f16",
    acc_dtype: str = "f32",
    fn: Any | None = None,
    call_args: tuple[Any, ...] = (),
    signature: Mapping[str, str] = (),
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Assemble a KernelIR v0.1 dict from Triton-Ascend compiled HIVM artifacts."""
    summary: HivmSummary = parse_hivm_summary(
        artifacts.hivm_ir,
        requested_multibuffer=artifacts.requested_multibuffer,
        requested_num_stages=artifacts.requested_num_stages,
        fallback_workspace_bytes=artifacts.workspace_size,
    )

    # Prefer tile dimensions directly from post-pass HIVM mmadL1 over config hints.
    cfg_bm = int(config.get("BLOCK_SIZE_M", config.get("BLOCK_M", config.get("BM", 128))))
    cfg_bn = int(config.get("BLOCK_SIZE_N", config.get("BLOCK_N", config.get("BN", 128))))
    cfg_bk = int(config.get("BLOCK_SIZE_K", config.get("BLOCK_K", config.get("BK", 64))))
    if summary.mma_ops:
        block_m, block_n, block_k = summary.mma_ops[0].tile_mnk
    else:
        block_m, block_n, block_k = cfg_bm, cfg_bn, cfg_bk

    elem_bits = dtype_bits(dtype)
    elem_bytes = elem_bits // 8
    size_a = M * K * elem_bytes
    size_b = K * N * elem_bytes
    base_a = 0
    base_b = _align_up(size_a)
    base_c = _align_up(base_b + size_b)

    num_pid_m = math.ceil(M / block_m)
    num_pid_n = math.ceil(N / block_n)
    uses_2d_grid = "tt.get_program_id y" in artifacts.ttir
    logical_grid = (num_pid_m, num_pid_n, 1) if uses_2d_grid else (num_pid_m * num_pid_n, 1, 1)
    k_trips = math.ceil(K / block_k)

    stride_am = strides_a[0] if strides_a is not None else K
    stride_bk = strides_b[0] if strides_b is not None else N
    group_m_raw = config.get("GROUP_SIZE_M", config.get("GROUP_M"))
    group_m = int(group_m_raw) if group_m_raw is not None else None

    logical_coords, used_linear_fallback, data_dependent = _recover_logical_tile_coords(
        artifacts.ttir,
        logical_grid,
        fn=fn,
        call_args=call_args,
        signature=signature,
        M=M,
        N=N,
        block_m=block_m,
        block_n=block_n,
        stride_am=stride_am,
        stride_bk=stride_bk,
        group_size_m=group_m,
    )

    target_arch = canonical_ascend_arch(artifacts.target_arch)
    diagnostics = list(summary.diagnostics)
    if target_arch != "ascend910b3" and "unvalidated_target_ascend" not in diagnostics:
        diagnostics.append("unvalidated_target_ascend")
    unsupported_features = list(summary.unsupported_features)
    if used_linear_fallback:
        diagnostics.append("linear_access_fallback")
    if data_dependent and "data_dependent_address" not in unsupported_features:
        unsupported_features.append("data_dependent_address")

    if summary.autoblockify_stride is not None and summary.autoblockify_stride > 0:
        phys_grid, outer_trips, ordered_coords = _reorder_for_autoblockify(
            logical_coords, summary.autoblockify_stride
        )
        grid = [phys_grid, 1, 1]
        if len(set(outer_trips)) == 1:
            outer_loop = {"trip_count_mode": "uniform", "trip_counts": outer_trips[0]}
        else:
            outer_loop = {"trip_count_mode": "per_cta", "trip_counts": outer_trips}
    else:
        ordered_coords = logical_coords
        grid = [len(ordered_coords), 1, 1]
        outer_loop = {"trip_count_mode": "uniform", "trip_counts": 1}

    tile_bytes_a = block_m * block_k * elem_bytes
    tile_bytes_b = block_k * block_n * elem_bytes
    tile_bytes_c = block_m * block_n * elem_bytes

    # Build strided_2d access maps in ordered_coords sequence.
    stride_a_bytes = K * elem_bytes
    stride_b_bytes = N * elem_bytes
    stride_c_bytes = N * elem_bytes
    step_a_bytes = block_k * elem_bytes
    step_b_bytes = block_k * stride_b_bytes
    k_masked = (K % block_k) != 0

    entries_a: list[list[Any]] = []
    entries_b: list[list[Any]] = []
    entries_c: list[list[Any]] = []
    for pid_m, pid_n in ordered_coords:
        row_start = pid_m * block_m
        col_start = pid_n * block_n
        eff_m = max(0, min(block_m, M - row_start))
        eff_n = max(0, min(block_n, N - col_start))
        m_masked = eff_m < block_m
        n_masked = eff_n < block_n
        entries_a.append(
            [
                pid_m,
                pid_n,
                base_a + row_start * stride_a_bytes,
                eff_m,
                min(block_k, K) * elem_bytes,
                stride_a_bytes,
                step_a_bytes,
                m_masked or k_masked,
            ]
        )
        entries_b.append(
            [
                pid_m,
                pid_n,
                base_b + col_start * elem_bytes,
                min(block_k, K),
                eff_n * elem_bytes,
                stride_b_bytes,
                step_b_bytes,
                n_masked or k_masked,
            ]
        )
        entries_c.append(
            [
                pid_m,
                pid_n,
                base_c + row_start * stride_c_bytes + col_start * elem_bytes,
                eff_m,
                eff_n * elem_bytes,
                stride_c_bytes,
                0,
                m_masked or n_masked,
            ]
        )

    access_map: dict[str, dict[str, Any]] = {
        "acc_A": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_a,
        },
        "acc_B": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_b,
        },
        "acc_C": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_c,
        },
    }

    tensors: list[dict[str, Any]] = [
        {
            "id": "A",
            "dtype": dtype,
            "element_bits": elem_bits,
            "shape": [M, K],
            "strides": [K, 1],
            "base_addr": base_a,
            "cache_policy": "default",
        },
        {
            "id": "B",
            "dtype": dtype,
            "element_bits": elem_bits,
            "shape": [K, N],
            "strides": [N, 1],
            "base_addr": base_b,
            "cache_policy": "default",
        },
        {
            "id": "C",
            "dtype": dtype,
            "element_bits": elem_bits,
            "shape": [M, N],
            "strides": [N, 1],
            "base_addr": base_c,
            "cache_policy": "streaming",
        },
    ]

    # Ascend 2D Nd2Nz/Fixpipe DMA issues one row-request per tile row (row_bytes = cols * elem_bytes).
    # Encoding transfer_width_bytes = row_bytes // 32 maps nominal_bytes / (32 * width) to the exact
    # 2D DMA row count (BM for A/C, BK for B) against the measured Nd2Nz row-issue rate.
    dma_width_a = max(1, (block_k * elem_bytes) // 32)
    dma_width_b = max(1, (block_n * elem_bytes) // 32)
    pipeline_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_A",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "A",
            "nominal_bytes_per_cta": tile_bytes_a,
            "transfer_width_bytes": dma_width_a,
            "access_ref": "acc_A",
        },
        {
            "op_id": "load_B",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "B",
            "nominal_bytes_per_cta": tile_bytes_b,
            "transfer_width_bytes": dma_width_b,
            "access_ref": "acc_B",
        },
        {
            "op_id": "l1_to_l0_AB",
            "kind": "smem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "bytes_per_cta": tile_bytes_a + tile_bytes_b,
        },
    ]

    # In mixed Cube+Vector kernels, inter-core GM workspace traffic stays L2-resident when
    # total workspace fits in L2; report it via vendor_extensions.ascend so the model core
    # charges L2 bounce bandwidth rather than write-through DRAM stores.
    ws_loop_bytes = summary.workspace_loop_store_bytes + summary.workspace_loop_load_bytes
    if summary.mix_mode == "mix" and ws_loop_bytes == 0 and summary.workspace_bytes > 0:
        slots = max(1, summary.workspace_multibuffer)
        ws_loop_bytes = 2 * (summary.workspace_bytes // slots)

    if summary.mma_ops:
        for idx, mma in enumerate(summary.mma_ops):
            pipeline_ops.append(
                {
                    "op_id": f"mmad_{idx}",
                    "kind": "mma",
                    "warp_group_id": 0,
                    "stage_offset": 0,
                    "op_class": mma.op_class,
                    "tile_mnk": list(mma.tile_mnk),
                    "inst_mnk": list(mma.inst_mnk),
                    "a_dtype": mma.a_dtype,
                    "b_dtype": mma.b_dtype,
                    "acc_dtype": mma.acc_dtype,
                    "count": mma.count,
                }
            )
    else:
        im, in_, ik = (16, 16, 16)
        cnt = max(1, math.ceil(block_m / im) * math.ceil(block_n / in_) * math.ceil(block_k / ik))
        pipeline_ops.append(
            {
                "op_id": "mmad_0",
                "kind": "mma",
                "warp_group_id": 0,
                "stage_offset": 0,
                "op_class": "cube.mmad",
                "tile_mnk": [block_m, block_n, block_k],
                "inst_mnk": [im, in_, ik],
                "a_dtype": dtype,
                "b_dtype": dtype,
                "acc_dtype": acc_dtype,
                "count": cnt,
            }
        )

    if summary.loop_vec_ops > 0:
        pipeline_ops.append(
            {
                "op_id": "loop_vec_alu",
                "kind": "alu",
                "warp_group_id": 0,
                "stage_offset": 0,
                "dtype": acc_dtype,
                "count": summary.loop_vec_ops * max(1, (block_m * block_k) // 32),
            }
        )
    if summary.loop_sfu_ops > 0:
        pipeline_ops.append(
            {
                "op_id": "loop_vec_sfu",
                "kind": "sfu",
                "warp_group_id": 0,
                "stage_offset": 0,
                "dtype": acc_dtype,
                "count": summary.loop_sfu_ops * max(1, (block_m * block_n) // 32),
            }
        )

    barrier_count = summary.loop_barriers + summary.loop_sync_blocks
    if barrier_count > 0:
        pipeline_ops.append(
            {
                "op_id": "pipe_sync",
                "kind": "barrier",
                "warp_group_id": 0,
                "stage_offset": 0,
                "count": barrier_count,
            }
        )

    epilogue_ops: list[dict[str, Any]] = [
        {
            "op_id": "fixpipe_store_C",
            "kind": "gmem_store",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "C",
            "nominal_bytes_per_cta": tile_bytes_c,
            "transfer_width_bytes": dma_width_b,
            "access_ref": "acc_C",
        }
    ]

    l1_used = summary.l1_bytes or (summary.num_stages * (tile_bytes_a + tile_bytes_b))
    l0c_used = summary.l0c_bytes or (block_m * block_n * 4)

    resources: dict[str, Any] = {
        "source": "assembler_report" if artifacts.hivm_ir else "estimated",
        "registers_per_thread": 1,
        "static_smem_bytes": l1_used,
        "stack_frame_bytes": 0,
        "spill_stores_bytes": 0,
        "spill_loads_bytes": 0,
        "spill_scope": "none",
        "barriers": barrier_count,
        "vendor_extensions": {
            "ascend": {
                "ub_bytes": summary.ub_bytes,
                "l0c_bytes": l0c_used,
                "workspace_bytes": summary.workspace_bytes,
                "workspace_loop_bytes": ws_loop_bytes,
                "workspace_multibuffer": summary.workspace_multibuffer,
                "cv_sync_hops_per_iter": summary.loop_cv_sync_hops,
            }
        },
    }

    return {
        "schema_version": "0.1",
        "metadata": {
            "kernel_name": artifacts.kernel_name,
            "target_vendor": "ascend",
            "target_arch": canonical_ascend_arch(artifacts.target_arch),
            "frontend": "triton_ascend",
            "toolchain": {
                "compiler": "triton-ascend",
                "compiler_version": artifacts.triton_version,
                "assembler": "bishengir-compile",
                "assembler_version": "hivm",
                "mix_mode": summary.mix_mode,
                "bs_task_type": artifacts.bs_task_type,
                "autoblockify_stride": summary.autoblockify_stride,
                "workspace_multibuffer": summary.workspace_multibuffer,
                "artifact_hash_version": ARTIFACT_HASH_VERSION,
                "ttir_sha256": artifacts.ttir_sha256,
                "hivm_sha256": artifacts.hivm_sha256,
            },
            "config": dict(config),
        },
        "launch": {
            "grid": grid,
            "cluster_shape": [1, 1, 1],
            "num_warps": 1,
            "warp_size": 32,
            "warp_groups": [{"id": 0, "role": "unified", "num_warps": 1}],
            "dynamic_smem_bytes": 0,
        },
        "resources": resources,
        "tensors": tensors,
        "body": {
            "outer_loop": outer_loop,
            "prologue": {"ops": []},
            "pipeline_loop": {
                "trip_count_mode": "uniform",
                "trip_counts": k_trips,
                "num_stages": summary.num_stages,
                "prefetch_distance": summary.prefetch_distance,
                "async_copy": summary.async_copy,
                "ops": pipeline_ops,
            },
            "epilogue": {
                "overlaps_next_tile": False,
                "ops": epilogue_ops,
            },
        },
        "access_map": access_map,
        "diagnostics": diagnostics,
        "unsupported_features": unsupported_features,
    }


def extract_ascend_gemm_kernel_ir(
    fn: Any,
    *,
    M: int,
    N: int,
    K: int,
    config: Mapping[str, Any],
    target_arch: str = "ascend910b3",
    dtype: str = "f16",
    acc_dtype: str = "f32",
    call_args: tuple[Any, ...] | None = None,
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
    strides_c: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compile a Triton GEMM kernel offline for Ascend NPU and emit KernelIR v0.1."""
    elem_bytes = dtype_bits(dtype) // 8
    base_a = 0x10000
    base_b = _align_up(base_a + M * K * elem_bytes)
    base_c = _align_up(base_b + K * N * elem_bytes)

    if call_args is None:
        call_args = _build_ascend_gemm_call_args(
            fn,
            dtype=dtype,
            M=M,
            N=N,
            K=K,
            base_a=base_a,
            base_b=base_b,
            base_c=base_c,
            strides_a=strides_a,
            strides_b=strides_b,
            strides_c=strides_c,
        )

    signature, constants = _mock_signature_for_fn(fn, call_args, config)
    num_stages = int(config.get("num_stages", 2))
    raw_mb = config.get("multibuffer")
    multibuffer = None if raw_mb is None else bool(raw_mb)

    cache_root = Path(cache_dir) if cache_dir is not None else default_aot_cache_dir()
    ir_key_payload = {
        "v": ASCEND_KERNEL_IR_CACHE_VERSION,
        "arch": canonical_ascend_arch(target_arch),
        "shape": [M, N, K],
        "dtype": dtype,
        "acc_dtype": acc_dtype,
        "config": _canonical_json(dict(config)),
        "src": hashlib.sha256(
            (getattr(fn, "src", None) or getattr(fn, "__name__", repr(fn))).encode("utf-8")
        ).hexdigest(),
    }
    ir_key = hashlib.sha256(
        json.dumps(ir_key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ir_cache_file = cache_root / "ascend_kernel_ir" / f"{ir_key}.json"
    if use_cache and ir_cache_file.is_file():
        try:
            return json.loads(ir_cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    artifacts = compile_triton_ascend_aot(
        fn,
        signature=signature,
        constants=constants,
        target_arch=target_arch,
        num_stages=num_stages,
        multibuffer=multibuffer,
        cache_dir=cache_root,
        use_cache=use_cache,
    )
    ir_dict = assemble_ascend_gemm_kernel_ir(
        artifacts,
        M=M,
        N=N,
        K=K,
        config=config,
        dtype=dtype,
        acc_dtype=acc_dtype,
        fn=fn,
        call_args=call_args,
        signature=signature,
        strides_a=strides_a,
        strides_b=strides_b,
    )
    if use_cache:
        try:
            ir_cache_file.parent.mkdir(parents=True, exist_ok=True)
            ir_cache_file.write_text(json.dumps(ir_dict), encoding="utf-8")
        except OSError:
            pass
    return ir_dict


def extract_ascend_gemm_kernel_ir_batch(
    fn: Any,
    *,
    M: int,
    N: int,
    K: int,
    configs: Sequence[Mapping[str, Any]],
    target_arch: str = "ascend910b3",
    dtype: str = "f16",
    acc_dtype: str = "f32",
    call_args: tuple[Any, ...] | None = None,
    strides_a: tuple[int, int] | None = None,
    strides_b: tuple[int, int] | None = None,
    strides_c: tuple[int, int] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[tuple[dict[str, Any], str] | Exception]:
    """Batch-extract (ir_dict, ir_json) for Ascend GEMM configs."""
    out: list[tuple[dict[str, Any], str] | Exception] = []
    for cfg in configs:
        try:
            ir_dict = extract_ascend_gemm_kernel_ir(
                fn,
                M=M,
                N=N,
                K=K,
                config=cfg,
                target_arch=target_arch,
                dtype=dtype,
                acc_dtype=acc_dtype,
                call_args=call_args,
                strides_a=strides_a,
                strides_b=strides_b,
                strides_c=strides_c,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            out.append((ir_dict, json.dumps(ir_dict)))
        except Exception as exc:
            out.append(exc)
    return out
