"""Extract VirtualGPU KernelIR from Triton forward fused-attention kernels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .ascend_compiler import is_ascend_arch
from .compiler import (
    ARTIFACT_HASH_VERSION,
    CompiledArtifacts,
    compile_triton_aot,
    parse_arch_number,
    specialize_triton_call,
)
from .scalar_eval import ScalarEvalResult, evaluate_ttgir_cta_offsets
from .triton_gemm import MockTensor, _align_up
from .ttgir import TtgirSummary, dtype_bits, parse_ttgir_summary


def _param_int_values_from_call(
    fn: Any,
    call_args: tuple[Any, ...],
    signature: dict[str, str],
) -> tuple[list[int | None], dict[str, int]]:
    """Map non-constexpr runtime parameters in tt.func order to their integer values."""
    param_int_values: list[int | None] = []
    runtime_int_args: dict[str, int] = {}
    for arg_name, val in zip(fn.arg_names, call_args):
        if signature.get(arg_name) == "constexpr":
            continue
        if isinstance(val, int) and not isinstance(val, bool):
            param_int_values.append(val)
            runtime_int_args[arg_name] = val
        else:
            param_int_values.append(None)
    return param_int_values, runtime_int_args


def _default_attention_call_args(
    fn: Any,
    Z: int,
    H: int,
    N_CTX: int,
    D: int,
    H_kv: int,
    dtype: str,
    strides_q: tuple[int, int, int, int],
    strides_k: tuple[int, int, int, int],
    strides_v: tuple[int, int, int, int],
    strides_o: tuple[int, int, int, int],
    sm_scale: float,
) -> tuple[Any, ...]:
    """Synthesize representative call arguments for a standard Triton attention kernel."""
    elem_bytes = dtype_bits(dtype) // 8
    size_q = Z * H * N_CTX * D * elem_bytes
    size_kv = Z * H_kv * N_CTX * D * elem_bytes
    base_q = 0x10000
    base_k = _align_up(base_q + size_q, 0x10000)
    base_v = _align_up(base_k + size_kv, 0x10000)
    base_o = _align_up(base_v + size_kv, 0x10000)

    q_t = MockTensor(dtype, base_q)
    k_t = MockTensor(dtype, base_k)
    v_t = MockTensor(dtype, base_v)
    o_t = MockTensor(dtype, base_o)

    arg_map: dict[str, Any] = {
        "q": q_t,
        "q_ptr": q_t,
        "k": k_t,
        "k_ptr": k_t,
        "v": v_t,
        "v_ptr": v_t,
        "out": o_t,
        "o": o_t,
        "o_ptr": o_t,
        "out_ptr": o_t,
        "sm_scale": sm_scale,
        "stride_qz": strides_q[0],
        "stride_qh": strides_q[1],
        "stride_qm": strides_q[2],
        "stride_qk": strides_q[3],
        "stride_qd": strides_q[3],
        "stride_kz": strides_k[0],
        "stride_kh": strides_k[1],
        "stride_kn": strides_k[2],
        "stride_kk": strides_k[3],
        "stride_kd": strides_k[3],
        "stride_vz": strides_v[0],
        "stride_vh": strides_v[1],
        "stride_vn": strides_v[2],
        "stride_vk": strides_v[3],
        "stride_vd": strides_v[3],
        "stride_oz": strides_o[0],
        "stride_oh": strides_o[1],
        "stride_om": strides_o[2],
        "stride_ok": strides_o[3],
        "stride_od": strides_o[3],
        "z": Z,
        "h": H,
        "h_kv": H_kv,
        "n_ctx": N_CTX,
        "seq_len": N_CTX,
        "head_dim": D,
    }

    positional: list[Any] = []
    for param in fn.params:
        if param.is_constexpr:
            continue
        low = param.name.lower()
        if low in arg_map:
            positional.append(arg_map[low])
        else:
            raise ValueError(
                f"Cannot infer default argument for parameter {param.name!r}; "
                "pass explicit `call_args` to extract_attention_kernel_ir."
            )
    return tuple(positional)


def _evaluate_attention_access(
    ttgir: str,
    grid: tuple[int, int, int],
    *,
    fn: Any,
    call_args: tuple[Any, ...],
    signature: dict[str, str],
    Z: int,
    H: int,
    N_CTX: int,
    D: int,
    strides_q: tuple[int, int, int, int],
    strides_k: tuple[int, int, int, int],
    strides_v: tuple[int, int, int, int],
    strides_o: tuple[int, int, int, int],
) -> tuple[ScalarEvalResult, bool]:
    """Run scalar evaluation over the 2D attention grid to recover per-CTA pointer offsets and trip counts."""
    runtime_ints: dict[str, int] = {
        "Z": Z,
        "H": H,
        "N_CTX": N_CTX,
        "D": D,
        "stride_qz": strides_q[0],
        "stride_qh": strides_q[1],
        "stride_qm": strides_q[2],
        "stride_qk": strides_q[3],
        "stride_kz": strides_k[0],
        "stride_kh": strides_k[1],
        "stride_kn": strides_k[2],
        "stride_kk": strides_k[3],
        "stride_vz": strides_v[0],
        "stride_vh": strides_v[1],
        "stride_vn": strides_v[2],
        "stride_vk": strides_v[3],
        "stride_oz": strides_o[0],
        "stride_oh": strides_o[1],
        "stride_om": strides_o[2],
        "stride_ok": strides_o[3],
    }
    param_int_values, extra_ints = _param_int_values_from_call(fn, call_args, signature)
    runtime_ints.update(extra_ints)
    res = evaluate_ttgir_cta_offsets(ttgir, grid, runtime_ints, param_int_values=param_int_values)
    grid_size = grid[0] * grid[1] * grid[2]
    all_ptrs_resolved = not res.has_nested_loop and all(
        len(res.ptr_elem_offsets.get(i, [])) == grid_size for i in (0, 1, 2, 3)
    )
    return res, all_ptrs_resolved


def _build_attention_access_map(
    *,
    grid_m: int,
    grid_hz: int,
    H: int,
    H_kv: int,
    N_CTX: int,
    D: int,
    block_m: int,
    block_n: int,
    elem_bytes: int,
    base_q: int,
    base_k: int,
    base_v: int,
    base_o: int,
    strides_q: tuple[int, int, int, int],
    strides_k: tuple[int, int, int, int],
    strides_v: tuple[int, int, int, int],
    strides_o: tuple[int, int, int, int],
    eval_res: ScalarEvalResult,
    all_ptrs_resolved: bool,
) -> dict[str, Any]:
    """Build strided_2d_per_outer_tile access_map entries for Q, K, V, Out."""
    row_bytes = D * elem_bytes
    q_row_stride_b = strides_q[2] * elem_bytes
    k_row_stride_b = strides_k[2] * elem_bytes
    v_row_stride_b = strides_v[2] * elem_bytes
    o_row_stride_b = strides_o[2] * elem_bytes
    k_step_delta_b = block_n * k_row_stride_b
    v_step_delta_b = block_n * v_row_stride_b
    kv_has_mask = (N_CTX % block_n) != 0
    kv_rows = min(block_n, N_CTX)
    gqa_group = max(1, H // H_kv)

    entries_q: list[list[Any]] = []
    entries_k: list[list[Any]] = []
    entries_v: list[list[Any]] = []
    entries_o: list[list[Any]] = []

    q_offs = eval_res.ptr_elem_offsets.get(0, [])
    k_offs = eval_res.ptr_elem_offsets.get(1, [])
    v_offs = eval_res.ptr_elem_offsets.get(2, [])
    o_offs = eval_res.ptr_elem_offsets.get(3, [])

    cta_idx = 0
    for off_hz in range(grid_hz):
        off_z = off_hz // H
        off_h = off_hz % H
        off_h_kv = off_h // gqa_group
        for start_m in range(grid_m):
            m_start = start_m * block_m
            q_rows = min(block_m, max(0, N_CTX - m_start))
            q_has_mask = (m_start + block_m) > N_CTX

            if all_ptrs_resolved:
                q_base = base_q + q_offs[cta_idx] * elem_bytes
                k_base = base_k + k_offs[cta_idx] * elem_bytes
                v_base = base_v + v_offs[cta_idx] * elem_bytes
                o_base = base_o + o_offs[cta_idx] * elem_bytes
            else:
                q_base = (
                    base_q
                    + (off_z * strides_q[0] + off_h * strides_q[1] + m_start * strides_q[2])
                    * elem_bytes
                )
                k_base = base_k + (off_z * strides_k[0] + off_h_kv * strides_k[1]) * elem_bytes
                v_base = base_v + (off_z * strides_v[0] + off_h_kv * strides_v[1]) * elem_bytes
                o_base = (
                    base_o
                    + (off_z * strides_o[0] + off_h * strides_o[1] + m_start * strides_o[2])
                    * elem_bytes
                )

            entries_q.append(
                [start_m, off_hz, q_base, q_rows, row_bytes, q_row_stride_b, 0, q_has_mask]
            )
            entries_k.append(
                [
                    start_m,
                    off_hz,
                    k_base,
                    kv_rows,
                    row_bytes,
                    k_row_stride_b,
                    k_step_delta_b,
                    kv_has_mask,
                ]
            )
            entries_v.append(
                [
                    start_m,
                    off_hz,
                    v_base,
                    kv_rows,
                    row_bytes,
                    v_row_stride_b,
                    v_step_delta_b,
                    kv_has_mask,
                ]
            )
            entries_o.append(
                [start_m, off_hz, o_base, q_rows, row_bytes, o_row_stride_b, 0, q_has_mask]
            )
            cta_idx += 1

    return {
        "acc_Q": {"encoding": "strided_2d_per_outer_tile", "entries": entries_q},
        "acc_K": {"encoding": "strided_2d_per_outer_tile", "entries": entries_k},
        "acc_V": {"encoding": "strided_2d_per_outer_tile", "entries": entries_v},
        "acc_O": {"encoding": "strided_2d_per_outer_tile", "entries": entries_o},
    }


def _assemble_attention_kernel_ir(
    *,
    fn: Any,
    artifacts: CompiledArtifacts,
    Z: int,
    H: int,
    H_kv: int,
    N_CTX: int,
    D: int,
    config: dict[str, int],
    dtype: str,
    causal: bool,
    call_args: tuple[Any, ...],
    signature: dict[str, str],
    strides_q: tuple[int, int, int, int],
    strides_k: tuple[int, int, int, int],
    strides_v: tuple[int, int, int, int],
    strides_o: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Assemble a KernelIR document for a compiled Triton forward attention kernel."""
    num_stages_req = int(config.get("num_stages", 3))

    elem_bits = dtype_bits(dtype)
    elem_bytes = elem_bits // 8
    size_q = Z * H * N_CTX * D * elem_bytes
    size_kv = Z * H_kv * N_CTX * D * elem_bytes
    base_q = 0
    base_k = _align_up(size_q)
    base_v = _align_up(base_k + size_kv)
    base_o = _align_up(base_v + size_kv)

    summary: TtgirSummary = parse_ttgir_summary(
        ttgir=artifacts.ttgir,
        ptx=artifacts.ptx,
        requested_num_stages=num_stages_req,
        element_bytes=elem_bytes,
    )

    block_m_cfg = config.get("BLOCK_M", config.get("BLOCK_SIZE_M", config.get("BM")))
    block_n_cfg = config.get("BLOCK_N", config.get("BLOCK_SIZE_N", config.get("BN")))
    if summary.mma_dots:
        # Derive tile (BM, BN) directly from the compiled QK^T tt.dot in TTGIR
        # so extraction is independent of Python constexpr parameter names.
        block_m, block_n = summary.mma_dots[0].tile_mnk[0], summary.mma_dots[0].tile_mnk[1]
    elif block_m_cfg is not None and block_n_cfg is not None:
        block_m, block_n = int(block_m_cfg), int(block_n_cfg)
    else:
        raise ValueError(
            "Could not determine attention tile dimensions (BM, BN) from TTGIR tt.dot "
            "or config keys (BLOCK_M/BLOCK_SIZE_M/BM, BLOCK_N/BLOCK_SIZE_N/BN)."
        )

    diagnostics = list(summary.diagnostics)
    unsupported_features = list(summary.unsupported_features)
    if artifacts.ptxas.ptxas_only_spill:
        diagnostics.append("ptxas_only_spill")

    grid_m = math.ceil(N_CTX / block_m)
    grid_hz = Z * H
    grid = (grid_m, grid_hz, 1)

    eval_res, all_ptrs_resolved = _evaluate_attention_access(
        artifacts.ttgir,
        grid,
        fn=fn,
        call_args=call_args,
        signature=signature,
        Z=Z,
        H=H,
        N_CTX=N_CTX,
        D=D,
        strides_q=strides_q,
        strides_k=strides_k,
        strides_v=strides_v,
        strides_o=strides_o,
    )
    if eval_res.has_nested_loop:
        diagnostics.append("persistent_kernel_unmodeled")
    if not all_ptrs_resolved:
        diagnostics.append("linear_access_fallback")
    if eval_res.data_dependent and "data_dependent_address" not in unsupported_features:
        unsupported_features.append("data_dependent_address")

    access_map = _build_attention_access_map(
        grid_m=grid_m,
        grid_hz=grid_hz,
        H=H,
        H_kv=H_kv,
        N_CTX=N_CTX,
        D=D,
        block_m=block_m,
        block_n=block_n,
        elem_bytes=elem_bytes,
        base_q=base_q,
        base_k=base_k,
        base_v=base_v,
        base_o=base_o,
        strides_q=strides_q,
        strides_k=strides_k,
        strides_v=strides_v,
        strides_o=strides_o,
        eval_res=eval_res,
        all_ptrs_resolved=all_ptrs_resolved,
    )

    # Determine pipeline_loop trip counts (uniform for non-causal; per_cta for causal).
    uniform_kv_trips = math.ceil(N_CTX / block_n)
    if causal:
        if (
            eval_res.loop_trip_counts is not None
            and len(eval_res.loop_trip_counts) == grid_m * grid_hz
        ):
            per_cta_trips = list(eval_res.loop_trip_counts)
        else:
            per_cta_trips = [
                min(uniform_kv_trips, math.ceil(min((start_m + 1) * block_m, N_CTX) / block_n))
                for _ in range(grid_hz)
                for start_m in range(grid_m)
            ]
        trip_count_mode = "per_cta"
        trip_counts: int | list[int] = per_cta_trips
    else:
        trip_count_mode = "uniform"
        trip_counts = uniform_kv_trips

    tile_bytes_q = block_m * D * elem_bytes
    tile_bytes_k = block_n * D * elem_bytes
    tile_bytes_v = block_n * D * elem_bytes
    tile_bytes_o = block_m * D * elem_bytes

    prologue_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_Q",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "Q",
            "nominal_bytes_per_cta": tile_bytes_q,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_Q",
        }
    ]
    if summary.prologue_smem_store_bytes > 0:
        prologue_ops.append(
            {
                "op_id": "smem_store_Q",
                "kind": "smem_store",
                "warp_group_id": 0,
                "stage_offset": 0,
                "bytes_per_cta": summary.prologue_smem_store_bytes,
            }
        )

    # The MMA operands are read with ldmatrix, once per warp that needs them: count those bytes from the
    # PTX and fall back to the TTGIR tile sizes (each tile once per CTA) only when the loop has none.
    ttgir_tile_bytes = summary.smem_load_bytes_per_iter or (
        tile_bytes_q + tile_bytes_k + tile_bytes_v
    )
    operand_bytes = summary.loop_ptx_ldmatrix_bytes or ttgir_tile_bytes
    smem_load_bytes = operand_bytes + summary.loop_ptx_ld_shared_bytes
    pipeline_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_K",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "K",
            "nominal_bytes_per_cta": tile_bytes_k,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_K",
        },
        {
            "op_id": "load_V",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "V",
            "nominal_bytes_per_cta": tile_bytes_v,
            "transfer_width_bytes": summary.transfer_width_bytes,
            "access_ref": "acc_V",
        },
        {
            "op_id": "smem_load_QKV",
            "kind": "smem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "bytes_per_cta": smem_load_bytes,
            "ttgir_tile_bytes_per_cta": ttgir_tile_bytes,
            "ptx_ldmatrix_bytes_per_cta": summary.loop_ptx_ldmatrix_bytes,
        },
    ]
    if summary.loop_ptx_st_shared_bytes > 0:
        pipeline_ops.append(
            {
                "op_id": "smem_store_loop",
                "kind": "smem_store",
                "warp_group_id": 0,
                "stage_offset": 0,
                "bytes_per_cta": summary.loop_ptx_st_shared_bytes,
            }
        )

    dot_names = ("mma_qk", "mma_pv")
    for idx, dot in enumerate(summary.mma_dots):
        op_id = dot_names[idx] if idx < len(dot_names) else f"mma_{idx}"
        pipeline_ops.append(
            {
                "op_id": op_id,
                "kind": "mma",
                "warp_group_id": 0,
                "stage_offset": 0,
                "op_class": dot.op_class,
                "tile_mnk": list(dot.tile_mnk),
                "inst_mnk": list(dot.inst_mnk),
                "a_dtype": dot.a_dtype,
                "b_dtype": dot.b_dtype,
                "acc_dtype": dot.acc_dtype,
                "count": dot.count,
            }
        )

    if summary.loop_sfu_per_warp > 0:
        pipeline_ops.append(
            {
                "op_id": "sfu_softmax",
                "kind": "sfu",
                "warp_group_id": 0,
                "stage_offset": 0,
                "dtype": "f32",
                "count": summary.loop_sfu_per_warp,
            }
        )

    if summary.loop_shfl_per_warp > 0:
        pipeline_ops.append(
            {
                "op_id": "shfl_softmax",
                "kind": "shfl",
                "warp_group_id": 0,
                "stage_offset": 0,
                "dtype": "f32",
                "count": summary.loop_shfl_per_warp,
            }
        )

    if summary.loop_alu_per_warp > 0:
        pipeline_ops.append(
            {
                "op_id": "alu_softmax",
                "kind": "alu",
                "warp_group_id": 0,
                "stage_offset": 0,
                "dtype": "f32",
                "count": summary.loop_alu_per_warp,
            }
        )

    pipeline_ops.append(
        {
            "op_id": "sync_loop",
            "kind": "barrier",
            "warp_group_id": 0,
            "stage_offset": 0,
            "count": max(1, summary.loop_ptx_barriers or summary.loop_barriers),
        }
    )

    epilogue_ops: list[dict[str, Any]] = [
        {
            "op_id": "store_O",
            "kind": "gmem_store",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "Out",
            "nominal_bytes_per_cta": tile_bytes_o,
            "transfer_width_bytes": summary.store_transfer_width_bytes
            or summary.transfer_width_bytes,
            "access_ref": "acc_O",
        }
    ]

    kernel_name = getattr(fn, "__name__", None) or getattr(
        getattr(fn, "fn", None), "__name__", "triton_attention"
    )

    return {
        "schema_version": "0.1",
        "metadata": {
            "kernel_name": kernel_name,
            "target_vendor": "nvidia",
            "target_arch": f"sm_{parse_arch_number(artifacts.target_arch)}",
            "frontend": "triton-aot",
            "toolchain": {
                "compiler": "triton",
                "compiler_version": artifacts.triton_version,
                "assembler": artifacts.ptxas.assembler,
                "assembler_version": artifacts.ptxas.assembler_version,
                "target_sm_compiled": artifacts.ptxas.target_sm,
                "enable_fp_fusion": artifacts.enable_fp_fusion,
                "artifact_hash_version": ARTIFACT_HASH_VERSION,
                "ptx_sha256": artifacts.ptx_sha256,
                "ttgir_sha256": artifacts.ttgir_sha256,
            },
            "config": {
                **config,
                "Z": Z,
                "H": H,
                "H_kv": H_kv,
                "N_CTX": N_CTX,
                "HEAD_DIM": D,
                "causal": causal,
            },
        },
        "launch": {
            "grid": list(grid),
            "cluster_shape": list(summary.cluster_shape),
            "num_warps": summary.num_warps,
            "warp_size": summary.warp_size,
            "warp_groups": [{"id": 0, "role": "unified", "num_warps": summary.num_warps}],
            "dynamic_smem_bytes": 0,
        },
        "resources": {
            "source": "assembler_report",
            "registers_per_thread": artifacts.ptxas.registers_per_thread,
            "static_smem_bytes": artifacts.static_smem_bytes,
            "stack_frame_bytes": artifacts.ptxas.stack_frame_bytes,
            "spill_stores_bytes": artifacts.ptxas.spill_stores_bytes,
            "spill_loads_bytes": artifacts.ptxas.spill_loads_bytes,
            "spill_scope": artifacts.ptxas.spill_scope,
            "barriers": artifacts.ptxas.barriers,
        },
        "tensors": [
            {
                "id": "Q",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H, N_CTX, D],
                "strides": list(strides_q),
                "base_addr": base_q,
                "cache_policy": "default",
            },
            {
                "id": "K",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H_kv, N_CTX, D],
                "strides": list(strides_k),
                "base_addr": base_k,
                "cache_policy": "default",
            },
            {
                "id": "V",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H_kv, N_CTX, D],
                "strides": list(strides_v),
                "base_addr": base_v,
                "cache_policy": "default",
            },
            {
                "id": "Out",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H, N_CTX, D],
                "strides": list(strides_o),
                "base_addr": base_o,
                "cache_policy": "default",
            },
        ],
        "body": {
            "outer_loop": {
                "trip_count_mode": "uniform",
                "trip_counts": 1,
            },
            "prologue": {"ops": prologue_ops},
            "pipeline_loop": {
                "trip_count_mode": trip_count_mode,
                "trip_counts": trip_counts,
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


def extract_attention_kernel_ir(
    fn: Any,
    Z: int,
    H: int,
    N_CTX: int,
    D: int,
    config: dict[str, int],
    target_arch: str,
    dtype: str = "f16",
    *,
    H_kv: int | None = None,
    causal: bool = False,
    sm_scale: float | None = None,
    call_args: tuple[Any, ...] | None = None,
    strides_q: tuple[int, int, int, int] | None = None,
    strides_k: tuple[int, int, int, int] | None = None,
    strides_v: tuple[int, int, int, int] | None = None,
    strides_o: tuple[int, int, int, int] | None = None,
    warp_size: int = 32,
    enable_fp_fusion: bool = True,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compile a Triton forward fused-attention kernel AOT and extract VirtualGPU KernelIR."""
    if is_ascend_arch(target_arch):
        from .triton_ascend_attention import extract_ascend_attention_kernel_ir

        return extract_ascend_attention_kernel_ir(
            fn,
            Z=Z,
            H=H,
            N_CTX=N_CTX,
            D=D,
            config=config,
            target_arch=target_arch,
            dtype=dtype,
            H_kv=H_kv,
            causal=causal,
            sm_scale=sm_scale,
            call_args=call_args,
            strides_q=strides_q,
            strides_k=strides_k,
            strides_v=strides_v,
            strides_o=strides_o,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    if hasattr(fn, "fn") and not hasattr(fn, "params"):
        fn = fn.fn

    resolved_h_kv = H if H_kv is None else H_kv
    resolved_scale = (1.0 / math.sqrt(D)) if sm_scale is None else sm_scale
    sq = strides_q or (H * N_CTX * D, N_CTX * D, D, 1)
    sk = strides_k or (resolved_h_kv * N_CTX * D, N_CTX * D, D, 1)
    sv = strides_v or (resolved_h_kv * N_CTX * D, N_CTX * D, D, 1)
    so = strides_o or (H * N_CTX * D, N_CTX * D, D, 1)

    num_warps = int(config.get("num_warps", 4))
    num_stages = int(config.get("num_stages", 3))

    args = (
        call_args
        if call_args is not None
        else _default_attention_call_args(
            fn, Z, H, N_CTX, D, resolved_h_kv, dtype, sq, sk, sv, so, resolved_scale
        )
    )

    param_names = {p.name for p in fn.params}
    call_kwargs: dict[str, Any] = {k: v for k, v in config.items() if k in param_names}
    if "HEAD_DIM" in param_names and "HEAD_DIM" not in call_kwargs:
        call_kwargs["HEAD_DIM"] = D
    if "CAUSAL" in param_names and "CAUSAL" not in call_kwargs:
        call_kwargs["CAUSAL"] = causal
    call_kwargs["num_warps"] = num_warps
    call_kwargs["num_stages"] = num_stages

    options_dict, signature, constexprs, attrs = specialize_triton_call(
        fn=fn,
        target_arch=target_arch,
        args=args,
        kwargs=call_kwargs,
        warp_size=warp_size,
        enable_fp_fusion=enable_fp_fusion,
    )

    artifacts = compile_triton_aot(
        fn=fn,
        signature=signature,
        constexprs=constexprs,
        target_arch=target_arch,
        num_warps=num_warps,
        num_stages=num_stages,
        attrs=attrs,
        options=options_dict,
        warp_size=warp_size,
        enable_fp_fusion=enable_fp_fusion,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    return _assemble_attention_kernel_ir(
        fn=fn,
        artifacts=artifacts,
        Z=Z,
        H=H,
        H_kv=resolved_h_kv,
        N_CTX=N_CTX,
        D=D,
        config=config,
        dtype=dtype,
        causal=causal,
        call_args=args,
        signature=signature,
        strides_q=sq,
        strides_k=sk,
        strides_v=sv,
        strides_o=so,
    )
