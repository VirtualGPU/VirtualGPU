"""Triton-Ascend forward fused-attention extractor emitting KernelIR v0.1 from HIVM MLIR."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
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
from virtualgpu.frontend.ttgir import dtype_bits

ASCEND_ATTN_KERNEL_IR_CACHE_VERSION = "3"


def _align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _build_ascend_attention_call_args(
    fn: Any,
    *,
    Z: int,
    H: int,
    H_kv: int,
    N_CTX: int,
    D: int,
    dtype: str,
    strides_q: tuple[int, int, int, int],
    strides_k: tuple[int, int, int, int],
    strides_v: tuple[int, int, int, int],
    strides_o: tuple[int, int, int, int],
    sm_scale: float,
) -> tuple[Any, ...]:
    """Synthesize representative runtime arguments for standard or validation Triton attention kernels."""
    elem_bytes = dtype_bits(dtype) // 8
    size_q = Z * H * N_CTX * D * elem_bytes
    size_kv = Z * H_kv * N_CTX * D * elem_bytes
    base_q = 0x10000
    base_k = _align_up(base_q + size_q, 0x10000)
    base_v = _align_up(base_k + size_kv, 0x10000)
    base_o = _align_up(base_v + size_kv, 0x10000)
    base_lse = _align_up(base_o + size_q, 0x10000)

    q_t = MockTensor(dtype, base_q)
    k_t = MockTensor(dtype, base_k)
    v_t = MockTensor(dtype, base_v)
    o_t = MockTensor(dtype, base_o)
    lse_t = MockTensor("f32", base_lse)

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
        "lse": lse_t,
        "lse_ptr": lse_t,
        "sm_scale": sm_scale,
        "scale_log2": sm_scale * math.log2(math.e),
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
        "hq": H,
        "h_kv": H_kv,
        "hk": H_kv,
        "n_ctx": N_CTX,
        "seq": N_CTX,
        "seq_len": N_CTX,
        "head_dim": D,
        "d": D,
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
                "pass explicit `call_args` to extract_ascend_attention_kernel_ir."
            )
    return tuple(positional)


def _reorder_attention_tiles_for_autoblockify(
    logical_tiles: list[tuple[int, int]],
    stride: int,
) -> tuple[int, list[int], list[tuple[int, int]]]:
    """Fold logical (start_m, off_hz) tiles into per-AI-Core persistent outer iterations."""
    total = len(logical_tiles)
    phys_grid = min(total, max(1, stride))
    trip_counts: list[int] = []
    ordered: list[tuple[int, int]] = []
    for core_id in range(phys_grid):
        core_tiles = logical_tiles[core_id::phys_grid]
        trip_counts.append(len(core_tiles))
        ordered.extend(core_tiles)
    return phys_grid, trip_counts, ordered


def assemble_ascend_attention_kernel_ir(
    artifacts: AscendCompiledArtifacts,
    *,
    Z: int,
    H: int,
    H_kv: int,
    N_CTX: int,
    D: int,
    config: Mapping[str, Any],
    dtype: str = "f16",
    acc_dtype: str = "f32",
    causal: bool = False,
    strides_q: tuple[int, int, int, int] | None = None,
    strides_k: tuple[int, int, int, int] | None = None,
    strides_v: tuple[int, int, int, int] | None = None,
    strides_o: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """Assemble a KernelIR v0.1 dict from Triton-Ascend compiled attention HIVM artifacts."""
    summary: HivmSummary = parse_hivm_summary(
        artifacts.hivm_ir,
        requested_multibuffer=artifacts.requested_multibuffer,
        requested_num_stages=artifacts.requested_num_stages,
        fallback_workspace_bytes=artifacts.workspace_size,
    )

    block_m_cfg = config.get("BLOCK_M", config.get("BLOCK_SIZE_M", config.get("BM")))
    block_n_cfg = config.get("BLOCK_N", config.get("BLOCK_SIZE_N", config.get("BN")))
    if block_m_cfg is not None and block_n_cfg is not None:
        block_m, block_n = int(block_m_cfg), int(block_n_cfg)
    elif summary.mma_ops:
        block_m, block_n = summary.mma_ops[0].tile_mnk[0], summary.mma_ops[0].tile_mnk[1]
    else:
        raise ValueError(
            "Could not determine attention tile dimensions (BM, BN) from config or HIVM mmadL1."
        )

    sq = strides_q or (H * N_CTX * D, N_CTX * D, D, 1)
    sk = strides_k or (H_kv * N_CTX * D, N_CTX * D, D, 1)
    sv = strides_v or (H_kv * N_CTX * D, N_CTX * D, D, 1)
    so = strides_o or (H * N_CTX * D, N_CTX * D, D, 1)

    elem_bits = dtype_bits(dtype)
    elem_bytes = elem_bits // 8
    size_q = Z * H * N_CTX * D * elem_bytes
    size_kv = Z * H_kv * N_CTX * D * elem_bytes
    base_q = 0
    base_k = _align_up(size_q)
    base_v = _align_up(base_k + size_kv)
    base_o = _align_up(base_v + size_kv)

    grid_m = math.ceil(N_CTX / block_m)
    grid_hz = Z * H
    logical_tiles: list[tuple[int, int]] = [
        (start_m, off_hz) for off_hz in range(grid_hz) for start_m in range(grid_m)
    ]

    if summary.autoblockify_stride is not None and summary.autoblockify_stride > 0:
        phys_grid, outer_trips, ordered_tiles = _reorder_attention_tiles_for_autoblockify(
            logical_tiles, summary.autoblockify_stride
        )
        grid = [phys_grid, 1, 1]
        if len(set(outer_trips)) == 1:
            outer_loop: dict[str, Any] = {
                "trip_count_mode": "uniform",
                "trip_counts": outer_trips[0],
            }
        else:
            outer_loop = {"trip_count_mode": "per_cta", "trip_counts": outer_trips}
    else:
        ordered_tiles = logical_tiles
        grid = [grid_m, grid_hz, 1]
        outer_loop = {"trip_count_mode": "uniform", "trip_counts": 1}

    uniform_kv_trips = math.ceil(N_CTX / block_n)
    if causal:
        per_tile_trips = [
            min(uniform_kv_trips, math.ceil(min((start_m + 1) * block_m, N_CTX) / block_n))
            for start_m, _off_hz in ordered_tiles
        ]
        if outer_loop["trip_count_mode"] == "uniform" and outer_loop["trip_counts"] == 1:
            pipeline_trip_mode = "per_cta"
        else:
            pipeline_trip_mode = "per_outer_iter"
        pipeline_trip_counts: int | list[int] = per_tile_trips
    else:
        pipeline_trip_mode = "uniform"
        pipeline_trip_counts = uniform_kv_trips

    # Build strided_2d_per_outer_tile access maps in ordered_tiles sequence.
    row_bytes = D * elem_bytes
    q_row_stride_b = sq[2] * elem_bytes
    k_row_stride_b = sk[2] * elem_bytes
    v_row_stride_b = sv[2] * elem_bytes
    o_row_stride_b = so[2] * elem_bytes
    k_step_delta_b = block_n * k_row_stride_b
    v_step_delta_b = block_n * v_row_stride_b
    kv_has_mask = (N_CTX % block_n) != 0
    kv_rows = min(block_n, N_CTX)
    gqa_group = max(1, H // H_kv)

    entries_q: list[list[Any]] = []
    entries_k: list[list[Any]] = []
    entries_v: list[list[Any]] = []
    entries_o: list[list[Any]] = []

    for start_m, off_hz in ordered_tiles:
        off_z = off_hz // H
        off_h = off_hz % H
        off_h_kv = off_h // gqa_group
        m_start = start_m * block_m
        q_rows = min(block_m, max(0, N_CTX - m_start))
        q_has_mask = (m_start + block_m) > N_CTX

        q_base = base_q + (off_z * sq[0] + off_h * sq[1] + m_start * sq[2]) * elem_bytes
        k_base = base_k + (off_z * sk[0] + off_h_kv * sk[1]) * elem_bytes
        v_base = base_v + (off_z * sv[0] + off_h_kv * sv[1]) * elem_bytes
        o_base = base_o + (off_z * so[0] + off_h * so[1] + m_start * so[2]) * elem_bytes

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

    access_map: dict[str, dict[str, Any]] = {
        "acc_Q": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_q,
        },
        "acc_K": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_k,
        },
        "acc_V": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_v,
        },
        "acc_O": {
            "encoding": "strided_2d_per_outer_tile",
            "sidecar_ref": None,
            "entries": entries_o,
        },
    }

    tile_bytes_q = block_m * D * elem_bytes
    tile_bytes_k = block_n * D * elem_bytes
    tile_bytes_v = block_n * D * elem_bytes
    tile_bytes_o = block_m * D * elem_bytes
    dma_width = max(1, (D * elem_bytes) // 32)

    prologue_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_Q",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "Q",
            "nominal_bytes_per_cta": tile_bytes_q,
            "transfer_width_bytes": dma_width,
            "access_ref": "acc_Q",
        }
    ]

    pipeline_ops: list[dict[str, Any]] = [
        {
            "op_id": "load_K",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "K",
            "nominal_bytes_per_cta": tile_bytes_k,
            "transfer_width_bytes": dma_width,
            "access_ref": "acc_K",
        },
        {
            "op_id": "load_V",
            "kind": "gmem_load",
            "warp_group_id": 0,
            "stage_offset": summary.prefetch_distance,
            "tensor_id": "V",
            "nominal_bytes_per_cta": tile_bytes_v,
            "transfer_width_bytes": dma_width,
            "access_ref": "acc_V",
        },
        {
            "op_id": "l1_to_l0_QKV",
            "kind": "smem_load",
            "warp_group_id": 0,
            "stage_offset": 0,
            "bytes_per_cta": tile_bytes_q + tile_bytes_k + tile_bytes_v,
        },
    ]

    # Each KV iteration executes two Cube MMAs: QK^T (BM x BN x D) and PV (BM x D x BN).
    # Causal kernels may lower to two sequential KV loops (off-diagonal + diagonal) with
    # 4 mmadL1 ops total in the IR; normalize to one (QK^T, PV) pair per KV step.
    im, in_, ik = (16, 16, 16)
    if len(summary.mma_ops) >= 2:
        qk_op, pv_op = summary.mma_ops[0], summary.mma_ops[1]
        mma_pair = [
            (
                "mma_qk",
                qk_op.tile_mnk,
                qk_op.inst_mnk,
                qk_op.a_dtype,
                qk_op.b_dtype,
                qk_op.acc_dtype,
                qk_op.count,
            ),
            (
                "mma_pv",
                pv_op.tile_mnk,
                pv_op.inst_mnk,
                pv_op.a_dtype,
                pv_op.b_dtype,
                pv_op.acc_dtype,
                pv_op.count,
            ),
        ]
    else:
        qk_cnt = max(1, math.ceil(block_m / im) * math.ceil(block_n / in_) * math.ceil(D / ik))
        pv_cnt = max(1, math.ceil(block_m / im) * math.ceil(D / in_) * math.ceil(block_n / ik))
        mma_pair = [
            ("mma_qk", (block_m, block_n, D), (im, in_, ik), dtype, dtype, acc_dtype, qk_cnt),
            ("mma_pv", (block_m, D, block_n), (im, in_, ik), dtype, dtype, acc_dtype, pv_cnt),
        ]

    for op_id, t_mnk, i_mnk, a_dt, b_dt, c_dt, cnt in mma_pair:
        pipeline_ops.append(
            {
                "op_id": op_id,
                "kind": "mma",
                "warp_group_id": 0,
                "stage_offset": 0,
                "op_class": "cube.mmad",
                "tile_mnk": list(t_mnk),
                "inst_mnk": list(i_mnk),
                "a_dtype": a_dt,
                "b_dtype": b_dt,
                "acc_dtype": c_dt,
                "count": cnt,
            }
        )

    # Per-KV-step AIV vector/SFU instruction counts from the steady-state loop.
    # BiShengIR auto-bind-sub-block splits BM rows across 2 AIV sub-cores per AI Core when
    # the AIV loop tile row count is smaller than BM; scale by 2x only when actually split.
    aiv_subblock_scale = (
        2.0 if (summary.mix_mode == "mix" and 0 < summary.aiv_loop_max_rows < block_m) else 1.0
    )
    if summary.loop_sfu_insts > 0.0:
        sfu_count = max(1, round(summary.loop_sfu_insts * aiv_subblock_scale))
    elif summary.loop_sfu_ops > 0:
        sfu_count = max(1, round(summary.loop_sfu_ops * (block_m * block_n / 32.0)))
    else:
        sfu_count = max(1, (block_m * block_n) // 32)
    pipeline_ops.append(
        {
            "op_id": "sfu_softmax",
            "kind": "sfu",
            "warp_group_id": 0,
            "stage_offset": 0,
            "dtype": acc_dtype,
            "count": sfu_count,
        }
    )

    if summary.loop_vec_insts > 0.0:
        alu_count = max(1, round(summary.loop_vec_insts * aiv_subblock_scale))
    elif summary.loop_vec_ops > 0:
        alu_count = max(1, round(summary.loop_vec_ops * (block_m * block_n / 32.0)))
    else:
        alu_count = max(1, (4 * block_m * block_n) // 32)
    pipeline_ops.append(
        {
            "op_id": "alu_softmax",
            "kind": "alu",
            "warp_group_id": 0,
            "stage_offset": 0,
            "dtype": acc_dtype,
            "count": alu_count,
        }
    )

    barrier_count = max(1, summary.loop_barriers + summary.loop_sync_blocks)
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
            "op_id": "store_O",
            "kind": "gmem_store",
            "warp_group_id": 0,
            "stage_offset": 0,
            "tensor_id": "Out",
            "nominal_bytes_per_cta": tile_bytes_o,
            "transfer_width_bytes": dma_width,
            "access_ref": "acc_O",
        }
    ]

    # Per-iteration AIC <-> AIV workspace bounce bytes (one store + one load per workspace slot).
    slots = max(1, summary.workspace_multibuffer)
    if summary.workspace_bytes > 0:
        ws_loop_bytes = 2 * (summary.workspace_bytes // slots)
    else:
        ws_loop_bytes = summary.workspace_loop_store_bytes + summary.workspace_loop_load_bytes

    l1_used = summary.l1_bytes or (
        summary.num_stages * (tile_bytes_q + tile_bytes_k + tile_bytes_v)
    )
    l0c_used = summary.l0c_bytes or (block_m * max(block_n, D) * 4)

    target_arch = canonical_ascend_arch(artifacts.target_arch)
    diagnostics = list(summary.diagnostics)
    if target_arch != "ascend910b3" and "unvalidated_target_ascend" not in diagnostics:
        diagnostics.append("unvalidated_target_ascend")

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
            "config": {
                **dict(config),
                "Z": Z,
                "H": H,
                "H_kv": H_kv,
                "N_CTX": N_CTX,
                "HEAD_DIM": D,
                "causal": causal,
            },
        },
        "launch": {
            "grid": grid,
            "cluster_shape": [1, 1, 1],
            "num_warps": 1,
            "warp_size": 32,
            "warp_groups": [{"id": 0, "role": "unified", "num_warps": 1}],
            "dynamic_smem_bytes": 0,
        },
        "resources": {
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
        },
        "tensors": [
            {
                "id": "Q",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H, N_CTX, D],
                "strides": list(sq),
                "base_addr": base_q,
                "cache_policy": "default",
            },
            {
                "id": "K",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H_kv, N_CTX, D],
                "strides": list(sk),
                "base_addr": base_k,
                "cache_policy": "default",
            },
            {
                "id": "V",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H_kv, N_CTX, D],
                "strides": list(sv),
                "base_addr": base_v,
                "cache_policy": "default",
            },
            {
                "id": "Out",
                "dtype": dtype,
                "element_bits": elem_bits,
                "shape": [Z, H, N_CTX, D],
                "strides": list(so),
                "base_addr": base_o,
                "cache_policy": "default",
            },
        ],
        "body": {
            "outer_loop": outer_loop,
            "prologue": {"ops": prologue_ops},
            "pipeline_loop": {
                "trip_count_mode": pipeline_trip_mode,
                "trip_counts": pipeline_trip_counts,
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
        "unsupported_features": list(summary.unsupported_features),
    }


def extract_ascend_attention_kernel_ir(
    fn: Any,
    Z: int,
    H: int,
    N_CTX: int,
    D: int,
    config: Mapping[str, Any],
    target_arch: str = "ascend910b3",
    dtype: str = "f16",
    *,
    H_kv: int | None = None,
    causal: bool = False,
    sm_scale: float | None = None,
    acc_dtype: str = "f32",
    call_args: tuple[Any, ...] | None = None,
    strides_q: tuple[int, int, int, int] | None = None,
    strides_k: tuple[int, int, int, int] | None = None,
    strides_v: tuple[int, int, int, int] | None = None,
    strides_o: tuple[int, int, int, int] | None = None,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Compile a Triton forward attention kernel offline for Ascend NPU and emit KernelIR v0.1."""
    if hasattr(fn, "fn") and not hasattr(fn, "params"):
        fn = fn.fn

    resolved_h_kv = H if H_kv is None else H_kv
    resolved_scale = (1.0 / math.sqrt(D)) if sm_scale is None else sm_scale
    sq = strides_q or (H * N_CTX * D, N_CTX * D, D, 1)
    sk = strides_k or (resolved_h_kv * N_CTX * D, N_CTX * D, D, 1)
    sv = strides_v or (resolved_h_kv * N_CTX * D, N_CTX * D, D, 1)
    so = strides_o or (H * N_CTX * D, N_CTX * D, D, 1)

    args = (
        call_args
        if call_args is not None
        else _build_ascend_attention_call_args(
            fn,
            Z=Z,
            H=H,
            H_kv=resolved_h_kv,
            N_CTX=N_CTX,
            D=D,
            dtype=dtype,
            strides_q=sq,
            strides_k=sk,
            strides_v=sv,
            strides_o=so,
            sm_scale=resolved_scale,
        )
    )

    # Populate constexpr parameters expected by either validation/attention.py or standard kernels.
    param_names = {p.name for p in fn.params}
    cfg_full: dict[str, Any] = dict(config)
    bm = cfg_full.get("BLOCK_M", cfg_full.get("BLOCK_SIZE_M", cfg_full.get("BM")))
    bn = cfg_full.get("BLOCK_N", cfg_full.get("BLOCK_SIZE_N", cfg_full.get("BN")))
    if bm is not None:
        for alias in ("BM", "BLOCK_M", "BLOCK_SIZE_M"):
            if alias in param_names and alias not in cfg_full:
                cfg_full[alias] = int(bm)
    if bn is not None:
        for alias in ("BN", "BLOCK_N", "BLOCK_SIZE_N"):
            if alias in param_names and alias not in cfg_full:
                cfg_full[alias] = int(bn)
    for alias, val in (
        ("HQ", H),
        ("HK", resolved_h_kv),
        ("SEQ", N_CTX),
        ("D", D),
        ("HEAD_DIM", D),
        ("CAUSAL", causal),
        ("IS_CAUSAL", causal),
        ("SCALE_LOG2", resolved_scale * math.log2(math.e)),
    ):
        if alias in param_names and alias not in cfg_full:
            cfg_full[alias] = val

    signature, constants = _mock_signature_for_fn(fn, args, cfg_full)
    num_stages = int(cfg_full.get("num_stages", 2))
    raw_mb = cfg_full.get("multibuffer")
    multibuffer = None if raw_mb is None else bool(raw_mb)

    cache_root = Path(cache_dir) if cache_dir is not None else default_aot_cache_dir()
    ir_key_payload = {
        "v": ASCEND_ATTN_KERNEL_IR_CACHE_VERSION,
        "arch": canonical_ascend_arch(target_arch),
        "shape": [Z, H, resolved_h_kv, N_CTX, D, causal],
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
    ir_cache_file = cache_root / "ascend_attn_ir" / f"{ir_key}.json"
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
    ir_dict = assemble_ascend_attention_kernel_ir(
        artifacts,
        Z=Z,
        H=H,
        H_kv=resolved_h_kv,
        N_CTX=N_CTX,
        D=D,
        config=config,
        dtype=dtype,
        acc_dtype=acc_dtype,
        causal=causal,
        strides_q=sq,
        strides_k=sk,
        strides_v=sv,
        strides_o=so,
    )
    if use_cache:
        try:
            ir_cache_file.parent.mkdir(parents=True, exist_ok=True)
            ir_cache_file.write_text(json.dumps(ir_dict), encoding="utf-8")
        except OSError:
            pass
    return ir_dict
