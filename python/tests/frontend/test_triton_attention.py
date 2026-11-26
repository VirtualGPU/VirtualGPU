"""Tests for the Triton forward fused-attention frontend extractor."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import triton
import triton.language as tl
from virtualgpu.frontend.triton_attention import extract_attention_kernel_ir
from virtualgpu.frontend.ttgir import parse_ttgir_summary

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "kernel_ir.schema.json"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@triton.jit
def _fused_attention_fwd(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        k_ptrs = (
            K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        )
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
        else:
            qk = qk * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v_ptrs = (
            V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        )
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


@pytest.mark.parametrize(
    ("head_dim", "block_m", "block_n", "num_stages", "expected_prefetch"),
    [
        (64, 64, 64, 1, 0),
        (64, 64, 64, 3, 1),
        (128, 64, 64, 3, 1),
        (128, 128, 64, 3, 1),
    ],
)
def test_non_causal_attention_extracts_valid_kernel_ir(
    head_dim: int,
    block_m: int,
    block_n: int,
    num_stages: int,
    expected_prefetch: int,
) -> None:
    schema = _load_schema()
    Z, H, N_CTX = 2, 4, 512
    ir = extract_attention_kernel_ir(
        _fused_attention_fwd,
        Z=Z,
        H=H,
        N_CTX=N_CTX,
        D=head_dim,
        config={"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 4, "num_stages": num_stages},
        target_arch="sm_89",
        dtype="f16",
        causal=False,
    )
    jsonschema.validate(instance=ir, schema=schema)

    assert "linear_access_fallback" not in ir["diagnostics"]
    assert ir["unsupported_features"] == []

    grid_m = N_CTX // block_m
    assert ir["launch"]["grid"] == [grid_m, Z * H, 1]

    # Prologue loads Q and stages it into SMEM
    pro_ops = {op["op_id"]: op for op in ir["body"]["prologue"]["ops"]}
    assert "load_Q" in pro_ops
    assert "smem_store_Q" in pro_ops
    assert pro_ops["smem_store_Q"]["bytes_per_cta"] == block_m * head_dim * 2

    # Pipeline loop: uniform trip count, two MMA ops (QK and PV), SFU + ALU, Q+K+V SMEM load
    pl = ir["body"]["pipeline_loop"]
    assert pl["trip_count_mode"] == "uniform"
    assert pl["trip_counts"] == N_CTX // block_n
    assert pl["prefetch_distance"] == expected_prefetch

    pl_ops = {op["op_id"]: op for op in pl["ops"]}
    assert "mma_qk" in pl_ops and "mma_pv" in pl_ops
    assert pl_ops["mma_qk"]["tile_mnk"] == [block_m, block_n, head_dim]
    assert pl_ops["mma_pv"]["tile_mnk"] == [block_m, head_dim, block_n]
    assert pl_ops["sfu_softmax"]["kind"] == "sfu"
    assert pl_ops["sfu_softmax"]["count"] > 0
    assert pl_ops["shfl_softmax"]["kind"] == "shfl"
    assert pl_ops["shfl_softmax"]["count"] > 0
    assert pl_ops["alu_softmax"]["kind"] == "alu"
    assert pl_ops["alu_softmax"]["count"] > 0
    # TTGIR local_load sum is at least K + V (and includes Q reload + any layout conversion in SMEM)
    assert pl_ops["smem_load_QKV"]["bytes_per_cta"] >= 2 * block_n * head_dim * 2
    if num_stages >= 3:
        assert pl_ops["smem_load_QKV"]["bytes_per_cta"] >= (block_m + 2 * block_n) * head_dim * 2

    # Verify access_map: CTAs in the same head share K/V base_byte_offset (L2 reuse),
    # while Q and Out advance by block_m * D * 2 per Q-block.
    q_entries = ir["access_map"]["acc_Q"]["entries"]
    k_entries = ir["access_map"]["acc_K"]["entries"]
    v_entries = ir["access_map"]["acc_V"]["entries"]
    o_entries = ir["access_map"]["acc_O"]["entries"]
    assert len(q_entries) == grid_m * Z * H
    assert q_entries[1][2] - q_entries[0][2] == block_m * head_dim * 2
    assert o_entries[1][2] - o_entries[0][2] == block_m * head_dim * 2
    assert k_entries[0][2] == k_entries[1][2]
    assert v_entries[0][2] == v_entries[1][2]
    assert k_entries[0][6] == block_n * head_dim * 2
    assert v_entries[0][6] == block_n * head_dim * 2


def test_causal_attention_recovers_per_cta_trip_counts_and_access_map() -> None:
    schema = _load_schema()
    Z, H, N_CTX, D = 1, 2, 256, 64
    block_m, block_n = 64, 64
    ir = extract_attention_kernel_ir(
        _fused_attention_fwd,
        Z=Z,
        H=H,
        N_CTX=N_CTX,
        D=D,
        config={"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 4, "num_stages": 3},
        target_arch="sm_89",
        dtype="f16",
        causal=True,
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert "linear_access_fallback" not in ir["diagnostics"]
    pl = ir["body"]["pipeline_loop"]
    assert pl["trip_count_mode"] == "per_cta"
    # 4 Q-blocks x 2 heads => [1, 2, 3, 4] repeated for each head
    assert pl["trip_counts"] == [1, 2, 3, 4, 1, 2, 3, 4]


@triton.jit
def _two_loop_causal_attention_fwd(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # Loop 1: off-diagonal unmasked blocks [0, start_m * BLOCK_M)
    for start_n in range(0, start_m * BLOCK_M, BLOCK_N):
        k = tl.load(
            K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(
            V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    # Loop 2: diagonal masked blocks [start_m * BLOCK_M, (start_m + 1) * BLOCK_M)
    for start_n in range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N):
        k = tl.load(
            K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        qk = qk + tl.where(mask, 0.0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(
            V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(
        Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(tl.float16),
    )


def test_two_loop_causal_attention_sums_per_cta_trip_counts() -> None:
    schema = _load_schema()
    Z, H, N_CTX, D = 1, 2, 512, 64
    block_m, block_n = 128, 64
    ir = extract_attention_kernel_ir(
        _two_loop_causal_attention_fwd,
        Z=Z,
        H=H,
        N_CTX=N_CTX,
        D=D,
        config={"BLOCK_M": block_m, "BLOCK_N": block_n, "num_warps": 4, "num_stages": 3},
        target_arch="sm_89",
        dtype="f16",
        causal=True,
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert "linear_access_fallback" not in ir["diagnostics"]
    pl = ir["body"]["pipeline_loop"]
    assert pl["trip_count_mode"] == "per_cta"
    assert pl["prefetch_distance"] == 1
    # 4 Q-blocks (BM=128, BN=64 => 2 KV blocks per Q-block) x 2 heads => [2, 4, 6, 8] x 2
    assert pl["trip_counts"] == [2, 4, 6, 8, 2, 4, 6, 8]


def test_split_async_wait_flagged_unsupported() -> None:
    fake_ttgir = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %k = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>\n"
        "  scf.for %i = %c0 to %c8 step %c1 {\n"
        "    %w1 = ttg.async_wait %tk {num = 2 : i32}\n"
        "    %w2 = ttg.async_wait %tv {num = 0 : i32}\n"
        "    %ck = ttg.async_copy_global_to_local %pk, %k : tensor<64x64x!tt.ptr<f16>>\n"
        "    %g1 = ttg.async_commit_group tokens %ck\n"
        "    %g2 = ttg.async_commit_group tokens %ck\n"
        "  }\n"
        "}\n"
    )
    summary = parse_ttgir_summary(fake_ttgir, "", requested_num_stages=3, element_bytes=2)
    assert "split_async_wait_unsupported" in summary.unsupported_features


@triton.jit
def _bm_bn_attention_fwd(
    Q,
    K,
    V,
    Out,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Z,
    H,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )
    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32) + 1.0
    acc = tl.zeros([BM, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    hi = (start_m + 1) * BM if CAUSAL else N_CTX
    for start_n in range(0, hi, BN):
        k = tl.load(
            K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if CAUSAL:
            qk = qk + tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0.0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(
            V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(
        Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(tl.float16),
        mask=offs_m[:, None] < N_CTX,
    )


def test_attention_derives_tile_from_ttgir_and_matches_gemm_toolchain_keys() -> None:
    """Kernels using BM/BN (not BLOCK_M/BLOCK_N) must derive exact grid, bytes, trips, and toolchain keys."""
    schema = _load_schema()
    ir_m128 = extract_attention_kernel_ir(
        _bm_bn_attention_fwd,
        Z=1,
        H=4,
        N_CTX=512,
        D=64,
        config={"BM": 128, "BN": 64, "num_warps": 4, "num_stages": 2},
        target_arch="sm_89",
        dtype="f16",
        causal=True,
    )
    jsonschema.validate(instance=ir_m128, schema=schema)
    assert ir_m128["launch"]["grid"] == [4, 4, 1]
    assert ir_m128["body"]["pipeline_loop"]["trip_counts"][:4] == [2, 4, 6, 8]

    # Toolchain keys must match triton_gemm / validation AOT-JIT verifier contract
    tc = ir_m128["metadata"]["toolchain"]
    assert tc["compiler"] == "triton"
    assert isinstance(tc["compiler_version"], str) and tc["compiler_version"]
    assert tc["target_sm_compiled"] == "sm_89"
    assert "ptx_sha256" in tc and "ttgir_sha256" in tc and "artifact_hash_version" in tc

    ir_n32 = extract_attention_kernel_ir(
        _bm_bn_attention_fwd,
        Z=1,
        H=4,
        N_CTX=512,
        D=64,
        config={"BM": 64, "BN": 32, "num_warps": 4, "num_stages": 2},
        target_arch="sm_89",
        dtype="f16",
        causal=True,
    )
    jsonschema.validate(instance=ir_n32, schema=schema)
    assert ir_n32["launch"]["grid"] == [8, 4, 1]
    pl_ops = {op["op_id"]: op for op in ir_n32["body"]["pipeline_loop"]["ops"]}
    assert pl_ops["load_K"]["nominal_bytes_per_cta"] == 32 * 64 * 2
    assert pl_ops["load_V"]["nominal_bytes_per_cta"] == 32 * 64 * 2


@pytest.mark.parametrize(
    ("bm", "num_warps", "expected_mma", "expect_override", "expect_smem_store"),
    [
        (64, 4, 256, False, True),
        (64, 8, 384, True, True),
        (128, 4, 512, False, False),
        (128, 8, 512, False, False),
    ],
)
def test_attention_ptx_ground_truth_mma_barriers_and_smem(
    bm: int,
    num_warps: int,
    expected_mma: int,
    expect_override: bool,
    expect_smem_store: bool,
) -> None:
    """BM=64/128 x 4/8 warps: MMA, barrier, and smem counts must match PTX loop ground truth."""
    schema = _load_schema()
    ir = extract_attention_kernel_ir(
        _bm_bn_attention_fwd,
        Z=2,
        H=4,
        N_CTX=512,
        D=128,
        config={"BM": bm, "BN": 32, "num_warps": num_warps, "num_stages": 3},
        target_arch="sm_89",
        dtype="f16",
        causal=False,
    )
    jsonschema.validate(instance=ir, schema=schema)
    ops = {op["op_id"]: op for op in ir["body"]["pipeline_loop"]["ops"]}
    assert ops["mma_qk"]["count"] + ops["mma_pv"]["count"] == expected_mma
    override_diags = [d for d in ir["diagnostics"] if d.startswith("mma_count_ptx_override:")]
    assert bool(override_diags) is expect_override
    if expect_override:
        assert override_diags == ["mma_count_ptx_override:ptx=384,tile=256"]
    if expect_smem_store:
        assert ops["sync_loop"]["count"] >= 6
        assert "smem_store_loop" in ops and ops["smem_store_loop"]["bytes_per_cta"] > 0
    else:
        assert ops["sync_loop"]["count"] == 2
        assert "smem_store_loop" not in ops
