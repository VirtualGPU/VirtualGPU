"""Offline CPU-only tests for the Triton frontend and KernelIR v0.1 extraction."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import jsonschema
import pytest
import triton
import triton.language as tl
from virtualgpu.frontend import (
    ARTIFACT_HASH_VERSION,
    MockTensor,
    artifact_hashes,
    compile_triton_aot,
    extract_gemm_kernel_ir,
    hash_ptx,
    hash_ttgir,
    parse_ptxas_verbose,
    specialize_triton_call,
)
from virtualgpu.frontend.ttgir import _extract_loop_body, parse_ttgir_summary

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "kernel_ir.schema.json"


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _matmul_tma_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N]
    )
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, k * BLOCK_SIZE_K])
        b = b_desc.load([k * BLOCK_SIZE_K, pid_n * BLOCK_SIZE_N])
        acc = tl.dot(a, b, acc)
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], acc.to(tl.float16))


@triton.jit
def _nonstandard_gemm(
    lhs,
    rhs,
    out,
    rows,
    cols,
    inner,
    l_s0,
    l_s1,
    r_s0,
    r_s1,
    o_s0,
    o_s1,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(inner, BK)):
        kk = k * BK + rk
        a = tl.load(
            lhs + rm[:, None] * l_s0 + kk[None, :] * l_s1,
            (rm[:, None] < rows) & (kk[None, :] < inner),
            other=0.0,
        )
        b = tl.load(
            rhs + kk[:, None] * r_s0 + rn[None, :] * r_s1,
            (kk[:, None] < inner) & (rn[None, :] < cols),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        out + rm[:, None] * o_s0 + rn[None, :] * o_s1,
        acc.to(tl.float16),
        (rm[:, None] < rows) & (rn[None, :] < cols),
    )


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_parse_ptxas_verbose_zero_and_ptxas_only_spill() -> None:
    clean = (
        "ptxas info    : 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info    : Used 128 registers, used 1 barriers, 416 bytes cmem[0]\n"
    )
    r_clean = parse_ptxas_verbose(
        clean, ptx="", assembler="ptxas", assembler_version="12.9", target_sm="sm_89"
    )
    assert r_clean.registers_per_thread == 128
    assert r_clean.barriers == 1
    assert r_clean.spill_scope == "none"
    assert not r_clean.ptxas_only_spill

    spilled = (
        "ptxas info    : 320 bytes stack frame, 328 bytes spill stores, 316 bytes spill loads\n"
        "ptxas info    : Used 255 registers, used 1 barriers, 320 bytes cumulative stack size\n"
    )
    r_spill = parse_ptxas_verbose(
        spilled, ptx="", assembler="ptxas", assembler_version="12.9", target_sm="sm_89"
    )
    assert r_spill.stack_frame_bytes == 320
    assert r_spill.spill_stores_bytes == 328
    assert r_spill.spill_loads_bytes == 316
    assert r_spill.spill_scope == "inner_loop_assumed"
    assert r_spill.ptxas_only_spill


@pytest.mark.parametrize("target_arch", ["sm_89", "sm_121"])
def test_extract_gemm_validates_against_schema(target_arch: str) -> None:
    schema = _load_schema()
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    ir = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=1024,
        N=1024,
        K=512,
        config=config,
        target_arch=target_arch,
    )
    jsonschema.validate(instance=ir, schema=schema)

    assert ir["metadata"]["target_arch"] == target_arch
    assert ir["launch"]["grid"] == [64, 1, 1]
    assert ir["resources"]["registers_per_thread"] > 0
    assert ir["resources"]["static_smem_bytes"] == 32768
    assert ir["body"]["prologue"]["ops"] == []
    assert ir["body"]["pipeline_loop"]["async_copy"] == "async_element"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 2
    assert ir["body"]["pipeline_loop"]["prefetch_distance"] == 1
    assert ir["unsupported_features"] == []
    assert ir["metadata"]["toolchain"]["enable_fp_fusion"] is True
    assert len(ir["metadata"]["toolchain"]["ptx_sha256"]) == 64
    assert len(ir["metadata"]["toolchain"]["ttgir_sha256"]) == 64

    mma_ops = [op for op in ir["body"]["pipeline_loop"]["ops"] if op["kind"] == "mma"]
    assert len(mma_ops) == 1
    assert mma_ops[0]["op_class"] == "mma.sync"
    assert mma_ops[0]["inst_mnk"] == [16, 8, 16]
    assert mma_ops[0]["count"] == 256


def test_extract_tma_gemm_sm121_uses_async_tensor() -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _matmul_tma_kernel,
        M=1024,
        N=1024,
        K=512,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_121",
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert ir["body"]["pipeline_loop"]["async_copy"] == "async_tensor"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 2
    assert ir["body"]["pipeline_loop"]["prefetch_distance"] == 1
    assert ir["unsupported_features"] == []


@pytest.mark.parametrize(
    ("req_stages", "expected_async", "expected_buffers", "expected_prefetch"),
    [
        (1, "none", 1, 0),
        (2, "async_element", 1, 0),
        (3, "async_element", 2, 1),
        (4, "async_element", 3, 2),
    ],
)
def test_cp_async_prefetch_distance_from_ptx_wait_group(
    req_stages: int,
    expected_async: str,
    expected_buffers: int,
    expected_prefetch: int,
) -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=1024,
        N=1024,
        K=512,
        config={
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": req_stages,
        },
        target_arch="sm_89",
    )
    jsonschema.validate(instance=ir, schema=schema)
    pl = ir["body"]["pipeline_loop"]
    assert pl["async_copy"] == expected_async
    assert pl["num_stages"] == expected_buffers
    assert pl["prefetch_distance"] == expected_prefetch
    assert ir["unsupported_features"] == []


def test_unaligned_shape_and_unpipelined_fallback_diagnostic() -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=300,
        N=300,
        K=100,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_89",
        # Misaligned pointer (not 16B aligned) causes Triton's runtime binder to
        # omit tt.divisibility=16, matching real runtime JIT fallback behavior.
        ptr_offset_a=0x10002,
        ptr_offset_b=0x20002,
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert ir["body"]["pipeline_loop"]["async_copy"] == "none"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 1
    assert "unpipelined_fallback" in ir["diagnostics"]
    # Tail tiles should be clamped and marked masked.
    last_a = ir["access_map"]["acc_A"]["entries"][-1]
    assert last_a[3] == 44  # 300 - 2*128
    assert last_a[7] is True


def test_runtime_specialization_parity_and_artifact_hashing() -> None:
    # Aligned pointers & 16-divisible dims vs misaligned pointer & non-unit stride
    args_aligned = (
        MockTensor("f16", ptr=0x10000),
        MockTensor("f16", ptr=0x20000),
        MockTensor("f16", ptr=0x30000),
        1024,
        1024,
        512,
        512,
        1,
        1024,
        1,
        1024,
        1,
    )
    kwargs = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }
    _, sig_a, consts_a, attrs_a = specialize_triton_call(
        _matmul_kernel, "sm_89", args_aligned, kwargs
    )
    assert sig_a["stride_ak"] == "constexpr"
    assert consts_a[(7,)] == 1
    assert ["tt.divisibility", 16] in attrs_a[(0,)]
    assert ["tt.divisibility", 16] in attrs_a[(3,)]

    args_unaligned = (
        MockTensor("f16", ptr=0x10002),
        MockTensor("f16", ptr=0x20000),
        MockTensor("f16", ptr=0x30000),
        300,
        300,
        100,
        100,
        2,
        300,
        1,
        300,
        1,
    )
    _, sig_u, _, attrs_u = specialize_triton_call(_matmul_kernel, "sm_89", args_unaligned, kwargs)
    assert sig_u["stride_ak"] == "i32"
    assert attrs_u[(0,)] == []
    assert attrs_u[(3,)] == []

    # Normalized hashes ignore comments, tab-separated .loc/.file, and debug sections
    ptx1 = (
        "// comment a\n"
        '.file\t1 "/path/a.py"\n'
        ".loc\t1 10 0\n"
        ".section .debug_info {\n"
        "  .b8 1, 2, 3\n"
        "}\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c;\n"
    )
    ptx2 = (
        "// different path\n"
        '.file\t1 "/other/b.py"\n'
        ".loc\t2 99 0\n"
        "  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32   d, a, b, c; // trailing\n"
    )
    assert hash_ptx(ptx1) == hash_ptx(ptx2)

    ttgir1 = (
        '#loc = loc("/a/b.py":1:1)\n'
        "#loc1 = loc(callsite(#loc at #loc))\n"
        'tt.func public @k(%a: !tt.ptr<f16> loc("a"(#loc))) {\n'
        "  %0 = tt.dot %a, %b, %c : tensor<128x32xf16> loc(callsite(#loc1 at #loc))\n"
        "}\n"
    )
    ttgir2 = (
        '#loc = loc("/other/path.py":99:5)\n'
        "#loc1 = loc(callsite(#loc at #loc))\n"
        'tt.func public @k(%a: !tt.ptr<f16> loc("a"(#loc))) {\n'
        "  %0 = tt.dot %a, %b, %c : tensor<128x32xf16> loc(#loc1)\n"
        "}\n"
    )
    assert hash_ttgir(ttgir1) == hash_ttgir(ttgir2)
    bundle = artifact_hashes(ptx1, ttgir1)
    assert bundle["artifact_hash_version"] == ARTIFACT_HASH_VERSION
    assert bundle["ptx_sha256"] == hash_ptx(ptx1)
    assert bundle["ttgir_sha256"] == hash_ttgir(ttgir1)


def test_nonstandard_param_names_use_triton_binder() -> None:
    schema = _load_schema()
    call_args = (
        MockTensor("f16", ptr=0x10000),
        MockTensor("f16", ptr=0x200000),
        MockTensor("f16", ptr=0x400000),
        1024,
        1024,
        512,
        512,
        1,
        1024,
        1,
        1024,
        1,
    )
    ir = extract_gemm_kernel_ir(
        _nonstandard_gemm,
        M=1024,
        N=1024,
        K=512,
        config={"BM": 128, "BN": 128, "BK": 32, "num_warps": 4, "num_stages": 3},
        target_arch="sm_89",
        call_args=call_args,
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert ir["body"]["pipeline_loop"]["async_copy"] == "async_element"
    assert ir["body"]["pipeline_loop"]["num_stages"] == 2
    assert ir["metadata"]["toolchain"]["artifact_hash_version"] == ARTIFACT_HASH_VERSION
    assert "linear_access_fallback" not in ir["diagnostics"]
    assert ir["launch"]["grid"] == [8, 8, 1]


def test_scalar_evaluator_recovers_group_size_m_swizzle() -> None:
    schema = _load_schema()
    # 1024x1024 with 128x128 tiles -> 8x8 tile grid.
    # With GROUP_SIZE_M = 4, first 4 CTAs traverse M=0..3 at N=0, then M=0..3 at N=1.
    ir_g4 = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=1024,
        N=1024,
        K=512,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_89",
    )
    jsonschema.validate(instance=ir_g4, schema=schema)
    assert "linear_access_fallback" not in ir_g4["diagnostics"]
    coords_g4 = [(e[0], e[1]) for e in ir_g4["access_map"]["acc_A"]["entries"][:10]]
    assert coords_g4 == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (0, 2),
        (1, 2),
    ]

    # With GROUP_SIZE_M = 1, traversal is row-major: (0,0), (0,1), (0,2), ...
    ir_g1 = extract_gemm_kernel_ir(
        _matmul_kernel,
        M=1024,
        N=1024,
        K=512,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_89",
    )
    jsonschema.validate(instance=ir_g1, schema=schema)
    assert "linear_access_fallback" not in ir_g1["diagnostics"]
    coords_g1 = [(e[0], e[1]) for e in ir_g1["access_map"]["acc_A"]["entries"][:10]]
    assert coords_g1 == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 0),
        (1, 1),
    ]


def test_ptx_and_ttgir_hash_invariant_across_file_paths_and_line_numbers() -> None:
    kernel_src = (
        "import triton\n"
        "import triton.language as tl\n\n"
        "@triton.jit\n"
        "def same_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,\n"
        "              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    grid_m = tl.cdiv(M, BM)\n"
        "    grid_n = tl.cdiv(N, BN)\n"
        "    group = pid // (GROUP_M * grid_n)\n"
        "    first_m = group * GROUP_M\n"
        "    group_size = tl.minimum(grid_m - first_m, GROUP_M)\n"
        "    local_id = pid % (GROUP_M * grid_n)\n"
        "    rows = (first_m + local_id % group_size) * BM + tl.arange(0, BM)\n"
        "    cols = (local_id // group_size) * BN + tl.arange(0, BN)\n"
        "    offsets = tl.arange(0, BK)\n"
        "    acc = tl.full((BM, BN), 0, tl.float32)\n"
        "    for step in range(tl.cdiv(K, BK)):\n"
        "        inner = step * BK + offsets\n"
        "        a = tl.load(A + rows[:, None] * K + inner[None, :], (rows[:, None] < M) & (inner[None, :] < K), other=0)\n"
        "        b = tl.load(B + inner[:, None] * N + cols[None, :], (inner[:, None] < K) & (cols[None, :] < N), other=0)\n"
        "        acc = tl.dot(a, b, acc)\n"
        "    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.float16), (rows[:, None] < M) & (cols[None, :] < N))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "dir_a" / "k1.py"
        p2 = Path(td) / "dir_b" / "k2.py"
        p1.parent.mkdir()
        p2.parent.mkdir()
        p1.write_text(kernel_src, encoding="utf-8")
        p2.write_text("\n\n\n\n" + kernel_src, encoding="utf-8")

        def _load(path: Path):
            spec = importlib.util.spec_from_file_location(f"_mod_{path.stem}", path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.same_gemm

        cfg = {
            "M": 1024,
            "N": 1024,
            "K": 512,
            "BM": 128,
            "BN": 128,
            "BK": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
        args = (
            MockTensor("f16", 0x10000),
            MockTensor("f16", 0x200000),
            MockTensor("f16", 0x400000),
        )
        ir1 = extract_gemm_kernel_ir(
            _load(p1), M=1024, N=1024, K=512, config=cfg, target_arch="sm_89", call_args=args
        )
        ir2 = extract_gemm_kernel_ir(
            _load(p2), M=1024, N=1024, K=512, config=cfg, target_arch="sm_89", call_args=args
        )
        assert (
            ir1["metadata"]["toolchain"]["ptx_sha256"] == ir2["metadata"]["toolchain"]["ptx_sha256"]
        )
        assert (
            ir1["metadata"]["toolchain"]["ttgir_sha256"]
            == ir2["metadata"]["toolchain"]["ttgir_sha256"]
        )


@triton.jit
def _persistent_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            kk = k * BLOCK_SIZE_K + offs_k
            a = tl.load(
                a_ptr + offs_am[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=kk[None, :] < K,
                other=0.0,
            )
            b = tl.load(
                b_ptr + kk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                mask=kk[:, None] < K,
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
        tl.store(
            c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, acc.to(tl.float16)
        )


@triton.jit
def _three_input_gemm(
    a0_ptr,
    a1_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a0_ptrs = a0_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a1_ptrs = a1_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        km = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a0 = tl.load(a0_ptrs, mask=km, other=0.0)
        a1 = tl.load(a1_ptrs, mask=km, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a0, b, acc)
        acc = tl.dot(a1, b, acc)
        a0_ptrs += BLOCK_SIZE_K * stride_ak
        a1_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    tl.store(
        c_ptr
        + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None] * stride_cm
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn,
        acc.to(tl.float16),
    )


def test_three_input_kernel_derives_prefetch_by_commit_group_count() -> None:
    """Verify prefetch_distance divides wait_group N by commits per stage (3, not hardcoded 2)."""
    args = (
        MockTensor("f16", 0x10000),
        MockTensor("f16", 0x210000),
        MockTensor("f16", 0x410000),
        MockTensor("f16", 0x610000),
        1024,
        1024,
        512,
        512,
        1,
        1024,
        1,
        1024,
        1,
    )
    for req_stages, expected_prefetch in ((2, 0), (3, 1), (4, 2)):
        kw = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "num_warps": 4,
            "num_stages": req_stages,
        }
        opts, sig, consts, attrs = specialize_triton_call(_three_input_gemm, "sm_89", args, kw)
        art = compile_triton_aot(
            _three_input_gemm, sig, consts, "sm_89", 4, req_stages, attrs=attrs, options=opts
        )
        loop_body = _extract_loop_body(art.ttgir)
        assert loop_body.count("ttg.async_commit_group") == 3
        summary = parse_ttgir_summary(
            art.ttgir, art.ptx, requested_num_stages=req_stages, element_bytes=2
        )
        assert summary.async_copy == "async_element"
        assert summary.num_stages == req_stages - 1
        assert summary.prefetch_distance == expected_prefetch
        assert "prefetch_distance_mismatch" not in summary.diagnostics
        assert summary.unsupported_features == []


def test_prefetch_derivation_diagnostics_and_unsupported_on_unresolved() -> None:
    # Synthetic mismatch between PTX wait_group and TTGIR async_wait
    fake_ttgir = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>\n"
        "  scf.for %i = %c0 to %c10 step %c1 {\n"
        "    %w = ttg.async_wait %t {num = 4 : i32}\n"
        "    %c = ttg.async_copy_global_to_local %p, %a : tensor<64x32x!tt.ptr<f16>>\n"
        "    %g1 = ttg.async_commit_group tokens %c\n"
        "    %g2 = ttg.async_commit_group tokens %c\n"
        "  }\n"
        "}\n"
    )
    fake_ptx = (
        "$L__loop:\n"
        "  cp.async.wait_group 2;\n"
        "  cp.async.commit_group;\n"
        "  cp.async.commit_group;\n"
        "  bra $L__loop;\n"
    )
    s_mis = parse_ttgir_summary(fake_ttgir, fake_ptx, requested_num_stages=3, element_bytes=2)
    assert s_mis.prefetch_distance == 1
    assert "prefetch_distance_mismatch" in s_mis.diagnostics

    # Unresolvable cp.async wait/commit must mark unsupported_features rather than guessing
    unresolved_cp = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>\n"
        "  scf.for %i = %c0 to %c10 step %c1 {\n"
        "    %c = ttg.async_copy_global_to_local %p, %a : tensor<64x32x!tt.ptr<f16>>\n"
        "  }\n"
        "}\n"
    )
    s_unres = parse_ttgir_summary(unresolved_cp, "", requested_num_stages=3, element_bytes=2)
    assert "prefetch_distance_unresolved" in s_unres.unsupported_features

    # TMA IR missing wait_barrier and PTX try_wait must mark unsupported_features
    fake_tma = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %a = ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16, #shared, #smem, mutable>\n"
        "  scf.for %i = %c0 to %c10 step %c1 {\n"
        "    ttng.async_tma_copy_global_to_local %d[%x, %y] %a, %b, %p\n"
        "  }\n"
        "}\n"
    )
    s_tma = parse_ttgir_summary(fake_tma, "", requested_num_stages=3, element_bytes=2)
    assert s_tma.prefetch_distance == 0
    assert "tma_prefetch_unresolved" in s_tma.unsupported_features


def test_tma_prefetch_accounts_for_arrive_vs_mma_order() -> None:
    # 1 prologue stage (2 TMA copies before loop):
    # Case A: wait -> MMA -> arrive/issue next TMA => 0 in-flight during MMA
    tma_issue_after_mma_ir = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>\n"
        "  ttng.async_tma_copy_global_to_local %da[%x, %y] %a, %b, %p\n"
        "  ttng.async_tma_copy_global_to_local %db[%x, %y] %a, %b, %p\n"
        "  scf.for %i = %c0 to %c10 step %c1 {\n"
        "    ttng.wait_barrier %b, %ph\n"
        "    %acc = tt.dot %x, %y, %z : tensor<64x32xf16, #op0> * tensor<32x64xf16, #op1> -> tensor<64x64xf32, #mma>\n"
        "    ttng.async_tma_copy_global_to_local %da[%x, %y] %a, %b, %p\n"
        "    ttng.async_tma_copy_global_to_local %db[%x, %y] %a, %b, %p\n"
        "  }\n"
        "}\n"
    )
    tma_issue_after_mma_ptx = (
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r0], 16384;\n"
        "$L__loop:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 complete, [%r0], %r1;\n"
        "  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c;\n"
        "  mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r0], 16384;\n"
        "  bra $L__loop;\n"
    )
    s_after = parse_ttgir_summary(
        tma_issue_after_mma_ir, tma_issue_after_mma_ptx, requested_num_stages=2, element_bytes=2
    )
    assert s_after.prefetch_distance == 0
    assert s_after.unsupported_features == []
    assert "prefetch_distance_mismatch" not in s_after.diagnostics

    # Case B: wait -> arrive/issue next TMA -> MMA => 1 in-flight during MMA
    tma_issue_before_mma_ir = (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
        "  %a = ttg.local_alloc : () -> !ttg.memdesc<1x64x32xf16, #shared, #smem, mutable>\n"
        "  ttng.async_tma_copy_global_to_local %da[%x, %y] %a, %b, %p\n"
        "  ttng.async_tma_copy_global_to_local %db[%x, %y] %a, %b, %p\n"
        "  scf.for %i = %c0 to %c10 step %c1 {\n"
        "    ttng.wait_barrier %b, %ph\n"
        "    ttng.async_tma_copy_global_to_local %da[%x, %y] %a, %b, %p\n"
        "    ttng.async_tma_copy_global_to_local %db[%x, %y] %a, %b, %p\n"
        "    %acc = tt.dot %x, %y, %z : tensor<64x32xf16, #op0> * tensor<32x64xf16, #op1> -> tensor<64x64xf32, #mma>\n"
        "  }\n"
        "}\n"
    )
    tma_issue_before_mma_ptx = (
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r0], 16384;\n"
        "$L__loop:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 complete, [%r0], %r1;\n"
        "  mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r0], 16384;\n"
        "  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 d, a, b, c;\n"
        "  bra $L__loop;\n"
    )
    s_before = parse_ttgir_summary(
        tma_issue_before_mma_ir, tma_issue_before_mma_ptx, requested_num_stages=2, element_bytes=2
    )
    assert s_before.prefetch_distance == 1
    assert s_before.unsupported_features == []
    assert "prefetch_distance_mismatch" not in s_before.diagnostics


def test_persistent_kernel_emits_explicit_diagnostic() -> None:
    schema = _load_schema()
    ir = extract_gemm_kernel_ir(
        _persistent_gemm,
        M=1024,
        N=1024,
        K=512,
        config={
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "NUM_SMS": 46,
            "num_warps": 4,
            "num_stages": 3,
        },
        target_arch="sm_89",
    )
    jsonschema.validate(instance=ir, schema=schema)
    assert "persistent_kernel_unmodeled" in ir["diagnostics"]
    assert "linear_access_fallback" in ir["diagnostics"]


@triton.jit
def _inloop_addr_gemm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * grid_n)
    first_m = group * GROUP_M
    group_size = tl.minimum(grid_m - first_m, GROUP_M)
    local_id = pid % (GROUP_M * grid_n)
    rows = (first_m + local_id % group_size) * BM + tl.arange(0, BM)
    cols = (local_id // group_size) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        inner = step * BK + offsets
        a = tl.load(
            A + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + inner[:, None] * N + cols[None, :],
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc.to(tl.float16),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def test_num_stages_1_2_3_recover_identical_access_map_without_fallback() -> None:
    """num_stages=1 (no software pipelining) must recover the exact same access_map as num_stages=2/3."""
    call_args = (
        MockTensor("f16", 0x10000),
        MockTensor("f16", 0x200000),
        MockTensor("f16", 0x400000),
    )
    irs_inloop = [
        extract_gemm_kernel_ir(
            _inloop_addr_gemm,
            M=256,
            N=256,
            K=256,
            config={
                "M": 256,
                "N": 256,
                "K": 256,
                "BM": 64,
                "BN": 64,
                "BK": 32,
                "GROUP_M": 4,
                "num_warps": 4,
                "num_stages": ns,
            },
            target_arch="sm_89",
            dtype="f16",
            call_args=call_args,
        )
        for ns in (1, 2, 3)
    ]
    for ns, ir in zip((1, 2, 3), irs_inloop):
        assert "linear_access_fallback" not in ir["diagnostics"], (
            f"num_stages={ns} fell back to linear access map"
        )
    assert irs_inloop[0]["access_map"] == irs_inloop[1]["access_map"] == irs_inloop[2]["access_map"]

    # Also verify pointer-iter_args stepping kernels across num_stages=1/2/3
    irs_ptr_step = [
        extract_gemm_kernel_ir(
            _matmul_kernel,
            M=256,
            N=256,
            K=256,
            config={
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
                "num_warps": 4,
                "num_stages": ns,
            },
            target_arch="sm_89",
            dtype="f16",
        )
        for ns in (1, 2, 3)
    ]
    for ns, ir in zip((1, 2, 3), irs_ptr_step):
        assert "linear_access_fallback" not in ir["diagnostics"], (
            f"iter_args num_stages={ns} fell back"
        )
    assert (
        irs_ptr_step[0]["access_map"]
        == irs_ptr_step[1]["access_map"]
        == irs_ptr_step[2]["access_map"]
    )


def test_transfer_width_bytes_from_ptx_aligned_vs_unaligned() -> None:
    """Post-hoc fix: transfer_width_bytes must reflect PTX load/store width (16 aligned vs 2 unaligned)."""
    # Verify MockTensor interns dtype singletons so Triton's native binder never sees reused IDs.
    assert MockTensor("f16").dtype is MockTensor("f16").dtype
    assert MockTensor("f16").dtype is not MockTensor("f32").dtype

    for ns in (1, 3):
        ir_aligned = extract_gemm_kernel_ir(
            _matmul_kernel,
            M=256,
            N=256,
            K=256,
            config={
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
                "num_warps": 4,
                "num_stages": ns,
            },
            target_arch="sm_89",
            dtype="f16",
        )
        pl_a = {op["op_id"]: op for op in ir_aligned["body"]["pipeline_loop"]["ops"]}
        ep_a = {op["op_id"]: op for op in ir_aligned["body"]["epilogue"]["ops"]}
        assert pl_a["load_A"]["transfer_width_bytes"] == 16
        assert pl_a["load_B"]["transfer_width_bytes"] == 16
        assert ep_a["store_C"]["transfer_width_bytes"] == 16

        ir_unaligned = extract_gemm_kernel_ir(
            _matmul_kernel,
            M=255,
            N=255,
            K=253,
            config={
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
                "num_warps": 4,
                "num_stages": ns,
            },
            target_arch="sm_89",
            dtype="f16",
        )
        pl_u = {op["op_id"]: op for op in ir_unaligned["body"]["pipeline_loop"]["ops"]}
        ep_u = {op["op_id"]: op for op in ir_unaligned["body"]["epilogue"]["ops"]}
        assert pl_u["load_A"]["transfer_width_bytes"] == 2
        assert pl_u["load_B"]["transfer_width_bytes"] == 2
        assert ep_u["store_C"]["transfer_width_bytes"] == 2


def test_mock_tensor_dtype_singleton_survives_gc_and_mixed_specialization() -> None:
    """Alternating mixed-dtype MockTensor specializations with gc.collect() must always match exact signatures."""
    import gc

    combos = [
        ("f16", "f32", "bf16", "*fp16", "*fp32", "*bf16"),
        ("f32", "bf16", "f16", "*fp32", "*bf16", "*fp16"),
        ("bf16", "f16", "f32", "*bf16", "*fp16", "*fp32"),
    ]
    kw = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 3,
    }
    for i in range(60):
        da, db, dc, ea, eb, ec = combos[i % len(combos)]
        args = (
            MockTensor(da, 0x10000),
            MockTensor(db, 0x200000),
            MockTensor(dc, 0x400000),
            128,
            128,
            128,
            128,
            1,
            128,
            1,
            128,
            1,
        )
        _, sig, _, _ = specialize_triton_call(_matmul_kernel, "sm_89", args, kw)
        assert sig["a_ptr"] == ea, f"iter {i}: expected a_ptr={ea}, got {sig['a_ptr']}"
        assert sig["b_ptr"] == eb, f"iter {i}: expected b_ptr={eb}, got {sig['b_ptr']}"
        assert sig["c_ptr"] == ec, f"iter {i}: expected c_ptr={ec}, got {sig['c_ptr']}"
        del args
        if i % 3 == 0:
            gc.collect()
