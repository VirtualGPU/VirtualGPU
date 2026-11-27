"""The shared memory bytes of ldmatrix in the PTX loop, per warp times the warps."""

import pytest
import triton
import triton.language as tl
from virtualgpu.frontend.ttgir import _extract_ptx_loop_ldmatrix_bytes

LOOP = """
$L__BB0_2:
	ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r1}, [%r10];
	ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r2, %r3}, [%r11];
	ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r4, %r5, %r6, %r7}, [%r12];
	ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r4, %r5, %r6, %r7}, [%r13];
	ld.shared.v4.b32 {%r20, %r21, %r22, %r23}, [%r14];
	mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1, %f2, %f3, %f4}, {%r4, %r5, %r6, %r7}, {%r2, %r3}, {%f1, %f2, %f3, %f4};
	@%p1 bra $L__BB0_2;
"""


@pytest.mark.parametrize("warps", [1, 4, 8])
def test_every_width_counts_per_warp_times_the_warps(warps):
    # 128 + 256 + 512 + 512 bytes per warp; ld.shared is counted elsewhere.
    assert _extract_ptx_loop_ldmatrix_bytes(LOOP, warps) == 1408 * warps


def test_a_loop_without_ldmatrix_counts_nothing():
    assert _extract_ptx_loop_ldmatrix_bytes("ld.shared.v4.b32 {%r1, %r2, %r3, %r4}, [%r5];", 8) == 0
    assert _extract_ptx_loop_ldmatrix_bytes("", 8) == 0


@triton.jit
def _gemm(
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
    rows = (pid // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    cols = (pid % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(tl.cdiv(K, BK)):
        inner = step * BK + offsets
        a = tl.load(A + rows[:, None] * K + inner[None, :])
        b = tl.load(B + inner[:, None] * N + cols[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.float16))


def test_gemm_and_attention_ir_carry_both_counts():
    """Compiled offline: the smem_load bytes are the ldmatrix bytes plus ld.shared, and the IR keeps
    the TTGIR tile bytes next to them."""
    from virtualgpu.frontend import MockTensor, extract_attention_kernel_ir, extract_gemm_kernel_ir

    from validation.attention import fused_attention

    f16 = tuple(MockTensor("f16") for _ in range(4))
    gemm = extract_gemm_kernel_ir(
        _gemm,
        M=1024,
        N=1024,
        K=1024,
        config=dict(BM=128, BN=128, BK=64, GROUP_M=8, num_warps=4, num_stages=3),
        target_arch="sm_89",
        call_args=(*f16[:3], 1024, 1024, 1024),
        use_cache=False,
    )
    hq, hk, seq, dim = 16, 16, 1024, 128
    attention = extract_attention_kernel_ir(
        fused_attention,
        1,
        hq,
        seq,
        dim,
        dict(num_warps=8, num_stages=2, BM=128, BN=64, SCALE_LOG2=0.1275),
        "sm_89",
        H_kv=hk,
        causal=False,
        call_args=(*f16, MockTensor("f32"), hq, hk, seq, dim),
        use_cache=False,
    )
    for ir, tile_bytes in ((gemm, (128 * 64 + 64 * 128) * 2), (attention, None)):
        (op,) = [o for o in ir["body"]["pipeline_loop"]["ops"] if o["kind"] == "smem_load"]
        assert op["ptx_ldmatrix_bytes_per_cta"] > op["ttgir_tile_bytes_per_cta"] > 0
        assert op["bytes_per_cta"] >= op["ptx_ldmatrix_bytes_per_cta"]
        if tile_bytes is not None:
            assert op["ttgir_tile_bytes_per_cta"] == tile_bytes
