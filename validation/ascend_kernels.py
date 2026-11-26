"""Module-level Triton kernels used by the Ascend 910B3 holdout protocol."""

import triton
import triton.language as tl


@triton.jit
def dense_gemm(
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


@triton.jit
def read_flush(Data, Sink, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(Data + offs)
    total = tl.sum(vals)
    if total == -1234567:
        tl.store(Sink + pid, total)
