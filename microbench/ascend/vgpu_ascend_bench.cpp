// AscendC microbenchmark suite for VirtualGPU hardware profiling (Ascend 910B / dav-c220).
//
// Build on an Ascend host with CANN 8+/9+:
//   bisheng -xcce microbench/ascend/vgpu_ascend_bench.cpp \
//     --npu-arch=dav-2201 -O3 -std=c++17 \
//     -I${ASCEND_HOME_PATH}/include \
//     -I${ASCEND_HOME_PATH}/tikcpp/tikcfw \
//     -I${ASCEND_HOME_PATH}/tikcpp/tikcfw/interface \
//     -I${ASCEND_HOME_PATH}/tikcpp/tikcfw/impl \
//     -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lruntime -lstdc++ -lm \
//     -o vgpu_ascend_bench
//
// Emits a single JSON object to stdout for microbench/ascend/collect_ascend_profile.py.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t TILE_BYTES = 65536;  // 64 KiB x 2 double buffer = 128 KiB <= 192 KiB UB
constexpr uint32_t TILE_FLOATS = TILE_BYTES / sizeof(float);

// ---------------------------------------------------------------------------
// 1. Dispatch / System Cycle Probe
// ---------------------------------------------------------------------------
__global__ __aicore__ void probe_dispatch_kernel(GM_ADDR out_info) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    uint32_t bid = GetBlockIdx();
    uint32_t bnum = GetBlockNum();
    uint32_t subid = GetSubBlockIdx();
    uint32_t subnum = GetSubBlockNum();
    uint64_t cyc = GetSystemCycle();

    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out_info), 128 * 16);

    TPipe pipe;
    TQue<QuePosition::VECOUT, 1> q;
    pipe.InitBuffer(q, 1, 64);
    LocalTensor<uint64_t> loc = q.AllocTensor<uint64_t>();
    loc.SetValue(0, bid);
    loc.SetValue(1, bnum);
    loc.SetValue(2, subid);
    loc.SetValue(3, subnum);
    loc.SetValue(4, cyc);
    q.EnQue(loc);
    LocalTensor<uint64_t> ready = q.DeQue<uint64_t>();
    if (bid < 128) {
        DataCopy(outGm[bid * 16], ready, 8);
    }
    q.FreeTensor(ready);
#endif
}

// ---------------------------------------------------------------------------
// 2. Empty Kernel & Device Delay Kernel (queued device-side launch overhead)
// ---------------------------------------------------------------------------
__global__ __aicore__ void empty_kernel(GM_ADDR sink) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    (void)sink;
#endif
}

__global__ __aicore__ void delay_kernel(GM_ADDR sink, uint32_t ticks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (GetBlockIdx() != 0) return;
    uint64_t t0 = GetSystemCycle();
    while (GetSystemCycle() - t0 < static_cast<uint64_t>(ticks)) {}
    (void)sink;
#endif
}

// ---------------------------------------------------------------------------
// 3. HBM / L2 Stream Read & Write (MTE2 / MTE3 double-buffered DMA)
// ---------------------------------------------------------------------------
__global__ __aicore__ void stream_read_kernel(GM_ADDR src, GM_ADDR sink,
                                              uint64_t total_floats, uint32_t repeat) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    uint32_t bid = GetBlockIdx();
    uint32_t bnum = GetBlockNum();
    uint32_t subnum = GetSubBlockNum();
    uint32_t workers = bnum * subnum;

    uint64_t total_tiles = total_floats / TILE_FLOATS;
    uint64_t tiles_per_w = total_tiles / workers;
    uint64_t rem = total_tiles % workers;
    uint64_t start_tile = static_cast<uint64_t>(bid) * tiles_per_w + (bid < rem ? bid : rem);
    uint64_t my_tiles = tiles_per_w + (bid < rem ? 1 : 0);

    GlobalTensor<float> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), total_floats);
    GlobalTensor<float> sinkGm;
    sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    pipe.InitBuffer(inQueue, 2, TILE_BYTES);

    float acc = 0.0f;
    for (uint32_t r = 0; r < repeat; ++r) {
        for (uint64_t t = 0; t < my_tiles; ++t) {
            LocalTensor<float> buf = inQueue.AllocTensor<float>();
            DataCopy(buf, srcGm[(start_tile + t) * TILE_FLOATS], TILE_FLOATS);
            inQueue.EnQue(buf);
            LocalTensor<float> out = inQueue.DeQue<float>();
            acc += out.GetValue(0);
            inQueue.FreeTensor(out);
        }
    }
    if (acc == 123456789.0f) {
        sinkGm.SetValue(bid & 63, acc);
    }
#endif
}

__global__ __aicore__ void stream_write_kernel(GM_ADDR dst, uint64_t total_floats,
                                               uint32_t repeat, float val) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    uint32_t bid = GetBlockIdx();
    uint32_t bnum = GetBlockNum();
    uint32_t subnum = GetSubBlockNum();
    uint32_t workers = bnum * subnum;

    uint64_t total_tiles = total_floats / TILE_FLOATS;
    uint64_t tiles_per_w = total_tiles / workers;
    uint64_t rem = total_tiles % workers;
    uint64_t start_tile = static_cast<uint64_t>(bid) * tiles_per_w + (bid < rem ? bid : rem);
    uint64_t my_tiles = tiles_per_w + (bid < rem ? 1 : 0);

    GlobalTensor<float> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst), total_floats);

    TPipe pipe;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(outQueue, 2, TILE_BYTES);

    LocalTensor<float> b0 = outQueue.AllocTensor<float>();
    LocalTensor<float> b1 = outQueue.AllocTensor<float>();
    Duplicate(b0, val, TILE_FLOATS);
    Duplicate(b1, val + 1.0f, TILE_FLOATS);
    outQueue.FreeTensor(b0);
    outQueue.FreeTensor(b1);

    for (uint32_t r = 0; r < repeat; ++r) {
        for (uint64_t t = 0; t < my_tiles; ++t) {
            LocalTensor<float> buf = outQueue.AllocTensor<float>();
            outQueue.EnQue(buf);
            LocalTensor<float> ready = outQueue.DeQue<float>();
            DataCopy(dstGm[(start_tile + t) * TILE_FLOATS], ready, TILE_FLOATS);
            outQueue.FreeTensor(ready);
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// 4. Pointer-Chasing GM Latency (L2 warm vs HBM cold)
// ---------------------------------------------------------------------------
__global__ __aicore__ void ptr_chase_kernel(GM_ADDR table, GM_ADDR sink,
                                            uint64_t table_elems, uint32_t hops) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (GetBlockIdx() != 0) return;

    GlobalTensor<uint32_t> tblGm;
    tblGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table), table_elems);
    GlobalTensor<uint32_t> sinkGm;
    sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sink), 16);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, 64);
    LocalTensor<uint32_t> loc = buf.Get<uint32_t>();

    uint32_t idx = 0;
    for (uint32_t h = 0; h < hops; ++h) {
        // 32-byte (8 x uint32) DMA read per hop; explicit MTE2 <-> S sync ensures strict dependency
        DataCopy(loc, tblGm[static_cast<uint64_t>(idx) * 32], 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        idx = loc.GetValue(0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    }
    if (idx == 0xdeadbeefu) {
        sinkGm.SetValue(0, idx);
    }
#endif
}

// ---------------------------------------------------------------------------
// 5. On-Chip UB Bandwidth & Latency
// ---------------------------------------------------------------------------
__global__ __aicore__ void ub_bw_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    constexpr uint32_t ELEMS = 16384;  // 64 KiB float32
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB;
    pipe.InitBuffer(bufA, ELEMS * sizeof(float));
    pipe.InitBuffer(bufB, ELEMS * sizeof(float));
    LocalTensor<float> a = bufA.Get<float>();
    LocalTensor<float> b = bufB.Get<float>();
    Duplicate(a, 1.25f, ELEMS);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < iters; ++i) {
        DataCopy(b, a, ELEMS);
        DataCopy(a, b, ELEMS);
    }
    PipeBarrier<PIPE_V>();
    float v = a.GetValue(0);
    if (v == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(GetBlockIdx() & 63, v);
    }
#endif
}

// ---------------------------------------------------------------------------
// 6. Cube MMA Throughput (FP16 & BF16 -> FP32, 128x128x64 resident in L0A/L0B/L0C)
// ---------------------------------------------------------------------------
template <typename InT>
__aicore__ inline void run_cube_mmad_bench(GM_ADDR sink, uint32_t iters) {
#ifdef __DAV_C220_CUBE__
    constexpr uint16_t M_TILE = 128;
    constexpr uint16_t N_TILE = 128;
    constexpr uint16_t K_TILE = 64;
    // L0A: 128x64x2 = 16 KiB, L0B: 64x128x2 = 16 KiB, L0C: 128x128x4 = 64 KiB
    TPipe pipe;
    TBuf<TPosition::A2> l0aBuf;
    TBuf<TPosition::B2> l0bBuf;
    TBuf<TPosition::CO1> l0cBuf;
    pipe.InitBuffer(l0aBuf, M_TILE * K_TILE * sizeof(InT));
    pipe.InitBuffer(l0bBuf, K_TILE * N_TILE * sizeof(InT));
    pipe.InitBuffer(l0cBuf, M_TILE * N_TILE * sizeof(float));

    LocalTensor<InT> l0a = l0aBuf.Get<InT>();
    LocalTensor<InT> l0b = l0bBuf.Get<InT>();
    LocalTensor<float> l0c = l0cBuf.Get<float>();

    MmadParams params;
    params.m = M_TILE;
    params.n = N_TILE;
    params.k = K_TILE;
    params.cmatrixInitVal = true;
    Mmad(l0c, l0a, l0b, params);
    params.cmatrixInitVal = false;
    for (uint32_t i = 1; i < iters; ++i) {
        Mmad(l0c, l0a, l0b, params);
    }
    PipeBarrier<PIPE_M>();
    (void)sink;
#else
    (void)sink;
    (void)iters;
#endif
}

__global__ __aicore__ void cube_mmad_f16_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    run_cube_mmad_bench<half>(sink, iters);
}

__global__ __aicore__ void cube_mmad_bf16_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    run_cube_mmad_bench<bfloat16_t>(sink, iters);
}

// ---------------------------------------------------------------------------
// 7. Vector FP32 FMA & SFU (Exp) Throughput in UB
// ---------------------------------------------------------------------------
__global__ __aicore__ void vec_fma_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    constexpr int32_t ELEMS = 12288;  // 48 KiB x 3 = 144 KiB <= 192 KiB UB
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB, bufC;
    pipe.InitBuffer(bufA, ELEMS * sizeof(float));
    pipe.InitBuffer(bufB, ELEMS * sizeof(float));
    pipe.InitBuffer(bufC, ELEMS * sizeof(float));
    LocalTensor<float> a = bufA.Get<float>();
    LocalTensor<float> b = bufB.Get<float>();
    LocalTensor<float> c = bufC.Get<float>();
    Duplicate(a, 1.0001f, ELEMS);
    Duplicate(b, 0.9999f, ELEMS);
    Duplicate(c, 0.5f, ELEMS);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < iters; ++i) {
        MulAddDst(c, a, b, ELEMS);
    }
    PipeBarrier<PIPE_V>();
    float v = c.GetValue(0);
    if (v == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(GetBlockIdx() & 63, v);
    }
#endif
}

__global__ __aicore__ void vec_exp_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    constexpr int32_t ELEMS = 12288;
    TPipe pipe;
    TBuf<TPosition::VECCALC> bufA, bufB;
    pipe.InitBuffer(bufA, ELEMS * sizeof(float));
    pipe.InitBuffer(bufB, ELEMS * sizeof(float));
    LocalTensor<float> a = bufA.Get<float>();
    LocalTensor<float> b = bufB.Get<float>();
    Duplicate(a, 0.01f, ELEMS);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < iters; ++i) {
        Exp(b, a, ELEMS);
    }
    PipeBarrier<PIPE_V>();
    float v = b.GetValue(0);
    if (v == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(GetBlockIdx() & 63, v);
    }
#endif
}

// ---------------------------------------------------------------------------
// 8. M2: On-Chip UB Pointer-Chase Latency (Scalar -> UB)
// ---------------------------------------------------------------------------
__global__ __aicore__ void ub_latency_kernel(GM_ADDR out_ticks, GM_ADDR sink, uint32_t hops) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (GetBlockIdx() != 0) return;

    constexpr uint32_t SLOTS = 128;
    constexpr uint32_t STRIDE = 8;  // 32 B per slot -> 4 KiB UB
    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    TBuf<TPosition::VECCALC> outBuf;
    pipe.InitBuffer(buf, SLOTS * STRIDE * sizeof(uint32_t));
    pipe.InitBuffer(outBuf, 64);
    LocalTensor<uint32_t> loc = buf.Get<uint32_t>();
    LocalTensor<uint64_t> outLoc = outBuf.Get<uint64_t>();

    // Single cycle visiting all 128 slots (gcd(37, 128) == 1)
    for (uint32_t i = 0; i < SLOTS; ++i) {
        loc.SetValue(i * STRIDE, ((i + 37u) & (SLOTS - 1u)) * STRIDE);
    }
    PipeBarrier<PIPE_ALL>();

    uint32_t idx = 0;
    uint64_t t0 = GetSystemCycle();
    for (uint32_t h = 0; h < hops; ++h) {
        idx = loc.GetValue(idx);
    }
    uint64_t t1 = GetSystemCycle();

    outLoc.SetValue(0, t1 - t0);
    outLoc.SetValue(1, static_cast<uint64_t>(idx));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out_ticks), 16);
    DataCopy(outGm, outLoc, 8);
    if (idx == 0xdeadbeefu) {
        GlobalTensor<uint32_t> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sink), 16);
        sinkGm.SetValue(0, idx);
    }
#endif
}

// ---------------------------------------------------------------------------
// 9. M2: MLP Bandwidth vs Bytes-in-Flight (1-token sync & 2-token async MTE2)
// ---------------------------------------------------------------------------
__global__ __aicore__ void mlp_sync_read_kernel(GM_ADDR src, GM_ADDR sink,
                                                uint64_t buf_floats, uint32_t chunk_floats,
                                                uint32_t steps, uint32_t dual_aiv) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (!dual_aiv && GetSubBlockIdx() != 0) return;
    uint32_t wid = dual_aiv ? GetBlockIdx() : (GetBlockIdx() / GetSubBlockNum());
    uint32_t workers = dual_aiv ? (GetBlockNum() * GetSubBlockNum()) : GetBlockNum();
    uint64_t mask = buf_floats - 1ULL;

    GlobalTensor<float> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), buf_floats);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, chunk_floats * sizeof(float));
    LocalTensor<float> loc = buf.Get<float>();

    uint64_t off = (static_cast<uint64_t>(wid) * chunk_floats) & mask;
    uint64_t step_stride = static_cast<uint64_t>(workers) * chunk_floats;
    float acc = 0.0f;

    for (uint32_t s = 0; s < steps; ++s) {
        DataCopy(loc, srcGm[off], chunk_floats);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        acc += loc.GetValue(0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        off = (off + step_stride) & mask;
    }
    if (acc == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(wid & 63, acc);
    }
#endif
}

__global__ __aicore__ void mlp_async_read_kernel(GM_ADDR src, GM_ADDR sink,
                                                 uint64_t buf_floats, uint32_t chunk_floats,
                                                 uint32_t steps, uint32_t dual_aiv) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (!dual_aiv && GetSubBlockIdx() != 0) return;
    uint32_t wid = dual_aiv ? GetBlockIdx() : (GetBlockIdx() / GetSubBlockNum());
    uint32_t workers = dual_aiv ? (GetBlockNum() * GetSubBlockNum()) : GetBlockNum();
    uint64_t mask = buf_floats - 1ULL;

    GlobalTensor<float> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), buf_floats);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf0, buf1;
    pipe.InitBuffer(buf0, chunk_floats * sizeof(float));
    pipe.InitBuffer(buf1, chunk_floats * sizeof(float));
    LocalTensor<float> loc0 = buf0.Get<float>();
    LocalTensor<float> loc1 = buf1.Get<float>();

    uint64_t off = (static_cast<uint64_t>(wid) * chunk_floats) & mask;
    uint64_t step_stride = static_cast<uint64_t>(workers) * chunk_floats;
    float acc = 0.0f;

    // Keep 2 chunks in flight per active worker.
    DataCopy(loc0, srcGm[off], chunk_floats);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    off = (off + step_stride) & mask;

    DataCopy(loc1, srcGm[off], chunk_floats);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
    off = (off + step_stride) & mask;

    for (uint32_t s = 2; s < steps; ++s) {
        if ((s & 1u) == 0u) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            acc += loc0.GetValue(0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopy(loc0, srcGm[off], chunk_floats);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
            acc += loc1.GetValue(0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            DataCopy(loc1, srcGm[off], chunk_floats);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        off = (off + step_stride) & mask;
    }
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    acc += loc0.GetValue(0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    acc += loc1.GetValue(0);

    if (acc == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(wid & 63, acc);
    }
#endif
}

// ---------------------------------------------------------------------------
// 10. M2: Loaded Pointer-Chase Latency vs Background Stream Bandwidth
// ---------------------------------------------------------------------------
__global__ __aicore__ void loaded_pchase_kernel(GM_ADDR table, GM_ADDR bg_src,
                                                GM_ADDR done_gm, GM_ADDR out_stats,
                                                GM_ADDR sink, uint64_t table_elems,
                                                uint64_t bg_floats, uint32_t hops,
                                                uint32_t warmup_ticks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    uint32_t bid = GetBlockIdx();
    uint32_t subnum = GetSubBlockNum();
    uint32_t subid = GetSubBlockIdx();
    uint32_t cid = bid / subnum;
    uint32_t ncores = GetBlockNum();

    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out_stats), 128 * 16);
    GlobalTensor<uint32_t> doneGm;
    doneGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(done_gm), 32);

    if (cid == 0) {
        if (subid != 0) return;

        GlobalTensor<uint32_t> tblGm;
        tblGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table), table_elems);

        TPipe pipe;
        TBuf<TPosition::VECCALC> buf, ctrlBuf, statBuf;
        pipe.InitBuffer(buf, 64);
        pipe.InitBuffer(ctrlBuf, 64);
        pipe.InitBuffer(statBuf, 64);
        LocalTensor<uint32_t> loc = buf.Get<uint32_t>();
        LocalTensor<uint32_t> ctrl = ctrlBuf.Get<uint32_t>();
        LocalTensor<uint64_t> stat = statBuf.Get<uint64_t>();

        // Let background streaming cores ramp up before timing the probe.
        if (ncores > 1 && warmup_ticks > 0) {
            uint64_t w0 = GetSystemCycle();
            while (GetSystemCycle() - w0 < static_cast<uint64_t>(warmup_ticks)) {}
        }

        uint32_t idx = 0;
        uint64_t t0 = GetSystemCycle();
        for (uint32_t h = 0; h < hops; ++h) {
            DataCopy(loc, tblGm[static_cast<uint64_t>(idx) * 32], 8);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            idx = loc.GetValue(0);
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        }
        uint64_t t1 = GetSystemCycle();

        for (uint32_t k = 0; k < 8; ++k) ctrl.SetValue(k, 1u);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopy(doneGm, ctrl, 8);

        stat.SetValue(0, t1 - t0);
        stat.SetValue(1, static_cast<uint64_t>(hops));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopy(outGm[0], stat, 8);

        if (idx == 0xdeadbeefu) {
            GlobalTensor<uint32_t> sinkGm;
            sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sink), 16);
            sinkGm.SetValue(0, idx);
        }
        return;
    }

    // Background streaming workers (cores 1..ncores-1)
    uint32_t bg_workers = (ncores - 1) * subnum;
    uint32_t bg_wid = (cid - 1) * subnum + subid;
    uint64_t total_tiles = bg_floats / TILE_FLOATS;
    uint64_t tiles_per_w = total_tiles / bg_workers;
    if (tiles_per_w == 0) tiles_per_w = 1;
    uint64_t tile = (static_cast<uint64_t>(bg_wid) * tiles_per_w) % total_tiles;

    GlobalTensor<float> bgGm;
    bgGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bg_src), bg_floats);

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;
    TBuf<TPosition::VECCALC> flagBuf, statBuf;
    pipe.InitBuffer(inQueue, 2, TILE_BYTES);
    pipe.InitBuffer(flagBuf, 64);
    pipe.InitBuffer(statBuf, 64);
    LocalTensor<uint32_t> flagLoc = flagBuf.Get<uint32_t>();
    LocalTensor<uint64_t> statLoc = statBuf.Get<uint64_t>();

    float acc = 0.0f;
    uint64_t tiles_done = 0;
    uint64_t t0 = GetSystemCycle();
    while (tiles_done < 262144ULL) {
        for (uint32_t b = 0; b < 32; ++b) {
            LocalTensor<float> buf = inQueue.AllocTensor<float>();
            DataCopy(buf, bgGm[tile * TILE_FLOATS], TILE_FLOATS);
            inQueue.EnQue(buf);
            LocalTensor<float> ready = inQueue.DeQue<float>();
            acc += ready.GetValue(0);
            inQueue.FreeTensor(ready);
            tile += 1;
            if (tile >= total_tiles) tile = 0;
        }
        tiles_done += 32;
        DataCopy(flagLoc, doneGm, 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
        uint32_t done = flagLoc.GetValue(0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
        if (done != 0u) break;
    }
    uint64_t t1 = GetSystemCycle();

    statLoc.SetValue(0, t1 - t0);
    statLoc.SetValue(1, tiles_done * static_cast<uint64_t>(TILE_BYTES));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    if (bid < 128) {
        DataCopy(outGm[bid * 16], statLoc, 8);
    }
    if (acc == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(bid & 63, acc);
    }
#endif
}

// ---------------------------------------------------------------------------
// 11. M2: L2 Miss-Merging by Sharing Degree (stagger=0 vs stagger=16)
// ---------------------------------------------------------------------------
__global__ __aicore__ void l2_merge_kernel(GM_ADDR src, GM_ADDR out_ticks, GM_ADDR sink,
                                           uint64_t buf_floats, uint32_t chunk_floats,
                                           uint32_t steps, uint32_t sharing,
                                           uint32_t stagger_ticks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (GetSubBlockIdx() != 0) return;
    uint32_t cid = GetBlockIdx() / GetSubBlockNum();
    uint32_t ncores = GetBlockNum();
    uint32_t group_id = cid / sharing;
    uint32_t rank = cid % sharing;
    uint32_t n_groups = ncores / sharing;
    uint64_t mask = buf_floats - 1ULL;

    GlobalTensor<float> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), buf_floats);
    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out_ticks), 64 * 16);

    TPipe pipe;
    TBuf<TPosition::VECCALC> buf, statBuf;
    pipe.InitBuffer(buf, chunk_floats * sizeof(float));
    pipe.InitBuffer(statBuf, 64);
    LocalTensor<float> loc = buf.Get<float>();
    LocalTensor<uint64_t> stat = statBuf.Get<uint64_t>();

    if (stagger_ticks > 0 && rank > 0) {
        uint64_t w0 = GetSystemCycle();
        while (GetSystemCycle() - w0 < static_cast<uint64_t>(stagger_ticks)) {}
    }

    uint64_t off = (static_cast<uint64_t>(group_id) * chunk_floats) & mask;
    uint64_t group_stride = static_cast<uint64_t>(n_groups) * chunk_floats;
    float acc = 0.0f;

    uint64_t t0 = GetSystemCycle();
    for (uint32_t s = 0; s < steps; ++s) {
        DataCopy(loc, srcGm[off], chunk_floats);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        acc += loc.GetValue(0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        off = (off + group_stride) & mask;
    }
    uint64_t t1 = GetSystemCycle();

    stat.SetValue(0, t1 - t0);
    stat.SetValue(1, static_cast<uint64_t>(steps));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    if (cid < 64) {
        DataCopy(outGm[cid * 16], stat, 8);
    }
    if (acc == 123456789.0f) {
        GlobalTensor<float> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sink), 64);
        sinkGm.SetValue(cid & 63, acc);
    }
#endif
}

// ---------------------------------------------------------------------------
// 12. M2: Per-Stage Pipeline Sync Overhead (AIC empty flags + AIV hot-L2 min RT)
// ---------------------------------------------------------------------------
__global__ __aicore__ void fixed_sync_cube_kernel(GM_ADDR sink, uint32_t tokens, uint32_t steps) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#ifdef __DAV_C220_CUBE__
    if (tokens <= 1u) {
        for (uint32_t s = 0; s < steps; ++s) {
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
    } else {
        // Double-buffered ping-pong event ring across MTE2 -> MTE1 -> M -> MTE1 -> MTE2
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        for (uint32_t s = 0; s < steps; ++s) {
            if ((s & 1u) == 0u) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    }
    (void)sink;
#else
    (void)sink;
    (void)tokens;
    (void)steps;
#endif
}

__global__ __aicore__ void stage_sync_hot_l2_kernel(GM_ADDR hot_src, GM_ADDR out_ticks,
                                                    GM_ADDR sink, uint32_t tokens,
                                                    uint32_t steps) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    if (GetBlockIdx() != 0) return;

    GlobalTensor<uint32_t> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(hot_src), 128);
    GlobalTensor<uint64_t> outGm;
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(out_ticks), 16);

    TPipe pipe;
    TBuf<TPosition::VECCALC> b0, b1, statBuf;
    pipe.InitBuffer(b0, 64);
    pipe.InitBuffer(b1, 64);
    pipe.InitBuffer(statBuf, 64);
    LocalTensor<uint32_t> loc0 = b0.Get<uint32_t>();
    LocalTensor<uint32_t> loc1 = b1.Get<uint32_t>();
    LocalTensor<uint64_t> stat = statBuf.Get<uint64_t>();

    // Warm the 32-byte line in L2.
    for (uint32_t w = 0; w < 16; ++w) {
        DataCopy(loc0, srcGm[0], 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    }

    uint32_t acc = 0;
    uint64_t t0 = GetSystemCycle();
    if (tokens <= 1u) {
        for (uint32_t s = 0; s < steps; ++s) {
            DataCopy(loc0, srcGm[0], 8);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            if (tokens == 0u) {
                acc ^= loc0.GetValue(0);
            }
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        }
    } else {
        DataCopy(loc0, srcGm[0], 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        DataCopy(loc1, srcGm[32], 8);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        for (uint32_t s = 2; s < steps; ++s) {
            if ((s & 1u) == 0u) {
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                DataCopy(loc0, srcGm[0], 8);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            } else {
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
                SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
                WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
                DataCopy(loc1, srcGm[32], 8);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
            }
        }
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
    }
    uint64_t t1 = GetSystemCycle();

    stat.SetValue(0, t1 - t0);
    stat.SetValue(1, static_cast<uint64_t>(steps));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(outGm, stat, 8);
    if (acc == 0xdeadbeefu) {
        GlobalTensor<uint32_t> sinkGm;
        sinkGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sink), 16);
        sinkGm.SetValue(0, acc);
    }
#endif
}

// ---------------------------------------------------------------------------
// 13. M2: Cross-Core AIC <-> AIV Sync Latency (FFTS+ CrossCoreSetFlag/WaitFlag)
// ---------------------------------------------------------------------------
__global__ __aicore__ void cv_sync_pingpong_kernel(GM_ADDR sink, uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    for (uint32_t i = 0; i < iters; ++i) {
#ifdef __DAV_C220_CUBE__
        CrossCoreSetFlag<0x2, PIPE_FIX>(0);
        CrossCoreWaitFlag(1);
#endif
#ifdef __DAV_C220_VEC__
        CrossCoreWaitFlag(0);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
#endif
    }
    (void)sink;
}

// ---------------------------------------------------------------------------
// 14. M2: AIC 2D Strided Nd2Nz Row-Issue Rate (GM ND -> L1 Fractal NZ)
// ---------------------------------------------------------------------------
template <int ROWS, int COLS>
__global__ __aicore__ void aic_nd2nz_issue_kernel(GM_ADDR src, uint32_t stride_elems,
                                                  uint32_t iters) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#ifdef __DAV_C220_CUBE__
    uint32_t cid = GetBlockIdx();
    TPipe pipe;
    TBuf<TPosition::A1> l1;
    pipe.InitBuffer(l1, ROWS * COLS * sizeof(half));
    LocalTensor<half> dst = l1.Get<half>();
    GlobalTensor<half> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(src), 128ULL * 1024 * 1024);

    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = ROWS;
    p.dValue = COLS;
    p.srcNdMatrixStride = 0;
    p.srcDValue = stride_elems;
    p.dstNzC0Stride = ROWS;
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;

    uint64_t base = static_cast<uint64_t>(cid) * ROWS * stride_elems;
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < iters; ++i) {
        uint64_t off = (base + (i & 31u) * COLS) & ((64ULL * 1024 * 1024) - 1);
        DataCopy(dst, gm[off], p);
        PipeBarrier<PIPE_MTE2>();
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

// ---------------------------------------------------------------------------
// Host Harness
// ---------------------------------------------------------------------------
#ifndef __CCE_KT_TEST__

#define ACL_CHECK(x)                                                             \
    do {                                                                         \
        aclError err__ = (x);                                                    \
        if (err__ != ACL_SUCCESS) {                                              \
            std::fprintf(stderr, "ACL error %d at %s:%d\n", (int)err__,          \
                         __FILE__, __LINE__);                                    \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static void print_samples(const std::vector<double>& vals) {
    std::printf("[");
    for (size_t i = 0; i < vals.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%.6f", vals[i]);
    }
    std::printf("]");
}

static std::vector<uint32_t> make_chase_table(uint64_t num_slots) {
    // Sattolo cycle over num_slots (stride = 32 uint32 = 128 B cache line per slot)
    std::vector<uint32_t> perm(num_slots);
    for (uint64_t i = 0; i < num_slots; ++i) perm[i] = static_cast<uint32_t>(i);
    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    for (uint64_t i = num_slots - 1; i > 0; --i) {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        uint64_t j = (rng * 0x2545F4914F6CDD1DULL) % i;
        std::swap(perm[i], perm[j]);
    }
    std::vector<uint32_t> table(num_slots * 32, 0);
    for (uint64_t i = 0; i < num_slots; ++i) {
        table[i * 32] = perm[i];
    }
    return table;
}

int main(int argc, char** argv) {
    int device_id = 0;
    int samples = 15;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            samples = std::atoi(argv[++i]);
        }
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    aclrtEvent ev_start = nullptr, ev_stop = nullptr;
    ACL_CHECK(aclrtCreateEvent(&ev_start));
    ACL_CHECK(aclrtCreateEvent(&ev_stop));

    const uint32_t ai_cores = 20;
    const uint64_t bytes_1g = 1024ULL * 1024ULL * 1024ULL;
    const uint64_t flush_bytes = 384ULL * 1024ULL * 1024ULL;

    void *d_buf = nullptr, *d_flush = nullptr, *d_sink = nullptr;
    ACL_CHECK(aclrtMalloc(&d_buf, bytes_1g, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_flush, flush_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_sink, 4096, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_buf, bytes_1g, 1, bytes_1g));
    ACL_CHECK(aclrtMemset(d_flush, flush_bytes, 2, flush_bytes));
    ACL_CHECK(aclrtMemset(d_sink, 4096, 0, 4096));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    auto time_ms = [&](auto launch_fn) -> double {
        ACL_CHECK(aclrtRecordEvent(ev_start, stream));
        launch_fn();
        ACL_CHECK(aclrtRecordEvent(ev_stop, stream));
        ACL_CHECK(aclrtSynchronizeEvent(ev_stop));
        float ms = 0.0f;
        ACL_CHECK(aclrtEventElapsedTime(&ms, ev_start, ev_stop));
        return static_cast<double>(ms);
    };

    const int warmup = 3;
    std::vector<uint32_t> active_list = {1, 2, 4, 8, 10, 15, 20};

    std::printf("{\n");
    std::printf("  \"device_id\": %d,\n", device_id);
    std::printf("  \"ai_core_cnt\": %u,\n", ai_cores);
    std::printf("  \"samples\": %d,\n", samples);

    // 1. HBM Read curve vs active AI Cores (cold L2)
    std::printf("  \"dram_read_by_active_cores\": [\n");
    for (size_t i = 0; i < active_list.size(); ++i) {
        uint32_t ac = active_list[i];
        for (int w = 0; w < warmup; ++w) {
            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
            stream_read_kernel<<<ac, nullptr, stream>>>(
                (uint8_t*)d_buf, (uint8_t*)d_sink, bytes_1g / 4, 1);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> gbps;
        gbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
            double ms = time_ms([&]() {
                stream_read_kernel<<<ac, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_sink, bytes_1g / 4, 1);
            });
            gbps.push_back(static_cast<double>(bytes_1g) / 1e9 / (ms / 1e3));
        }
        std::printf("    {\"active_cores\": %u, \"gbps\": ", ac);
        print_samples(gbps);
        std::printf("}%s\n", (i + 1 < active_list.size()) ? "," : "");
    }
    std::printf("  ],\n");

    // 2. HBM Write at 20 AI Cores (cold L2)
    {
        for (int w = 0; w < warmup; ++w) {
            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
            stream_write_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_buf, bytes_1g / 4, 1, 1.0f + w);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> gbps;
        gbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
            double ms = time_ms([&]() {
                stream_write_kernel<<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, bytes_1g / 4, 1, 2.0f + s);
            });
            gbps.push_back(static_cast<double>(bytes_1g) / 1e9 / (ms / 1e3));
        }
        std::printf("  \"dram_write_gbps\": ");
        print_samples(gbps);
        std::printf(",\n");
    }

    // 3. L2 Read curve vs active AI Cores (64 MiB warm buffer x 8 repeats = 512 MiB)
    {
        const uint64_t l2_buf_bytes = 64ULL * 1024ULL * 1024ULL;
        const uint32_t rep = 8;
        std::printf("  \"l2_read_by_active_cores\": [\n");
        for (size_t i = 0; i < active_list.size(); ++i) {
            uint32_t ac = active_list[i];
            for (int w = 0; w < warmup; ++w) {
                stream_read_kernel<<<ac, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_sink, l2_buf_bytes / 4, rep);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> gbps;
            gbps.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                double ms = time_ms([&]() {
                    stream_read_kernel<<<ac, nullptr, stream>>>(
                        (uint8_t*)d_buf, (uint8_t*)d_sink, l2_buf_bytes / 4, rep);
                });
                gbps.push_back(static_cast<double>(l2_buf_bytes * rep) / 1e9 / (ms / 1e3));
            }
            std::printf("    {\"active_cores\": %u, \"gbps\": ", ac);
            print_samples(gbps);
            std::printf("}%s\n", (i + 1 < active_list.size()) ? "," : "");
        }
        std::printf("  ],\n");

        // L2 Write at 20 AI Cores
        for (int w = 0; w < warmup; ++w) {
            stream_write_kernel<<<ai_cores, nullptr, stream>>>(
                (uint8_t*)d_buf, l2_buf_bytes / 4, rep, 1.0f + w);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> wgbps;
        wgbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                stream_write_kernel<<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, l2_buf_bytes / 4, rep, 2.0f + s);
            });
            wgbps.push_back(static_cast<double>(l2_buf_bytes * rep) / 1e9 / (ms / 1e3));
        }
        std::printf("  \"l2_write_gbps\": ");
        print_samples(wgbps);
        std::printf(",\n");
    }

    // 4. L2 working-set rescan sweep (20 AI Cores)
    {
        std::vector<uint64_t> sizes_mb = {16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 512, 768, 1024};
        std::printf("  \"l2_rescan_sweep\": [\n");
        for (size_t i = 0; i < sizes_mb.size(); ++i) {
            uint64_t mb = sizes_mb[i];
            uint64_t sz = mb * 1024ULL * 1024ULL;
            uint32_t rep = (mb <= 64) ? 8 : (mb <= 256 ? 4 : 1);
            for (int w = 0; w < warmup; ++w) {
                stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_sink, sz / 4, rep);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> gbps;
            gbps.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                double ms = time_ms([&]() {
                    stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                        (uint8_t*)d_buf, (uint8_t*)d_sink, sz / 4, rep);
                });
                gbps.push_back(static_cast<double>(sz * rep) / 1e9 / (ms / 1e3));
            }
            std::printf("    {\"working_set_bytes\": %llu, \"gbps\": ", (unsigned long long)sz);
            print_samples(gbps);
            std::printf("}%s\n", (i + 1 < sizes_mb.size()) ? "," : "");
        }
        std::printf("  ],\n");
    }

    // 5. Pointer-chasing latency: L2 warm (4 MiB) vs HBM cold (512 MiB)
    {
        const uint32_t hops = 32768;
        auto run_chase = [&](uint64_t buf_bytes) -> std::vector<double> {
            uint64_t slots = buf_bytes / 128;
            auto h_tbl = make_chase_table(slots);
            ACL_CHECK(aclrtMemcpy(d_buf, buf_bytes, h_tbl.data(), buf_bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            for (int w = 0; w < 2; ++w) {
                ptr_chase_kernel<<<1, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_sink, slots * 32, hops);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> ns;
            ns.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                if (buf_bytes >= 256ULL * 1024ULL * 1024ULL) {
                    stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                        (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                }
                double ms = time_ms([&]() {
                    ptr_chase_kernel<<<1, nullptr, stream>>>(
                        (uint8_t*)d_buf, (uint8_t*)d_sink, slots * 32, hops);
                });
                ns.push_back(ms * 1e6 / static_cast<double>(hops));
            }
            return ns;
        };
        auto l2_lat = run_chase(4ULL * 1024ULL * 1024ULL);
        auto hbm_lat = run_chase(512ULL * 1024ULL * 1024ULL);
        std::printf("  \"l2_latency_ns\": ");
        print_samples(l2_lat);
        std::printf(",\n");
        std::printf("  \"dram_latency_ns\": ");
        print_samples(hbm_lat);
        std::printf(",\n");
    }

    // 6. On-chip UB bandwidth per AI Core (2 AIVs on 1 AI Core)
    {
        const uint32_t iters = 4096;
        // Each iter moves 2 x 64 KiB read+write = 256 KiB per AIV, x2 AIV per AI Core = 512 KiB/iter
        const double bytes_per_core = static_cast<double>(iters) * 2.0 * 65536.0 * 2.0 * 2.0;
        for (int w = 0; w < warmup; ++w) {
            ub_bw_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, iters);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> gbps;
        gbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                ub_bw_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, iters);
            });
            gbps.push_back(bytes_per_core / 1e9 / (ms / 1e3));
        }
        std::printf("  \"smem_bw_per_core_gbps\": ");
        print_samples(gbps);
        std::printf(",\n");
    }

    // 7. Cube MMA throughput (FP16 and BF16 -> FP32) on 20 AIC cores
    {
        const uint32_t iters = 8192;
        const double flops_per_core = 2.0 * 128.0 * 128.0 * 64.0 * static_cast<double>(iters);

        for (int w = 0; w < warmup; ++w) {
            cube_mmad_f16_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> f16_tf;
        f16_tf.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                cube_mmad_f16_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
            });
            f16_tf.push_back(flops_per_core / 1e12 / (ms / 1e3));
        }
        std::printf("  \"cube_mmad_f16_tflops_per_core\": ");
        print_samples(f16_tf);
        std::printf(",\n");

        for (int w = 0; w < warmup; ++w) {
            cube_mmad_bf16_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> bf16_tf;
        bf16_tf.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                cube_mmad_bf16_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
            });
            bf16_tf.push_back(flops_per_core / 1e12 / (ms / 1e3));
        }
        std::printf("  \"cube_mmad_bf16_tflops_per_core\": ");
        print_samples(bf16_tf);
        std::printf(",\n");
    }

    // 8. Vector FP32 FMA & SFU (Exp) on 20 AI Cores (40 AIVs)
    {
        const uint32_t iters = 4096;
        // 2 AIVs per AI Core, 12288 fp32 elements per MulAddDst, 2 FLOPs per FMA
        const double total_fma_flops = static_cast<double>(ai_cores) * 2.0 * 12288.0 * 2.0 * iters;
        const double per_core_ginsts = 2.0 * (12288.0 / 32.0) * iters / 1e9;  // 32-lane equiv insts
        for (int w = 0; w < warmup; ++w) {
            vec_fma_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> fma_tf, issue_gips;
        fma_tf.reserve(samples);
        issue_gips.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                vec_fma_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
            });
            fma_tf.push_back(total_fma_flops / 1e12 / (ms / 1e3));
            issue_gips.push_back(per_core_ginsts / (ms / 1e3));
        }
        std::printf("  \"fp32_fma_tflops\": ");
        print_samples(fma_tf);
        std::printf(",\n");
        std::printf("  \"non_mma_issue_rate_per_core_gips\": ");
        print_samples(issue_gips);
        std::printf(",\n");

        const double total_exp_ops = static_cast<double>(ai_cores) * 2.0 * 12288.0 * iters;
        const double per_core_sfu_ginsts = 2.0 * (12288.0 / 32.0) * iters / 1e9;
        for (int w = 0; w < warmup; ++w) {
            vec_exp_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> sfu_tf, sfu_gips;
        sfu_tf.reserve(samples);
        sfu_gips.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            double ms = time_ms([&]() {
                vec_exp_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink, iters);
            });
            sfu_tf.push_back(total_exp_ops / 1e12 / (ms / 1e3));
            sfu_gips.push_back(per_core_sfu_ginsts / (ms / 1e3));
        }
        std::printf("  \"sfu_tflops\": ");
        print_samples(sfu_tf);
        std::printf(",\n");
        std::printf("  \"sfu_issue_rate_per_core_gips\": ");
        print_samples(sfu_gips);
        std::printf(",\n");
    }

    // 9. Launch overhead: queued device-side dispatch (behind a 50 ms delay kernel so host
    //    finishes queuing all 200 launches before execution starts) and un-queued host batch.
    {
        const int reps = 200;
        const uint32_t delay_ticks = 2500000;  // ~50 ms at 50 MHz system counter
        for (int w = 0; w < 20; ++w) {
            empty_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> dev_1, dev_20, host_1;
        dev_1.reserve(samples);
        dev_20.reserve(samples);
        host_1.reserve(samples);
        for (int s = 0; s < samples; ++s) {
            delay_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, delay_ticks);
            double ms1 = time_ms([&]() {
                for (int r = 0; r < reps; ++r) empty_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink);
            });
            delay_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, delay_ticks);
            double ms20 = time_ms([&]() {
                for (int r = 0; r < reps; ++r) empty_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_sink);
            });
            double hms1 = time_ms([&]() {
                for (int r = 0; r < reps; ++r) empty_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink);
            });
            dev_1.push_back(ms1 * 1e3 / reps);
            dev_20.push_back(ms20 * 1e3 / reps);
            host_1.push_back(hms1 * 1e3 / reps);
        }
        std::printf("  \"launch_overhead_1core_us\": ");
        print_samples(dev_1);
        std::printf(",\n");
        std::printf("  \"launch_overhead_20core_us\": ");
        print_samples(dev_20);
        std::printf(",\n");
        std::printf("  \"host_launch_overhead_us\": ");
        print_samples(host_1);
        std::printf(",\n");
    }

    // 10. CTA Dispatch first-wave trace (20 AI Cores)
    {
        void* d_info = nullptr;
        ACL_CHECK(aclrtMalloc(&d_info, 128 * 16 * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemset(d_info, 128 * 16 * sizeof(uint64_t), 0xFF, 128 * 16 * sizeof(uint64_t)));
        probe_dispatch_kernel<<<ai_cores, nullptr, stream>>>((uint8_t*)d_info);
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<uint64_t> h(128 * 16);
        ACL_CHECK(aclrtMemcpy(h.data(), h.size() * 8, d_info, h.size() * 8,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        uint64_t min_cyc = ~0ULL;
        for (uint32_t i = 0; i < ai_cores; ++i) {
            uint64_t c = h[(i * 2) * 16 + 4];
            if (c < min_cyc) min_cyc = c;
        }
        std::printf("  \"dispatch_first_wave\": [\n");
        for (uint32_t i = 0; i < ai_cores; ++i) {
            uint64_t c = h[(i * 2) * 16 + 4];
            // System counter on Ascend 910B ticks at 50 MHz (20 ns/tick)
            double start_ns = static_cast<double>(c - min_cyc) * 20.0;
            std::printf("    {\"cta_id\": %u, \"sm_id\": %u, \"start_ns\": %.1f}%s\n",
                        i, i, start_ns, (i + 1 < ai_cores) ? "," : "");
        }
        std::printf("  ],\n");
        aclrtFree(d_info);
    }

    // -----------------------------------------------------------------------
    // M2 Calibration Suite
    // -----------------------------------------------------------------------
    const int m2_samples = std::max(3, std::min(samples, 9));
    const uint64_t bg_bytes = 512ULL * 1024ULL * 1024ULL;
    const uint64_t l2_mlp_bytes = 32ULL * 1024ULL * 1024ULL;
    void *d_bg = nullptr, *d_done = nullptr, *d_stats = nullptr;
    ACL_CHECK(aclrtMalloc(&d_bg, bg_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_done, 256, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_stats, 128 * 16 * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_bg, bg_bytes, 5, bg_bytes));
    ACL_CHECK(aclrtMemset(d_done, 256, 0, 256));
    ACL_CHECK(aclrtMemset(d_stats, 128 * 16 * sizeof(uint64_t), 0, 128 * 16 * sizeof(uint64_t)));
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 11. M2: On-chip UB latency (Scalar -> UB pointer chase)
    {
        const uint32_t hops = 100000;
        for (int w = 0; w < 2; ++w) {
            ub_latency_kernel<<<1, nullptr, stream>>>((uint8_t*)d_stats, (uint8_t*)d_sink, hops);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        std::vector<double> ub_lat;
        ub_lat.reserve(samples);
        std::vector<uint64_t> h(16, 0);
        for (int s = 0; s < samples; ++s) {
            ub_latency_kernel<<<1, nullptr, stream>>>((uint8_t*)d_stats, (uint8_t*)d_sink, hops);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK(aclrtMemcpy(h.data(), h.size() * 8, d_stats, h.size() * 8,
                                  ACL_MEMCPY_DEVICE_TO_HOST));
            ub_lat.push_back(static_cast<double>(h[0]) * 20.0 / static_cast<double>(hops));
        }
        std::printf("  \"smem_latency_ns\": ");
        print_samples(ub_lat);
        std::printf(",\n");
    }

    // 12. M2: MLP Bandwidth Curves vs Bytes-in-Flight (DRAM 512 MiB cold & L2 32 MiB warm)
    {
        struct MlpCfg {
            uint32_t cores;
            uint32_t bytes_per_core;
        };
        std::vector<MlpCfg> pts = {
            {1, 4096},    {2, 4096},    {4, 4096},    {8, 4096},
            {12, 4096},   {16, 4096},   {20, 4096},   {12, 8192},
            {16, 8192},   {20, 8192},   {12, 16384},  {16, 16384},
            {20, 16384},  {12, 32768},  {16, 32768},  {20, 32768},
            {12, 65536},  {16, 65536},  {20, 65536},  {16, 131072},
            {20, 131072}, {16, 262144}, {20, 262144},
        };
        auto run_mlp_suite = [&](const char* key, bool is_dram) {
            const uint64_t buf_bytes = is_dram ? bg_bytes : l2_mlp_bytes;
            const uint64_t buf_floats = buf_bytes / sizeof(float);
            void* buf_ptr = is_dram ? d_bg : d_buf;

            std::printf("  \"%s\": [\n", key);
            bool first = true;
            for (const char* mode : {"cp_async", "ld_global_v4"}) {
                const bool is_async = (std::strcmp(mode, "cp_async") == 0);
                for (const auto& pt : pts) {
                    const uint32_t cores = pt.cores;
                    const uint32_t bpc = pt.bytes_per_core;
                    const uint64_t bif = static_cast<uint64_t>(cores) * bpc;
                    // For > 128 KiB/core, use both AIV sub-cores per AI Core (each holds bpc/2).
                    const uint32_t dual_aiv = (bpc > 131072u) ? 1u : 0u;
                    const uint32_t workers = dual_aiv ? (cores * 2u) : cores;
                    const uint32_t bytes_per_worker = bpc / (dual_aiv ? 2u : 1u);
                    // For async (2 tokens in flight per worker), each DMA chunk is half.
                    const uint32_t chunk_bytes = is_async ? (bytes_per_worker / 2u) : bytes_per_worker;
                    const uint32_t chunk_floats = chunk_bytes / sizeof(float);
                    // For DRAM keep total <= buf_bytes so every read is a cold HBM fetch.
                    const uint64_t target_bytes = is_dram
                        ? std::min(buf_bytes, std::max<uint64_t>(32ULL * 1024ULL * 1024ULL, bif * 128ULL))
                        : std::max<uint64_t>(128ULL * 1024ULL * 1024ULL, bif * 256ULL);
                    const uint32_t steps = std::max<uint32_t>(
                        8u, static_cast<uint32_t>(target_bytes / (static_cast<uint64_t>(workers) * chunk_bytes)));
                    const double actual_bytes =
                        static_cast<double>(workers) * static_cast<double>(chunk_bytes) * static_cast<double>(steps);

                    auto launch_once = [&]() {
                        if (is_async) {
                            mlp_async_read_kernel<<<cores, nullptr, stream>>>(
                                (uint8_t*)buf_ptr, (uint8_t*)d_sink, buf_floats, chunk_floats, steps, dual_aiv);
                        } else {
                            mlp_sync_read_kernel<<<cores, nullptr, stream>>>(
                                (uint8_t*)buf_ptr, (uint8_t*)d_sink, buf_floats, chunk_floats, steps, dual_aiv);
                        }
                    };

                    for (int w = 0; w < 2; ++w) {
                        if (is_dram) {
                            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                        }
                        launch_once();
                    }
                    ACL_CHECK(aclrtSynchronizeStream(stream));

                    std::vector<double> gbps;
                    gbps.reserve(m2_samples);
                    for (int s = 0; s < m2_samples; ++s) {
                        if (is_dram) {
                            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                        }
                        double ms = time_ms(launch_once);
                        gbps.push_back(actual_bytes / 1e9 / (ms / 1e3));
                    }
                    if (!first) std::printf(",\n");
                    first = false;
                    std::printf("    {\"issue_mode\": \"%s\", \"bytes_in_flight\": %llu, "
                                "\"active_ctas\": %u, \"bytes_per_cta\": %u, \"gbps\": ",
                                mode, (unsigned long long)bif, cores, bpc);
                    print_samples(gbps);
                    std::printf("}");
                }
            }
            std::printf("\n  ],\n");
        };
        run_mlp_suite("dram_bw_by_bytes_in_flight", true);
        run_mlp_suite("l2_bw_by_bytes_in_flight", false);
    }

    // 13. M2: Loaded Latency & Latency-vs-BW Curves (L2 warm & DRAM cold)
    {
        std::vector<uint32_t> bg_core_list = {0, 1, 2, 4, 8, 12, 16, 19};
        auto run_loaded_lat = [&](const char* key_curve, const char* key_loaded,
                                  uint64_t chase_bytes, uint64_t bg_scan_bytes,
                                  uint32_t hops, bool flush_l2) {
            uint64_t slots = chase_bytes / 128;
            auto h_tbl = make_chase_table(slots);
            ACL_CHECK(aclrtMemcpy(d_buf, chase_bytes, h_tbl.data(), chase_bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtSynchronizeStream(stream));

            std::vector<double> loaded_samples;
            std::vector<uint64_t> h_stats(128 * 16, 0);

            std::printf("  \"%s\": [\n", key_curve);
            for (size_t i = 0; i < bg_core_list.size(); ++i) {
                uint32_t bgc = bg_core_list[i];
                uint32_t total_cores = 1 + bgc;
                // Warm L2 chase table and L2 background buffer if warm-L2 test
                if (!flush_l2) {
                    ptr_chase_kernel<<<1, nullptr, stream>>>(
                        (uint8_t*)d_buf, (uint8_t*)d_sink, slots * 32, 4096);
                    if (bgc > 0) {
                        stream_read_kernel<<<bgc, nullptr, stream>>>(
                            (uint8_t*)d_bg, (uint8_t*)d_sink, bg_scan_bytes / 4, 2);
                    }
                    ACL_CHECK(aclrtSynchronizeStream(stream));
                }

                std::vector<double> lat_s, bw_s;
                lat_s.reserve(m2_samples);
                bw_s.reserve(m2_samples);
                for (int s = 0; s < m2_samples; ++s) {
                    if (flush_l2) {
                        stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                            (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                    }
                    ACL_CHECK(aclrtMemset(d_done, 256, 0, 256));
                    ACL_CHECK(aclrtMemset(d_stats, 128 * 16 * sizeof(uint64_t), 0,
                                          128 * 16 * sizeof(uint64_t)));
                    loaded_pchase_kernel<<<total_cores, nullptr, stream>>>(
                        (uint8_t*)d_buf, (uint8_t*)d_bg, (uint8_t*)d_done,
                        (uint8_t*)d_stats, (uint8_t*)d_sink,
                        slots * 32, bg_scan_bytes / 4, hops, 15000);
                    ACL_CHECK(aclrtSynchronizeStream(stream));
                    ACL_CHECK(aclrtMemcpy(h_stats.data(), h_stats.size() * 8, d_stats,
                                          h_stats.size() * 8, ACL_MEMCPY_DEVICE_TO_HOST));
                    double lat_ns = static_cast<double>(h_stats[0]) * 20.0 / static_cast<double>(hops);
                    double total_bw = 0.0;
                    for (uint32_t bid = 2; bid < total_cores * 2; ++bid) {
                        uint64_t ticks = h_stats[bid * 16 + 0];
                        uint64_t bytes = h_stats[bid * 16 + 1];
                        if (ticks > 0 && bytes > 0) {
                            total_bw += static_cast<double>(bytes) / 1e9 /
                                        (static_cast<double>(ticks) * 20e-9);
                        }
                    }
                    lat_s.push_back(lat_ns);
                    bw_s.push_back(total_bw);
                }
                std::vector<double> bw_sorted = bw_s;
                std::sort(bw_sorted.begin(), bw_sorted.end());
                double med_bw = bw_sorted[bw_sorted.size() / 2];
                if (i + 1 == bg_core_list.size()) {
                    loaded_samples = lat_s;
                }
                std::printf("    {\"bg_cores\": %u, \"achieved_bw_gbps\": %.3f, \"latency_ns\": ",
                            bgc, med_bw);
                print_samples(lat_s);
                std::printf("}%s\n", (i + 1 < bg_core_list.size()) ? "," : "");
            }
            std::printf("  ],\n");
            std::printf("  \"%s\": ", key_loaded);
            print_samples(loaded_samples);
            std::printf(",\n");
        };

        run_loaded_lat("l2_latency_by_bw", "l2_latency_loaded_ns",
                       4ULL * 1024ULL * 1024ULL, 32ULL * 1024ULL * 1024ULL, 16384, false);
        run_loaded_lat("dram_latency_by_bw", "dram_latency_loaded_ns",
                       512ULL * 1024ULL * 1024ULL, bg_bytes, 8192, true);
    }

    // 14. M2: L2 Miss-Merging by Sharing Degree (cold L2 DRAM buffer)
    {
        const std::vector<uint32_t> n_list = {4, 8, 16, 20};
        const std::vector<uint32_t> r_list = {1, 2, 4, 5, 10, 20};
        const std::vector<uint32_t> k_list = {4096, 16384, 65536};
        const uint32_t steps = 128;
        const int merge_samples = std::max(3, std::min(samples, 7));
        std::vector<uint64_t> h_ticks(64 * 16, 0);

        std::printf("  \"l2_merge_by_sharing\": [\n");
        bool first = true;
        for (uint32_t n : n_list) {
            for (uint32_t r : r_list) {
                if (n % r != 0) continue;
                for (uint32_t k : k_list) {
                    const uint32_t chunk_floats = k / sizeof(float);
                    std::vector<uint32_t> staggers = (r == 1) ? std::vector<uint32_t>{0}
                                                              : std::vector<uint32_t>{0, 16};
                    for (uint32_t stag : staggers) {
                        // ~300 ns per stagger stage -> 15 ticks at 50 MHz
                        uint32_t stag_ticks = stag * 15u;
                        double total_l2_bytes = static_cast<double>(n) * k * steps;
                        double unique_dram_bytes = total_l2_bytes / static_cast<double>(r);

                        for (int w = 0; w < 1; ++w) {
                            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                            l2_merge_kernel<<<n, nullptr, stream>>>(
                                (uint8_t*)d_bg, (uint8_t*)d_stats, (uint8_t*)d_sink,
                                bg_bytes / 4, chunk_floats, steps, r, stag_ticks);
                        }
                        ACL_CHECK(aclrtSynchronizeStream(stream));

                        std::vector<double> turn_s, dram_s, l2_s;
                        turn_s.reserve(merge_samples);
                        dram_s.reserve(merge_samples);
                        l2_s.reserve(merge_samples);
                        for (int s = 0; s < merge_samples; ++s) {
                            stream_read_kernel<<<ai_cores, nullptr, stream>>>(
                                (uint8_t*)d_flush, (uint8_t*)d_sink, flush_bytes / 4, 1);
                            double ms = time_ms([&]() {
                                l2_merge_kernel<<<n, nullptr, stream>>>(
                                    (uint8_t*)d_bg, (uint8_t*)d_stats, (uint8_t*)d_sink,
                                    bg_bytes / 4, chunk_floats, steps, r, stag_ticks);
                            });
                            ACL_CHECK(aclrtMemcpy(h_ticks.data(), n * 16 * 8, d_stats,
                                                  n * 16 * 8, ACL_MEMCPY_DEVICE_TO_HOST));
                            double sum_turn_ns = 0.0;
                            for (uint32_t c = 0; c < n; ++c) {
                                sum_turn_ns += static_cast<double>(h_ticks[c * 16]) * 20.0 / steps;
                            }
                            turn_s.push_back(sum_turn_ns / static_cast<double>(n));
                            dram_s.push_back(unique_dram_bytes / 1e9 / (ms / 1e3));
                            l2_s.push_back(total_l2_bytes / 1e9 / (ms / 1e3));
                        }
                        if (!first) std::printf(",\n");
                        first = false;
                        std::printf("    {\"active_ctas\": %u, \"sharing\": %u, "
                                    "\"bytes_per_cta\": %u, \"stagger_stages\": %u, \"turnaround_ns\": ",
                                    n, r, k, stag);
                        print_samples(turn_s);
                        std::printf(", \"dram_unique_bw_gbps\": ");
                        print_samples(dram_s);
                        std::printf(", \"l2_to_sm_bw_gbps\": ");
                        print_samples(l2_s);
                        std::printf("}");
                    }
                }
            }
        }
        std::printf("\n  ],\n");
    }

    // 15. M2: Pipeline Sync Overhead (AIC fixed_sync_ns via 2N - N diff + AIV hot-L2 stage sync)
    {
        const uint32_t base_steps = 500000;
        std::printf("  \"fixed_sync_ns\": [\n");
        std::vector<uint32_t> tok_list = {1, 2};
        for (size_t i = 0; i < tok_list.size(); ++i) {
            uint32_t tok = tok_list[i];
            for (int w = 0; w < 2; ++w) {
                fixed_sync_cube_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, tok, base_steps);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> ns_s;
            ns_s.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                double ms1 = time_ms([&]() {
                    fixed_sync_cube_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, tok, base_steps);
                });
                double ms2 = time_ms([&]() {
                    fixed_sync_cube_kernel<<<1, nullptr, stream>>>((uint8_t*)d_sink, tok, base_steps * 2u);
                });
                double delta_ns = std::max(0.1, (ms2 - ms1) * 1e6 / static_cast<double>(base_steps));
                ns_s.push_back(delta_ns);
            }
            std::printf("    {\"num_warps\": 1, \"tokens\": %u, \"overhead_ns\": ", tok);
            print_samples(ns_s);
            std::printf("}%s\n", (i + 1 < tok_list.size()) ? "," : "");
        }
        std::printf("  ],\n");

        const uint32_t hot_steps = 16384;
        std::vector<uint32_t> stage_toks = {0, 1, 2};
        std::vector<double> min_rt_samples;
        std::vector<uint64_t> h(16, 0);
        std::printf("  \"stage_sync_overhead_ns\": [\n");
        for (size_t i = 0; i < stage_toks.size(); ++i) {
            uint32_t tok = stage_toks[i];
            const char* mode = (tok == 0) ? "ld_global_v4" : "cp_async";
            for (int w = 0; w < 2; ++w) {
                stage_sync_hot_l2_kernel<<<1, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_stats, (uint8_t*)d_sink, tok, hot_steps);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> ns_s;
            ns_s.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                stage_sync_hot_l2_kernel<<<1, nullptr, stream>>>(
                    (uint8_t*)d_buf, (uint8_t*)d_stats, (uint8_t*)d_sink, tok, hot_steps);
                ACL_CHECK(aclrtSynchronizeStream(stream));
                ACL_CHECK(aclrtMemcpy(h.data(), h.size() * 8, d_stats, h.size() * 8,
                                      ACL_MEMCPY_DEVICE_TO_HOST));
                ns_s.push_back(static_cast<double>(h[0]) * 20.0 / static_cast<double>(hot_steps));
            }
            if (tok == 1) min_rt_samples = ns_s;
            std::printf("    {\"issue_mode\": \"%s\", \"num_warps\": 1, \"tokens\": %u, \"overhead_ns\": ",
                        mode, tok);
            print_samples(ns_s);
            std::printf("}%s\n", (i + 1 < stage_toks.size()) ? "," : "");
        }
        std::printf("  ],\n");
        std::printf("  \"min_async_roundtrip_ns\": ");
        print_samples(min_rt_samples);
        std::printf(",\n");
    }

    // 16. M2: Cross-Core AIC <-> AIV Sync Handoff Latency vs Active AI Cores
    {
        const uint32_t cv_steps = 5000;
        std::printf("  \"cv_sync_hop_ns_by_active_cores\": [\n");
        for (size_t i = 0; i < active_list.size(); ++i) {
            uint32_t ac = active_list[i];
            for (int w = 0; w < 2; ++w) {
                cv_sync_pingpong_kernel<<<ac, nullptr, stream>>>((uint8_t*)d_sink, cv_steps);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> hop_ns_s;
            hop_ns_s.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                double ms1 = time_ms([&]() {
                    cv_sync_pingpong_kernel<<<ac, nullptr, stream>>>((uint8_t*)d_sink, cv_steps);
                });
                double ms2 = time_ms([&]() {
                    cv_sync_pingpong_kernel<<<ac, nullptr, stream>>>(
                        (uint8_t*)d_sink, cv_steps * 2u);
                });
                double hop_ns =
                    std::max(0.1, (ms2 - ms1) * 1e6 / (static_cast<double>(cv_steps) * 2.0));
                hop_ns_s.push_back(hop_ns);
            }
            std::printf("    {\"active_cores\": %u, \"hop_ns\": ", ac);
            print_samples(hop_ns_s);
            std::printf("}%s\n", (i + 1 < active_list.size()) ? "," : "");
        }
        std::printf("  ],\n");
    }

    // 17. M2: AIC 2D Strided Nd2Nz Row-Issue Rate (load_width_bytes = row_bytes / 32)
    {
        const uint32_t nd2nz_iters = 2048;
        const uint32_t nd2nz_stride = 4608;
        auto run_nd2nz = [&](int rows, int cols, auto launch_fn, bool trailing_comma) {
            for (int w = 0; w < 2; ++w) launch_fn();
            ACL_CHECK(aclrtSynchronizeStream(stream));
            std::vector<double> rate_s, bw_s;
            rate_s.reserve(samples);
            bw_s.reserve(samples);
            const double row_bytes = static_cast<double>(cols) * 2.0;
            const double tile_bytes = static_cast<double>(rows) * row_bytes;
            for (int s = 0; s < samples; ++s) {
                double ms = time_ms(launch_fn);
                double ns_per_iter = ms * 1e6 / static_cast<double>(nd2nz_iters);
                rate_s.push_back(static_cast<double>(rows) / ns_per_iter);
                bw_s.push_back(tile_bytes / ns_per_iter);
            }
            int width_b = static_cast<int>(row_bytes) / 32;
            uint64_t bif = static_cast<uint64_t>(ai_cores) * static_cast<uint64_t>(rows) *
                           static_cast<uint64_t>(cols) * 2ULL;
            std::printf(
                "    {\"load_width_bytes\": %d, \"batch_loads_per_thread\": %d, "
                "\"num_warps\": 1, \"bytes_in_flight\": %llu, "
                "\"issue_rate_gwarp_inst_per_sm\": ",
                width_b, rows, (unsigned long long)bif);
            print_samples(rate_s);
            std::printf(", \"delivered_bw_per_sm_gbps\": ");
            print_samples(bw_s);
            std::printf("}%s\n", trailing_comma ? "," : "");
        };
        std::printf("  \"gmem_load_issue_controlled\": [\n");
        run_nd2nz(
            128, 32,
            [&]() {
                aic_nd2nz_issue_kernel<128, 32><<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, nd2nz_stride, nd2nz_iters);
            },
            true);
        run_nd2nz(
            128, 64,
            [&]() {
                aic_nd2nz_issue_kernel<128, 64><<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, nd2nz_stride, nd2nz_iters);
            },
            true);
        run_nd2nz(
            128, 128,
            [&]() {
                aic_nd2nz_issue_kernel<128, 128><<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, nd2nz_stride, nd2nz_iters);
            },
            true);
        run_nd2nz(
            64, 256,
            [&]() {
                aic_nd2nz_issue_kernel<64, 256><<<ai_cores, nullptr, stream>>>(
                    (uint8_t*)d_buf, nd2nz_stride, nd2nz_iters);
            },
            false);
        std::printf("  ]\n");
    }

    aclrtFree(d_bg);
    aclrtFree(d_done);
    aclrtFree(d_stats);

    std::printf("}\n");

    aclrtDestroyEvent(ev_start);
    aclrtDestroyEvent(ev_stop);
    aclrtFree(d_buf);
    aclrtFree(d_flush);
    aclrtFree(d_sink);
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    return 0;
}
#endif
