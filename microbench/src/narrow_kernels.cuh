// Narrow register-staged global loads (H5). Included after m2_kernels.cuh.
//
// gmem-narrow: every warp owns N rows and walks along them 32 elements per
// batch, one load per row per thread, so a warp load is 32 consecutive
// elements of one row. A batch depends on the previous one. The row pitch is
// either a multiple of 128 B (aligned) or that plus one element (offset), so
// offset rows start at every element offset like a GEMM tile with odd K.
//
// sync-loop-narrow: the same rows staged through registers into shared
// memory, then ldmatrix and an MMA chain, one stage per iteration (d = 0).

#include <vector>

namespace narrow {

constexpr int kMaskNone = 0;
constexpr int kMaskTrue = 1;  // predicate computed at run time, always true
constexpr int kMaskTail = 2;  // lanes 24..31 off

template <int W, bool CG>
__device__ __forceinline__ uint32_t load(const uint8_t* p, bool pred) {
  uint32_t v = 0u;
  const uint32_t on = pred ? 1u : 0u;
  if constexpr (W == 1) {
    if constexpr (CG) {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.cg.u8 %0, [%1];\n}\n"
                   : "+r"(v) : "l"(p), "r"(on));
    } else {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.u8 %0, [%1];\n}\n"
                   : "+r"(v) : "l"(p), "r"(on));
    }
  } else if constexpr (W == 2) {
    uint16_t h = 0;
    if constexpr (CG) {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.cg.u16 %0, [%1];\n}\n"
                   : "+h"(h) : "l"(p), "r"(on));
    } else {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.u16 %0, [%1];\n}\n"
                   : "+h"(h) : "l"(p), "r"(on));
    }
    v = h;
  } else if constexpr (W == 4) {
    if constexpr (CG) {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.cg.u32 %0, [%1];\n}\n"
                   : "+r"(v) : "l"(p), "r"(on));
    } else {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %2, 0;\n @p ld.global.u32 %0, [%1];\n}\n"
                   : "+r"(v) : "l"(p), "r"(on));
    }
  } else {
    uint32_t x = 0u, y = 0u, z = 0u, w = 0u;
    if constexpr (CG) {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %5, 0;\n @p ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n}\n"
                   : "+r"(x), "+r"(y), "+r"(z), "+r"(w) : "l"(p), "r"(on));
    } else {
      asm volatile("{\n .reg .pred p;\n setp.ne.u32 p, %5, 0;\n @p ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n}\n"
                   : "+r"(x), "+r"(y), "+r"(z), "+r"(w) : "l"(p), "r"(on));
    }
    v = x ^ y ^ z ^ w;
  }
  return v;
}

template <int W, int MASK>
__device__ __forceinline__ bool lane_on(uint32_t lane, uint32_t elem, uint32_t limit) {
  if constexpr (MASK == kMaskNone) return true;
  else if constexpr (MASK == kMaskTrue) return elem < limit;
  else return lane < 24u;
}

template <int W, int N, bool CG, int MASK>
__global__ void gmem_narrow_kernel(const uint8_t* __restrict__ src, uint32_t* __restrict__ sink,
                                   size_t pitch, int rows_per_cta, int share, int batches,
                                   int sweeps, uint32_t step, uint32_t limit) {
  extern __shared__ uint8_t pad[];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
  const size_t row0 = static_cast<size_t>(blockIdx.x / share) * rows_per_cta + warp * N;
  const uint8_t* base = src + row0 * pitch;
  uint32_t acc = 0u;
  uint32_t dep = 0u;
  for (int s = 0; s < sweeps; ++s) {
    for (int b = 0; b < batches; ++b) {
      const uint32_t elem = static_cast<uint32_t>(b) * step + lane + dep;
      uint32_t x = 0u;
      #pragma unroll
      for (int u = 0; u < N; ++u) {
        x ^= load<W, CG>(base + u * pitch + static_cast<size_t>(elem) * W,
                         lane_on<W, MASK>(lane, elem, limit));
      }
      acc ^= x;
      dep = (x == 0x0000beefu) ? 1u : 0u;
    }
  }
  if (acc == 0x0000beefu) {
    pad[0] = 1;
    sink[blockIdx.x * blockDim.x + threadIdx.x] = acc + pad[0];
  }
}

struct Timer {
  cudaEvent_t start{}, stop{};
  Timer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  template <typename F>
  double ns(F&& f) {
    CUDA_CHECK(cudaEventRecord(start));
    f();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms) * 1e6;
  }
};

// Shared memory per CTA that leaves room for `cps` CTAs per SM and no more.
inline int pad_bytes(const cudaDeviceProp& prop, int cps) {
  const int per_sm = static_cast<int>(prop.sharedMemPerMultiprocessor);
  const int reserved = static_cast<int>(prop.reservedSharedMemPerBlock);
  const int bytes = (per_sm / cps - reserved - (cps > 1 ? 128 : 0)) / 128 * 128;
  return std::min(bytes, static_cast<int>(prop.sharedMemPerBlockOptin));
}

inline size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

constexpr size_t kDramBytes = 256ULL << 20;   // requested bytes per DRAM sample
constexpr size_t kL2Set = 12ULL << 20;        // L2-hot working set, far above L1
constexpr size_t kL2Bytes = 256ULL << 20;     // requested bytes per L2 sample
constexpr size_t kBufferBytes = 1088ULL << 20;

template <typename Kernel>
void set_smem(Kernel k, int bytes) {
  CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

template <int W, int N, bool CG, int MASK>
void gmem_narrow_point(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf,
                       uint32_t* sink, const uint4* flush, uint4* flush_sink, size_t flush_vec4,
                       bool dram, bool offset, int cps, int warps, int share, int step_bytes,
                       int samples, int warmup, bool& first, size_t dram_bytes = kDramBytes) {
  auto kernel = gmem_narrow_kernel<W, N, CG, MASK>;
  const int smem = pad_bytes(prop, cps);
  set_smem(kernel, smem);
  int fit = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, warps * 32, smem));
  const int sms = prop.multiProcessorCount;
  const int grid = sms * cps;
  const int rows_per_cta = warps * N;
  const size_t rows = static_cast<size_t>(grid / share) * rows_per_cta;
  const size_t seg = 32ULL * W;
  int batches = 0, sweeps = 1;
  // A batch moves `step_bytes` along the row: one warp load (32 W) or a
  // full 128 B line, so that no two batches share an L1 line.
  const size_t step = static_cast<size_t>(step_bytes);
  if (dram) {
    batches = static_cast<int>(std::max<size_t>(1, dram_bytes / (static_cast<size_t>(grid) * rows_per_cta * seg)));
  } else {
    batches = static_cast<int>(std::max<size_t>(1, kL2Set / (rows * step)));
    sweeps = static_cast<int>(std::max<size_t>(1, kL2Bytes / (static_cast<size_t>(grid) * rows_per_cta * seg * batches)));
  }
  const size_t pitch = round_up(static_cast<size_t>(batches) * step + seg, 128) + (offset ? W : 0);
  if (rows * pitch > kBufferBytes) {
    std::fprintf(stderr, "gmem-narrow: buffer too small\n");
    std::exit(3);
  }
  const uint32_t step_elems = static_cast<uint32_t>(step / W);
  const uint32_t limit = static_cast<uint32_t>(batches) * step_elems + 1u;
  auto launch = [&]() {
    kernel<<<grid, warps * 32, smem>>>(buf, sink, pitch, rows_per_cta, share, batches, sweeps,
                                       step_elems, limit);
  };
  auto clean = [&]() {
    if (dram) l2_read_cg_kernel<<<sms * 4, 512>>>(flush, flush_sink, flush_vec4, 1);
  };
  for (int i = 0; i < warmup; ++i) { clean(); launch(); }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ns;
  for (int i = 0; i < samples; ++i) {
    clean();
    ns.push_back(timer.ns(launch));
  }
  CUDA_CHECK(cudaGetLastError());
  std::printf("%s\n    {\"width_bytes\": %d, \"mask\": \"%s\", \"aligned\": %s, \"cache\": \"%s\", "
              "\"level\": \"%s\", \"ctas_per_sm\": %d, \"ctas_per_sm_fit\": %d, \"num_warps\": %d, "
              "\"loads_in_flight\": %d, \"share\": %d, \"column_step_bytes\": %d, \"grid\": %d, \"rows_per_cta\": %d, "
              "\"pitch_bytes\": %zu, \"batches\": %d, \"sweeps\": %d, \"smem_bytes\": %d, "
              "\"dram_request_bytes\": %zu, \"elapsed_ns\": [",
              first ? "" : ",", W, MASK == kMaskNone ? "none" : (MASK == kMaskTrue ? "true" : "tail"),
              offset ? "false" : "true", CG ? "cg" : "default", dram ? "dram" : "l2", cps, fit, warps, N,
              share, step_bytes, grid, rows_per_cta, pitch, batches, sweeps, smem, dram_bytes);
  for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.1f", i ? ", " : "", ns[i]);
  std::printf("]}");
  first = false;
}

template <int W, int N, bool CG, int MASK>
void gmem_narrow_sweep(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, uint32_t* sink,
                       const uint4* flush, uint4* flush_sink, size_t flush_vec4, int samples,
                       int warmup, bool& first) {
  for (int level = 0; level < 2; ++level)
    for (int offset = 0; offset < 2; ++offset)
      for (int cps = 1; cps <= 2; ++cps)
        for (int warps = 4; warps <= 8; warps *= 2)
          gmem_narrow_point<W, N, CG, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4,
                                            level == 1, offset == 1, cps, warps, 1, 32 * W, samples,
                                            warmup, first);
}

template <int W, int N, int MASK>
void gmem_narrow_caches(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, uint32_t* sink,
                        const uint4* flush, uint4* flush_sink, size_t flush_vec4, int samples,
                        int warmup, bool& first) {
  gmem_narrow_sweep<W, N, false, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
  gmem_narrow_sweep<W, N, true, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
}

template <int W, int MASK>
void gmem_narrow_depths(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, uint32_t* sink,
                        const uint4* flush, uint4* flush_sink, size_t flush_vec4, int samples,
                        int warmup, bool& first) {
  gmem_narrow_caches<W, 8, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
  gmem_narrow_caches<W, 16, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
  gmem_narrow_caches<W, 32, MASK>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
}

// ---------------------------------------------------------------------------
// sync-loop-narrow
// ---------------------------------------------------------------------------

// One stage is ROWS rows of 128 B. A W-byte warp load covers 32 * W bytes:
// half a row at W = 2, four rows at W = 16.
template <int W, int ROWS, int WARPS, int MMA_ITERS>
__global__ void sync_loop_narrow_kernel(const uint8_t* __restrict__ src, float* __restrict__ sink,
                                        size_t pitch, int steps, int distinct) {
  extern __shared__ uint4 stage[];
  constexpr int ROW_BYTES = 128;
  constexpr int STAGE_BYTES = ROWS * ROW_BYTES;
  constexpr int THREADS = WARPS * 32;
  constexpr int PER_THREAD = STAGE_BYTES / (THREADS * W);
  constexpr int LD_N = STAGE_BYTES / (WARPS * 512);
  static_assert(PER_THREAD >= 1, "stage too small");
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint8_t* base = src + static_cast<size_t>(blockIdx.x) * ROWS * pitch;
  uint8_t* smem = reinterpret_cast<uint8_t*>(stage);

  uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  uint32_t live = 0u;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int step = 0; step < steps; ++step) {
    const uint8_t* tile = base + static_cast<size_t>(step % distinct) * ROW_BYTES;
    if constexpr (W == 2) {
      uint16_t v[PER_THREAD];
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) {
        const int e = i * THREADS + tid;  // element in the stage
        const int row = e / (ROW_BYTES / 2), col = e % (ROW_BYTES / 2);
        asm volatile("ld.global.u16 %0, [%1];\n" : "=h"(v[i]) : "l"(tile + row * pitch + col * 2));
      }
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) {
        const int e = i * THREADS + tid;
        const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem + e * 2));
        asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(s), "h"(v[i]) : "memory");
      }
    } else {
      uint4 v[PER_THREAD];
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) {
        const int e = i * THREADS + tid;  // 16-byte vector in the stage
        const int row = e / (ROW_BYTES / 16), col = e % (ROW_BYTES / 16);
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(v[i].x), "=r"(v[i].y), "=r"(v[i].z), "=r"(v[i].w)
                     : "l"(tile + row * pitch + col * 16));
      }
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) {
        const int e = i * THREADS + tid;
        const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem + e * 16));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(s), "r"(v[i].x), "r"(v[i].y), "r"(v[i].z), "r"(v[i].w) : "memory");
      }
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      const uint32_t s = static_cast<uint32_t>(
          __cvta_generic_to_shared(stage + (static_cast<size_t>(r) * WARPS + warp) * 32 + lane));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s) : "memory");
      live ^= r0 ^ r1 ^ r2 ^ r3;
      if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
    }
    #pragma unroll 1
    for (int m = 0; m < MMA_ITERS; ++m) {
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
  }
  // Keep every ldmatrix live: `live` is not provably constant.
  const float sum = static_cast<float>(live & 1u) + c0 + c1 + c2 + c3;
  if (live == 0x9e3779b9u || sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

template <int W, int ROWS, int MMA_ITERS>
void sync_loop_narrow_point(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, float* sink,
                            const uint4* flush, uint4* flush_sink, size_t flush_vec4, bool dram,
                            bool offset, int cps, int samples, int warmup, bool& first,
                            size_t dram_bytes = kDramBytes) {
  constexpr int WARPS = 4;
  constexpr int STAGE_BYTES = ROWS * 128;
  auto kernel = sync_loop_narrow_kernel<W, ROWS, WARPS, MMA_ITERS>;
  const int smem = std::max(STAGE_BYTES, pad_bytes(prop, cps));
  set_smem(kernel, smem);
  int fit = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, WARPS * 32, smem));
  const int sms = prop.multiProcessorCount;
  const int grid = sms * cps;
  // Rows of one CTA hold all its stages side by side: stage s is bytes
  // [s * 128, s * 128 + 128) of each row, like K steps of a GEMM tile.
  const size_t requested = dram ? dram_bytes : kL2Bytes;
  int steps = static_cast<int>(std::max<size_t>(1, requested / (static_cast<size_t>(grid) * STAGE_BYTES)));
  int distinct = steps;
  if (!dram) {
    distinct = static_cast<int>(std::max<size_t>(1, kL2Set / (static_cast<size_t>(grid) * STAGE_BYTES)));
    steps = std::max(1, steps / distinct) * distinct;
  }
  const size_t pitch = round_up(static_cast<size_t>(distinct) * 128, 128) + (offset ? W : 0);
  if (static_cast<size_t>(grid) * ROWS * pitch > kBufferBytes) {
    std::fprintf(stderr, "sync-loop-narrow: buffer too small\n");
    std::exit(3);
  }
  // L2-hot stages wrap within `distinct` columns.
  auto launch = [&]() { kernel<<<grid, WARPS * 32, smem>>>(buf, sink, pitch, steps, distinct); };
  auto clean = [&]() {
    if (dram) l2_read_cg_kernel<<<sms * 4, 512>>>(flush, flush_sink, flush_vec4, 1);
  };
  for (int i = 0; i < warmup; ++i) { clean(); launch(); }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ns;
  for (int i = 0; i < samples; ++i) {
    clean();
    ns.push_back(timer.ns(launch));
  }
  CUDA_CHECK(cudaGetLastError());
  std::printf("%s\n    {\"width_bytes\": %d, \"aligned\": %s, \"level\": \"%s\", \"ctas_per_sm\": %d, "
              "\"ctas_per_sm_fit\": %d, \"num_warps\": %d, \"stage_bytes\": %d, \"mma_per_warp\": %d, "
              "\"grid\": %d, \"pitch_bytes\": %zu, \"steps\": %d, \"distinct_steps\": %d, "
              "\"smem_bytes\": %d, \"elapsed_ns\": [",
              first ? "" : ",", W, offset ? "false" : "true", dram ? "dram" : "l2", cps, fit, WARPS,
              STAGE_BYTES, MMA_ITERS, grid, pitch, steps, distinct, smem);
  for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.1f", i ? ", " : "", ns[i]);
  std::printf("]}");
  first = false;
}

template <int W, int ROWS, int MMA_ITERS>
void sync_loop_narrow_sweep(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, float* sink,
                            const uint4* flush, uint4* flush_sink, size_t flush_vec4, int samples,
                            int warmup, bool& first) {
  for (int level = 0; level < 2; ++level)
    for (int offset = 0; offset < (W == 2 ? 2 : 1); ++offset)
      for (int cps = 1; cps <= 3; ++cps)
        sync_loop_narrow_point<W, ROWS, MMA_ITERS>(prop, timer, buf, sink, flush, flush_sink, flush_vec4,
                                                   level == 1, offset == 1, cps, samples, warmup, first);
}

template <int W, int MMA_ITERS>
void sync_loop_narrow_stages(const cudaDeviceProp& prop, Timer& timer, const uint8_t* buf, float* sink,
                             const uint4* flush, uint4* flush_sink, size_t flush_vec4, int samples,
                             int warmup, bool& first) {
  sync_loop_narrow_sweep<W, 32, MMA_ITERS>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
  sync_loop_narrow_sweep<W, 64, MMA_ITERS>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
  sync_loop_narrow_sweep<W, 128, MMA_ITERS>(prop, timer, buf, sink, flush, flush_sink, flush_vec4, samples, warmup, first);
}

struct Buffers {
  uint8_t* data = nullptr;
  uint4* flush = nullptr;
  uint4* flush_sink = nullptr;
  uint32_t* sink = nullptr;
  float* fsink = nullptr;
  size_t flush_vec4 = 0;
  Buffers() {
    const size_t flush_bytes = 256ULL << 20;
    flush_vec4 = flush_bytes / sizeof(uint4);
    CUDA_CHECK(cudaMalloc(&data, kBufferBytes));
    CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
    CUDA_CHECK(cudaMalloc(&flush_sink, 1024 * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&sink, 1 << 20));
    CUDA_CHECK(cudaMalloc(&fsink, 1 << 20));
    CUDA_CHECK(cudaMemset(data, 0x3c, kBufferBytes));
    CUDA_CHECK(cudaMemset(flush, 25, flush_bytes));
  }
  ~Buffers() {
    cudaFree(data);
    cudaFree(flush);
    cudaFree(flush_sink);
    cudaFree(sink);
    cudaFree(fsink);
  }
};

template <int W, int MASK>
void line_step_masks(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                     int warmup, bool& first) {
  for (int cg = 0; cg < 2; ++cg)
    for (int level = 0; level < 2; ++level)
      for (int offset = 0; offset < 2; ++offset)
        for (int cps = 1; cps <= 2; ++cps)
          for (int warps = 4; warps <= 8; warps *= 2) {
            if (cg)
              gmem_narrow_point<W, 16, true, MASK>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                   b.flush_vec4, level == 1, offset == 1, cps, warps, 1,
                                                   128, samples, warmup, first);
            else
              gmem_narrow_point<W, 16, false, MASK>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                    b.flush_vec4, level == 1, offset == 1, cps, warps, 1,
                                                    128, samples, warmup, first);
          }
}

}  // namespace narrow

static int run_gmem_narrow(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"gmem_load_narrow\": [", samples);
  gmem_narrow_depths<1, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<1, kMaskTrue>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<2, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<2, kMaskTrue>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<2, kMaskTail>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<4, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<4, kMaskTrue>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<4, kMaskTail>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<16, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  gmem_narrow_depths<16, kMaskTrue>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  // One line per batch for the loads narrower than a line, 16 in flight.
  line_step_masks<1, kMaskNone>(prop, timer, b, samples, warmup, first);
  line_step_masks<1, kMaskTrue>(prop, timer, b, samples, warmup, first);
  line_step_masks<2, kMaskNone>(prop, timer, b, samples, warmup, first);
  line_step_masks<2, kMaskTrue>(prop, timer, b, samples, warmup, first);
  line_step_masks<2, kMaskTail>(prop, timer, b, samples, warmup, first);
  // Sharing: four CTAs read the same rows, DRAM only.
  for (int offset = 0; offset < 2; ++offset)
    for (int cps = 1; cps <= 2; ++cps) {
      gmem_narrow_point<2, 16, false, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                 b.flush_vec4, true, offset == 1, cps, 8, 4, 64, samples, warmup, first);
      gmem_narrow_point<16, 16, false, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                  b.flush_vec4, true, offset == 1, cps, 8, 4, 512, samples, warmup, first);
    }
  std::printf("\n  ]\n}\n");
  return 0;
}

static int run_sync_loop_narrow(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"sync_loop_narrow\": [", samples);
  sync_loop_narrow_stages<2, 0>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  sync_loop_narrow_stages<2, 8>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  sync_loop_narrow_stages<2, 32>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  sync_loop_narrow_stages<16, 0>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  sync_loop_narrow_stages<16, 8>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  sync_loop_narrow_stages<16, 32>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink, b.flush_vec4, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}
