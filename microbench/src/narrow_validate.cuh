// Follow-ups to the narrow-load benchmarks (H5). Included after narrow_kernels.cuh.
//
// gmem-narrow-validate and sync-loop-narrow-validate: new points for the
// DRAM-side narrow request form, none of them measured before.
// stage-narrow: register staging split into its parts (global loads only,
// shared memory stores only, both), to place the per-element cost.

namespace narrow {

template <int W, int N>
void validate_depth(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                    int warmup, bool& first) {
  // DRAM: two request volumes per sample, neither measured before.
  for (size_t mib : {128, 512})
    for (int offset = 0; offset < 2; ++offset)
      for (int cps : {1, 3})
        for (int warps = 4; warps <= 8; warps *= 2)
          gmem_narrow_point<W, N, true, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                   b.flush_vec4, true, offset == 1, cps, warps, 1,
                                                   32 * W, samples, warmup, first, mib << 20);
  // Hot L2 regression points.
  for (int offset = 0; offset < 2; ++offset)
    for (int cps : {1, 3})
      for (int warps = 4; warps <= 8; warps *= 2)
        gmem_narrow_point<W, N, true, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink,
                                                 b.flush_vec4, false, offset == 1, cps, warps, 1,
                                                 32 * W, samples, warmup, first);
}

template <int W>
void validate_width(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                    int warmup, bool& first) {
  validate_depth<W, 4>(prop, timer, b, samples, warmup, first);
  validate_depth<W, 12>(prop, timer, b, samples, warmup, first);
  validate_depth<W, 24>(prop, timer, b, samples, warmup, first);
}

template <int W, int ROWS, int MMA_ITERS>
void validate_sync(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                   int warmup, bool& first) {
  for (int offset = 0; offset < (W == 2 ? 2 : 1); ++offset)
    for (int cps : {1, 2, 4})
      sync_loop_narrow_point<W, ROWS, MMA_ITERS>(prop, timer, b.data, b.fsink, b.flush, b.flush_sink,
                                                 b.flush_vec4, true, offset == 1, cps, samples, warmup,
                                                 first);
}

// ---------------------------------------------------------------------------
// stage-narrow
// ---------------------------------------------------------------------------

constexpr int kLoadOnly = 0;
constexpr int kStoreOnly = 1;
constexpr int kLoadStore = 2;

template <int W>
struct Elem;
template <>
struct Elem<1> { using T = uint16_t; };
template <>
struct Elem<2> { using T = uint16_t; };
template <>
struct Elem<4> { using T = uint32_t; };
template <>
struct Elem<16> { using T = uint4; };

template <int W>
__device__ __forceinline__ typename Elem<W>::T ldg(const uint8_t* p) {
  typename Elem<W>::T v;
  if constexpr (W == 1) {
    asm volatile("ld.global.u8 %0, [%1];\n" : "=h"(v) : "l"(p));
  } else if constexpr (W == 2) {
    asm volatile("ld.global.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
  } else if constexpr (W == 4) {
    asm volatile("ld.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  } else {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  }
  return v;
}

template <int W>
__device__ __forceinline__ void sts(uint32_t addr, typename Elem<W>::T v) {
  if constexpr (W == 1) {
    asm volatile("st.shared.u8 [%0], %1;\n" :: "r"(addr), "h"(v) : "memory");
  } else if constexpr (W == 2) {
    asm volatile("st.shared.u16 [%0], %1;\n" :: "r"(addr), "h"(v) : "memory");
  } else if constexpr (W == 4) {
    asm volatile("st.shared.u32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory");
  } else {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
  }
}

template <int W>
__device__ __forceinline__ uint32_t fold(typename Elem<W>::T v) {
  if constexpr (W == 16) return v.x ^ v.y ^ v.z ^ v.w;
  else return static_cast<uint32_t>(v);
}

template <int W>
__device__ __forceinline__ typename Elem<W>::T make(uint32_t x) {
  if constexpr (W == 16) return make_uint4(x, x ^ 1u, x ^ 2u, x ^ 3u);
  else return static_cast<typename Elem<W>::T>(x);
}

// One stage is ROWS rows of 128 B, element e of the stage at byte e * W, as
// in sync-loop-narrow. The variant keeps or drops the global loads and the
// shared memory stores; the barrier, ldmatrix and MMA chain always run.
template <int VARIANT, int W, int ROWS, int MMA_ITERS>
__global__ void stage_narrow_kernel(const uint8_t* __restrict__ src, float* __restrict__ sink,
                                    size_t pitch, int steps, int distinct) {
  extern __shared__ uint4 stage[];
  constexpr int WARPS = 4;
  constexpr int THREADS = WARPS * 32;
  constexpr int STAGE_BYTES = ROWS * 128;
  constexpr int PER_THREAD = STAGE_BYTES / (THREADS * W);
  constexpr int LD_N = STAGE_BYTES / (WARPS * 512);
  using T = typename Elem<W>::T;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint8_t* base = src + static_cast<size_t>(blockIdx.x) * ROWS * pitch;
  uint8_t* smem = reinterpret_cast<uint8_t*>(stage);
  uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  uint32_t live = 0u;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int step = 0; step < steps; ++step) {
    const uint8_t* tile = base + static_cast<size_t>(step % distinct) * 128;
    T v[PER_THREAD];
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      const int e = i * THREADS + tid;
      if constexpr (VARIANT == kStoreOnly) {
        v[i] = make<W>(static_cast<uint32_t>(step) * 2654435761u + e);
      } else {
        const int row = e / (128 / W), col = e % (128 / W);
        v[i] = ldg<W>(tile + row * pitch + col * W);
      }
    }
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      const int e = i * THREADS + tid;
      if constexpr (VARIANT == kLoadOnly) {
        live ^= fold<W>(v[i]);
      } else {
        sts<W>(static_cast<uint32_t>(__cvta_generic_to_shared(smem + e * W)), v[i]);
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
  const float sum = static_cast<float>(live & 1u) + c0 + c1 + c2 + c3;
  if (live == 0x9e3779b9u || sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

template <int VARIANT, int W, int ROWS, int MMA_ITERS>
void stage_narrow_point(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int cps,
                        int samples, int warmup, bool& first) {
  constexpr int WARPS = 4;
  constexpr int STAGE_BYTES = ROWS * 128;
  auto kernel = stage_narrow_kernel<VARIANT, W, ROWS, MMA_ITERS>;
  const int smem = std::max(STAGE_BYTES, pad_bytes(prop, cps));
  set_smem(kernel, smem);
  int fit = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, WARPS * 32, smem));
  const int grid = prop.multiProcessorCount * cps;
  const int distinct = static_cast<int>(std::max<size_t>(1, kL2Set / (static_cast<size_t>(grid) * STAGE_BYTES)));
  int steps = static_cast<int>(std::max<size_t>(1, kL2Bytes / (static_cast<size_t>(grid) * STAGE_BYTES)));
  steps = std::max(1, steps / distinct) * distinct;
  const size_t pitch = round_up(static_cast<size_t>(distinct) * 128, 128);
  auto launch = [&]() { kernel<<<grid, WARPS * 32, smem>>>(b.data, b.fsink, pitch, steps, distinct); };
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ns;
  for (int i = 0; i < samples; ++i) ns.push_back(timer.ns(launch));
  CUDA_CHECK(cudaGetLastError());
  const char* names[] = {"load_only", "store_only", "load_store"};
  std::printf("%s\n    {\"variant\": \"%s\", \"width_bytes\": %d, \"ctas_per_sm\": %d, \"ctas_per_sm_fit\": %d, "
              "\"num_warps\": %d, \"stage_bytes\": %d, \"mma_per_warp\": %d, \"grid\": %d, "
              "\"pitch_bytes\": %zu, \"steps\": %d, \"distinct_steps\": %d, \"smem_bytes\": %d, \"elapsed_ns\": [",
              first ? "" : ",", names[VARIANT], W, cps, fit, WARPS, STAGE_BYTES, MMA_ITERS, grid, pitch, steps,
              distinct, smem);
  for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.1f", i ? ", " : "", ns[i]);
  std::printf("]}");
  first = false;
}

template <int VARIANT, int W, int ROWS>
void stage_narrow_cps(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                      int warmup, bool& first) {
  for (int cps = 1; cps <= 3; ++cps) {
    stage_narrow_point<VARIANT, W, ROWS, 0>(prop, timer, b, cps, samples, warmup, first);
    stage_narrow_point<VARIANT, W, ROWS, 32>(prop, timer, b, cps, samples, warmup, first);
  }
}

template <int VARIANT, int W>
void stage_narrow_stages(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                         int warmup, bool& first) {
  stage_narrow_cps<VARIANT, W, 32>(prop, timer, b, samples, warmup, first);
  stage_narrow_cps<VARIANT, W, 128>(prop, timer, b, samples, warmup, first);
}

template <int VARIANT>
void stage_narrow_widths(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                         int warmup, bool& first) {
  stage_narrow_stages<VARIANT, 2>(prop, timer, b, samples, warmup, first);
  stage_narrow_stages<VARIANT, 4>(prop, timer, b, samples, warmup, first);
  stage_narrow_stages<VARIANT, 16>(prop, timer, b, samples, warmup, first);
}

// stage-narrow-batched: like stage-narrow, but every thread issues all its
// loads before its first shared memory store, as the Triton GEMM kernels do:
// each store address carries `gate`, the XOR of every loaded value masked by
// the run-time zero `zero`. Stores go to Triton's swizzled layout for 128 B
// rows (16 B chunks XORed with the row index, 8 phases).
__device__ __forceinline__ uint32_t swizzled(int row, int byte) {
  return static_cast<uint32_t>(row * 128 + ((((byte >> 4) ^ (row & 7)) << 4) | (byte & 15)));
}

template <int VARIANT, int W, int ROWS, int MMA_ITERS, int WARPS = 4>
__global__ void stage_batched_kernel(const uint8_t* __restrict__ src, float* __restrict__ sink,
                                     size_t pitch, int steps, int distinct, uint32_t zero) {
  extern __shared__ uint4 stage[];
  constexpr int THREADS = WARPS * 32;
  constexpr int STAGE_BYTES = ROWS * 128;
  constexpr int PER_THREAD = STAGE_BYTES / (THREADS * W);
  constexpr int LD_N = STAGE_BYTES / (WARPS * 512);
  using T = typename Elem<W>::T;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint8_t* base = src + static_cast<size_t>(blockIdx.x) * ROWS * pitch;
  const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(stage));
  uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  uint32_t live = 0u;
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int step = 0; step < steps; ++step) {
    const uint8_t* tile = base + static_cast<size_t>(step % distinct) * 128;
    T v[PER_THREAD];
    uint32_t gate = 0u;
    #pragma unroll
    for (int i = 0; i < PER_THREAD; ++i) {
      const int e = i * THREADS + tid;
      if constexpr (VARIANT == kStoreOnly) {
        v[i] = make<W>(static_cast<uint32_t>(step) * 2654435761u + e);
      } else {
        const int row = e / (128 / W), col = e % (128 / W);
        v[i] = ldg<W>(tile + row * pitch + col * W);
      }
    }
    if constexpr (VARIANT != kStoreOnly) {
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) gate ^= fold<W>(v[i]);
      gate &= zero;
      live ^= gate;
    }
    if constexpr (VARIANT != kLoadOnly) {
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) {
        const int e = i * THREADS + tid;
        const int row = e / (128 / W), byte = (e % (128 / W)) * W;
        sts<W>(smem + swizzled(row, byte) + gate, v[i]);
      }
    } else {
      #pragma unroll
      for (int i = 0; i < PER_THREAD; ++i) live ^= fold<W>(v[i]);
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
  const float sum = static_cast<float>(live & 1u) + c0 + c1 + c2 + c3;
  if (live == 0x9e3779b9u || sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

template <int VARIANT, int W, int ROWS, int MMA_ITERS, int WARPS = 4>
void stage_batched_point(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int cps,
                         int samples, int warmup, bool& first, size_t l2_bytes = kL2Bytes) {
  constexpr int STAGE_BYTES = ROWS * 128;
  auto kernel = stage_batched_kernel<VARIANT, W, ROWS, MMA_ITERS, WARPS>;
  const int smem = std::max(STAGE_BYTES, pad_bytes(prop, cps));
  set_smem(kernel, smem);
  int fit = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, WARPS * 32, smem));
  const int grid = prop.multiProcessorCount * cps;
  const int distinct = static_cast<int>(std::max<size_t>(1, kL2Set / (static_cast<size_t>(grid) * STAGE_BYTES)));
  int steps = static_cast<int>(std::max<size_t>(1, l2_bytes / (static_cast<size_t>(grid) * STAGE_BYTES)));
  steps = std::max(1, steps / distinct) * distinct;
  const size_t pitch = round_up(static_cast<size_t>(distinct) * 128, 128);
  auto launch = [&]() { kernel<<<grid, WARPS * 32, smem>>>(b.data, b.fsink, pitch, steps, distinct, 0u); };
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ns;
  for (int i = 0; i < samples; ++i) ns.push_back(timer.ns(launch));
  CUDA_CHECK(cudaGetLastError());
  const char* names[] = {"load_only", "store_only", "load_store"};
  std::printf("%s\n    {\"variant\": \"%s\", \"width_bytes\": %d, \"ctas_per_sm\": %d, \"ctas_per_sm_fit\": %d, "
              "\"num_warps\": %d, \"stage_bytes\": %d, \"mma_per_warp\": %d, \"grid\": %d, "
              "\"pitch_bytes\": %zu, \"steps\": %d, \"distinct_steps\": %d, \"smem_bytes\": %d, \"elapsed_ns\": [",
              first ? "" : ",", names[VARIANT], W, cps, fit, WARPS, STAGE_BYTES, MMA_ITERS, grid, pitch, steps,
              distinct, smem);
  for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.1f", i ? ", " : "", ns[i]);
  std::printf("]}");
  first = false;
}

template <int VARIANT, int W, int ROWS>
void stage_batched_cps(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                       int warmup, bool& first) {
  for (int cps = 1; cps <= 3; ++cps) {
    stage_batched_point<VARIANT, W, ROWS, 0>(prop, timer, b, cps, samples, warmup, first);
    stage_batched_point<VARIANT, W, ROWS, 32>(prop, timer, b, cps, samples, warmup, first);
  }
}

template <int VARIANT, int W>
void stage_batched_stages(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                          int warmup, bool& first) {
  stage_batched_cps<VARIANT, W, 32>(prop, timer, b, samples, warmup, first);
  stage_batched_cps<VARIANT, W, 64>(prop, timer, b, samples, warmup, first);
  stage_batched_cps<VARIANT, W, 128>(prop, timer, b, samples, warmup, first);
}

template <int VARIANT>
void stage_batched_widths(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                          int warmup, bool& first) {
  stage_batched_stages<VARIANT, 2>(prop, timer, b, samples, warmup, first);
  stage_batched_stages<VARIANT, 4>(prop, timer, b, samples, warmup, first);
  stage_batched_stages<VARIANT, 16>(prop, timer, b, samples, warmup, first);
}

}  // namespace narrow

static int run_gmem_narrow_validate(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"gmem_load_narrow\": [", samples);
  // The calibration point of the DRAM-side form, measured in the same session.
  gmem_narrow_point<1, 16, true, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4,
                                            true, false, 2, 8, 1, 32, samples, warmup, first);
  validate_width<1>(prop, timer, b, samples, warmup, first);
  validate_width<2>(prop, timer, b, samples, warmup, first);
  validate_width<4>(prop, timer, b, samples, warmup, first);
  validate_width<16>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

// The calibration point of the DRAM-side form alone, run as its own process
// between repeats of gmem-narrow-validate.
static int run_gmem_narrow_calibrate(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"gmem_load_narrow\": [", samples);
  gmem_narrow_point<1, 16, true, kMaskNone>(prop, timer, b.data, b.sink, b.flush, b.flush_sink, b.flush_vec4,
                                            true, false, 2, 8, 1, 32, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

static int run_sync_loop_narrow_validate(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"sync_loop_narrow\": [", samples);
  validate_sync<2, 16, 0>(prop, timer, b, samples, warmup, first);
  validate_sync<2, 16, 16>(prop, timer, b, samples, warmup, first);
  validate_sync<2, 96, 0>(prop, timer, b, samples, warmup, first);
  validate_sync<2, 96, 16>(prop, timer, b, samples, warmup, first);
  validate_sync<16, 16, 0>(prop, timer, b, samples, warmup, first);
  validate_sync<16, 16, 16>(prop, timer, b, samples, warmup, first);
  validate_sync<16, 96, 0>(prop, timer, b, samples, warmup, first);
  validate_sync<16, 96, 16>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

static int run_stage_narrow(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"stage_narrow\": [", samples);
  stage_narrow_widths<kLoadOnly>(prop, timer, b, samples, warmup, first);
  stage_narrow_widths<kStoreOnly>(prop, timer, b, samples, warmup, first);
  stage_narrow_widths<kLoadStore>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

namespace narrow {

// New points for the register staging forms: 8 warps, 8 and 12 KiB stages,
// 1 to 16 B, MMA chains of 0 and 8, at 1 CTA per SM and at the most CTAs
// per SM that fit (`high`).
template <int VARIANT, int W, int ROWS, int MMA_ITERS>
void staging_validate_point(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                            int warmup, bool& first) {
  for (int cps : {1, 3})
    stage_batched_point<VARIANT, W, ROWS, MMA_ITERS, 8>(prop, timer, b, cps, samples, warmup, first);
}

template <int VARIANT, int W>
void staging_validate_width(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                            int warmup, bool& first) {
  staging_validate_point<VARIANT, W, 64, 0>(prop, timer, b, samples, warmup, first);
  staging_validate_point<VARIANT, W, 64, 8>(prop, timer, b, samples, warmup, first);
  staging_validate_point<VARIANT, W, 96, 0>(prop, timer, b, samples, warmup, first);
  staging_validate_point<VARIANT, W, 96, 8>(prop, timer, b, samples, warmup, first);
}

template <int VARIANT>
void staging_validate_variant(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                              int warmup, bool& first) {
  staging_validate_width<VARIANT, 1>(prop, timer, b, samples, warmup, first);
  staging_validate_width<VARIANT, 2>(prop, timer, b, samples, warmup, first);
  staging_validate_width<VARIANT, 4>(prop, timer, b, samples, warmup, first);
  staging_validate_width<VARIANT, 16>(prop, timer, b, samples, warmup, first);
}

}  // namespace narrow

static int run_stage_narrow_validate(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"stage_narrow\": [", samples);
  // The two calibration points, remeasured in the same session.
  stage_batched_point<kStoreOnly, 2, 128, 0>(prop, timer, b, 3, samples, warmup, first);
  stage_batched_point<kLoadOnly, 2, 32, 0>(prop, timer, b, 1, samples, warmup, first);
  staging_validate_variant<kLoadOnly>(prop, timer, b, samples, warmup, first);
  staging_validate_variant<kStoreOnly>(prop, timer, b, samples, warmup, first);
  staging_validate_variant<kLoadStore>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

static int run_stage_narrow_batched(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"stage_narrow\": [", samples);
  stage_batched_widths<kLoadOnly>(prop, timer, b, samples, warmup, first);
  stage_batched_widths<kStoreOnly>(prop, timer, b, samples, warmup, first);
  stage_batched_widths<kLoadStore>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

namespace narrow {

// Last H5 staging round: new points for b1 + b2 (4 warps, 6 and 10 KiB
// stages, 2 and 4 CTAs per SM, 1 to 16 B, MMA chains of 0 and 8). Each sample
// moves kK1Bytes so that every launch runs well over 1 ms.
constexpr size_t kK1Bytes = 9ULL << 30;
template <int VARIANT, int W, int ROWS, int MMA_ITERS>
void staging_k1_point(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                      int warmup, bool& first) {
  for (int cps : {2, 4})
    stage_batched_point<VARIANT, W, ROWS, MMA_ITERS, 4>(prop, timer, b, cps, samples, warmup, first,
                                                        kK1Bytes);
}

template <int VARIANT, int W>
void staging_k1_width(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                      int warmup, bool& first) {
  staging_k1_point<VARIANT, W, 48, 0>(prop, timer, b, samples, warmup, first);
  staging_k1_point<VARIANT, W, 48, 8>(prop, timer, b, samples, warmup, first);
  staging_k1_point<VARIANT, W, 80, 0>(prop, timer, b, samples, warmup, first);
  staging_k1_point<VARIANT, W, 80, 8>(prop, timer, b, samples, warmup, first);
}

template <int VARIANT>
void staging_k1_variant(const cudaDeviceProp& prop, Timer& timer, const Buffers& b, int samples,
                        int warmup, bool& first) {
  staging_k1_width<VARIANT, 1>(prop, timer, b, samples, warmup, first);
  staging_k1_width<VARIANT, 2>(prop, timer, b, samples, warmup, first);
  staging_k1_width<VARIANT, 4>(prop, timer, b, samples, warmup, first);
  staging_k1_width<VARIANT, 16>(prop, timer, b, samples, warmup, first);
}

}  // namespace narrow

static int run_stage_narrow_k1(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"stage_narrow\": [", samples);
  staging_k1_variant<kLoadOnly>(prop, timer, b, samples, warmup, first);
  staging_k1_variant<kStoreOnly>(prop, timer, b, samples, warmup, first);
  staging_k1_variant<kLoadStore>(prop, timer, b, samples, warmup, first);
  std::printf("\n  ]\n}\n");
  return 0;
}

// The two calibration points of b1 and b2, run as their own process.
static int run_stage_narrow_k1_calibrate(int device, int samples, int warmup) {
  using namespace narrow;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  Buffers b;
  Timer timer;
  bool first = true;
  std::printf("{\n  \"samples\": %d,\n  \"stage_narrow\": [", samples);
  stage_batched_point<kStoreOnly, 2, 128, 0>(prop, timer, b, 3, samples, warmup, first, kK1Bytes);
  stage_batched_point<kLoadOnly, 2, 32, 0>(prop, timer, b, 1, samples, warmup, first, kK1Bytes);
  std::printf("\n  ]\n}\n");
  return 0;
}
