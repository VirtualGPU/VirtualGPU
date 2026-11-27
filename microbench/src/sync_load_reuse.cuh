// sync-load-reuse: L1 reuse of synchronous loads between the CTAs of an SM (diagnostic, not
// part of a profile). Registration: docs/results/analysis/v0.3-sync-load-gemm-plan.md (as amended).
//
// The single-buffer register-staged loop of Triton's num_stages = 1 GEMM: load A, barrier,
// load B, store both to shared memory, barrier, a fixed block of ldmatrix and mma.sync. A
// stage is 16 KiB per CTA (8 KiB A, 8 KiB B), 4 warps. Each CTA walks its own region of the
// source in 16 KiB steps; with `shared` the region is the CTA's SM (%smid), so the CTAs of
// one SM read the same addresses in each step; otherwise it is the CTA's index. Each CTA
// records its SM so co-residency is checked, not assumed. Each CTA requests enough dynamic
// shared memory that only the largest carveout fits; it uses 16 KiB of it.

namespace sync_load_reuse {

constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kStageU4 = 16384 / 16;                // 1024 uint4 per stage
constexpr int kHalfVecs = kStageU4 / 2 / kThreads;  // 4 loads per thread for A, 4 for B
constexpr int kMmaIters = 2;                         // 8 mma.sync per warp per step

template <bool CG>
__device__ __forceinline__ uint4 load(const uint4* p) {
  uint4 v;
  if constexpr (CG) {
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  } else {
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  }
  return v;
}

__device__ __forceinline__ void store_shared(const uint4* s, uint4 v) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(s));
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w) : "memory");
}

template <bool CG>
__global__ void kernel(const uint4* __restrict__ src, float* __restrict__ sink,
                       int* __restrict__ smid_out, size_t region_u4, int steps_per_region,
                       int shared, int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (tid == 0) smid_out[blockIdx.x] = static_cast<int>(smid);
  const size_t region = shared ? smid : blockIdx.x;
  const uint4* base = src + region * region_u4;
  uint32_t a0 = 0x3c003c00u, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, live = 0u;
  float c[4][4] = {};

  for (int s = 0; s < steps; ++s) {
    const uint4* stage = base + static_cast<size_t>(s % steps_per_region) * kStageU4;
    uint4 va[kHalfVecs], vb[kHalfVecs];
    #pragma unroll
    for (int v = 0; v < kHalfVecs; ++v) va[v] = load<CG>(stage + v * kThreads + tid);
    asm volatile("bar.sync 0;" ::: "memory");
    #pragma unroll
    for (int v = 0; v < kHalfVecs; ++v) vb[v] = load<CG>(stage + kStageU4 / 2 + v * kThreads + tid);
    #pragma unroll
    for (int v = 0; v < kHalfVecs; ++v) {
      store_shared(smem + v * kThreads + tid, va[v]);
      store_shared(smem + kStageU4 / 2 + v * kThreads + tid, vb[v]);
    }
    asm volatile("bar.sync 0;" ::: "memory");
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t sa = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem + (r * kWarps + warp) * 32 + lane));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sa) : "memory");
      live ^= r0 ^ r1 ^ r2 ^ r3;
      if (r == 0) { b0 = r0; b1 = r1; } else { a0 = r0; a1 = r1; a2 = r2; a3 = r3; }
    }
    #pragma unroll
    for (int m = 0; m < kMmaIters; ++m) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  float sum = static_cast<float>(live);
  for (int j = 0; j < 4; ++j) sum += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (sum == -1.0f) sink[blockIdx.x * kThreads + tid] = sum;
}

}  // namespace sync_load_reuse

static int run_sync_load_reuse(int device, int samples, int warmup, LaunchSchedule* schedule) {
  namespace slr = sync_load_reuse;
  if (!schedule) {
    std::fprintf(stderr, "sync-load-reuse runs only under a launch schedule (pilot or frozen)\n");
    return 2;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2 = static_cast<size_t>(prop.l2CacheSize);
  // DRAM: at least 16 times L2 in total over the CTAs of the disjoint arm at 2 per SM.
  const size_t dram_region = std::max<size_t>(4 * LaunchSchedule::MiB, 16 * l2 / (2 * sm_count));
  const size_t l2_region = 64 * 1024;  // 4 stages; 6 MiB over 96 CTAs, inside GB10's L2
  const size_t src_bytes = dram_region * 2 * sm_count;
  const size_t flush_bytes = 4 * l2;
  uint4 *src = nullptr, *flush = nullptr;
  float* sink = nullptr;
  int* smid = nullptr;
  CUDA_CHECK(cudaMalloc(&src, src_bytes));
  CUDA_CHECK(cudaMemset(src, 0x3c, src_bytes));
  CUDA_CHECK(cudaMalloc(&flush, flush_bytes));
  CUDA_CHECK(cudaMemset(flush, 0, flush_bytes));
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(2 * sm_count) * slr::kThreads * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&smid, static_cast<size_t>(2 * sm_count) * sizeof(int)));
  // The request that leaves only the largest carveout: 2 x (50176 + 1024 reserved) = 102400 B
  // at two CTAs per SM, 101376 + 1024 at one.
  for (auto k : {slr::kernel<true>, slr::kernel<false>}) {
    CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376));
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"l2_bytes\": %zu,\n  \"points\": [",
              samples, sm_count, l2);
  bool first = true;
  for (const char* level : {"l2", "dram"}) {
    const bool dram = std::string(level) == "dram";
    const size_t region = dram ? dram_region : l2_region;
    for (int ctas : {1, 2}) {
      for (const char* sharing : {"disjoint", "shared"}) {
        const bool shared = std::string(sharing) == "shared";
        if (ctas == 1 && shared) continue;  // the same as disjoint with one CTA per SM
        for (const char* load : {"default", "cg"}) {
          const bool cg = std::string(load) == "cg";
          const int grid = ctas * sm_count;
          const size_t smem = ctas == 2 ? 50176 : 101376;
          std::printf("%s\n    {\"level\": \"%s\", \"ctas_per_sm\": %d, \"sharing\": \"%s\", "
                      "\"load\": \"%s\", \"region_bytes\": %zu, \"dynamic_smem_bytes\": %zu",
                      first ? "" : ",", level, ctas, sharing, load, region, smem);
          first = false;
          auto prepare = [&] {
            if (dram) {
              l2_read_cg_kernel<<<sm_count * 4, 256>>>(flush, reinterpret_cast<uint4*>(sink),
                                                        flush_bytes / sizeof(uint4), 1);
            }
          };
          auto launch = [&](int steps) {
            const size_t region_u4 = region / sizeof(uint4);
            const int per_region = static_cast<int>(region / 16384);
            if (cg) slr::kernel<true><<<grid, slr::kThreads, smem>>>(src, sink, smid, region_u4, per_region, shared, steps);
            else slr::kernel<false><<<grid, slr::kThreads, smem>>>(src, sink, smid, region_u4, per_region, shared, steps);
          };
          auto timed = schedule->time_point(1024, 1, 1 << 28, warmup, samples, start, stop, prepare, launch);
          std::vector<int> where(grid);
          CUDA_CHECK(cudaMemcpy(where.data(), smid, grid * sizeof(int), cudaMemcpyDeviceToHost));
          std::vector<int> per_sm(sm_count, 0);
          for (int s : where) if (s >= 0 && s < sm_count) ++per_sm[s];
          const auto [lo, hi] = std::minmax_element(per_sm.begin(), per_sm.end());
          std::printf(", \"ctas_per_sm_observed\": [%d, %d], \"ns_per_step_samples\": [", *lo, *hi);
          for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
            std::printf("%s%.4f", k ? ", " : "", timed.launch_ns[k] / timed.length);
          }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(src));
  CUDA_CHECK(cudaFree(flush));
  CUDA_CHECK(cudaFree(sink));
  CUDA_CHECK(cudaFree(smid));
  return 0;
}
