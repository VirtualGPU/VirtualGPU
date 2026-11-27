#include <type_traits>

// local-bw: local memory bandwidth per SM, L1 and L2 tiers (diagnostic, not part of a
// profile). Registration: docs/results/analysis/v0.3-local-pipe-bw-plan.md.
//
// One CTA per SM. Each thread owns a runtime-indexed local frame, so the compiler emits real
// LDL/STL. All lanes of a warp use the same frame offset, as ptxas spill reloads do
// (LDL [R1 + imm]), so one warp instruction moves 32 consecutive elements. Each step makes 8
// independent accesses at idx + k (k = 0..7), then advances idx by 8.

namespace local_bw {

template <int FRAME_BYTES, int WIDTH, bool STORE>
__global__ void kernel(uint32_t* __restrict__ sink, int steps) {
  using T = typename std::conditional<WIDTH == 16, uint4, uint32_t>::type;
  constexpr int N = FRAME_BYTES / WIDTH;
  static_assert(N >= 2 && (N & (N - 1)) == 0, "frame must hold a power of two elements");
  constexpr uint32_t mask = N - 1;
  T loc[N];
  const uint32_t tid = threadIdx.x, warp = tid >> 5;
  for (int i = 0; i < N; ++i) {
    if constexpr (WIDTH == 16) loc[i] = make_uint4(i, i + tid, i ^ tid, 1u);
    else loc[i] = static_cast<uint32_t>(i + tid);
  }
  uint32_t idx = (warp * 7u) & mask;
  uint32_t acc[8] = {};
  #pragma unroll 1
  for (int s = 0; s < steps; ++s) {
    idx = __shfl_sync(0xffffffffu, idx, 0);  // warp-uniform offset
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      const uint32_t j = (idx + static_cast<uint32_t>(k)) & mask;
      if constexpr (STORE) {
        if constexpr (WIDTH == 16) loc[j] = make_uint4(acc[k], s, k, tid);
        else loc[j] = acc[k] + static_cast<uint32_t>(s);
        acc[k] += 1u;
      } else {
        if constexpr (WIDTH == 16) {
          // An explicit ld.local.v4: the frame's address reaches the asm, so the compiler
          // keeps the frame in local memory instead of registers.
          uint64_t local;
          asm("cvta.to.local.u64 %0, %1;" : "=l"(local) : "l"(reinterpret_cast<uint64_t>(&loc[j])));
          uint32_t x, y, z, w;
          asm volatile("ld.local.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(local) : "memory");
          acc[k] ^= x ^ y ^ z ^ w;
        } else {
          acc[k] ^= loc[j];
        }
      }
    }
    idx = (idx + 8u) & mask;
  }
  uint32_t out = 0u;
  for (int k = 0; k < 8; ++k) out ^= acc[k];
  if constexpr (WIDTH == 16) out ^= loc[(idx + out) & mask].x;
  else out ^= loc[(idx + out) & mask];
  if (out == 0xdeadbeefu) sink[blockIdx.x * blockDim.x + tid] = out;
}

struct Point {
  bool store;
  int width, warps, frame;
  const char* tier;
  int length;
  std::vector<std::vector<double>> launch_ns;  // per allocation
  std::vector<double> mhz_before, mhz_after;
};

// The registered points (as amended): the L1 curves, the L1 frame check and the L2 tier.
static std::vector<Point> points() {
  std::vector<Point> out;
  const int curve[][2] = {{1, 128}, {2, 128}, {4, 128}, {8, 128}, {16, 64}, {32, 32}};
  for (bool store : {false, true})
    for (auto& c : curve) out.push_back({store, 4, c[0], c[1], "l1", 0, {}, {}, {}});
  // 16 B loads: 128 B frames only, which hold 8 distinct addresses (amended registration).
  for (int warps : {1, 2, 4, 8}) out.push_back({false, 16, warps, 128, "l1", 0, {}, {}, {}});
  for (int frame : {32, 64}) out.push_back({false, 4, 8, frame, "l1", 0, {}, {}, {}});
  for (auto [warps, frame] : {std::pair<int, int>{16, 1024}, {8, 2048}, {4, 4096}})
    out.push_back({false, 4, warps, frame, "l2", 0, {}, {}, {}});
  return out;
}

template <int F, int W, bool S>
static void set_max_l1() {
  CUDA_CHECK(cudaFuncSetAttribute(kernel<F, W, S>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
}

static void prepare_kernels() {
  set_max_l1<32, 4, false>(); set_max_l1<64, 4, false>(); set_max_l1<128, 4, false>();
  set_max_l1<1024, 4, false>(); set_max_l1<2048, 4, false>(); set_max_l1<4096, 4, false>();
  set_max_l1<32, 4, true>(); set_max_l1<64, 4, true>(); set_max_l1<128, 4, true>();
  set_max_l1<128, 16, false>();
}

static void launch(const Point& p, int grid, uint32_t* sink, int steps) {
  const int threads = p.warps * 32;
  if (p.store) {
    if (p.frame == 32) kernel<32, 4, true><<<grid, threads>>>(sink, steps);
    else if (p.frame == 64) kernel<64, 4, true><<<grid, threads>>>(sink, steps);
    else kernel<128, 4, true><<<grid, threads>>>(sink, steps);
  } else if (p.width == 16) {
    kernel<128, 16, false><<<grid, threads>>>(sink, steps);
  } else {
    switch (p.frame) {
      case 32: kernel<32, 4, false><<<grid, threads>>>(sink, steps); break;
      case 64: kernel<64, 4, false><<<grid, threads>>>(sink, steps); break;
      case 128: kernel<128, 4, false><<<grid, threads>>>(sink, steps); break;
      case 1024: kernel<1024, 4, false><<<grid, threads>>>(sink, steps); break;
      case 2048: kernel<2048, 4, false><<<grid, threads>>>(sink, steps); break;
      default: kernel<4096, 4, false><<<grid, threads>>>(sink, steps); break;
    }
  }
}

static double bytes_per_step(const Point& p) { return static_cast<double>(p.warps) * 32 * 8 * p.width; }

static void print_head(const Point& p) {
  std::printf("{\"op\": \"%s\", \"width_bytes\": %d, \"tier\": \"%s\", \"num_warps\": %d, "
              "\"frame_bytes_per_thread\": %d, \"footprint_bytes_per_sm\": %d",
              p.store ? "store" : "load", p.width, p.tier, p.warps, p.frame, p.warps * 32 * p.frame);
}

}  // namespace local_bw

// Local memory allocations every frozen point is read in (as spill-latency's reload points).
constexpr int kLocalBwAllocations = 15;

static int run_local_bw(int device, int samples, int warmup, LaunchSchedule* schedule) {
  namespace lb = local_bw;
  if (!schedule) {
    std::fprintf(stderr, "local-bw runs only under a launch schedule (pilot or frozen)\n");
    return 2;
  }
  std::vector<lb::Point> pts = lb::points();
  int sm_count = 0;
  uint32_t* sink = nullptr;
  cudaEvent_t ev_start, ev_stop, cal_start, cal_stop;
  auto open_context = [&]() {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    sm_count = prop.multiProcessorCount;
    CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(sm_count) * 1024 * sizeof(uint32_t)));
    for (cudaEvent_t* e : {&ev_start, &ev_stop, &cal_start, &cal_stop}) CUDA_CHECK(cudaEventCreate(e));
    lb::prepare_kernels();
  };
  auto close_context = [&]() {
    for (cudaEvent_t e : {ev_start, ev_stop, cal_start, cal_stop}) CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(sink));
  };
  auto rates = [&](const lb::Point& p, const std::vector<double>& ns) {
    std::printf(", \"gbps_per_sm_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i)
      std::printf("%s%.4f", i ? ", " : "", lb::bytes_per_step(p) * p.length / ns[i]);
    std::printf("]");
  };

  open_context();
  if (schedule->is_pilot()) {
    std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"points\": [", samples, sm_count);
    for (size_t i = 0; i < pts.size(); ++i) {
      lb::Point& p = pts[i];
      std::printf("%s\n    ", i ? "," : "");
      lb::print_head(p);
      schedule->launch_target_ns = std::string(p.tier) == "l2" ? 2.0e6 : LaunchSchedule::pilot_min_ns;
      auto timed = schedule->time_point(1024, 1, 1 << 30, warmup, samples, ev_start, ev_stop, [] {},
                                        [&](int steps) { lb::launch(p, sm_count, sink, steps); });
      p.length = timed.length;
      rates(p, timed.launch_ns);
      std::printf("}");
    }
    schedule->launch_target_ns = LaunchSchedule::pilot_min_ns;
    std::printf("\n  ]\n}\n");
    close_context();
    return 0;
  }

  for (lb::Point& p : pts) p.length = schedule->frozen_length(1, 1 << 30);
  auto read = [&](lb::Point& p) {
    p.mhz_before.push_back(1e3 * measure_cycles_per_ns(cal_start, cal_stop));
    for (int w = 0; w < warmup; ++w) lb::launch(p, sm_count, sink, p.length);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> ns;
    for (int i = 0; i < samples; ++i) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      lb::launch(p, sm_count, sink, p.length);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      ns.push_back(static_cast<double>(ms) * 1e6);
    }
    CUDA_CHECK(cudaGetLastError());
    p.mhz_after.push_back(1e3 * measure_cycles_per_ns(cal_start, cal_stop));
    p.launch_ns.push_back(ns);
  };
  for (int a = 0; a < kLocalBwAllocations; ++a) {
    if (a > 0) {
      close_context();
      CUDA_CHECK(cudaDeviceReset());
      open_context();
    }
    for (lb::Point& p : pts) read(p);
  }
  close_context();

  auto mean = [](const std::vector<double>& v) {
    double t = 0.0;
    for (double x : v) t += x;
    return t / static_cast<double>(v.size());
  };
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"allocations\": %d,\n  \"points\": [",
              samples, sm_count, kLocalBwAllocations);
  for (size_t i = 0; i < pts.size(); ++i) {
    lb::Point& p = pts[i];
    // Sample j is the mean over allocations of their j-th smallest launch, so the median of
    // the samples is the mean of the per-allocation medians (15 samples).
    std::vector<double> rank(samples, 0.0);
    for (std::vector<double> v : p.launch_ns) {
      std::sort(v.begin(), v.end());
      for (int j = 0; j < samples; ++j) rank[j] += v[j] / static_cast<double>(p.launch_ns.size());
    }
    std::printf("%s\n    ", i ? "," : "");
    lb::print_head(p);
    std::printf(", \"sm_mhz_before\": %.1f, \"sm_mhz_after\": %.1f, \"steps\": %d, \"launch_ns_samples\": [",
                mean(p.mhz_before), mean(p.mhz_after), p.length);
    for (int j = 0; j < samples; ++j) std::printf("%s%.1f", j ? ", " : "", rank[j]);
    std::printf("]");
    rates(p, rank);
    std::printf(", \"allocation_launch_ns\": [");
    for (size_t a = 0; a < p.launch_ns.size(); ++a) {
      std::printf("%s[", a ? ", " : "");
      for (int j = 0; j < samples; ++j) std::printf("%s%.1f", j ? ", " : "", p.launch_ns[a][j]);
      std::printf("]");
    }
    std::printf("]}");
  }
  std::printf("\n  ]\n}\n");
  return 0;
}
