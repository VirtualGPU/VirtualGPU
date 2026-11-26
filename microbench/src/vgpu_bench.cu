#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

static __device__ __forceinline__ uint32_t read_smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

static __device__ __forceinline__ uint64_t read_globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void l2_flush_clean_kernel(const uint4* __restrict__ buf,
                                      uint4* __restrict__ sink,
                                      size_t n) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (size_t i = idx; i < n; i += stride) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(buf + i));
    acc ^= (v.x + v.y + v.z + v.w);
  }
  if (acc == 0xdeadbeefu) {
    sink[idx & 1023u] = make_uint4(acc, acc, acc, acc);
  }
}

__global__ void l2_flush_kernel(uint4* __restrict__ buf, size_t n, uint32_t seed) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint4 val = make_uint4(seed, seed ^ 0x55555555u, seed ^ 0xaaaaaaaau, seed + 1u);
  for (size_t i = idx; i < n; i += stride) {
    buf[i] = val;
  }
}

__global__ void dram_read_kernel(const float4* __restrict__ src,
                                 float4* __restrict__ sink,
                                 size_t n_vec4,
                                 uint32_t* __restrict__ sm_seen) {
  if (threadIdx.x == 0) {
    atomicAdd(&sm_seen[read_smid()], 1u);
  }
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const uint4* src_u4 = reinterpret_cast<const uint4*>(src);

  // 4 independent 128-bit .cg loads in flight per thread to saturate DRAM MSHRs.
  uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  size_t i = tid;
  const size_t stride4 = 4 * stride;
  for (; i + 3 * stride < n_vec4; i += stride4) {
    uint4 v0, v1, v2, v3;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                 : "l"(src_u4 + i));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                 : "l"(src_u4 + i + stride));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v2.x), "=r"(v2.y), "=r"(v2.z), "=r"(v2.w)
                 : "l"(src_u4 + i + 2 * stride));
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v3.x), "=r"(v3.y), "=r"(v3.z), "=r"(v3.w)
                 : "l"(src_u4 + i + 3 * stride));
    acc0 ^= v0.x ^ v1.y ^ v2.z ^ v3.w;
    acc1 ^= v0.y ^ v1.z ^ v2.w ^ v3.x;
    acc2 ^= v0.z ^ v1.w ^ v2.x ^ v3.y;
    acc3 ^= v0.w ^ v1.x ^ v2.y ^ v3.z;
  }
  for (; i < n_vec4; i += stride) {
    uint4 v0;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                 : "l"(src_u4 + i));
    acc0 ^= v0.x ^ v0.y ^ v0.z ^ v0.w;
  }

  if ((acc0 ^ acc1 ^ acc2 ^ acc3) == 0xdeadbeefu) {
    sink[tid & 1023u] = make_float4(static_cast<float>(acc0), static_cast<float>(acc1),
                                    static_cast<float>(acc2), static_cast<float>(acc3));
  }
}

__global__ void dram_write_kernel(float4* __restrict__ dst,
                                  size_t n_vec4,
                                  float val,
                                  uint32_t* __restrict__ sm_seen) {
  if (threadIdx.x == 0) {
    atomicAdd(&sm_seen[read_smid()], 1u);
  }
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  float4 v = make_float4(val, val + 1.f, val + 2.f, val + 3.f);
  for (size_t i = tid; i < n_vec4; i += stride) {
    dst[i] = v;
  }
}

struct DispatchRecord {
  uint32_t cta_id;
  uint32_t sm_id;
  uint64_t start_ns;
  uint64_t end_ns;
};

__global__ void cta_dispatch_kernel(DispatchRecord* __restrict__ out,
                                    uint64_t min_duration_ns) {
  extern __shared__ uint8_t dyn_smem[];
  if (threadIdx.x == 0) {
    uint64_t t0 = read_globaltimer();
    uint32_t sm = read_smid();
    // Hold the CTA resident briefly so concurrent wave placement is observable.
    uint64_t t1 = t0;
    while (t1 - t0 < min_duration_ns) {
      t1 = read_globaltimer();
    }
    if (dyn_smem[0] == 255) {
      out[blockIdx.x].cta_id = 0;
    }
    out[blockIdx.x].cta_id = blockIdx.x;
    out[blockIdx.x].sm_id = sm;
    out[blockIdx.x].start_ns = t0;
    out[blockIdx.x].end_ns = t1;
  }
}

static void print_json_escaped(const char* s) {
  std::putchar('"');
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
    if (*p == '"' || *p == '\\') {
      std::putchar('\\');
    }
    std::putchar(*p);
  }
  std::putchar('"');
}

static int run_query(int device) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int runtime_ver = 0;
  int driver_ver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));
  CUDA_CHECK(cudaDriverGetVersion(&driver_ver));

  int max_blocks_per_sm = 0;
  int reserved_smem_per_block = 0;
  int max_smem_optin = 0;
  int clock_khz = 0;
  int mem_clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&reserved_smem_per_block, cudaDevAttrReservedSharedMemoryPerBlock, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
  cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);

  std::printf("{\n");
  std::printf("  \"device\": %d,\n", device);
  std::printf("  \"name\": ");
  print_json_escaped(prop.name);
  std::printf(",\n");
  std::printf("  \"major\": %d,\n", prop.major);
  std::printf("  \"minor\": %d,\n", prop.minor);
  std::printf("  \"integrated\": %s,\n", prop.integrated ? "true" : "false");
  std::printf("  \"runtime_version\": %d,\n", runtime_ver);
  std::printf("  \"driver_version\": %d,\n", driver_ver);
  std::printf("  \"sm_count\": %d,\n", prop.multiProcessorCount);
  std::printf("  \"warp_size\": %d,\n", prop.warpSize);
  std::printf("  \"max_threads_per_sm\": %d,\n", prop.maxThreadsPerMultiProcessor);
  std::printf("  \"max_warps_per_sm\": %d,\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
  std::printf("  \"max_blocks_per_sm\": %d,\n", max_blocks_per_sm);
  std::printf("  \"max_threads_per_block\": %d,\n", prop.maxThreadsPerBlock);
  std::printf("  \"registers_per_sm\": %d,\n", prop.regsPerMultiprocessor);
  std::printf("  \"registers_per_block\": %d,\n", prop.regsPerBlock);
  std::printf("  \"shared_mem_per_sm_bytes\": %zu,\n", prop.sharedMemPerMultiprocessor);
  std::printf("  \"shared_mem_per_block_bytes\": %zu,\n", prop.sharedMemPerBlock);
  std::printf("  \"max_shared_mem_per_block_optin_bytes\": %d,\n", max_smem_optin);
  std::printf("  \"reserved_shared_mem_per_cta_bytes\": %d,\n", reserved_smem_per_block);
  std::printf("  \"l2_size_bytes\": %d,\n", prop.l2CacheSize);
  std::printf("  \"total_global_mem_bytes\": %zu,\n", prop.totalGlobalMem);
  std::printf("  \"memory_bus_width_bits\": %d,\n", prop.memoryBusWidth);
  std::printf("  \"clock_khz\": %d,\n", clock_khz);
  std::printf("  \"mem_clock_khz\": %d\n", mem_clock_khz);
  std::printf("}\n");
  return 0;
}

static int run_dram_bw(int device, int samples, int warmup, size_t buffer_mb) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int sm_count = prop.multiProcessorCount;
  const size_t bytes = buffer_mb * 1024ULL * 1024ULL;
  const size_t n_vec4 = bytes / sizeof(float4);

  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 32ULL * 1024ULL * 1024ULL);
  const size_t flush_bytes = std::max<size_t>(l2_bytes * 2ULL, 96ULL * 1024ULL * 1024ULL);
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);

  float4* d_buf = nullptr;
  float4* d_sink = nullptr;
  uint4* d_flush = nullptr;
  uint32_t* d_sm_seen = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sm_seen, static_cast<size_t>(sm_count) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_buf, 1, bytes));
  CUDA_CHECK(cudaMemset(d_flush, 2, flush_bytes));

  std::vector<int> targets;
  auto add_target = [&](int s) {
    s = std::max(1, std::min(sm_count, s));
    if (std::find(targets.begin(), targets.end(), s) == targets.end()) {
      targets.push_back(s);
    }
  };
  add_target(1);
  add_target(2);
  add_target(4);
  add_target(sm_count / 4);
  add_target(sm_count / 2);
  add_target(sm_count);
  std::sort(targets.begin(), targets.end());

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const int threads = 512;
  const int flush_blocks = sm_count * 4;
  std::vector<uint32_t> h_sm_seen(static_cast<size_t>(sm_count), 0);

  auto measure_curve = [&](bool is_write, const std::vector<int>& sm_list) {
    std::printf("[");
    for (size_t ti = 0; ti < sm_list.size(); ++ti) {
      int target_sms = sm_list[ti];
      // Launch 1 CTA per target SM (512 threads = 16 warps/SM) for partial-SM points;
      // at full GPU, launch 4 CTAs/SM to saturate memory controllers.
      int grid = (target_sms == sm_count) ? (sm_count * 4) : target_sms;

      for (int w = 0; w < warmup; ++w) {
        l2_flush_clean_kernel<<<flush_blocks, 256>>>(d_flush, reinterpret_cast<uint4*>(d_sink), flush_vec4);
        if (is_write) {
          dram_write_kernel<<<grid, threads>>>(d_buf, n_vec4, 1.0f + w, d_sm_seen);
        } else {
          dram_read_kernel<<<grid, threads>>>(d_buf, d_sink, n_vec4, d_sm_seen);
        }
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> gbps_samples;
      gbps_samples.reserve(samples);
      int observed_sms = 0;

      for (int s = 0; s < samples; ++s) {
        l2_flush_clean_kernel<<<flush_blocks, 256>>>(d_flush, reinterpret_cast<uint4*>(d_sink), flush_vec4);
        CUDA_CHECK(cudaMemset(d_sm_seen, 0, static_cast<size_t>(sm_count) * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (is_write) {
          dram_write_kernel<<<grid, threads>>>(d_buf, n_vec4, 2.0f + s, d_sm_seen);
        } else {
          dram_read_kernel<<<grid, threads>>>(d_buf, d_sink, n_vec4, d_sm_seen);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double gbps = (static_cast<double>(bytes) / 1e9) / (static_cast<double>(ms) / 1e3);
        gbps_samples.push_back(gbps);

        if (s == 0) {
          CUDA_CHECK(cudaMemcpy(h_sm_seen.data(), d_sm_seen,
                                static_cast<size_t>(sm_count) * sizeof(uint32_t),
                                cudaMemcpyDeviceToHost));
          observed_sms = 0;
          for (uint32_t c : h_sm_seen) {
            if (c > 0) ++observed_sms;
          }
        }
      }

      if (ti > 0) std::printf(",");
      std::printf("\n      {\"target_sms\": %d, \"observed_sms\": %d, \"grid\": %d, \"gbps_samples\": [",
                  target_sms, observed_sms, grid);
      for (size_t k = 0; k < gbps_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.3f", gbps_samples[k]);
      }
      std::printf("]}");
    }
    std::printf("\n    ]");
  };

  std::printf("{\n");
  std::printf("  \"buffer_bytes\": %zu,\n", bytes);
  std::printf("  \"flush_bytes\": %zu,\n", flush_bytes);
  std::printf("  \"samples\": %d,\n", samples);
  std::printf("  \"read_curve\": ");
  measure_curve(false, targets);
  std::printf(",\n  \"write_full\": ");
  measure_curve(true, std::vector<int>{sm_count});
  std::printf("\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sm_seen));
  return 0;
}

static int run_cta_dispatch(int device, int repeats) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  int reserved_smem = 0;
  int max_smem_optin = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&reserved_smem, cudaDevAttrReservedSharedMemoryPerBlock, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  const int sm_count = prop.multiProcessorCount;
  const size_t smem_per_sm = prop.sharedMemPerMultiprocessor;

  // Choose dynamic smem sizes to force ~1 CTA/SM and ~2 CTAs/SM.
  size_t smem_1cta = std::min<size_t>(
      static_cast<size_t>(max_smem_optin),
      (smem_per_sm > static_cast<size_t>(reserved_smem) + 4096)
          ? (smem_per_sm * 2 / 3)
          : 0);
  smem_1cta = (smem_1cta / 1024) * 1024;

  size_t smem_2cta = std::min<size_t>(
      static_cast<size_t>(max_smem_optin),
      (smem_per_sm / 3 / 1024) * 1024);

  CUDA_CHECK(cudaFuncSetAttribute(
      cta_dispatch_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_1cta)));

  struct ProbeCfg {
    int ctas_per_sm_target;
    size_t dyn_smem;
    int waves;
  };
  std::vector<ProbeCfg> cfgs = {
      {1, smem_1cta, 1},
      {1, smem_1cta, 2},
      {1, smem_1cta, 4},
      {2, smem_2cta, 1},
      {2, smem_2cta, 2},
  };

  std::printf("{\n  \"sm_count\": %d,\n  \"repeats\": %d,\n  \"probes\": [\n",
              sm_count, repeats);

  for (size_t ci = 0; ci < cfgs.size(); ++ci) {
    const auto& cfg = cfgs[ci];
    int active_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, cta_dispatch_kernel, 128, cfg.dyn_smem));
    if (active_blocks < 1) active_blocks = 1;

    const int grid = sm_count * active_blocks * cfg.waves;
    const int first_wave_ctas = std::min(grid, sm_count * active_blocks);
    DispatchRecord* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(grid) * sizeof(DispatchRecord)));
    std::vector<DispatchRecord> h_out(static_cast<size_t>(grid));

    // Warmup launch.
    cta_dispatch_kernel<<<grid, 128, cfg.dyn_smem>>>(d_out, 20000ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::vector<DispatchRecord>> runs;
    runs.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
      cta_dispatch_kernel<<<grid, 128, cfg.dyn_smem>>>(d_out, 30000ULL);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                            static_cast<size_t>(grid) * sizeof(DispatchRecord),
                            cudaMemcpyDeviceToHost));
      runs.push_back(h_out);
    }

    // Measure run-to-run consistency of first-wave CTA->SM assignment.
    int total_cmp = 0;
    int matched = 0;
    for (int r = 1; r < repeats; ++r) {
      for (int c = 0; c < first_wave_ctas; ++c) {
        ++total_cmp;
        if (runs[r][c].sm_id == runs[0][c].sm_id) ++matched;
      }
    }
    double consistency = (total_cmp > 0) ? (static_cast<double>(matched) / total_cmp) : 1.0;

    // Normalize start_ns relative to earliest CTA in run 0.
    uint64_t min_t0 = runs[0][0].start_ns;
    for (int c = 1; c < first_wave_ctas; ++c) {
      if (runs[0][c].start_ns < min_t0) min_t0 = runs[0][c].start_ns;
    }

    if (ci > 0) std::printf(",\n");
    std::printf("    {\n");
    std::printf("      \"grid_dims\": [%d, 1, 1],\n", grid);
    std::printf("      \"ctas_per_sm\": %d,\n", active_blocks);
    std::printf("      \"waves_approx\": %.2f,\n",
                static_cast<double>(grid) / (sm_count * active_blocks));
    std::printf("      \"order_consistency\": %.4f,\n", consistency);
    std::printf("      \"first_wave_trace\": [");
    for (int c = 0; c < first_wave_ctas; ++c) {
      if (c > 0) std::printf(", ");
      std::printf("{\"cta_id\": %u, \"sm_id\": %u, \"start_ns\": %llu}",
                  runs[0][c].cta_id, runs[0][c].sm_id,
                  static_cast<unsigned long long>(runs[0][c].start_ns - min_t0));
    }
    std::printf("]\n    }");

    CUDA_CHECK(cudaFree(d_out));
  }

  std::printf("\n  ]\n}\n");
  return 0;
}

#include "m2_kernels.cuh"

int main(int argc, char** argv) {
  std::string mode = "query";
  int device = 0;
  int samples = 15;
  int warmup = 3;
  int repeats = 5;
  size_t buffer_mb = 256;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = std::atoi(argv[++i]);
    } else if (arg == "--samples" && i + 1 < argc) {
      samples = std::atoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (arg == "--repeats" && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (arg == "--buffer-mb" && i + 1 < argc) {
      buffer_mb = static_cast<size_t>(std::atoll(argv[++i]));
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  CUDA_CHECK(cudaSetDevice(device));
  if (mode == "query") {
    return run_query(device);
  }
  if (mode == "dram-bw") {
    return run_dram_bw(device, samples, warmup, buffer_mb);
  }
  if (mode == "cta-dispatch") {
    return run_cta_dispatch(device, repeats);
  }
  if (mode == "mma") {
    return run_mma(device, samples, warmup);
  }
  if (mode == "l2") {
    return run_l2(device, samples, warmup);
  }
  if (mode == "cp-async") {
    return run_cp_async(device, samples, warmup);
  }
  if (mode == "latency") {
    return run_latency(device, samples);
  }
  if (mode == "launch") {
    return run_launch(device, samples, warmup);
  }
  if (mode == "latency-curve") {
    return run_latency_curve(device, samples);
  }
  if (mode == "l2-rescan") {
    return run_l2_rescan(device, samples);
  }
  if (mode == "mlp") {
    return run_mlp(device, samples, warmup);
  }
  if (mode == "closed-loop") {
    return run_closed_loop(device, samples, warmup);
  }
  if (mode == "l2-merge") {
    return run_l2_merge(device, samples, warmup);
  }
  if (mode == "stage-overhead") {
    return run_stage_overhead(device, samples, warmup);
  }
  if (mode == "closed-loop-mma") {
    return run_closed_loop_mma(device, samples, warmup);
  }
  if (mode == "mma-chain") {
    return run_mma_chain(device, samples, warmup);
  }
  if (mode == "serial-fill") {
    return run_serial_fill(device, samples, warmup);
  }
  if (mode == "sync-loop") {
    return run_sync_loop(device, samples, warmup);
  }
  if (mode == "gmem-issue") {
    return run_gmem_issue(device, samples, warmup);
  }
  if (mode == "spill-counter-probe" || mode == "spill-counters") {
    return run_spill_counters(device, mode == "spill-counter-probe");
  }
  if (mode == "spill-latency") {
    return run_spill_latency(device, samples, warmup);
  }
  if (mode == "strided-mlp") {
    return run_strided_mlp(device, samples, warmup);
  }
  if (mode == "sfu-shfl") {
    return run_sfu_shfl(device, samples, warmup);
  }
  if (mode == "mma-softmax-chain") {
    return run_mma_softmax_chain(device, samples, warmup);
  }
  if (mode == "sync-loop-ablation") {
    return run_sync_loop_ablation(device, samples, warmup);
  }
  if (mode == "gmem-issue-controlled") {
    return run_gmem_issue_controlled(device, samples, warmup);
  }
  if (mode == "l1-carveout") {
    return run_l1_carveout(device, samples, warmup);
  }
  if (mode == "sync-loop-multicta") {
    return run_sync_loop_multicta(device, samples, warmup);
  }
  if (mode == "fill-vs-loop-ablation") {
    return run_fill_vs_loop_ablation(device, samples, warmup);
  }
  if (mode == "sync-loop-reconcile") {
    return run_sync_loop_reconcile(device, samples, warmup);
  }
  if (mode == "sync-loop-overlap") {
    return run_sync_loop_overlap(device, samples, warmup);
  }
  if (mode == "sync-loop-share") {
    return run_sync_loop_share(device, samples, warmup);
  }
  if (mode == "sync-loop-bytes") {
    return run_sync_loop_bytes(device, samples, warmup);
  }
  if (mode == "sync-loop-miss") {
    return run_sync_loop_miss(device, samples, warmup);
  }
  std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
  return 2;
}
