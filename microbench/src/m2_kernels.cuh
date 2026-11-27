#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <map>
#include <vector>

// ---------------------------------------------------------------------------
// 1. MMA throughput vs warps per SM (mma.sync.aligned.m16n8k16)
// ---------------------------------------------------------------------------

template <bool IS_BF16>
__global__ void mma_m16n8k16_kernel(float* __restrict__ sink, int iters) {
  uint32_t a0 = 0x3c003c00u + (threadIdx.x & 7u);
  uint32_t a1 = 0x3c003c00u;
  uint32_t a2 = 0x3c003c00u;
  uint32_t a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u;
  uint32_t b1 = 0x3c003c00u;

  // 4 independent accumulators per warp to hide MMA pipeline latency.
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int i = 0; i < iters; ++i) {
    if constexpr (!IS_BF16) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33;
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
  }
}

__global__ void fp32_fma_kernel(float* __restrict__ sink, int iters) {
  float a = 1.0001f + 1e-6f * threadIdx.x;
  float b = 0.9999f;
  float r0 = 0.1f, r1 = 0.2f, r2 = 0.3f, r3 = 0.4f;
  float r4 = 0.5f, r5 = 0.6f, r6 = 0.7f, r7 = 0.8f;
  for (int i = 0; i < iters; ++i) {
    r0 = fmaf(r0, a, b);
    r1 = fmaf(r1, a, b);
    r2 = fmaf(r2, a, b);
    r3 = fmaf(r3, a, b);
    r4 = fmaf(r4, a, b);
    r5 = fmaf(r5, a, b);
    r6 = fmaf(r6, a, b);
    r7 = fmaf(r7, a, b);
  }
  float s = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
  if (s == -999999.0f) {
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

static int run_mma(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int sm_count = prop.multiProcessorCount;
  const int max_warps_per_block = prop.maxThreadsPerBlock / prop.warpSize;
  const int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  std::vector<int> warp_counts;
  for (int w : {1, 2, 4, 8, 16, 32}) {
    if (w <= max_warps_per_block && w <= max_warps_per_sm) {
      warp_counts.push_back(w);
    }
  }

  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * prop.maxThreadsPerBlock * sizeof(float)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  auto measure_mma_variant = [&](const char* key, bool is_bf16) {
    std::printf("    \"%s\": [", key);
    for (size_t wi = 0; wi < warp_counts.size(); ++wi) {
      int warps = warp_counts[wi];
      int threads = warps * 32;
      // Scale iterations so each launch takes ~1-3 ms regardless of warp count.
      int iters = std::max(4000, 32000 / std::min(warps, 8));
      auto launch = [&](int length) {
        if (!is_bf16) {
          mma_m16n8k16_kernel<false><<<sm_count, threads>>>(d_sink, length);
        } else {
          mma_m16n8k16_kernel<true><<<sm_count, threads>>>(d_sink, length);
        }
      };
      if (schedule) {
        std::printf("%s\n      {\"mma_warps_per_sm\": %d", wi ? "," : "", warps);
        auto timed = schedule->time_point(iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {}, launch);
        const double flops = static_cast<double>(warps) * timed.length * 4.0 * 4096.0;
        std::printf(", \"tflops_per_sm_samples\": [");
        for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
          std::printf("%s%.4f", k ? ", " : "", flops / 1e3 / timed.launch_ns[k]);
        }
        std::printf("]}");
        continue;
      }
      double flops_per_sm = static_cast<double>(warps) * iters * 4.0 * 4096.0;

      for (int w = 0; w < warmup; ++w) {
        if (!is_bf16) {
          mma_m16n8k16_kernel<false><<<sm_count, threads>>>(d_sink, iters);
        } else {
          mma_m16n8k16_kernel<true><<<sm_count, threads>>>(d_sink, iters);
        }
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> tflops_samples;
      tflops_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (!is_bf16) {
          mma_m16n8k16_kernel<false><<<sm_count, threads>>>(d_sink, iters);
        } else {
          mma_m16n8k16_kernel<true><<<sm_count, threads>>>(d_sink, iters);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double tflops_per_sm = (flops_per_sm / 1e12) / (static_cast<double>(ms) / 1e3);
        tflops_samples.push_back(tflops_per_sm);
      }

      if (wi > 0) std::printf(",");
      std::printf("\n      {\"mma_warps_per_sm\": %d, \"tflops_per_sm_samples\": [", warps);
      for (size_t k = 0; k < tflops_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.4f", tflops_samples[k]);
      }
      std::printf("]}");
    }
    std::printf("\n    ]");
  };

  // Measure FP32 FMA total TFLOP/s (16 warps/SM, 8 FMAs = 16 FLOPs per thread per iter).
  int fma_warps = std::min(16, max_warps_per_block);
  int fma_threads = fma_warps * 32;
  int fma_iters = 40000;
  if (schedule) {
    // Emission order: both MMA tables, then FP32 FMA.
    std::printf("{\n  \"sm_count\": %d,\n  \"samples\": %d,\n  \"mma\": {\n", sm_count, samples);
    measure_mma_variant("mma.sync:f16:f16:f32:16x8x16", false);
    std::printf(",\n");
    measure_mma_variant("mma.sync:bf16:bf16:f32:16x8x16", true);
    std::printf("\n  },\n  \"fp32_fma\": {\"warps_per_sm\": %d", fma_warps);
    auto timed = schedule->time_point(fma_iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {},
                                      [&](int length) { fp32_fma_kernel<<<sm_count, fma_threads>>>(d_sink, length); });
    const double flops = static_cast<double>(sm_count) * fma_threads * timed.length * 16.0;
    std::printf("},\n  \"fp32_fma_tflops_samples\": [");
    for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
      std::printf("%s%.4f", k ? ", " : "", flops / 1e3 / timed.launch_ns[k]);
    }
    std::printf("]\n}\n");
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
  }
  double fma_total_flops = static_cast<double>(sm_count) * fma_threads * fma_iters * 16.0;
  for (int w = 0; w < warmup; ++w) {
    fp32_fma_kernel<<<sm_count, fma_threads>>>(d_sink, fma_iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> fp32_samples;
  fp32_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    fp32_fma_kernel<<<sm_count, fma_threads>>>(d_sink, fma_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    fp32_samples.push_back((fma_total_flops / 1e12) / (static_cast<double>(ms) / 1e3));
  }

  std::printf("{\n  \"sm_count\": %d,\n  \"samples\": %d,\n  \"mma\": {\n", sm_count, samples);
  measure_mma_variant("mma.sync:f16:f16:f32:16x8x16", false);
  std::printf(",\n");
  measure_mma_variant("mma.sync:bf16:bf16:f32:16x8x16", true);
  std::printf("\n  },\n  \"fp32_fma_tflops_samples\": [");
  for (size_t k = 0; k < fp32_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.4f", fp32_samples[k]);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 2. L2 bandwidth vs active SMs & L2 effective capacity working-set sweep
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint4 ld_cg_u4(const uint4* ptr) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

static __device__ __forceinline__ void st_cg_u4(uint4* ptr, uint4 v) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__global__ void l2_read_cg_kernel(const uint4* __restrict__ src,
                                  uint4* __restrict__ sink,
                                  size_t n_vec4,
                                  int passes) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (int p = 0; p < passes; ++p) {
    for (size_t i = tid; i < n_vec4; i += stride) {
      uint4 v = ld_cg_u4(src + i);
      acc ^= (v.x + v.y + v.z + v.w);
    }
  }
  if (acc == 0xdeadbeefu) {
    sink[tid & 1023u] = make_uint4(acc, acc, acc, acc);
  }
}

__global__ void l2_read_cg_ilp8_kernel(const uint4* __restrict__ src,
                                       uint4* __restrict__ sink,
                                       size_t vec4_mask,
                                       int iters) {
  const uint32_t tid = threadIdx.x;
  const uint32_t threads = blockDim.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * threads * 8ULL + tid) & vec4_mask;
  const size_t step = static_cast<size_t>(threads) * 8ULL;
  uint32_t acc = 0u;

  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      uint4 v = ld_cg_u4(src + ((idx + static_cast<size_t>(u) * threads) & vec4_mask));
      acc ^= (v.x ^ v.y ^ v.z ^ v.w);
    }
    idx = (idx + step) & vec4_mask;
  }

  if (acc == 0x0000beefu) {
    sink[tid & 1023u] = make_uint4(acc, acc, acc, acc);
  }
}

__global__ void l2_write_cg_kernel(uint4* __restrict__ dst,
                                   size_t n_vec4,
                                   int passes,
                                   uint32_t seed) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint4 v = make_uint4(seed, seed + 1u, seed + 2u, seed + 3u);
  for (int p = 0; p < passes; ++p) {
    for (size_t i = tid; i < n_vec4; i += stride) {
      st_cg_u4(dst + i, v);
    }
  }
}

template <int STAGES, int SUB_PER_STAGE>
__global__ void l2_read_cp_async_kernel(const uint4* __restrict__ src,
                                        uint4* __restrict__ sink,
                                        size_t vec4_mask,
                                        int steps) {
  extern __shared__ uint4 smem[];
  uint32_t tid = threadIdx.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & vec4_mask;
  size_t stride = static_cast<size_t>(gridDim.x) * 256ULL;

  auto issue_group = [&](int slot) {
    #pragma unroll
    for (int sub = 0; sub < SUB_PER_STAGE; ++sub) {
      size_t smem_off = (static_cast<size_t>(slot) * SUB_PER_STAGE + sub) * 256ULL + tid;
      uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[smem_off]));
      const void* gptr = src + idx;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :
                   : "r"(smem_addr), "l"(gptr)
                   : "memory");
      idx = (idx + stride) & vec4_mask;
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue_group(s);
  }
  for (int step = STAGES; step < steps; ++step) {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(STAGES - 1));
    issue_group(step % STAGES);
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

static int run_l2(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  // Working set for pure L2 BW: ~25% of L2, clamped to [2 MB, 12 MB].
  size_t l2_ws_bytes = std::max<size_t>(2ULL * 1024ULL * 1024ULL,
                                        std::min<size_t>(12ULL * 1024ULL * 1024ULL, l2_bytes / 4));
  l2_ws_bytes = (l2_ws_bytes / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);

  // Power-of-two warm L2 working set for cp.async active-SM sweep (< 0.35x L2 on all SKUs).
  const size_t cp_ws_bytes = (l2_bytes >= 24ULL * 1024ULL * 1024ULL)
                                 ? (8ULL * 1024ULL * 1024ULL)
                                 : ((l2_bytes >= 12ULL * 1024ULL * 1024ULL)
                                        ? (4ULL * 1024ULL * 1024ULL)
                                        : (1ULL * 1024ULL * 1024ULL));

  // Sweep sizes from ~0.1x to ~2.5x L2 to locate the effective L2 capacity cliff.
  std::vector<size_t> sweep_bytes;
  for (double frac : {0.10, 0.25, 0.50, 0.65, 0.75, 0.85, 0.95, 1.05, 1.20, 1.50, 2.00, 2.50}) {
    size_t sz = static_cast<size_t>((l2_bytes * frac) / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);
    sz = std::max<size_t>(1ULL * 1024ULL * 1024ULL, sz);
    if (sweep_bytes.empty() || sz > sweep_bytes.back()) {
      sweep_bytes.push_back(sz);
    }
  }
  size_t max_bytes = std::max(std::max(l2_ws_bytes, cp_ws_bytes), sweep_bytes.back());

  uint4* d_buf = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_buf, 3, max_bytes));

  std::vector<int> targets;
  auto add_t = [&](int s) {
    s = std::max(1, std::min(sm_count, s));
    if (std::find(targets.begin(), targets.end(), s) == targets.end()) targets.push_back(s);
  };
  add_t(1);
  add_t(2);
  add_t(4);
  add_t(8);
  add_t(16);
  add_t(sm_count / 2);
  add_t(32);
  add_t(sm_count);
  std::sort(targets.begin(), targets.end());

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const int threads = 512;
  const size_t l2_vec4 = l2_ws_bytes / sizeof(uint4);
  // 4 MiB power-of-two hot L2 buffer for ILP-8 L2 read ceiling sweep (fits in 24-36 MiB L2).
  const size_t ilp8_ws_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t ilp8_mask = (ilp8_ws_bytes / sizeof(uint4)) - 1ULL;
  const size_t cp_mask = (cp_ws_bytes / sizeof(uint4)) - 1ULL;

  std::printf("{\n  \"l2_spec_bytes\": %zu,\n  \"l2_bw_ws_bytes\": %zu,\n  \"samples\": %d,\n",
              l2_bytes, l2_ws_bytes, samples);

  // 1a) L2 read BW vs active SMs via register loads (16 warps/SM, 16 B ld.global.cg, ILP 8).
  std::printf("  \"read_curve\": [");
  for (size_t ti = 0; ti < targets.size(); ++ti) {
    int target_sms = targets[ti];
    int iters = (target_sms <= 4) ? 4096 : 2048;
    if (schedule) {
      auto launch = [&](int count) {
        l2_read_cg_ilp8_kernel<<<target_sms, threads>>>(d_buf, d_sink, ilp8_mask, count);
      };
      std::printf("%s\n    {\"target_sms\": %d", ti ? "," : "", target_sms);
      schedule->measure(iters, 1, std::numeric_limits<int>::max(), target_sms * threads * 8ULL * 16,
                        ilp8_ws_bytes, warmup, samples, ev_start, ev_stop, [] {}, launch);
      std::printf("}");
      continue;
    }
    double total_bytes = static_cast<double>(target_sms) * static_cast<double>(threads) * 8.0 * 16.0 * static_cast<double>(iters);

    for (int w = 0; w < warmup; ++w) {
      l2_read_cg_ilp8_kernel<<<target_sms, threads>>>(d_buf, d_sink, ilp8_mask, iters);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gbps;
    gbps.reserve(samples);
    for (int s = 0; s < samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      l2_read_cg_ilp8_kernel<<<target_sms, threads>>>(d_buf, d_sink, ilp8_mask, iters);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      gbps.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }

    if (ti > 0) std::printf(",");
    std::printf("\n    {\"target_sms\": %d, \"gbps_samples\": [", target_sms);
    for (size_t k = 0; k < gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", gbps[k]);
    }
    std::printf("]}");
  }
  std::printf("\n  ],\n");

  // 1b) L2 read BW vs active SMs via cp.async.cg (bypasses SM register scoreboard limits).
  // Partial-SM points launch 1 CTA/SM (32 KiB in flight/SM); full-GPU launches 3 CTAs/SM
  // (16 KiB/CTA = 48 KiB/SM across 24 warps/SM, fitting within <= 64 KiB SMEM on all arches).
  std::printf("  \"read_curve_cp_async\": [");
  if (prop.major >= 8) {
    for (size_t ti = 0; ti < targets.size(); ++ti) {
      int target_sms = targets[ti];
      bool full_gpu = (target_sms == sm_count);
      int grid = full_gpu ? (sm_count * 3) : target_sms;
      int sub = full_gpu ? 1 : 2;
      size_t dyn_smem = 4ULL * static_cast<size_t>(sub) * 4096ULL;
      size_t step_bytes = static_cast<size_t>(grid) * static_cast<size_t>(sub) * 4096ULL;
      size_t target_total = std::max<size_t>(32ULL * 1024ULL * 1024ULL, step_bytes * 2048ULL);
      int steps = std::max(64, static_cast<int>(target_total / step_bytes));
      double total_bytes = static_cast<double>(step_bytes) * static_cast<double>(steps);

      auto launch_cp = [&](int count) {
        if (full_gpu) {
          l2_read_cp_async_kernel<4, 1><<<grid, 256, dyn_smem>>>(d_buf, d_sink, cp_mask, count);
        } else {
          l2_read_cp_async_kernel<4, 2><<<grid, 256, dyn_smem>>>(d_buf, d_sink, cp_mask, count);
        }
      };

      if (schedule) {
        std::printf("%s\n    {\"target_sms\": %d", ti ? "," : "", target_sms);
        schedule->measure(steps, 1, std::numeric_limits<int>::max(), step_bytes, cp_ws_bytes,
                          warmup, samples, ev_start, ev_stop, [] {}, launch_cp);
        std::printf("}");
        continue;
      }

      for (int w = 0; w < warmup; ++w) {
        launch_cp(steps);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> gbps;
      gbps.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch_cp(steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        gbps.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
      }

      if (ti > 0) std::printf(",");
      std::printf("\n    {\"target_sms\": %d, \"gbps_samples\": [", target_sms);
      for (size_t k = 0; k < gbps.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.3f", gbps[k]);
      }
      std::printf("]}");
    }
  }
  std::printf("\n  ],\n");

  // 2) Full-GPU L2 write BW.
  {
    int grid = sm_count * 4;
    int passes = 64;
    double total_bytes = static_cast<double>(l2_ws_bytes) * passes;
    for (int w = 0; w < warmup; ++w) {
      l2_write_cg_kernel<<<grid, threads>>>(d_buf, l2_vec4, 4, static_cast<uint32_t>(w));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (schedule) {
      auto launch = [&](int count) {
        l2_write_cg_kernel<<<grid, threads>>>(d_buf, l2_vec4, count, 10u);
      };
      std::printf("  \"write_full\": {\"working_set_bytes\": %zu", l2_ws_bytes);
      const auto timed = schedule->measure(passes, 1, std::numeric_limits<int>::max(),
                                           l2_ws_bytes, l2_ws_bytes, 0, samples,
                                           ev_start, ev_stop, [] {}, launch);
      std::printf("},\n  \"write_full_gbps_samples\": ");
      LaunchSchedule::print_bandwidth_samples(timed, l2_ws_bytes);
      std::printf(",\n");
    } else {
      std::printf("  \"write_full_gbps_samples\": [");
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        l2_write_cg_kernel<<<grid, threads>>>(d_buf, l2_vec4, passes, static_cast<uint32_t>(s + 10));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        if (s > 0) std::printf(", ");
        std::printf("%.3f", (total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
      }
      std::printf("],\n");
    }
  }

  // 3) Working-set sweep for effective L2 capacity.
  std::printf("  \"capacity_sweep\": [");
  const int sweep_grid = sm_count * 4;
  for (size_t si = 0; si < sweep_bytes.size(); ++si) {
    size_t ws = sweep_bytes[si];
    size_t n_vec4 = ws / sizeof(uint4);
    int passes = std::max(4, static_cast<int>((512ULL * 1024ULL * 1024ULL) / ws));
    double total_bytes = static_cast<double>(ws) * passes;

    for (int w = 0; w < 2; ++w) {
      l2_read_cg_kernel<<<sweep_grid, threads>>>(d_buf, d_sink, n_vec4, 2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (schedule) {
      auto launch = [&](int count) {
        l2_read_cg_kernel<<<sweep_grid, threads>>>(d_buf, d_sink, n_vec4, count);
      };
      std::printf("%s\n    {\"working_set_bytes\": %zu", si ? "," : "", ws);
      auto timed = schedule->measure(passes, 4, std::numeric_limits<int>::max(), ws, ws,
                                     0, samples, ev_start, ev_stop, [] {}, launch);
      auto& durations = timed.launch_ns;
      std::sort(durations.begin(), durations.end());
      const double transferred = static_cast<double>(timed.length) * ws;
      const size_t mid = durations.size() / 2;
      const double median_bw =
          (transferred / durations[(durations.size() - 1) / 2] + transferred / durations[mid]) / 2;
      std::printf(", \"median_gbps\": %.6f}", median_bw);
      continue;
    }

    std::vector<double> gbps;
    gbps.reserve(5);
    for (int s = 0; s < 5; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      l2_read_cg_kernel<<<sweep_grid, threads>>>(d_buf, d_sink, n_vec4, passes);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      gbps.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }
    std::sort(gbps.begin(), gbps.end());
    double med = gbps[gbps.size() / 2];
    if (si > 0) std::printf(",");
    std::printf("\n    {\"working_set_bytes\": %zu, \"median_gbps\": %.3f}", ws, med);
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 3. cp.async bandwidth per SM & shared memory bandwidth per SM
// ---------------------------------------------------------------------------

__global__ void cp_async_bw_kernel(const uint4* __restrict__ src,
                                   uint4* __restrict__ sink,
                                   size_t n_vec4_mask,
                                   int iters) {
  __shared__ uint4 smem[4 * 256];  // 4 stages x 4 KB = 16 KB
  uint32_t tid = threadIdx.x;
  size_t base = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t step = static_cast<size_t>(gridDim.x) * 256ULL;

  for (int i = 0; i < iters; ++i) {
    uint32_t stage = static_cast<uint32_t>(i & 3);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[stage * 256 + tid]));
    const void* gptr = src + ((base + static_cast<size_t>(i) * step) & n_vec4_mask);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    asm volatile("cp.async.commit_group;\n" ::);
    if (i >= 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    }
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

__global__ void smem_bw_kernel(uint4* __restrict__ sink, int iters, uint32_t seed) {
  // 512 threads (16 warps/SM) x 4 independent 128-bit loads per iter, 32 KB smem, read-only loop.
  // Rotate the byte offset across 4 distinct slots per iteration so ptxas cannot hoist loads out of the loop.
  __shared__ uint4 smem[512 * 4];
  uint32_t tid = threadIdx.x;
  smem[tid] = make_uint4(tid + seed, tid + 1u, tid + 2u, tid + 3u);
  smem[tid + 512] = make_uint4(tid + 4u + seed, tid + 5u, tid + 6u, tid + 7u);
  smem[tid + 1024] = make_uint4(tid + 8u + seed, tid + 9u, tid + 10u, tid + 11u);
  smem[tid + 1536] = make_uint4(tid + 12u + seed, tid + 13u, tid + 14u, tid + 15u);
  __syncthreads();

  uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[0]));
  uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

  for (int i = 0; i < iters; ++i) {
    uint32_t idx0 = (tid + static_cast<uint32_t>(i)) & 511u;
    uint32_t a0 = smem_base + (idx0 << 4);
    uint32_t a1 = a0 + 8192u;
    uint32_t a2 = a0 + 16384u;
    uint32_t a3 = a0 + 24576u;

    uint4 v0, v1, v2, v3;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                 : "r"(a0)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                 : "r"(a1)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v2.x), "=r"(v2.y), "=r"(v2.z), "=r"(v2.w)
                 : "r"(a2)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v3.x), "=r"(v3.y), "=r"(v3.z), "=r"(v3.w)
                 : "r"(a3)
                 : "memory");
    acc0 += v0.x + v1.y + v2.z + v3.w;
    acc1 += v0.y + v1.z + v2.w + v3.x;
    acc2 += v0.z + v1.w + v2.x + v3.y;
    acc3 += v0.w + v1.x + v2.y + v3.z;
  }
  if ((acc0 + acc1 + acc2 + acc3) == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = make_uint4(acc0, acc1, acc2, acc3);
  }
}

static int run_cp_async(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Power-of-two L2-resident source buffer (8 MB = 524288 uint4).
  const size_t n_vec4 = 524288ULL;
  const size_t n_vec4_mask = n_vec4 - 1ULL;
  uint4* d_src = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, n_vec4 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_src, 5, n_vec4 * sizeof(uint4)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // 1) cp.async per-SM BW (1 CTA/SM, 256 threads, 16 B/thread/iter).
  const int cp_iters = 16384;
  const double cp_bytes_per_sm = static_cast<double>(256) * 16.0 * cp_iters;
  if (schedule) {
    // Legacy launches already take about 1.5 ms; the schedule only records lengths.
    auto rates = [&](const LaunchSchedule::Timed& timed, double bytes_per_iter) {
      for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
        std::printf("%s%.3f", k ? ", " : "", bytes_per_iter * timed.length / timed.launch_ns[k]);
      }
    };
    std::printf("{\n  \"sm_count\": %d,\n  \"samples\": %d,\n  \"cp_async\": {\"threads_per_sm\": 256",
                sm_count, samples);
    auto cp = schedule->time_point(
        cp_iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {}, [&](int length) {
          cp_async_bw_kernel<<<sm_count, 256>>>(d_src, d_sink, n_vec4_mask, length);
        });
    std::printf("},\n  \"cp_async_bw_per_sm_gbps_samples\": [");
    rates(cp, 256.0 * 16.0);
    uint32_t seed = 1;
    std::printf("],\n  \"smem\": {\"threads_per_sm\": 512");
    auto sm = schedule->time_point(16384, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {},
                                   [&](int length) {
                                     smem_bw_kernel<<<sm_count, 512>>>(d_sink, length, seed++);
                                   });
    std::printf("},\n  \"smem_bw_per_sm_gbps_samples\": [");
    rates(sm, 512.0 * 64.0);
    std::printf("]\n}\n");
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_sink));
    return 0;
  }
  for (int w = 0; w < warmup; ++w) {
    cp_async_bw_kernel<<<sm_count, 256>>>(d_src, d_sink, n_vec4_mask, 1024);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> cp_samples;
  cp_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    cp_async_bw_kernel<<<sm_count, 256>>>(d_src, d_sink, n_vec4_mask, cp_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    cp_samples.push_back((cp_bytes_per_sm / 1e9) / (static_cast<double>(ms) / 1e3));
  }

  // 2) Shared memory per-SM BW (512 threads = 16 warps/SM, 4 x 16B = 64 B/thread/iter, read-only).
  const int sm_iters = 16384;
  const double sm_bytes_per_sm = static_cast<double>(512) * 64.0 * sm_iters;
  for (int w = 0; w < warmup; ++w) {
    smem_bw_kernel<<<sm_count, 512>>>(d_sink, 1024, static_cast<uint32_t>(w));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> smem_samples;
  smem_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    smem_bw_kernel<<<sm_count, 512>>>(d_sink, sm_iters, static_cast<uint32_t>(s + 1));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    smem_samples.push_back((sm_bytes_per_sm / 1e9) / (static_cast<double>(ms) / 1e3));
  }

  std::printf("{\n  \"sm_count\": %d,\n  \"cp_async_bw_per_sm_gbps_samples\": [", sm_count);
  for (size_t k = 0; k < cp_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.3f", cp_samples[k]);
  }
  std::printf("],\n  \"smem_bw_per_sm_gbps_samples\": [");
  for (size_t k = 0; k < smem_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.3f", smem_samples[k]);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 4. Load latency: smem, L2 (unloaded & loaded), DRAM (unloaded & loaded)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint64_t ld_cg_u64(const uint64_t* ptr) {
  uint64_t v;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(ptr));
  return v;
}

__global__ void smem_latency_kernel(double* __restrict__ out_ns, int steps) {
  __shared__ uint32_t chain[1024];
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < 1024; ++i) {
      chain[i] = (i + 37u) & 1023u;
    }
    uint32_t idx = 0;
    for (int w = 0; w < 256; ++w) idx = chain[idx];
    uint64_t t0 = read_globaltimer();
    for (int i = 0; i < steps; ++i) {
      idx = chain[idx];
    }
    uint64_t t1 = read_globaltimer();
    *out_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
    if (idx == 999999u) chain[0] = idx;
  }
}

__global__ void pchase_latency_kernel(const uint64_t* __restrict__ chain,
                                      uint64_t* __restrict__ sink,
                                      double* __restrict__ out_ns,
                                      int steps,
                                      const uint4* __restrict__ bg_buf,
                                      size_t bg_vec4,
                                      volatile int* __restrict__ done_flag) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      uint64_t idx = 0;
      uint64_t t0 = read_globaltimer();
      for (int i = 0; i < steps; ++i) {
        idx = ld_cg_u64(chain + idx);
      }
      uint64_t t1 = read_globaltimer();
      *out_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
      if (done_flag) {
        atomicExch(const_cast<int*>(done_flag), 1);
      }
      *sink = idx;
    }
  } else if (bg_buf != nullptr && done_flag != nullptr) {
    size_t tid = static_cast<size_t>(blockIdx.x - 1) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x - 1) * blockDim.x;
    uint32_t acc = 0;
    size_t i = tid % bg_vec4;
    while (*done_flag == 0) {
      for (int k = 0; k < 16; ++k) {
        uint4 v = ld_cg_u4(bg_buf + i);
        acc ^= v.x;
        i += stride;
        if (i >= bg_vec4) i -= bg_vec4;
      }
    }
    if (acc == 0xdeadbeefu) *sink = acc;
  }
}

static std::vector<uint64_t> build_random_chain(size_t bytes, size_t stride_bytes, uint32_t seed) {
  size_t n_u64 = bytes / sizeof(uint64_t);
  size_t step_u64 = stride_bytes / sizeof(uint64_t);
  size_t n_nodes = n_u64 / step_u64;

  std::vector<size_t> perm(n_nodes);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);

  std::vector<uint64_t> buf(n_u64, 0);
  for (size_t i = 0; i < n_nodes; ++i) {
    size_t cur = perm[i] * step_u64;
    size_t nxt = perm[(i + 1) % n_nodes] * step_u64;
    buf[cur] = static_cast<uint64_t>(nxt);
  }
  return buf;
}

static int run_latency(int device, int samples, int warmup = 1, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  const size_t l2_chain_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_chain_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);
  const size_t bg_l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t bg_dram_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);

  auto h_l2_chain = build_random_chain(l2_chain_bytes, 256, 101);
  auto h_dram_chain = build_random_chain(dram_chain_bytes, 256, 202);

  uint64_t* d_l2_chain = nullptr;
  uint64_t* d_dram_chain = nullptr;
  uint4* d_bg_dram = nullptr;
  uint64_t* d_sink = nullptr;
  double* d_out_ns = nullptr;
  int* d_done = nullptr;

  CUDA_CHECK(cudaMalloc(&d_l2_chain, l2_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_dram_chain, dram_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_bg_dram, bg_dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_out_ns, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_l2_chain, h_l2_chain.data(), l2_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dram_chain, h_dram_chain.data(), dram_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bg_dram, 7, bg_dram_bytes));

  auto print_arr = [](const char* name, const std::vector<double>& v, bool comma) {
    std::printf("  \"%s\": [", name);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) std::printf(", ");
      std::printf("%.2f", v[i]);
    }
    std::printf("]%s\n", comma ? "," : "");
  };

  if (schedule) {
    // Each probe times its chase inside the kernel; the length is the chase steps and the
    // window is the per-step latency the kernel writes times the steps.
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    int current = 0;
    auto read_window = [&]() {
      double v = 0.0;
      CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
      return v * current;
    };
    uint32_t seed = 0;
    auto probe = [&](const char* name, const char* key, int legacy, auto prepare, auto launch, bool comma) {
      std::printf("  \"%s\": {\"probe\": \"%s\"", key, name);
      auto timed = schedule->time_point(
          legacy, 1, 1 << 24, std::max(warmup, 1), samples, ev_start, ev_stop, prepare,
          [&](int length) {
            current = length;
            launch(length);
          },
          read_window);
      std::printf(", \"window_ns_samples\": [");
      for (size_t k = 0; k < timed.observed.size(); ++k) std::printf("%s%.1f", k ? ", " : "", timed.observed[k]);
      std::printf("]},\n  \"%s_ns_samples\": [", key);
      for (size_t k = 0; k < timed.observed.size(); ++k) {
        std::printf("%s%.2f", k ? ", " : "", timed.observed[k] / timed.length);
      }
      std::printf("]%s\n", comma ? "," : "");
    };
    auto none = [] {};
    std::printf("{\n  \"samples\": %d,\n", samples);
    probe("smem", "smem_latency", 50000, none,
          [&](int n) { smem_latency_kernel<<<1, 32>>>(d_out_ns, n); }, true);
    pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, 16384, nullptr, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    probe("l2", "l2_latency", 16384, none,
          [&](int n) { pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, n, nullptr, 0, nullptr); },
          true);
    probe("l2_loaded", "l2_latency_loaded", 8192,
          [&] { CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int))); },
          [&](int n) {
            pchase_latency_kernel<<<sm_count, 256>>>(d_l2_chain, d_sink, d_out_ns, n, d_bg_dram,
                                                     bg_l2_bytes / sizeof(uint4), d_done);
          },
          true);
    probe("dram", "dram_latency", 8192,
          [&] { l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), seed++); },
          [&](int n) { pchase_latency_kernel<<<1, 32>>>(d_dram_chain, d_sink, d_out_ns, n, nullptr, 0, nullptr); },
          true);
    probe("dram_loaded", "dram_latency_loaded", 4096,
          [&] {
            l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), seed++);
            CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
          },
          [&](int n) {
            pchase_latency_kernel<<<sm_count, 256>>>(d_dram_chain, d_sink, d_out_ns, n, d_bg_dram,
                                                     bg_dram_bytes / sizeof(uint4), d_done);
          },
          false);
    std::printf("}\n");
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_l2_chain));
    CUDA_CHECK(cudaFree(d_dram_chain));
    CUDA_CHECK(cudaFree(d_bg_dram));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_out_ns));
    CUDA_CHECK(cudaFree(d_done));
    return 0;
  }

  // 1) Shared memory latency.
  std::vector<double> smem_ns;
  smem_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    smem_latency_kernel<<<1, 32>>>(d_out_ns, 50000);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    smem_ns.push_back(v);
  }

  // 2) Unloaded L2 latency (warm L2).
  pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, 16384, nullptr, 0, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> l2_ns;
  l2_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, 16384, nullptr, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    l2_ns.push_back(v);
  }

  // 3) Loaded L2 latency (background CTAs hammer 8 MB L2).
  std::vector<double> l2_loaded_ns;
  l2_loaded_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
    pchase_latency_kernel<<<sm_count, 256>>>(d_l2_chain, d_sink, d_out_ns, 8192,
                                             d_bg_dram, bg_l2_bytes / sizeof(uint4), d_done);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    l2_loaded_ns.push_back(v);
  }

  // 4) Unloaded DRAM latency (flush L2 before each sample).
  std::vector<double> dram_ns;
  dram_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), static_cast<uint32_t>(s));
    pchase_latency_kernel<<<1, 32>>>(d_dram_chain, d_sink, d_out_ns, 8192, nullptr, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    dram_ns.push_back(v);
  }

  // 5) Loaded DRAM latency (background CTAs hammer >=3x L2 DRAM).
  std::vector<double> dram_loaded_ns;
  dram_loaded_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), static_cast<uint32_t>(s + 50));
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
    pchase_latency_kernel<<<sm_count, 256>>>(d_dram_chain, d_sink, d_out_ns, 4096,
                                             d_bg_dram, bg_dram_bytes / sizeof(uint4), d_done);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    dram_loaded_ns.push_back(v);
  }

  std::printf("{\n");
  print_arr("smem_latency_ns_samples", smem_ns, true);
  print_arr("l2_latency_ns_samples", l2_ns, true);
  print_arr("l2_latency_loaded_ns_samples", l2_loaded_ns, true);
  print_arr("dram_latency_ns_samples", dram_ns, true);
  print_arr("dram_latency_loaded_ns_samples", dram_loaded_ns, false);
  std::printf("}\n");

  CUDA_CHECK(cudaFree(d_l2_chain));
  CUDA_CHECK(cudaFree(d_dram_chain));
  CUDA_CHECK(cudaFree(d_bg_dram));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out_ns));
  CUDA_CHECK(cudaFree(d_done));
  return 0;
}

// ---------------------------------------------------------------------------
// 5. Launch overhead & per-CTA launch latency
// ---------------------------------------------------------------------------

__global__ void empty_kernel() {}

// Launch overheads under a launch schedule: the length is the number of launches per
// timed batch. Graphs are built per length during warmup, outside the timed window.
static int run_launch_scheduled(int samples, int warmup, LaunchSchedule* schedule) {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const int batch = 500;
  const int large_grid = 4096;
  warmup = std::max(warmup, 1);
  std::map<std::pair<int, int>, std::pair<cudaGraph_t, cudaGraphExec_t>> graphs;
  auto graph = [&](int grid, int length) {
    auto key = std::make_pair(grid, length);
    auto it = graphs.find(key);
    if (it != graphs.end()) return it->second.second;
    cudaGraph_t g = nullptr;
    cudaGraphExec_t e = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < length; ++i) empty_kernel<<<grid, 1, 0, stream>>>();
    CUDA_CHECK(cudaStreamEndCapture(stream, &g));
    CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
    graphs[key] = {g, e};
    return e;
  };
  // time_point records its events on the legacy stream, so batches run there too.
  auto run_graph = [&](int grid, int length) { CUDA_CHECK(cudaGraphLaunch(graph(grid, length), 0)); };
  auto print_per_launch = [](const std::vector<double>& ns, int length, double scale) {
    for (size_t k = 0; k < ns.size(); ++k) std::printf("%s%.4f", k ? ", " : "", ns[k] / length * scale);
  };

  std::printf("{\n  \"samples\": %d,\n  \"legacy_batch\": %d,\n  \"large_grid\": %d,\n", samples, batch,
              large_grid);
  std::printf("  \"graph_launch\": {\"ctas\": 1");
  auto small = schedule->time_point(batch, 1, 1 << 20, warmup, samples, ev_start, ev_stop, [] {},
                                    [&](int length) { run_graph(1, length); });
  std::printf("},\n  \"device_launch_overhead_us_samples\": [");
  print_per_launch(small.launch_ns, small.length, 1e-3);

  // Per-CTA slope: sized on the 1-CTA graph, then the 4096-CTA graph of the same count.
  std::printf("],\n  \"per_cta\": {\"ctas\": 1, \"large_ctas\": %d", large_grid);
  auto slope = schedule->time_point(batch, 1, 1 << 20, warmup, samples, ev_start, ev_stop, [] {},
                                    [&](int length) { run_graph(1, length); });
  for (int w = 0; w < warmup; ++w) run_graph(large_grid, slope.length);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> large_ns;
  for (int k = 0; k < samples; ++k) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    run_graph(large_grid, slope.length);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    large_ns.push_back(static_cast<double>(ms) * 1e6);
  }
  std::printf(", \"large_graph_ns_samples\": [");
  for (size_t k = 0; k < large_ns.size(); ++k) std::printf("%s%.6f", k ? ", " : "", large_ns[k]);
  std::printf("]},\n  \"per_cta_launch_latency_ns_samples\": [");
  for (size_t k = 0; k < large_ns.size(); ++k) {
    const double diff = (large_ns[k] - slope.launch_ns[k]) / slope.length;
    std::printf("%s%.4f", k ? ", " : "", std::max(0.05, diff / (large_grid - 1)));
  }

  std::printf("],\n  \"stream_launch\": {\"ctas\": 1");
  auto host = schedule->time_point(batch, 1, 1 << 20, warmup, samples, ev_start, ev_stop, [] {},
                                   [&](int length) {
                                     for (int i = 0; i < length; ++i) empty_kernel<<<1, 1>>>();
                                   });
  std::printf("},\n  \"host_launch_overhead_us_samples\": [");
  print_per_launch(host.launch_ns, host.length, 1e-3);
  std::printf("]\n}\n");

  for (auto& kv : graphs) {
    CUDA_CHECK(cudaGraphExecDestroy(kv.second.second));
    CUDA_CHECK(cudaGraphDestroy(kv.second.first));
  }
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

static int run_launch(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  (void)device;
  if (schedule) return run_launch_scheduled(samples, warmup, schedule);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const int batch = 500;
  const int large_grid = 4096;

  // Capture two CUDA Graphs (1-CTA and large-grid) to measure device-side launch overhead
  // matching the validation runner's graph and event timing.
  cudaGraph_t graph_small = nullptr, graph_large = nullptr;
  cudaGraphExec_t exec_small = nullptr, exec_large = nullptr;

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < batch; ++i) empty_kernel<<<1, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_small));
  CUDA_CHECK(cudaGraphInstantiate(&exec_small, graph_small, nullptr, nullptr, 0));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < batch; ++i) empty_kernel<<<large_grid, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_large));
  CUDA_CHECK(cudaGraphInstantiate(&exec_large, graph_large, nullptr, nullptr, 0));

  for (int w = 0; w < warmup; ++w) {
    CUDA_CHECK(cudaGraphLaunch(exec_small, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_large, stream));
    for (int i = 0; i < 50; ++i) empty_kernel<<<1, 1, 0, stream>>>();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> device_launch_us;
  std::vector<double> host_launch_us;
  std::vector<double> per_cta_ns;
  device_launch_us.reserve(samples);
  host_launch_us.reserve(samples);
  per_cta_ns.reserve(samples);

  for (int s = 0; s < samples; ++s) {
    // 1) Device-side CUDA Graph launch overhead (1 CTA).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_small, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_dev_small = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dev_small, ev_start, ev_stop));
    device_launch_us.push_back((static_cast<double>(ms_dev_small) * 1e3) / batch);

    // 2) Device-side CUDA Graph per-CTA launch slope (4096 CTAs vs 1 CTA).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_large, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_dev_large = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dev_large, ev_start, ev_stop));
    double diff_ns = ((static_cast<double>(ms_dev_large) - static_cast<double>(ms_dev_small)) * 1e6) / batch;
    per_cta_ns.push_back(std::max(0.05, diff_ns / static_cast<double>(large_grid - 1)));

    // 3) Host-submitted stream batch launch overhead (includes WSL/driver submission).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < batch; ++i) empty_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_host = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_host, ev_start, ev_stop));
    host_launch_us.push_back((static_cast<double>(ms_host) * 1e3) / batch);
  }

  auto print_vec = [](const char* key, const std::vector<double>& v, bool comma) {
    std::printf("  \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) std::printf(", ");
      std::printf("%.3f", v[i]);
    }
    std::printf("]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"batch\": %d,\n  \"large_grid\": %d,\n", batch, large_grid);
  print_vec("device_launch_overhead_us_samples", device_launch_us, true);
  print_vec("host_launch_overhead_us_samples", host_launch_us, true);
  print_vec("per_cta_launch_latency_ns_samples", per_cta_ns, false);
  std::printf("}\n");

  CUDA_CHECK(cudaGraphExecDestroy(exec_small));
  CUDA_CHECK(cudaGraphExecDestroy(exec_large));
  CUDA_CHECK(cudaGraphDestroy(graph_small));
  CUDA_CHECK(cudaGraphDestroy(graph_large));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

// ---------------------------------------------------------------------------
// 6. Latency vs achieved bandwidth curves (DRAM & L2) for Little's Law model
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int ld_acq_s32(const volatile int* ptr) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

struct BgCtaStats {
  uint64_t bytes;
  uint64_t t_start_ns;
  uint64_t t_end_ns;
};

__global__ void pchase_bw_curve_kernel(const uint64_t* __restrict__ chain,
                                       uint64_t* __restrict__ sink,
                                       double* __restrict__ out_latency_ns,
                                       BgCtaStats* __restrict__ out_bg_stats,
                                       int steps,
                                       const uint4* __restrict__ bg_buf,
                                       size_t bg_vec4,
                                       int bg_active_threads,
                                       volatile int* __restrict__ start_flag,
                                       volatile int* __restrict__ done_flag,
                                       unsigned long long* __restrict__ chunk_counter) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      uint64_t idx = 0;
      for (int w = 0; w < 256; ++w) {
        idx = ld_cg_u64(chain + idx);
      }
      if (start_flag) {
        atomicExch(const_cast<int*>(start_flag), 1);
      }
      uint64_t t0 = read_globaltimer();
      for (int i = 0; i < steps; ++i) {
        idx = ld_cg_u64(chain + idx);
      }
      uint64_t t1 = read_globaltimer();
      if (done_flag) {
        atomicExch(const_cast<int*>(done_flag), 1);
      }
      *out_latency_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
      *sink = idx;
    }
  } else if (bg_buf != nullptr && done_flag != nullptr) {
    if (static_cast<int>(threadIdx.x) >= bg_active_threads) return;
    __shared__ size_t s_chunk_base;
    // Each chunk is 32 * bg_active_threads uint4s; CTAs claim disjoint chunks so
    // no CTA reads data recently fetched into L2 by another CTA.
    const size_t chunk_vec4 = 32ULL * static_cast<size_t>(bg_active_threads);
    const size_t raw_chunks = bg_vec4 / chunk_vec4;
    const size_t n_chunks = (raw_chunks > 0ULL) ? raw_chunks : 1ULL;
    uint32_t acc = 0;

    auto claim_and_read = [&]() {
      if (threadIdx.x == 0) {
        unsigned long long cid = atomicAdd(chunk_counter, 1ULL);
        s_chunk_base = (static_cast<size_t>(cid) % n_chunks) * chunk_vec4;
      }
      __syncthreads();
      size_t base = s_chunk_base + threadIdx.x;
      for (int k = 0; k < 32; ++k) {
        uint4 v = ld_cg_u4(bg_buf + base + static_cast<size_t>(k) * static_cast<size_t>(bg_active_threads));
        acc ^= v.x;
      }
    };

    while (ld_acq_s32(start_flag) == 0 && ld_acq_s32(done_flag) == 0) {
      claim_and_read();
    }

    __syncthreads();
    uint64_t bg_t0 = (threadIdx.x == 0) ? read_globaltimer() : 0;
    unsigned long long loops = 0;
    while (ld_acq_s32(done_flag) == 0) {
      claim_and_read();
      ++loops;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t bg_t1 = read_globaltimer();
      uint64_t cta_bytes = static_cast<uint64_t>(loops) * chunk_vec4 * 16ULL;
      out_bg_stats[blockIdx.x - 1] = BgCtaStats{cta_bytes, bg_t0, bg_t1};
    }
    if (acc == 0xdeadbeefu) *sink = acc;
  }
}

static int run_latency_curve(int device, int samples, int warmup = 1,
                             LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  const size_t l2_chain_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_chain_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);
  const size_t bg_l2_bytes = 8ULL * 1024ULL * 1024ULL;
  // Use >= 256 MiB (> 7x L2 on RTX 4070) so background DRAM traffic has zero L2 reuse.
  const size_t bg_dram_bytes = std::max<size_t>(256ULL * 1024ULL * 1024ULL, l2_bytes * 6ULL);

  auto h_l2_chain = build_random_chain(l2_chain_bytes, 256, 303);
  auto h_dram_chain = build_random_chain(dram_chain_bytes, 256, 404);

  uint64_t* d_l2_chain = nullptr;
  uint64_t* d_dram_chain = nullptr;
  uint4* d_bg_buf = nullptr;
  uint4* d_sink_vec = nullptr;
  uint64_t* d_sink = nullptr;
  double* d_lat_ns = nullptr;
  BgCtaStats* d_bg_stats = nullptr;
  int* d_start = nullptr;
  int* d_done = nullptr;
  unsigned long long* d_chunk_counter = nullptr;

  int max_bg = std::max(1, sm_count - 1);
  CUDA_CHECK(cudaMalloc(&d_l2_chain, l2_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_dram_chain, dram_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_bg_buf, bg_dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_vec, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_lat_ns, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_bg_stats, static_cast<size_t>(max_bg) * sizeof(BgCtaStats)));
  CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_chunk_counter, sizeof(unsigned long long)));

  CUDA_CHECK(cudaMemcpy(d_l2_chain, h_l2_chain.data(), l2_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dram_chain, h_dram_chain.data(), dram_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bg_buf, 9, bg_dram_bytes));

  struct LoadLevel {
    int bg_blocks;
    int bg_threads;
  };
  std::vector<LoadLevel> levels = {
      {0, 0},
      {1, 64},
      {1, 256},
      {std::min(2, max_bg), 256},
      {std::min(4, max_bg), 256},
      {std::min(8, max_bg), 256},
      {std::min(16, max_bg), 256},
      {max_bg, 256},
  };
  std::vector<BgCtaStats> h_bg_stats(static_cast<size_t>(max_bg));

  auto measure_levels = [&](const char* key, bool is_dram, bool comma) {
    const uint64_t* chain = is_dram ? d_dram_chain : d_l2_chain;
    const size_t bg_vec4 = (is_dram ? bg_dram_bytes : bg_l2_bytes) / sizeof(uint4);
    const int steps = is_dram ? 4096 : 8192;

    if (!is_dram) {
      CUDA_CHECK(cudaMemset(d_start, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_chunk_counter, 0, sizeof(unsigned long long)));
      pchase_bw_curve_kernel<<<1, 32>>>(chain, d_sink, d_lat_ns, d_bg_stats,
                                        4096, nullptr, 0, 0, d_start, d_done, d_chunk_counter);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::printf("  \"%s\": [", key);
    for (size_t li = 0; li < levels.size(); ++li) {
      int bg_b = levels[li].bg_blocks;
      int bg_t = levels[li].bg_threads;
      int grid = 1 + bg_b;
      int threads = (bg_b == 0) ? 32 : bg_t;

      // Background bandwidth of the last launch: total bytes over the wall span of the CTAs.
      auto background_gbps = [&]() {
        if (bg_b == 0) return 0.0;
        CUDA_CHECK(cudaMemcpy(h_bg_stats.data(), d_bg_stats, static_cast<size_t>(bg_b) * sizeof(BgCtaStats),
                              cudaMemcpyDeviceToHost));
        uint64_t total_bytes = 0, min_t0 = UINT64_MAX, max_t1 = 0;
        for (int b = 0; b < bg_b; ++b) {
          total_bytes += h_bg_stats[b].bytes;
          if (h_bg_stats[b].t_start_ns > 0 && h_bg_stats[b].t_start_ns < min_t0) min_t0 = h_bg_stats[b].t_start_ns;
          if (h_bg_stats[b].t_end_ns > max_t1) max_t1 = h_bg_stats[b].t_end_ns;
        }
        return max_t1 > min_t0 ? static_cast<double>(total_bytes) / static_cast<double>(max_t1 - min_t0) : 0.0;
      };
      if (schedule) {
        // Length = chase steps; window = the per-step latency the kernel writes times the steps.
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        int current = 0;
        std::vector<double> bw_all;
        std::printf("%s\n    {\"bg_blocks\": %d, \"bg_threads\": %d", li ? "," : "", bg_b, bg_t);
        auto timed = schedule->time_point(
            steps, 1, 1 << 24, std::max(warmup, 1), samples, ev_start, ev_stop,
            [&] {
              if (is_dram) {
                l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_bg_buf, d_sink_vec, bg_dram_bytes / sizeof(uint4), 1);
              }
              CUDA_CHECK(cudaMemset(d_start, 0, sizeof(int)));
              CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
              CUDA_CHECK(cudaMemset(d_chunk_counter, 0, sizeof(unsigned long long)));
              if (bg_b > 0) CUDA_CHECK(cudaMemset(d_bg_stats, 0, static_cast<size_t>(bg_b) * sizeof(BgCtaStats)));
            },
            [&](int length) {
              current = length;
              pchase_bw_curve_kernel<<<grid, threads>>>(chain, d_sink, d_lat_ns, d_bg_stats, length,
                                                        (bg_b > 0) ? d_bg_buf : nullptr, bg_vec4, bg_t, d_start,
                                                        d_done, d_chunk_counter);
            },
            [&]() {
              double lat = 0.0;
              CUDA_CHECK(cudaMemcpy(&lat, d_lat_ns, sizeof(double), cudaMemcpyDeviceToHost));
              bw_all.push_back(background_gbps());
              return lat * current;
            });
        const size_t n = timed.observed.size();
        std::printf(", \"window_ns_samples\": [");
        for (size_t k = 0; k < n; ++k) std::printf("%s%.1f", k ? ", " : "", timed.observed[k]);
        // The unloaded level has no background, so no bandwidth list (zero is not a sample).
        if (bg_b > 0) {
          std::printf("], \"achieved_bw_gbps_samples\": [");
          for (size_t k = bw_all.size() - n; k < bw_all.size(); ++k) {
            std::printf("%s%.3f", k + n > bw_all.size() ? ", " : "", bw_all[k]);
          }
        }
        std::printf("], \"latency_ns_samples\": [");
        for (size_t k = 0; k < n; ++k) std::printf("%s%.2f", k ? ", " : "", timed.observed[k] / timed.length);
        std::printf("]}");
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        continue;
      }

      std::vector<double> bw_samples;
      std::vector<double> lat_samples;
      bw_samples.reserve(samples);
      lat_samples.reserve(samples);

      for (int s = 0; s < samples; ++s) {
        if (is_dram) {
          // Clean read-only L2 flush avoids dirty write-backs during measurement.
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_bg_buf, d_sink_vec, bg_dram_bytes / sizeof(uint4), 1);
        }
        CUDA_CHECK(cudaMemset(d_start, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_chunk_counter, 0, sizeof(unsigned long long)));
        if (bg_b > 0) {
          CUDA_CHECK(cudaMemset(d_bg_stats, 0, static_cast<size_t>(bg_b) * sizeof(BgCtaStats)));
        }

        pchase_bw_curve_kernel<<<grid, threads>>>(
            chain, d_sink, d_lat_ns, d_bg_stats, steps,
            (bg_b > 0) ? d_bg_buf : nullptr, bg_vec4, bg_t, d_start, d_done, d_chunk_counter);
        CUDA_CHECK(cudaDeviceSynchronize());

        double lat_ns = 0.0;
        CUDA_CHECK(cudaMemcpy(&lat_ns, d_lat_ns, sizeof(double), cudaMemcpyDeviceToHost));
        double bw_gbps = 0.0;
        if (bg_b > 0) {
          CUDA_CHECK(cudaMemcpy(h_bg_stats.data(), d_bg_stats,
                                static_cast<size_t>(bg_b) * sizeof(BgCtaStats),
                                cudaMemcpyDeviceToHost));
          // Total background bytes divided by global wall span [min(t_start), max(t_end)].
          uint64_t total_bytes = 0;
          uint64_t min_t0 = UINT64_MAX;
          uint64_t max_t1 = 0;
          for (int b = 0; b < bg_b; ++b) {
            total_bytes += h_bg_stats[b].bytes;
            if (h_bg_stats[b].t_start_ns > 0 && h_bg_stats[b].t_start_ns < min_t0) {
              min_t0 = h_bg_stats[b].t_start_ns;
            }
            if (h_bg_stats[b].t_end_ns > max_t1) {
              max_t1 = h_bg_stats[b].t_end_ns;
            }
          }
          if (max_t1 > min_t0) {
            bw_gbps = static_cast<double>(total_bytes) / static_cast<double>(max_t1 - min_t0);
          }
        }
        bw_samples.push_back(bw_gbps);
        lat_samples.push_back(lat_ns);
      }

      if (li > 0) std::printf(",");
      std::printf("\n    {\"bg_blocks\": %d, \"bg_threads\": %d, \"achieved_bw_gbps_samples\": [",
                  bg_b, bg_t);
      for (size_t k = 0; k < bw_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.3f", bw_samples[k]);
      }
      std::printf("], \"latency_ns_samples\": [");
      for (size_t k = 0; k < lat_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.2f", lat_samples[k]);
      }
      std::printf("]}");
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n", samples);
  measure_levels("dram_latency_by_bw", true, true);
  measure_levels("l2_latency_by_bw", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaFree(d_l2_chain));
  CUDA_CHECK(cudaFree(d_dram_chain));
  CUDA_CHECK(cudaFree(d_bg_buf));
  CUDA_CHECK(cudaFree(d_sink_vec));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_lat_ns));
  CUDA_CHECK(cudaFree(d_bg_stats));
  CUDA_CHECK(cudaFree(d_start));
  CUDA_CHECK(cudaFree(d_done));
  CUDA_CHECK(cudaFree(d_chunk_counter));
  return 0;
}

// ---------------------------------------------------------------------------
// 7. L2 rescan hit rate vs working-set size (0.25x to 4.0x L2)
// ---------------------------------------------------------------------------

static int run_l2_rescan(int device, int samples, int warmup = 1, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  // Dense sampling around the L2 thrashing cliff (0.96x - 1.25x L2).
  std::vector<double> ratios = {
      0.25, 0.50, 0.75, 0.90, 0.96, 1.00, 1.04, 1.08, 1.12, 1.16, 1.25, 1.50, 2.00, 3.00, 4.00};
  std::vector<size_t> ws_list;
  for (double r : ratios) {
    size_t sz = static_cast<size_t>((l2_bytes * r) / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);
    sz = std::max<size_t>(1ULL * 1024ULL * 1024ULL, sz);
    if (ws_list.empty() || sz > ws_list.back()) {
      ws_list.push_back(sz);
    }
  }
  const size_t max_ws = ws_list.back();
  const size_t flush_bytes = std::max<size_t>(l2_bytes * 3ULL, 128ULL * 1024ULL * 1024ULL);

  uint4* d_buf = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, max_ws));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_buf, 11, max_ws));
  CUDA_CHECK(cudaMemset(d_flush, 12, flush_bytes));

  cudaEvent_t ev0, ev1, ev2;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventCreate(&ev2));

  const int grid = sm_count * 4;
  const int threads = 512;
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);

  std::printf("{\n  \"l2_spec_bytes\": %zu,\n  \"samples\": %d,\n  \"points\": [",
              l2_bytes, samples);

  for (size_t wi = 0; wi < ws_list.size(); ++wi) {
    size_t ws = ws_list[wi];
    size_t ws_vec4 = ws / sizeof(uint4);
    if (schedule) {
      // One sample is a group of K repeats of flush, pass 1, pass 2, each pass one launch and
      // the flush outside the pass timing. The length is K; the window is the summed pass-2
      // time of the group; the pass bandwidths are the medians over the group.
      std::vector<cudaEvent_t> marks;
      std::vector<double> p1_all, p2_all;
      int current = 0;
      auto ensure = [&](int k) {
        while (static_cast<int>(marks.size()) < 3 * k) {
          cudaEvent_t e;
          CUDA_CHECK(cudaEventCreate(&e));
          marks.push_back(e);
        }
      };
      auto group = [&](int k) {
        current = k;
        ensure(k);
        for (int g = 0; g < k; ++g) {
          l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_vec4, 1);
          CUDA_CHECK(cudaEventRecord(marks[3 * g]));
          l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
          CUDA_CHECK(cudaEventRecord(marks[3 * g + 1]));
          l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
          CUDA_CHECK(cudaEventRecord(marks[3 * g + 2]));
        }
      };
      auto observe = [&]() {
        std::vector<double> p1, p2;
        double pass2_ns = 0.0;
        for (int g = 0; g < current; ++g) {
          float a = 0.0f, b = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&a, marks[3 * g], marks[3 * g + 1]));
          CUDA_CHECK(cudaEventElapsedTime(&b, marks[3 * g + 1], marks[3 * g + 2]));
          p1.push_back(static_cast<double>(ws) / (a * 1e6));
          p2.push_back(static_cast<double>(ws) / (b * 1e6));
          pass2_ns += b * 1e6;
        }
        std::sort(p1.begin(), p1.end());
        std::sort(p2.begin(), p2.end());
        auto mid = [](const std::vector<double>& v) {
          return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
        };
        p1_all.push_back(mid(p1));
        p2_all.push_back(mid(p2));
        return pass2_ns;
      };
      std::printf("%s\n    {\"working_set_bytes\": %zu, \"ratio_to_l2\": %.4f", wi ? "," : "", ws,
                  static_cast<double>(ws) / static_cast<double>(l2_bytes));
      // In GB10's L2 bistable band the pass-2 time differs up to twice between processes,
      // so the pilot sizes those points to twice the floor.
      const double ratio = static_cast<double>(ws) / static_cast<double>(l2_bytes);
      const bool band = ratio >= 0.75 && ratio <= 1.25;
      schedule->observed_target_ns = band ? 2.0 * LaunchSchedule::pilot_min_ns : LaunchSchedule::pilot_min_ns;
      auto timed = schedule->time_point(1, 1, 1 << 16, std::max(warmup, 1), samples, ev0, ev1, [] {}, group,
                                        observe);
      schedule->observed_target_ns = LaunchSchedule::pilot_min_ns;
      const size_t n = timed.observed.size();
      auto tail = [&](const char* key, const std::vector<double>& v) {
        std::printf(", \"%s\": [", key);
        for (size_t k = v.size() - n; k < v.size(); ++k) std::printf("%s%.3f", k + n > v.size() ? ", " : "", v[k]);
        std::printf("]");
      };
      std::printf(", \"window_ns_samples\": [");
      for (size_t k = 0; k < n; ++k) std::printf("%s%.1f", k ? ", " : "", timed.observed[k]);
      std::printf("]");
      tail("pass1_bw_gbps_samples", p1_all);
      tail("pass2_bw_gbps_samples", p2_all);
      std::printf("}");
      for (cudaEvent_t e : marks) CUDA_CHECK(cudaEventDestroy(e));
      continue;
    }

    // Warmup (read-only flush leaves L2 clean with disjoint data, avoiding dirty write-backs).
    l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_vec4, 1);
    l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
    l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> p1_gbps, p2_gbps;
    p1_gbps.reserve(samples);
    p2_gbps.reserve(samples);

    for (int s = 0; s < samples; ++s) {
      l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev0));
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev1));
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev2));
      CUDA_CHECK(cudaEventSynchronize(ev2));

      float ms1 = 0.0f, ms2 = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms1, ev0, ev1));
      CUDA_CHECK(cudaEventElapsedTime(&ms2, ev1, ev2));
      p1_gbps.push_back((static_cast<double>(ws) / 1e9) / (static_cast<double>(ms1) / 1e3));
      p2_gbps.push_back((static_cast<double>(ws) / 1e9) / (static_cast<double>(ms2) / 1e3));
    }

    if (wi > 0) std::printf(",");
    std::printf("\n    {\"working_set_bytes\": %zu, \"ratio_to_l2\": %.4f, \"pass1_bw_gbps_samples\": [",
                ws, static_cast<double>(ws) / static_cast<double>(l2_bytes));
    for (size_t k = 0; k < p1_gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", p1_gbps[k]);
    }
    std::printf("], \"pass2_bw_gbps_samples\": [");
    for (size_t k = 0; k < p2_gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", p2_gbps[k]);
    }
    std::printf("]}");
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaEventDestroy(ev2));
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 8. MLP curves: delivered BW vs bytes in flight (cp.async & ld.global.v4)
// ---------------------------------------------------------------------------

template <int STAGES, int SUB_PER_STAGE>
__global__ void mlp_cp_async_kernel(const uint4* __restrict__ src,
                                    uint4* __restrict__ sink,
                                    size_t n_vec4_mask,
                                    int steps) {
  // Disjoint shared memory slot per (stage, sub) so every cp.async writes a unique address.
  extern __shared__ uint4 smem[];
  uint32_t tid = threadIdx.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t stride = static_cast<size_t>(gridDim.x) * 256ULL;

  auto issue_group = [&](int slot) {
    #pragma unroll
    for (int sub = 0; sub < SUB_PER_STAGE; ++sub) {
      size_t smem_off = (static_cast<size_t>(slot) * SUB_PER_STAGE + sub) * 256ULL + tid;
      uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[smem_off]));
      const void* gptr = src + idx;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr) : "memory");
      idx = (idx + stride) & n_vec4_mask;
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
  };

  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue_group(s);
  }

  for (int step = STAGES; step < steps; ++step) {
    if constexpr (STAGES == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (STAGES == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (STAGES == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    issue_group(step % STAGES);
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

template <int UNROLL>
__global__ void mlp_ld_v4_kernel(const uint4* __restrict__ src,
                                 uint4* __restrict__ sink,
                                 size_t n_vec4_mask,
                                 int outer_iters) {
  uint32_t tid = threadIdx.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t stride = static_cast<size_t>(gridDim.x) * 256ULL;
  uint32_t acc = 0;

  for (int it = 0; it < outer_iters; ++it) {
    uint32_t round_xor = 0;
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      uint4 v = ld_cg_u4(src + ((idx + static_cast<size_t>(u) * stride) & n_vec4_mask));
      round_xor ^= (v.x ^ v.y ^ v.z ^ v.w);
    }
    acc ^= round_xor;
    // Data dependency forces all UNROLL loads to return before issuing the next batch,
    // holding exactly UNROLL * 256 * 16 B in flight per CTA.
    size_t dep = (round_xor == 0xdeadbeefu) ? 1ULL : 0ULL;
    idx = (idx + static_cast<size_t>(UNROLL) * stride + dep) & n_vec4_mask;
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

// One MLP launch: `steps` cp.async stages or ld.global.v4 batches of `mult` x 4 KiB per CTA.
static void launch_mlp(bool is_cp, int mult, int ctas, const uint4* buf, uint4* sink, size_t mask, int steps) {
  if (is_cp) {
    const int cp_sub = (mult <= 4) ? 1 : (mult / 4);
    const int cp_stages = (mult <= 4) ? mult : 4;
    const size_t smem = static_cast<size_t>(cp_stages) * static_cast<size_t>(cp_sub) * 4096ULL;
    if (mult == 1) mlp_cp_async_kernel<1, 1><<<ctas, 256, smem>>>(buf, sink, mask, steps);
    else if (mult == 2) mlp_cp_async_kernel<2, 1><<<ctas, 256, smem>>>(buf, sink, mask, steps);
    else if (mult == 4) mlp_cp_async_kernel<4, 1><<<ctas, 256, smem>>>(buf, sink, mask, steps);
    else mlp_cp_async_kernel<4, 2><<<ctas, 256, smem>>>(buf, sink, mask, steps);
  } else {
    if (mult == 1) mlp_ld_v4_kernel<1><<<ctas, 256>>>(buf, sink, mask, steps);
    else if (mult == 2) mlp_ld_v4_kernel<2><<<ctas, 256>>>(buf, sink, mask, steps);
    else if (mult == 4) mlp_ld_v4_kernel<4><<<ctas, 256>>>(buf, sink, mask, steps);
    else mlp_ld_v4_kernel<8><<<ctas, 256>>>(buf, sink, mask, steps);
  }
}

static int run_mlp(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Power-of-two buffers for cheap bitmask wrap:
  // DRAM = 256 MiB (> 7x L2 on RTX 4070), L2 = 8 MiB (< 0.25x L2).
  const size_t dram_bytes = (schedule ? 1024ULL : 256ULL) * LaunchSchedule::MiB;
  const size_t flush_bytes = 256ULL * LaunchSchedule::MiB;
  const size_t l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 13, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 14, flush_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 15, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  struct MlpPoint {
    int ctas;
    int mult;  // Total 4 KB units in flight per CTA (1, 2, 4, 8, 12, 16)
  };
  // Dense log-spaced coverage across 32-256 KB (>= 4 points per octave) and extended to >= 6 MB
  // (up to 8.83 MB on 46-SM RTX 4070 with 138 concurrent CTAs = 3 CTAs/SM).
  std::vector<MlpPoint> raw_pts = {
      {1, 1},   // 4 KB
      {2, 1},   // 8 KB
      {4, 1},   // 16 KB
      {8, 1},   // 32 KB
      {10, 1},  // 40 KB
      {12, 1},  // 48 KB
      {14, 1},  // 56 KB
      {16, 1},  // 64 KB
      {19, 1},  // 76 KB
      {22, 1},  // 88 KB
      {26, 1},  // 104 KB
      {std::min(32, sm_count), 1},  // 128 KB
      {std::min(38, sm_count), 1},  // 152 KB
      {sm_count, 1},                // 184 KB
      {std::min(28, sm_count), 2},  // 224 KB
      {std::min(32, sm_count), 2},  // 256 KB
      {sm_count, 2},                // 368 KB
      {std::min(32, sm_count), 4},  // 512 KB
      {sm_count, 4},                // 736 KB
      {sm_count * 2, 4},            // 1472 KB
      {sm_count * 3, 4},            // 2208 KB
      {sm_count * 3, 8},            // 4416 KB (32 KB/CTA x 3 CTAs/SM = 96 KB/SM)
      {sm_count * 4, 8},            // 5888 KB
      {sm_count * 5, 8},            // 7360 KB (> 7 MiB)
  };
  // Deduplicate and sort strictly by bytes_in_flight = ctas * mult * 4096.
  std::sort(raw_pts.begin(), raw_pts.end(), [](const MlpPoint& a, const MlpPoint& b) {
    return (a.ctas * a.mult) < (b.ctas * b.mult);
  });
  std::vector<MlpPoint> pts;
  for (const auto& p : raw_pts) {
    if (pts.empty() || (p.ctas * p.mult) > (pts.back().ctas * pts.back().mult)) {
      pts.push_back(p);
    }
  }

  auto run_suite = [&](const char* key, bool is_dram, bool comma) {
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    const size_t flush_vec4 = flush_bytes / sizeof(uint4);

    std::printf("  \"%s\": [", key);
    bool first_out = true;
    for (const char* mode_name : {"cp_async", "ld_global_v4"}) {
      bool is_cp = (std::strcmp(mode_name, "cp_async") == 0);
      for (const auto& pt : pts) {
        int ctas = pt.ctas;
        int mult = pt.mult;
        size_t bytes_per_cta = static_cast<size_t>(mult) * 4096ULL;
        size_t bif = static_cast<size_t>(ctas) * bytes_per_cta;

        // Pick total bytes transferred so each launch takes ~0.4-2.0 ms.
        size_t target_bytes = is_dram
                                  ? std::min(dram_bytes, std::max<size_t>(16ULL * 1024ULL * 1024ULL, bif * 128ULL))
                                  : std::max<size_t>(64ULL * 1024ULL * 1024ULL, bif * 256ULL);
        int cp_sub = (mult <= 4) ? 1 : (mult / 4);
        int cp_stages = (mult <= 4) ? mult : 4;
        int cp_steps = std::max(cp_stages * 4, static_cast<int>(target_bytes / (static_cast<size_t>(ctas) * 4096ULL * cp_sub)));
        int ld_iters = std::max(4, static_cast<int>(target_bytes / (static_cast<size_t>(ctas) * 4096ULL * mult)));
        double actual_bytes = is_cp
                                  ? (static_cast<double>(ctas) * 4096.0 * cp_sub * cp_steps)
                                  : (static_cast<double>(ctas) * 4096.0 * mult * ld_iters);

        auto launch_once = [&]() { launch_mlp(is_cp, mult, ctas, buf, d_sink, mask, is_cp ? cp_steps : ld_iters); };

        if (schedule) {
          auto prepare = [&]() {
            if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          };
          auto launch = [&](int count) {
            cp_steps = ld_iters = count;
            launch_once();
          };
          const size_t bytes_per_step = static_cast<size_t>(ctas) * 4096ULL * (is_cp ? cp_sub : mult);
          // A step consumes a contiguous grid stripe. For ld.global.v4, the
          // initialized uniform uint4 values keep the load-dependent offset zero.
          const int max_steps = is_dram ? static_cast<int>(dram_bytes / bytes_per_step)
                                       : std::numeric_limits<int>::max();
          std::printf("%s\n    {\"issue_mode\": \"%s\", \"bytes_in_flight\": %zu, \"active_ctas\": %d, \"bytes_per_cta\": %zu",
                      first_out ? "" : ",", mode_name, bif, ctas, bytes_per_cta);
          first_out = false;
          const int minimum = is_cp ? cp_stages * 4 : 4;
          const size_t pilot_bytes = is_dram ? 16ULL * LaunchSchedule::MiB : 64ULL * LaunchSchedule::MiB;
          const int initial = std::max(minimum, static_cast<int>(pilot_bytes / bytes_per_step));
          schedule->measure(initial, minimum, max_steps, bytes_per_step, is_dram ? dram_bytes : l2_bytes,
                            warmup, samples, ev_start, ev_stop, prepare, launch);
          std::printf("}");
          continue;
        }

        for (int w = 0; w < warmup; ++w) {
          if (is_dram) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          }
          launch_once();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> gbps;
        gbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          if (is_dram) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          }
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          gbps.push_back((actual_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
        }

        if (!first_out) std::printf(",");
        first_out = false;
        std::printf("\n    {\"issue_mode\": \"%s\", \"bytes_in_flight\": %zu, \"active_ctas\": %d, \"bytes_per_cta\": %zu, \"gbps_samples\": [",
                    mode_name, bif, ctas, bytes_per_cta);
        for (size_t k = 0; k < gbps.size(); ++k) {
          if (k > 0) std::printf(", ");
          std::printf("%.3f", gbps[k]);
        }
        std::printf("]}");
      }
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n", samples);
  run_suite("dram_bw_by_bytes_in_flight", true, true);
  run_suite("l2_bw_by_bytes_in_flight", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// MLP points in a given order (the GB10 slow-segment timing study): each line of
// `path` is `field issue_mode bytes_in_flight active_ctas bytes_per_cta steps`,
// field being dram_bw_by_bytes_in_flight or l2_bw_by_bytes_in_flight. Buffers,
// flush, kernels and timing are those of run_mlp under a launch schedule; each
// sample adds host CLOCK_MONOTONIC stamps and one stream event before its flush,
// and each point the SM clock before and after.
static double measure_cycles_per_ns(cudaEvent_t ev_start, cudaEvent_t ev_stop);

static long long monotonic_ns(clockid_t clock = CLOCK_MONOTONIC) {
  timespec t{};
  clock_gettime(clock, &t);
  return static_cast<long long>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}
static const long long g_entry_monotonic_ns = monotonic_ns();
static const long long g_entry_realtime_ns = monotonic_ns(CLOCK_REALTIME);

static int run_mlp_order(int device, int samples, int warmup, const std::string& path) {
  struct Point {
    std::string field, issue_mode;
    size_t bytes_in_flight;
    int ctas, bytes_per_cta, steps;
  };
  std::vector<Point> points;
  std::ifstream input(path);
  for (Point p; input >> p.field >> p.issue_mode >> p.bytes_in_flight >> p.ctas >> p.bytes_per_cta >> p.steps;)
    points.push_back(p);
  if (!input.eof() || points.empty()) {
    std::fprintf(stderr, "mlp-order: cannot read %s\n", path.c_str());
    return 2;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t dram_bytes = 1024ULL * LaunchSchedule::MiB;
  const size_t flush_bytes = 256ULL * LaunchSchedule::MiB;
  const size_t l2_bytes = 8ULL * 1024ULL * 1024ULL;
  uint4 *d_dram = nullptr, *d_flush = nullptr, *d_l2 = nullptr, *d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 13, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 14, flush_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 15, l2_bytes));
  cudaEvent_t ev_flush, ev_start, ev_stop, cal_start, cal_stop;
  for (cudaEvent_t* e : {&ev_flush, &ev_start, &ev_stop, &cal_start, &cal_stop}) CUDA_CHECK(cudaEventCreate(e));

  std::printf("{\n  \"mode\": \"mlp-order\",\n  \"entry_monotonic_ns\": %lld,\n  \"entry_realtime_ns\": %lld,\n"
              "  \"samples\": %d,\n  \"warmup\": %d,\n  \"points\": [",
              g_entry_monotonic_ns, g_entry_realtime_ns, samples, warmup);
  for (size_t pi = 0; pi < points.size(); ++pi) {
    const Point& p = points[pi];
    const bool is_dram = p.field == "dram_bw_by_bytes_in_flight";
    const bool is_cp = p.issue_mode == "cp_async";
    const int mult = p.bytes_per_cta / 4096;
    if ((!is_dram && p.field != "l2_bw_by_bytes_in_flight") || (!is_cp && p.issue_mode != "ld_global_v4") ||
        (mult != 1 && mult != 2 && mult != 4 && mult != 8) || p.bytes_per_cta % 4096 != 0 ||
        p.bytes_in_flight != static_cast<size_t>(p.ctas) * p.bytes_per_cta) {
      std::fprintf(stderr, "mlp-order: invalid point %zu\n", pi);
      return 2;
    }
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = (is_dram ? dram_bytes : l2_bytes) / sizeof(uint4) - 1ULL;
    const size_t bytes_per_step = static_cast<size_t>(p.ctas) * 4096ULL * (is_cp ? (mult <= 4 ? 1 : mult / 4) : mult);
    if (is_dram && static_cast<size_t>(p.steps) * bytes_per_step > dram_bytes) {
      std::fprintf(stderr, "mlp-order: point %zu would wrap the DRAM buffer\n", pi);
      return 2;
    }
    auto prepare = [&]() {
      if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_bytes / sizeof(uint4), 1);
    };
    auto launch = [&]() { launch_mlp(is_cp, mult, p.ctas, buf, d_sink, mask, p.steps); };
    const double mhz_before = 1e3 * measure_cycles_per_ns(cal_start, cal_stop);
    for (int w = 0; w < warmup; ++w) {
      prepare();
      launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<long long> host_start, launch_end;
    std::vector<double> flush_ns, launch_ns;
    for (int s = 0; s < samples; ++s) {
      host_start.push_back(monotonic_ns());
      CUDA_CHECK(cudaEventRecord(ev_flush));
      prepare();
      CUDA_CHECK(cudaEventRecord(ev_start));
      launch();
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      launch_end.push_back(monotonic_ns());
      float ms = 0, fms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      CUDA_CHECK(cudaEventElapsedTime(&fms, ev_flush, ev_start));
      launch_ns.push_back(static_cast<double>(ms) * 1e6);
      flush_ns.push_back(static_cast<double>(fms) * 1e6);
    }
    CUDA_CHECK(cudaGetLastError());
    const double mhz_after = 1e3 * measure_cycles_per_ns(cal_start, cal_stop);
    const double transfer = static_cast<double>(p.steps) * bytes_per_step;
    std::printf("%s\n    {\"exec_index\": %zu, \"field\": \"%s\", \"issue_mode\": \"%s\", \"bytes_in_flight\": %zu, "
                "\"active_ctas\": %d, \"bytes_per_cta\": %d, \"steps\": %d, \"transfer_bytes\": %.0f, "
                "\"sm_mhz_before\": %.1f, \"sm_mhz_after\": %.1f",
                pi ? "," : "", pi, p.field.c_str(), p.issue_mode.c_str(), p.bytes_in_flight, p.ctas, p.bytes_per_cta,
                p.steps, transfer, mhz_before, mhz_after);
    auto ints = [](const char* key, const std::vector<long long>& v) {
      std::printf(", \"%s\": [", key);
      for (size_t i = 0; i < v.size(); ++i) std::printf("%s%lld", i ? ", " : "", v[i]);
      std::printf("]");
    };
    ints("host_start_ns", host_start);
    ints("launch_end_ns", launch_end);
    std::vector<long long> flush_end;
    for (int s = 0; s < samples; ++s) flush_end.push_back(launch_end[s] - static_cast<long long>(launch_ns[s]));
    ints("flush_end_ns", flush_end);
    std::printf(", \"flush_ns\": [");
    for (int s = 0; s < samples; ++s) std::printf("%s%.1f", s ? ", " : "", flush_ns[s]);
    std::printf("], \"launch_ns\": [");
    for (int s = 0; s < samples; ++s) std::printf("%s%.1f", s ? ", " : "", launch_ns[s]);
    std::printf("], \"gbps_samples\": [");
    for (int s = 0; s < samples; ++s) std::printf("%s%.3f", s ? ", " : "", transfer / launch_ns[s]);
    std::printf("]}");
  }
  std::printf("\n  ]\n}\n");
  for (cudaEvent_t e : {ev_flush, ev_start, ev_stop, cal_start, cal_stop}) CUDA_CHECK(cudaEventDestroy(e));
  for (uint4* b : {d_dram, d_flush, d_l2, d_sink}) CUDA_CHECK(cudaFree(b));
  return 0;
}

// ---------------------------------------------------------------------------
// 9. Closed-loop MLP benchmark: D tokens circulating between DRAM and compute
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void spin_cycles(uint64_t cycles) {
  if (cycles == 0) return;
  uint64_t t0 = static_cast<uint64_t>(clock64());
  while (static_cast<uint64_t>(clock64()) - t0 < cycles) {
  }
}

__global__ void calibrate_clock_kernel(uint64_t cycles, uint64_t* __restrict__ sink) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    uint64_t t0 = static_cast<uint64_t>(clock64());
    while (static_cast<uint64_t>(clock64()) - t0 < cycles) {
    }
    *sink = static_cast<uint64_t>(clock64()) - t0;
  }
}

static double measure_cycles_per_ns(cudaEvent_t ev_start, cudaEvent_t ev_stop) {
  uint64_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  const uint64_t cal_cycles = 5000000ULL;  // ~1.7-2.0 ms
  calibrate_clock_kernel<<<1, 32>>>(cal_cycles, d_sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  calibrate_clock_kernel<<<1, 32>>>(cal_cycles, d_sink);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float ms = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return static_cast<double>(cal_cycles) / (static_cast<double>(ms) * 1e6);
}

template <int TOKENS, int VECS_PER_THREAD>
__global__ void closed_loop_cp_async_kernel(const uint4* __restrict__ src,
                                            uint4* __restrict__ sink,
                                            size_t n_vec4_mask,
                                            int steps,
                                            uint64_t compute_cycles) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(slot_ptr + static_cast<size_t>(v) * 256ULL + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  // Prefill TOKENS stages into the ring buffer.
  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    // Wait until the oldest stage arrives, leaving at most (TOKENS - 1) stages in flight.
    if constexpr (TOKENS == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (TOKENS == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (TOKENS == 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    } else if constexpr (TOKENS == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    // Hold the arrived stage buffer during compute before releasing and refilling it.
    spin_cycles(compute_cycles);
    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

template <int VECS_PER_THREAD>
__global__ void closed_loop_ld_v4_kernel(const uint4* __restrict__ src,
                                         uint4* __restrict__ sink,
                                         size_t n_vec4_mask,
                                         int steps,
                                         uint64_t compute_cycles) {
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;
  uint32_t acc = 0;

  for (int step = 0; step < steps; ++step) {
    uint32_t round_xor = 0;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint4 val = ld_cg_u4(src + ((base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask));
      round_xor ^= (val.x ^ val.y ^ val.z ^ val.w);
    }
    acc ^= round_xor;
    // Force all loads in this stage to complete before compute_cycles begins.
    uint64_t dep_cyc = (round_xor == 0xdeadbeefu) ? 1ULL : 0ULL;
    spin_cycles(compute_cycles + dep_cyc);
    base = (base + grid_stage_vec4 + static_cast<size_t>(dep_cyc)) & n_vec4_mask;
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

static int run_closed_loop(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  // Opt-in to 64 KB dynamic shared memory for tokens=4, stage_bytes=16384.
  CUDA_CHECK(cudaFuncSetAttribute(
      closed_loop_cp_async_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 17, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 18, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 19, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> token_list = {0, 1, 2, 3, 4};
  const std::vector<int> cta_list = {16, 32, sm_count, sm_count * 2, 128};
  const std::vector<int> stage_bytes_list = {4096, 8192, 16384};
  const std::vector<uint64_t> compute_ns_list = {0ULL, 250ULL, 500ULL, 1000ULL, 2000ULL};

  auto max_resident_ctas = [&](int tokens, int vecs, size_t dyn_smem) -> int {
    int max_blk = 1;
    if (tokens == 0) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<1>, 256, 0);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<2>, 256, 0);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<4>, 256, 0);
    } else if (tokens == 1) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 4>, 256, dyn_smem);
    } else if (tokens == 2) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 4>, 256, dyn_smem);
    } else if (tokens == 3) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 4>, 256, dyn_smem);
    } else {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 4>, 256, dyn_smem);
    }
    return sm_count * max_blk;
  };

  auto run_level = [&](const char* key, bool is_dram, bool comma) {
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    std::printf("  \"%s\": [", key);
    bool first = true;

    for (int tokens : token_list) {
      const char* mode_name = (tokens == 0) ? "ld_global_v4" : "cp_async";
      for (int ctas : cta_list) {
        for (int stage_bytes : stage_bytes_list) {
          const int vecs = stage_bytes / 4096;
          const size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
          // Skip configs that exceed single-wave residency so every point is truly concurrent.
          if (ctas > max_resident_ctas(tokens, vecs, dyn_smem)) continue;

          for (uint64_t tc_ns : compute_ns_list) {
            const uint64_t tc_cyc = static_cast<uint64_t>(static_cast<double>(tc_ns) * cycles_per_ns);
            const double bw_ref = is_dram ? 450.0 : 1400.0;
            double est_bw_step_ns = (static_cast<double>(ctas) * stage_bytes) / bw_ref;
            double est_step_ns = std::max({static_cast<double>(tc_ns), est_bw_step_ns, is_dram ? 250.0 : 120.0});
            int steps = std::max(256, std::min(1024, static_cast<int>(500000.0 / est_step_ns)));

            auto launch_once = [&]() {
              if (tokens == 0) {
                if (vecs == 1) closed_loop_ld_v4_kernel<1><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_ld_v4_kernel<2><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_ld_v4_kernel<4><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 1) {
                if (vecs == 1) closed_loop_cp_async_kernel<1, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<1, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<1, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 2) {
                if (vecs == 1) closed_loop_cp_async_kernel<2, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<2, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<2, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 3) {
                if (vecs == 1) closed_loop_cp_async_kernel<3, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<3, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<3, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else {
                if (vecs == 1) closed_loop_cp_async_kernel<4, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<4, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<4, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              }
            };

            auto prepare = [&]() {
              if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
            };
            std::vector<double> launch_ns;
            if (!first) std::printf(",");
            first = false;
            std::printf(
                "\n    {\"issue_mode\": \"%s\", \"tokens\": %d, \"active_ctas\": %d, "
                "\"stage_bytes\": %d, \"compute_ns\": %llu",
                mode_name, tokens, ctas, stage_bytes, static_cast<unsigned long long>(tc_ns));
            if (schedule) {
              auto timed = schedule->time_point(steps, 1, 1 << 22, warmup, samples, ev_start, ev_stop,
                                                prepare, [&](int length) { steps = length; launch_once(); });
              steps = timed.length;
              launch_ns = timed.launch_ns;
            } else {
              for (int w = 0; w < warmup; ++w) {
                prepare();
                launch_once();
              }
              CUDA_CHECK(cudaDeviceSynchronize());
              for (int s = 0; s < samples; ++s) {
                prepare();
                CUDA_CHECK(cudaEventRecord(ev_start));
                launch_once();
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                launch_ns.push_back(static_cast<double>(ms) * 1e6);
              }
            }
            const double total_bytes = static_cast<double>(ctas) * static_cast<double>(stage_bytes) * steps;
            std::vector<double> bw_samples;
            std::vector<double> turn_samples;
            for (double ns : launch_ns) {
              bw_samples.push_back(total_bytes / ns);
              turn_samples.push_back(ns / static_cast<double>(steps));
            }
            std::printf(", \"bw_gbps_samples\": [");
            for (size_t k = 0; k < bw_samples.size(); ++k) {
              if (k > 0) std::printf(", ");
              std::printf("%.3f", bw_samples[k]);
            }
            std::printf("], \"turnaround_ns_samples\": [");
            for (size_t k = 0; k < turn_samples.size(); ++k) {
              if (k > 0) std::printf(", ");
              std::printf("%.2f", turn_samples[k]);
            }
            std::printf("]}");
          }
        }
      }
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n", samples, cycles_per_ns);
  run_level("dram_closed_loop", true, true);
  run_level("l2_closed_loop", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 10. L2 miss-merging benchmark: sharing degree R with stagger=0 vs stagger=16
// ---------------------------------------------------------------------------

template <int VECS_PER_THREAD>
__global__ void l2_merge_cp_async_kernel(const uint4* __restrict__ src,
                                         uint4* __restrict__ sink,
                                         double* __restrict__ out_cta_turn_cycles,
                                         size_t n_vec4_mask,
                                         int steps,
                                         int sharing,
                                         uint64_t stagger_delay_cycles) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const int group_id = blockIdx.x / sharing;
  const int rank = blockIdx.x % sharing;
  const int n_groups = gridDim.x / sharing;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t group_stride_vec4 = static_cast<size_t>(n_groups) * stage_vec4;
  size_t base = (static_cast<size_t>(group_id) * stage_vec4 + tid) & n_vec4_mask;

  // When staggered, followers wait briefly so rank 0 fetches data into L2 first (resident hit).
  if (stagger_delay_cycles > 0 && rank > 0) {
    spin_cycles(stagger_delay_cycles);
  }
  __syncthreads();
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;

  for (int step = 0; step < steps; ++step) {
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem + static_cast<size_t>(v) * 256ULL + tid));
      const void* gptr = src + ((base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    base = (base + group_stride_vec4) & n_vec4_mask;
  }

  __syncthreads();
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cta_turn_cycles[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

static int run_l2_merge(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // With a schedule, leaders read cold data from a 1 GiB buffer without wrapping.
  const size_t flush_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t dram_bytes = schedule ? 1024ULL * LaunchSchedule::MiB : flush_bytes;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  double* d_cta_turn = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  // One slot per CTA; the largest launch below is sm_count * 2. A schedule times its
  // samples in one loop, so each launch writes its own row of a ring of `ring` rows
  // and the last rows hold the timed samples.
  const size_t max_ctas = std::max<size_t>(256, static_cast<size_t>(sm_count) * 2);
  // With a schedule a sample is the median of kGroup launches (l2-merge groups registration).
  constexpr int kGroup = 9;
  const int ring = schedule ? std::max(samples * kGroup, 3) : 1;
  CUDA_CHECK(cudaMalloc(&d_cta_turn, max_ctas * ring * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_dram, 21, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 22, flush_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> n_list = {16, 23, sm_count, sm_count * 2};
  const std::vector<int> r_list = {1, 2, 4, 8, 16, 23};
  const std::vector<int> k_list = {4096, 8192, 16384};

  std::vector<double> h_cta_turn(max_ctas, 0.0);
  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"l2_merge_by_sharing\": [",
              samples, cycles_per_ns);
  bool first = true;

  for (int n : n_list) {
    for (int r : r_list) {
      if (n % r != 0) continue;
      for (int k : k_list) {
        const int vecs = k / 4096;
        const size_t dyn_smem = static_cast<size_t>(k);
        std::vector<int> staggers = (r == 1) ? std::vector<int>{0} : std::vector<int>{0, 16};
        for (int stag : staggers) {
          // ~300 ns per staggered stage so rank 0 populates L2 ahead of followers.
          uint64_t stag_cyc = static_cast<uint64_t>(static_cast<double>(stag) * 300.0 * cycles_per_ns);
          int steps = 256;
          const double total_l2_bytes = static_cast<double>(n) * static_cast<double>(k) * steps;
          double* turn_row = d_cta_turn;
          auto launch_once = [&]() {
            if (vecs == 1) l2_merge_cp_async_kernel<1><<<n, 256, dyn_smem>>>(d_dram, d_sink, turn_row, dram_mask, steps, r, stag_cyc);
            else if (vecs == 2) l2_merge_cp_async_kernel<2><<<n, 256, dyn_smem>>>(d_dram, d_sink, turn_row, dram_mask, steps, r, stag_cyc);
            else l2_merge_cp_async_kernel<4><<<n, 256, dyn_smem>>>(d_dram, d_sink, turn_row, dram_mask, steps, r, stag_cyc);
          };
          auto mean_turn_ns = [&](const double* row) {
            double avg_cyc = 0.0;
            for (int c = 0; c < n; ++c) avg_cyc += row[c];
            return avg_cyc / static_cast<double>(n) / cycles_per_ns;
          };

          if (schedule) {
            std::printf("%s\n    {\"active_ctas\": %d, \"sharing\": %d, \"bytes_per_cta\": %d, \"stagger_stages\": %d",
                        first ? "" : ",", n, r, k, stag);
            first = false;
            long calls = 0;
            auto prepare = [&]() { l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1); };
            auto launch = [&](int length) {
              steps = length;
              turn_row = d_cta_turn + static_cast<size_t>(calls++ % ring) * max_ctas;
              launch_once();
            };
            // Each step reads a fresh stripe of one stage per group; no wrap within a launch.
            const size_t stripe = static_cast<size_t>(n / r) * static_cast<size_t>(k);
            const int maximum = static_cast<int>(std::min<size_t>(dram_bytes / stripe, 1u << 30));
            schedule->group = kGroup;
            auto timed = schedule->time_point(256, 1, maximum, warmup, samples, ev_start, ev_stop, prepare, launch);
            schedule->group = 1;
            std::vector<double> rows(max_ctas * ring);
            CUDA_CHECK(cudaMemcpy(rows.data(), d_cta_turn, rows.size() * sizeof(double), cudaMemcpyDeviceToHost));
            const double l2_bytes = static_cast<double>(n) * static_cast<double>(k) * timed.length;
            // Per launch (every launch in a capture, the sizing launches' last ones in a pilot),
            // then the median over each group for the sample lists.
            const std::vector<double>& per_launch = timed.launch_each.empty() ? timed.launch_ns : timed.launch_each;
            const int g = timed.launch_each.empty() ? 1 : kGroup;
            std::vector<double> turn_each, l2_each;
            const long count = static_cast<long>(per_launch.size());
            for (long i = 0; i < count; ++i) {
              turn_each.push_back(mean_turn_ns(rows.data() + static_cast<size_t>((calls - count + i) % ring) * max_ctas));
              l2_each.push_back(l2_bytes / per_launch[i]);
            }
            const auto turn = LaunchSchedule::group_medians(turn_each, g);
            const auto l2_bw = LaunchSchedule::group_medians(l2_each, g);
            std::vector<double> dram_bw;
            for (double b : l2_bw) dram_bw.push_back(b / static_cast<double>(r));
            auto list = [](const char* key, const std::vector<double>& v, const char* fmt) {
              std::printf(", \"%s\": [", key);
              for (size_t i = 0; i < v.size(); ++i) {
                if (i > 0) std::printf(", ");
                std::printf(fmt, v[i]);
              }
              std::printf("]");
            };
            list("turnaround_ns_samples", turn, "%.2f");
            list("dram_unique_bw_gbps_samples", dram_bw, "%.3f");
            list("l2_to_sm_bw_gbps_samples", l2_bw, "%.3f");
            if (!timed.launch_each.empty()) {
              list("turnaround_ns_each", turn_each, "%.2f");
              std::vector<double> sorted_each = timed.launch_each;
              std::sort(sorted_each.begin(), sorted_each.end());
              const size_t m = sorted_each.size();
              const double med = m % 2 ? sorted_each[m / 2] : 0.5 * (sorted_each[m / 2 - 1] + sorted_each[m / 2]);
              long slow = 0;
              for (double x : timed.launch_each) slow += x > 1.15 * med;
              std::printf(", \"slow_launch_share\": %.4f", static_cast<double>(slow) / static_cast<double>(m));
            }
            std::printf("}");
            continue;
          }

          for (int w = 0; w < warmup; ++w) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
            launch_once();
          }
          CUDA_CHECK(cudaDeviceSynchronize());

          std::vector<double> turn_ns_s, dram_bw_s, l2_bw_s;
          turn_ns_s.reserve(samples);
          dram_bw_s.reserve(samples);
          l2_bw_s.reserve(samples);

          for (int s = 0; s < samples; ++s) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
            CUDA_CHECK(cudaEventRecord(ev_start));
            launch_once();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            CUDA_CHECK(cudaMemcpy(h_cta_turn.data(), d_cta_turn,
                                  static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
            double l2_bw = (total_l2_bytes / 1e9) / (static_cast<double>(ms) / 1e3);
            turn_ns_s.push_back(mean_turn_ns(h_cta_turn.data()));
            l2_bw_s.push_back(l2_bw);
            dram_bw_s.push_back(l2_bw / static_cast<double>(r));
          }

          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"active_ctas\": %d, \"sharing\": %d, \"bytes_per_cta\": %d, \"stagger_stages\": %d, "
              "\"turnaround_ns_samples\": [",
              n, r, k, stag);
          for (size_t i = 0; i < turn_ns_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.2f", turn_ns_s[i]);
          }
          std::printf("], \"dram_unique_bw_gbps_samples\": [");
          for (size_t i = 0; i < dram_bw_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.3f", dram_bw_s[i]);
          }
          std::printf("], \"l2_to_sm_bw_gbps_samples\": [");
          for (size_t i = 0; i < l2_bw_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.3f", l2_bw_s[i]);
          }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cta_turn));
  return 0;
}

// ---------------------------------------------------------------------------
// 11. Per-stage fixed sync/pipeline overhead (hot cache, minimal data)
// ---------------------------------------------------------------------------

template <int TOKENS>
__global__ void stage_overhead_cp_async_kernel(const uint4* __restrict__ hot_src,
                                               uint4* __restrict__ sink,
                                               double* __restrict__ out_cycles_per_stage,
                                               int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  // Each thread reads its own 16B slot in a tiny hot L2-resident buffer (<= 4 KB/CTA, zero bank contention).
  const void* gptr = hot_src + static_cast<size_t>(blockIdx.x) * nthreads + tid;

  auto issue_tiny = [&](int slot) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[slot * nthreads + tid]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr) : "memory");
    asm volatile("cp.async.commit_group;\n" :: : "memory");
  };

  // Warmup and prefill TOKENS stages.
  for (int w = 0; w < 16; ++w) {
    issue_tiny(w % TOKENS);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
  }
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    issue_tiny(s);
  }
  __syncthreads();
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (TOKENS == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (TOKENS == 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    } else if constexpr (TOKENS == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    __syncthreads();
    issue_tiny(step % TOKENS);
  }

  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cycles_per_stage[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

__global__ void stage_overhead_ld_v4_kernel(const uint4* __restrict__ hot_src,
                                            uint4* __restrict__ sink,
                                            double* __restrict__ out_cycles_per_stage,
                                            int steps) {
  const uint32_t tid = threadIdx.x;
  const uint4* gptr = hot_src + (tid & 31u);
  uint32_t acc = 0;

  for (int w = 0; w < 16; ++w) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(gptr));
    acc ^= v.x;
    __syncthreads();
  }

  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(gptr + ((acc == 0xdeadbeefu) ? 1u : 0u)));
    acc ^= (v.x ^ v.y ^ v.z ^ v.w);
    __syncthreads();
  }
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cycles_per_stage[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

template <int TOKENS>
__global__ void fixed_sync_empty_kernel(double* __restrict__ out_cycles_per_stage, int steps) {
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  __syncthreads();
  uint64_t t0 = (threadIdx.x == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    __syncthreads();
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    out_cycles_per_stage[0] = static_cast<double>(static_cast<uint64_t>(clock64()) - t0) / static_cast<double>(steps);
  }
}

__global__ void min_async_roundtrip_kernel(const uint4* __restrict__ hot_src,
                                           uint4* __restrict__ sink,
                                           double* __restrict__ out_cycles_per_stage,
                                           int steps) {
  __shared__ uint4 smem[32];
  uint32_t tid = threadIdx.x;
  uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[0]));
  for (int w = 0; w < 16; ++w) {
    if (tid == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(hot_src) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
  }
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    if (tid == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(hot_src) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
  }
  if (tid == 0) {
    out_cycles_per_stage[0] = static_cast<double>(static_cast<uint64_t>(clock64()) - t0) / static_cast<double>(steps);
    if (smem[0].x == 0xdeadbeefu) sink[0] = smem[0];
  }
}

static int run_stage_overhead(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Run on 1 CTA so per-stage sync/pipeline overhead is isolated from cross-SM L2 contention.
  const int grid = 1;
  uint4* d_hot = nullptr;
  uint4* d_sink = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hot, 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_hot, 3, 512 * sizeof(uint4)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> token_list = {0, 1, 2, 3, 4};
  const int steps = 4096;

  // With a schedule each probe is a timed point: the pilot raises its loop steps until a
  // launch takes 1.25 ms, and the point prints the in-kernel window (cycles over the steps
  // times the steps) next to the per-step value, which stays the window over the steps.
  auto scheduled = [&](int initial, auto launch) {
    int length = initial;
    auto timed = schedule->time_point(initial, 1, 1 << 24, warmup, samples, ev_start, ev_stop, [] {},
                                      [&](int n) { length = n; launch(n); },
                                      [&] {
                                        double cyc = 0.0;
                                        CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
                                        return cyc * length / cycles_per_ns;
                                      });
    std::printf(", \"window_ns_samples\": [");
    for (size_t i = 0; i < timed.observed.size(); ++i) std::printf("%s%.1f", i ? ", " : "", timed.observed[i]);
    std::printf("], \"overhead_ns_samples\": [");
    for (size_t i = 0; i < timed.observed.size(); ++i)
      std::printf("%s%.2f", i ? ", " : "", timed.observed[i] / timed.length);
    std::printf("]}");
    return timed;
  };

  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"stage_sync_overhead_ns\": [",
              samples, cycles_per_ns);
  bool first = true;

  for (int warps : warp_list) {
    int threads = warps * 32;
    for (int tokens : token_list) {
      const char* mode_name = (tokens == 0) ? "ld_global_v4" : "cp_async";
      size_t dyn_smem = static_cast<size_t>(std::max(1, tokens)) * static_cast<size_t>(threads) * sizeof(uint4);

      auto launch_n = [&](int n) {
        if (tokens == 0) {
          stage_overhead_ld_v4_kernel<<<grid, threads>>>(d_hot, d_sink, d_out, n);
        } else if (tokens == 1) {
          stage_overhead_cp_async_kernel<1><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, n);
        } else if (tokens == 2) {
          stage_overhead_cp_async_kernel<2><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, n);
        } else if (tokens == 3) {
          stage_overhead_cp_async_kernel<3><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, n);
        } else {
          stage_overhead_cp_async_kernel<4><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, n);
        }
      };
      auto launch_once = [&]() { launch_n(steps); };
      if (schedule) {
        std::printf("%s\n    {\"issue_mode\": \"%s\", \"num_warps\": %d, \"tokens\": %d", first ? "" : ",",
                    mode_name, warps, tokens);
        first = false;
        scheduled(steps, launch_n);
        continue;
      }

      for (int w = 0; w < warmup; ++w) launch_once();
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> ns_samples;
      ns_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());
        double cyc = 0.0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        ns_samples.push_back(cyc / cycles_per_ns);
      }

      if (!first) std::printf(",");
      first = false;
      std::printf("\n    {\"issue_mode\": \"%s\", \"num_warps\": %d, \"tokens\": %d, \"overhead_ns_samples\": [",
                  mode_name, warps, tokens);
      for (size_t i = 0; i < ns_samples.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%.2f", ns_samples[i]);
      }
      std::printf("]}");
    }
  }
  std::printf("\n  ],\n  \"fixed_sync_ns\": [");
  bool first_fs = true;
  for (int warps : warp_list) {
    int threads = warps * 32;
    for (int tokens : {1, 2, 3, 4}) {
      auto launch_fs_n = [&](int n) {
        if (tokens == 1) fixed_sync_empty_kernel<1><<<grid, threads>>>(d_out, n);
        else if (tokens == 2) fixed_sync_empty_kernel<2><<<grid, threads>>>(d_out, n);
        else if (tokens == 3) fixed_sync_empty_kernel<3><<<grid, threads>>>(d_out, n);
        else fixed_sync_empty_kernel<4><<<grid, threads>>>(d_out, n);
      };
      auto launch_fs = [&]() { launch_fs_n(steps); };
      if (schedule) {
        std::printf("%s\n    {\"num_warps\": %d, \"tokens\": %d", first_fs ? "" : ",", warps, tokens);
        first_fs = false;
        scheduled(steps, launch_fs_n);
        continue;
      }
      for (int w = 0; w < warmup; ++w) launch_fs();
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<double> fs_samples;
      fs_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        launch_fs();
        CUDA_CHECK(cudaDeviceSynchronize());
        double cyc = 0.0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        fs_samples.push_back(cyc / cycles_per_ns);
      }
      if (!first_fs) std::printf(",");
      first_fs = false;
      std::printf("\n    {\"num_warps\": %d, \"tokens\": %d, \"overhead_ns_samples\": [", warps, tokens);
      for (size_t i = 0; i < fs_samples.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%.2f", fs_samples[i]);
      }
      std::printf("]}");
    }
  }

  if (schedule) {
    std::printf("\n  ],\n  \"min_async_roundtrip\": {\"num_warps\": 4");
    auto timed = scheduled(steps, [&](int n) { min_async_roundtrip_kernel<<<1, 128>>>(d_hot, d_sink, d_out, n); });
    std::printf(",\n  \"min_async_roundtrip_ns_samples\": [");
    for (size_t i = 0; i < timed.observed.size(); ++i)
      std::printf("%s%.2f", i ? ", " : "", timed.observed[i] / timed.length);
    std::printf("]\n}\n");
  } else {
    for (int w = 0; w < warmup; ++w) min_async_roundtrip_kernel<<<1, 128>>>(d_hot, d_sink, d_out, steps);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("\n  ],\n  \"min_async_roundtrip_ns_samples\": [");
    for (int s = 0; s < samples; ++s) {
      min_async_roundtrip_kernel<<<1, 128>>>(d_hot, d_sink, d_out, steps);
      CUDA_CHECK(cudaDeviceSynchronize());
      double cyc = 0.0;
      CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
      if (s > 0) std::printf(", ");
      std::printf("%.2f", cyc / cycles_per_ns);
    }
    std::printf("]\n}\n");
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_hot));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

// ---------------------------------------------------------------------------
// 12. Closed-loop DRAM MLP with real MMA compute contention (1/2/3 CTAs/SM)
// ---------------------------------------------------------------------------

template <int TOKENS>
__global__ void closed_loop_mma_kernel(const uint4* __restrict__ src,
                                       float* __restrict__ sink,
                                       size_t n_vec4_mask,
                                       int steps,
                                       int mma_iters) {
  // 128 threads (4 warps/CTA), 8192 B per stage (4 x 16B per thread).
  // Max dynamic smem = 4 * 8192 = 32 KB/CTA -> 3 CTAs/SM = 96 KB/SM (single wave on RTX 4070).
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = 512ULL;  // 8192 B / 16 B
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  uint32_t a0 = 0x3c003c00u + (tid & 7u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(slot_ptr + v * 128 + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 128ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");

    for (int m = 0; m < mma_iters; ++m) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(smem[tid].x);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int TOKENS, int LD_PASSES>
__global__ void closed_loop_mma_8w_kernel(const uint4* __restrict__ src,
                                          float* __restrict__ sink,
                                          size_t n_vec4_mask,
                                          int steps,
                                          int mma_iters) {
  // 256 threads (8 warps/CTA), 8192 B per stage (2 x 16B per thread).
  // Max dynamic smem = 4 * 8192 = 32 KB/CTA -> 2 CTAs/SM = 64 KB/SM (single wave on RTX 4070).
  // LD_PASSES in {0, 4, 8} => 0 / 16384 / 32768 B of ldmatrix.x4 reads per stage per CTA.
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  const size_t stage_vec4 = 512ULL;  // 8192 B / 16 B
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  uint32_t a0 = 0x3c003c00u + (tid & 7u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < 2; ++v) {
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(slot_ptr + v * 256 + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");

    if constexpr (LD_PASSES > 0) {
      asm volatile("bar.sync 0;\n" ::: "memory");
      uint4* cur_slot = smem + static_cast<size_t>(step % TOKENS) * stage_vec4;
      #pragma unroll
      for (int p = 0; p < LD_PASSES; ++p) {
        size_t off_u4 = (static_cast<size_t>(p & 1) * 8ULL + warp_id) * 32ULL + lane;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(cur_slot + off_u4));
        uint32_t r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
        live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
        if (p == LD_PASSES - 2) {
          b0 = r0;
          b1 = r1;
        } else if (p == LD_PASSES - 1) {
          a0 = r0;
          a1 = r1;
          a2 = r2;
          a3 = r3;
        }
      }
    }

    for (int m = 0; m < mma_iters; ++m) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    if constexpr (LD_PASSES > 0) {
      asm volatile("bar.sync 0;\n" ::: "memory");
    }

    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(smem[tid].x ^ live_acc);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int LD_PASSES>
static void dispatch_closed_loop_mma_8w(int tokens, int active_ctas, size_t dyn_smem,
                                        const uint4* d_dram, float* d_sink_f,
                                        size_t dram_mask, int steps, int mma_iters) {
  if (tokens == 1) closed_loop_mma_8w_kernel<1, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
  else if (tokens == 2) closed_loop_mma_8w_kernel<2, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
  else closed_loop_mma_8w_kernel<4, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
}

static int run_closed_loop_mma(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 4 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 24, dram_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Calibrate solo 4-warp MMA time per iteration (1 CTA/SM across all SMs, no memory traffic).
  const int cal_iters = 20000;
  for (int w = 0; w < 2; ++w) {
    mma_m16n8k16_kernel<false><<<sm_count, 128>>>(d_sink_f, cal_iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  mma_m16n8k16_kernel<false><<<sm_count, 128>>>(d_sink_f, cal_iters);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float cal_ms = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cal_ms, ev_start, ev_stop));
  const double ns_per_mma_iter = (static_cast<double>(cal_ms) * 1e6) / static_cast<double>(cal_iters);
  // 4 warps * 4 unrolled m16n8k16 * 4096 FLOPs = 65536 FLOPs per CTA per iter.
  const double flops_per_cta_iter = 4.0 * 4.0 * 4096.0;
  const double peak_tflops_per_sm = (flops_per_cta_iter / 1e12) / (ns_per_mma_iter / 1e9);

  const int stage_bytes = 8192;
  const std::vector<int> ctas_per_sm_list = {1, 2, 3};
  auto flush = [&]() { l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1); };
  // With a schedule: size and time one point, then print its rates (the caller has
  // printed the point's fields).
  auto scheduled = [&](int& steps, auto launch, int active_ctas, double flops_per_step_per_sm, double peak) {
    auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, flush,
                                      [&](int length) { steps = length; launch(); });
    const double bytes = static_cast<double>(active_ctas) * stage_bytes * timed.length;
    const double flops = flops_per_step_per_sm * timed.length;
    const auto& ns = timed.launch_ns;
    std::printf(", \"bw_gbps_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", bytes / ns[i]);
    std::printf("], \"turnaround_ns_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i] / timed.length);
    std::printf("], \"mma_tflops_per_sm_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", flops / 1e3 / ns[i]);
    std::printf("], \"tensor_util_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", peak > 0.0 ? flops / 1e3 / ns[i] / peak : 0.0);
    std::printf("]}");
  };
  const std::vector<int> token_list = {1, 2, 3, 4};
  const std::vector<double> target_tc_ns_list = {0.0, 250.0, 500.0, 1000.0, 2000.0};

  std::printf("{\n  \"samples\": %d,\n  \"ns_per_mma_iter\": %.3f,\n  \"peak_tflops_per_sm\": %.4f,\n  \"dram_closed_loop_mma\": [",
              samples, ns_per_mma_iter, peak_tflops_per_sm);
  bool first = true;

  for (int cps : ctas_per_sm_list) {
    int active_ctas = sm_count * cps;
    for (int tokens : token_list) {
      size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
      for (double target_tc : target_tc_ns_list) {
        int mma_iters = (target_tc <= 0.0) ? 0 : std::max(1, static_cast<int>(target_tc / ns_per_mma_iter + 0.5));
        if (schedule) mma_iters = schedule->parameter(mma_iters);  // frozen at the pilot's calibration
        double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_mma_iter;
        int steps = 256;

        auto launch_once = [&]() {
          if (tokens == 1) closed_loop_mma_kernel<1><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else if (tokens == 2) closed_loop_mma_kernel<2><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else if (tokens == 3) closed_loop_mma_kernel<3><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else closed_loop_mma_kernel<4><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
        };
        if (schedule) {
          std::printf("%s\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 4, "
                      "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": 0, \"mma_iters_per_stage\": %d, "
                      "\"solo_compute_ns\": %.2f",
                      first ? "" : ",", cps, active_ctas, tokens, stage_bytes, mma_iters, solo_tc_ns);
          first = false;
          scheduled(steps, launch_once, active_ctas, cps * flops_per_cta_iter * mma_iters, peak_tflops_per_sm);
          continue;
        }
        double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(stage_bytes) * steps;
        double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * steps;

        for (int w = 0; w < warmup; ++w) {
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
          launch_once();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> bw_s, turn_s, tf_s, util_s;
        bw_s.reserve(samples);
        turn_s.reserve(samples);
        tf_s.reserve(samples);
        util_s.reserve(samples);

        for (int s = 0; s < samples; ++s) {
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          double sec = static_cast<double>(ms) / 1e3;
          double tf_per_sm = (total_flops_per_sm / 1e12) / sec;
          bw_s.push_back((total_bytes / 1e9) / sec);
          turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
          tf_s.push_back(tf_per_sm);
          util_s.push_back((peak_tflops_per_sm > 0.0) ? (tf_per_sm / peak_tflops_per_sm) : 0.0);
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 4, "
            "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": 0, \"mma_iters_per_stage\": %d, "
            "\"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
            cps, active_ctas, tokens, stage_bytes, mma_iters, solo_tc_ns);
        for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
        std::printf("], \"turnaround_ns_samples\": [");
        for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
        std::printf("], \"mma_tflops_per_sm_samples\": [");
        for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
        std::printf("], \"tensor_util_samples\": [");
        for (size_t i = 0; i < util_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", util_s[i]); }
        std::printf("]}");
      }
    }
  }

  // 8-warp contention sweep: ctas_per_sm in {1, 2}, tokens in {1, 2, 4},
  // ldmatrix_bytes_per_stage in {0, 16384, 32768}.
  const int cal_iters_8w = 10000;
  for (int w = 0; w < 2; ++w) {
    mma_m16n8k16_kernel<false><<<sm_count, 256>>>(d_sink_f, cal_iters_8w);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  mma_m16n8k16_kernel<false><<<sm_count, 256>>>(d_sink_f, cal_iters_8w);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float cal_ms_8w = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cal_ms_8w, ev_start, ev_stop));
  const double ns_per_mma_iter_8w = (static_cast<double>(cal_ms_8w) * 1e6) / static_cast<double>(cal_iters_8w);
  const double flops_per_cta_iter_8w = 8.0 * 4.0 * 4096.0;
  const double peak_tflops_per_sm_8w = (flops_per_cta_iter_8w / 1e12) / (ns_per_mma_iter_8w / 1e9);

  for (int ld_passes : {0, 4, 8}) {
    const int ldm_bytes = ld_passes * 4096;
    for (int cps : {1, 2}) {
      int active_ctas = sm_count * cps;
      for (int tokens : {1, 2, 4}) {
        size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
        for (double target_tc : target_tc_ns_list) {
          int mma_iters = (target_tc <= 0.0) ? 0 : std::max(1, static_cast<int>(target_tc / ns_per_mma_iter_8w + 0.5));
          if (schedule) mma_iters = schedule->parameter(mma_iters);  // frozen at the pilot's calibration
          double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_mma_iter_8w;
          int steps = 256;

          auto launch_8w = [&]() {
            if (ld_passes == 0) dispatch_closed_loop_mma_8w<0>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else if (ld_passes == 4) dispatch_closed_loop_mma_8w<4>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else dispatch_closed_loop_mma_8w<8>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
          };
          if (schedule) {
            std::printf("%s\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 8, "
                        "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": %d, \"mma_iters_per_stage\": %d, "
                        "\"solo_compute_ns\": %.2f",
                        first ? "" : ",", cps, active_ctas, tokens, stage_bytes, ldm_bytes, mma_iters, solo_tc_ns);
            first = false;
            scheduled(steps, launch_8w, active_ctas, cps * flops_per_cta_iter_8w * mma_iters, peak_tflops_per_sm_8w);
            continue;
          }
          double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(stage_bytes) * steps;
          double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter_8w * mma_iters * steps;

          for (int w = 0; w < warmup; ++w) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
            launch_8w();
          }
          CUDA_CHECK(cudaDeviceSynchronize());

          std::vector<double> bw_s, turn_s, tf_s, util_s;
          bw_s.reserve(samples);
          turn_s.reserve(samples);
          tf_s.reserve(samples);
          util_s.reserve(samples);

          for (int s = 0; s < samples; ++s) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
            CUDA_CHECK(cudaEventRecord(ev_start));
            launch_8w();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            double sec = static_cast<double>(ms) / 1e3;
            double tf_per_sm = (total_flops_per_sm / 1e12) / sec;
            bw_s.push_back((total_bytes / 1e9) / sec);
            turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
            tf_s.push_back(tf_per_sm);
            util_s.push_back((peak_tflops_per_sm_8w > 0.0) ? (tf_per_sm / peak_tflops_per_sm_8w) : 0.0);
          }

          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 8, "
              "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": %d, \"mma_iters_per_stage\": %d, "
              "\"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
              cps, active_ctas, tokens, stage_bytes, ldm_bytes, mma_iters, solo_tc_ns);
          for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
          std::printf("], \"turnaround_ns_samples\": [");
          for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
          std::printf("], \"mma_tflops_per_sm_samples\": [");
          for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
          std::printf("], \"tensor_util_samples\": [");
          for (size_t i = 0; i < util_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", util_s[i]); }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 13. MMA chain length & ldmatrix serial stage overhead (1 CTA/SM, no GMEM)
// ---------------------------------------------------------------------------

template <int M_CHAIN, int LD_PER_WARP, bool DEPENDENT>
__global__ void mma_chain_stage_kernel(const uint4* __restrict__ init_src,
                                       float* __restrict__ sink,
                                       int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint32_t num_warps = blockDim.x >> 5;
  const size_t tile_u4 = static_cast<size_t>(LD_PER_WARP) * static_cast<size_t>(num_warps) * 32ULL;

  // Initialize shared memory from runtime buffer (filled with f16 1.0) so ptxas cannot constant-fold.
  if constexpr (LD_PER_WARP > 0) {
    for (size_t i = tid; i < 2ULL * tile_u4; i += blockDim.x) {
      smem[i] = init_src[i];
    }
    __syncthreads();
  }

  constexpr int REGS_N = (LD_PER_WARP > 0) ? LD_PER_WARP : 1;
  uint32_t reg[REGS_N][4];
  #pragma unroll
  for (int r = 0; r < REGS_N; ++r) {
    reg[r][0] = 0x3c003c00u;
    reg[r][1] = 0x3c003c00u;
    reg[r][2] = 0x3c003c00u;
    reg[r][3] = 0x3c003c00u;
  }
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (LD_PER_WARP > 0) {
      const size_t slot_base = static_cast<size_t>(step & 1) * tile_u4;
      #pragma unroll
      for (int r = 0; r < LD_PER_WARP; ++r) {
        size_t off_u4 = slot_base + (static_cast<size_t>(r) * num_warps + warp_id) * 32ULL + lane;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(reg[r][0]), "=r"(reg[r][1]), "=r"(reg[r][2]), "=r"(reg[r][3])
                     : "r"(saddr) : "memory");
      }
    }

    if constexpr (DEPENDENT) {
      #pragma unroll
      for (int i = 0; i < M_CHAIN; ++i) {
        const int ra = (LD_PER_WARP > 0) ? (i % REGS_N) : 0;
        const int rb = (LD_PER_WARP > 0) ? ((i + 1) % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[ra][0]), "r"(reg[ra][1]), "r"(reg[ra][2]), "r"(reg[ra][3]),
              "r"(reg[rb][0]), "r"(reg[rb][1]));
      }
    } else {
      constexpr int FULL_GROUPS = M_CHAIN / 4;
      constexpr int REM = M_CHAIN % 4;
      #pragma unroll
      for (int g = 0; g < FULL_GROUPS; ++g) {
        const int i0 = g * 4;
        const int ra0 = (LD_PER_WARP > 0) ? ((i0 + 0) % REGS_N) : 0;
        const int rb0 = (LD_PER_WARP > 0) ? ((i0 + 1) % REGS_N) : 0;
        const int ra1 = (LD_PER_WARP > 0) ? ((i0 + 1) % REGS_N) : 0;
        const int rb1 = (LD_PER_WARP > 0) ? ((i0 + 2) % REGS_N) : 0;
        const int ra2 = (LD_PER_WARP > 0) ? ((i0 + 2) % REGS_N) : 0;
        const int rb2 = (LD_PER_WARP > 0) ? ((i0 + 3) % REGS_N) : 0;
        const int ra3 = (LD_PER_WARP > 0) ? ((i0 + 3) % REGS_N) : 0;
        const int rb3 = (LD_PER_WARP > 0) ? ((i0 + 4) % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[ra0][0]), "r"(reg[ra0][1]), "r"(reg[ra0][2]), "r"(reg[ra0][3]),
              "r"(reg[rb0][0]), "r"(reg[rb0][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
            : "r"(reg[ra1][0]), "r"(reg[ra1][1]), "r"(reg[ra1][2]), "r"(reg[ra1][3]),
              "r"(reg[rb1][0]), "r"(reg[rb1][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
            : "r"(reg[ra2][0]), "r"(reg[ra2][1]), "r"(reg[ra2][2]), "r"(reg[ra2][3]),
              "r"(reg[rb2][0]), "r"(reg[rb2][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
            : "r"(reg[ra3][0]), "r"(reg[ra3][1]), "r"(reg[ra3][2]), "r"(reg[ra3][3]),
              "r"(reg[rb3][0]), "r"(reg[rb3][1]));
      }
      if constexpr (REM >= 1) {
        const int rb0 = (LD_PER_WARP > 0) ? (1 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[0][0]), "r"(reg[0][1]), "r"(reg[0][2]), "r"(reg[0][3]),
              "r"(reg[rb0][0]), "r"(reg[rb0][1]));
      }
      if constexpr (REM >= 2) {
        const int ra1 = (LD_PER_WARP > 0) ? (1 % REGS_N) : 0;
        const int rb1 = (LD_PER_WARP > 0) ? (2 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
            : "r"(reg[ra1][0]), "r"(reg[ra1][1]), "r"(reg[ra1][2]), "r"(reg[ra1][3]),
              "r"(reg[rb1][0]), "r"(reg[rb1][1]));
      }
      if constexpr (REM >= 3) {
        const int ra2 = (LD_PER_WARP > 0) ? (2 % REGS_N) : 0;
        const int rb2 = (LD_PER_WARP > 0) ? (3 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
            : "r"(reg[ra2][0]), "r"(reg[ra2][1]), "r"(reg[ra2][2]), "r"(reg[ra2][3]),
              "r"(reg[rb2][0]), "r"(reg[rb2][1]));
      }
    }

    // Keep any ldmatrix outputs live when M_CHAIN < LD_PER_WARP so ptxas cannot elide them.
    if constexpr (LD_PER_WARP > M_CHAIN) {
      #pragma unroll
      for (int r = M_CHAIN; r < LD_PER_WARP; ++r) {
        live_acc ^= (reg[r][0] ^ reg[r][1] ^ reg[r][2] ^ reg[r][3]);
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int LD_PER_WARP, bool DEPENDENT>
static void dispatch_mma_chain_len(int m_chain, int grid, int threads, size_t dyn_smem,
                                   const uint4* d_init, float* d_sink, int steps) {
  switch (m_chain) {
    case 1: mma_chain_stage_kernel<1, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 2: mma_chain_stage_kernel<2, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 4: mma_chain_stage_kernel<4, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 8: mma_chain_stage_kernel<8, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 16: mma_chain_stage_kernel<16, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 32: mma_chain_stage_kernel<32, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 64: mma_chain_stage_kernel<64, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 128: mma_chain_stage_kernel<128, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    default: break;
  }
}

template <bool DEPENDENT>
static void dispatch_mma_chain(int ld_per_warp, int m_chain, int grid, int threads, size_t dyn_smem,
                               const uint4* d_init, float* d_sink, int steps) {
  switch (ld_per_warp) {
    case 0: dispatch_mma_chain_len<0, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 2: dispatch_mma_chain_len<2, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 3: dispatch_mma_chain_len<3, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 4: dispatch_mma_chain_len<4, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 6: dispatch_mma_chain_len<6, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 8: dispatch_mma_chain_len<8, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    default: break;
  }
}

static int run_mma_chain(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t max_init_u4 = (2ULL * 16384ULL) / sizeof(uint4);
  std::vector<uint4> h_init(max_init_u4, make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u));
  uint4* d_init = nullptr;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_init, max_init_u4 * sizeof(uint4)));
  CUDA_CHECK(cudaMemcpy(d_init, h_init.data(), max_init_u4 * sizeof(uint4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  struct ChainConfig {
    const char* variant;
    int tile_bytes;
    bool dependent;
  };
  const std::vector<ChainConfig> configs = {
      {"mma_only_independent", 0, false},
      {"mma_only_dependent", 0, true},
      {"ldmatrix_independent", 8192, false},
      {"ldmatrix_independent", 12288, false},
      {"ldmatrix_independent", 16384, false},
      {"ldmatrix_dependent", 8192, true},
      {"ldmatrix_dependent", 12288, true},
      {"ldmatrix_dependent", 16384, true},
  };
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> m_list = {1, 2, 4, 8, 16, 32, 64, 128};

  std::printf("{\n  \"samples\": %d,\n  \"mma_chain_latency\": [", samples);
  bool first = true;

  for (const auto& cfg : configs) {
    for (int warps : warp_list) {
      int threads = warps * 32;
      int ld_per_warp = cfg.tile_bytes / (warps * 512);
      size_t dyn_smem = static_cast<size_t>(cfg.tile_bytes) * 2ULL;

      for (int m : m_list) {
        // Keep each kernel launch ~0.35 ms to avoid GPU Boost thermal/power droop at large m.
        int est_ns = std::max(20, (warps / 4) * 12 * m + 30);
        int steps = std::max(128, std::min(16384, 350000 / est_ns));
        auto launch_once = [&]() {
          if (cfg.dependent) {
            dispatch_mma_chain<true>(ld_per_warp, m, sm_count, threads, dyn_smem, d_init, d_sink, steps);
          } else {
            dispatch_mma_chain<false>(ld_per_warp, m, sm_count, threads, dyn_smem, d_init, d_sink, steps);
          }
        };
        if (schedule) {
          std::printf("%s\n    {\"variant\": \"%s\", \"num_warps\": %d, \"tile_bytes\": %d, "
                      "\"ldmatrix_x4_per_warp\": %d, \"mma_chain_len\": %d",
                      first ? "" : ",", cfg.variant, warps, cfg.tile_bytes, ld_per_warp, m);
          first = false;
          auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, [] {},
                                            [&](int length) { steps = length; launch_once(); });
          std::printf(", \"stage_time_ns_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i)
            std::printf("%s%.3f", i ? ", " : "", timed.launch_ns[i] / timed.length);
          std::printf("]}");
          continue;
        }

        for (int w = 0; w < warmup; ++w) launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ns_samples;
        ns_samples.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          ns_samples.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"variant\": \"%s\", \"num_warps\": %d, \"tile_bytes\": %d, "
            "\"ldmatrix_x4_per_warp\": %d, \"mma_chain_len\": %d, \"stage_time_ns_samples\": [",
            cfg.variant, warps, cfg.tile_bytes, ld_per_warp, m);
        for (size_t i = 0; i < ns_samples.size(); ++i) {
          if (i) std::printf(", ");
          std::printf("%.3f", ns_samples[i]);
        }
        std::printf("]}");
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_init));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 14. Serial stage fill & ldmatrix read (1 CTA/SM, hot L2, S=1, no MMA)
// ---------------------------------------------------------------------------

template <int STAGE_BYTES, int NUM_WARPS, bool USE_CP_ASYNC, bool WITH_LDMATRIX>
__global__ void serial_stage_fill_kernel(const uint4* __restrict__ src,
                                         uint32_t* __restrict__ sink,
                                         size_t l2_mask,
                                         int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & l2_mask;
  uint32_t acc = 0u;

  for (int step = 0; step < steps; ++step) {
    if constexpr (USE_CP_ASYNC) {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (tid < 128) {
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid));
          const void* gptr = src + ((base + tid) & l2_mask);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
        }
      } else {
        constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
          const void* gptr = src + ((base + off) & l2_mask);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
        }
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (tid < 128) {
          uint4 val = ld_cg_u4(src + ((base + tid) & l2_mask));
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                       :: "r"(saddr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
        }
      } else {
        constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
        uint4 vals[VECS];
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          vals[v] = ld_cg_u4(src + ((base + off) & l2_mask));
        }
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                       :: "r"(saddr), "r"(vals[v].x), "r"(vals[v].y), "r"(vals[v].z), "r"(vals[v].w) : "memory");
        }
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    if constexpr (WITH_LDMATRIX) {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (warp_id < 4) {
          size_t off_u4 = static_cast<size_t>(warp_id) * 32ULL + lane;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
          uint32_t r0, r1, r2, r3;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
          acc ^= (r0 ^ r1 ^ r2 ^ r3);
        }
      } else {
        constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
        #pragma unroll
        for (int r = 0; r < LD_N; ++r) {
          size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
          uint32_t r0, r1, r2, r3;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
          acc ^= (r0 ^ r1 ^ r2 ^ r3);
        }
      }
      asm volatile("bar.sync 0;\n" ::: "memory");
    }

    base = (base + grid_stage_u4) & l2_mask;
  }

  uint4 tail = smem[tid & (STAGE_U4 - 1ULL)];
  acc ^= (tail.x ^ tail.y ^ tail.z ^ tail.w);
  if (acc == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = acc;
  }
}

template <int NUM_WARPS, bool USE_CP_ASYNC, bool WITH_LDMATRIX>
static void dispatch_serial_fill_bytes(int stage_bytes, int grid, const uint4* d_l2,
                                       uint32_t* d_sink, size_t l2_mask, int steps) {
  const int threads = NUM_WARPS * 32;
  const size_t dyn_smem = static_cast<size_t>(stage_bytes);
  switch (stage_bytes) {
    case 2048:
      serial_stage_fill_kernel<2048, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 4096:
      serial_stage_fill_kernel<4096, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 8192:
      serial_stage_fill_kernel<8192, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 16384:
      serial_stage_fill_kernel<16384, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    default:
      break;
  }
}

static int run_serial_fill(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // 4 MiB working set (<< 36 MiB L2 on RTX 4070), pre-warmed in L2 before timing.
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  uint4* d_l2 = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> grid_list = {1, sm_count};
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> bytes_list = {2048, 4096, 8192, 16384};

  std::printf("{\n  \"samples\": %d,\n  \"serial_stage_fill\": [", samples);
  bool first = true;

  for (int grid : grid_list) {
    for (bool use_cp : {true, false}) {
      const char* t_mode = use_cp ? "cp_async" : "ld_st";
      for (bool with_ldm : {false, true}) {
        const char* r_mode = with_ldm ? "fill_and_ldmatrix" : "fill_only";
        for (int warps : warp_list) {
          for (int sb : bytes_list) {
            int steps = 4096;
            auto launch_once = [&]() {
              if (warps == 4) {
                if (use_cp && !with_ldm) dispatch_serial_fill_bytes<4, true, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (use_cp && with_ldm) dispatch_serial_fill_bytes<4, true, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (!use_cp && !with_ldm) dispatch_serial_fill_bytes<4, false, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else dispatch_serial_fill_bytes<4, false, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
              } else {
                if (use_cp && !with_ldm) dispatch_serial_fill_bytes<8, true, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (use_cp && with_ldm) dispatch_serial_fill_bytes<8, true, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (!use_cp && !with_ldm) dispatch_serial_fill_bytes<8, false, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else dispatch_serial_fill_bytes<8, false, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
              }
            };

            if (schedule) {
              std::printf("%s\n    {\"transfer_mode\": \"%s\", \"read_mode\": \"%s\", \"num_warps\": %d, "
                          "\"stage_bytes\": %d, \"active_ctas\": %d",
                          first ? "" : ",", t_mode, r_mode, warps, sb, grid);
              first = false;
              auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, [] {},
                                                [&](int length) { steps = length; launch_once(); });
              const auto& ns = timed.launch_ns;
              std::printf(", \"stage_time_ns_samples\": [");
              for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", ns[i] / timed.length);
              std::printf("], \"effective_bytes_per_ns_samples\": [");
              for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", static_cast<double>(sb) * timed.length / ns[i]);
              std::printf("]}");
              continue;
            }
            for (int w = 0; w < warmup; ++w) launch_once();
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> ns_samples;
            std::vector<double> bpns_samples;
            ns_samples.reserve(samples);
            bpns_samples.reserve(samples);
            for (int s = 0; s < samples; ++s) {
              CUDA_CHECK(cudaEventRecord(ev_start));
              launch_once();
              CUDA_CHECK(cudaEventRecord(ev_stop));
              CUDA_CHECK(cudaEventSynchronize(ev_stop));
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
              double stage_ns = (static_cast<double>(ms) * 1e6) / static_cast<double>(steps);
              ns_samples.push_back(stage_ns);
              bpns_samples.push_back(static_cast<double>(sb) / stage_ns);
            }

            if (!first) std::printf(",");
            first = false;
            std::printf(
                "\n    {\"transfer_mode\": \"%s\", \"read_mode\": \"%s\", \"num_warps\": %d, "
                "\"stage_bytes\": %d, \"active_ctas\": %d, \"stage_time_ns_samples\": [",
                t_mode, r_mode, warps, sb, grid);
            for (size_t i = 0; i < ns_samples.size(); ++i) {
              if (i) std::printf(", ");
              std::printf("%.3f", ns_samples[i]);
            }
            std::printf("], \"effective_bytes_per_ns_samples\": [");
            for (size_t i = 0; i < bpns_samples.size(); ++i) {
              if (i) std::printf(", ");
              std::printf("%.3f", bpns_samples[i]);
            }
            std::printf("]}");
          }
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 15. Post-hoc S=1 synchronous (ld.global -> st.shared) vs cp.async closed loop
//     with bar.sync -> ldmatrix -> real MMA chain -> bar.sync
// ---------------------------------------------------------------------------

template <int STAGE_BYTES, int NUM_WARPS, bool USE_CP_ASYNC>
__global__ void sync_stage_loop_kernel(const uint4* __restrict__ src,
                                       float* __restrict__ sink,
                                       size_t buf_mask,
                                       int steps,
                                       int mma_iters) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;

  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (USE_CP_ASYNC) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      uint4 vals[VECS];
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        vals[v] = ld_cg_u4(src + ((base + off) & buf_mask));
      }
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(saddr), "r"(vals[v].x), "r"(vals[v].y), "r"(vals[v].z), "r"(vals[v].w) : "memory");
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if constexpr (LD_N == 1) {
        a0 = r0; a1 = r1; a2 = r2; a3 = r3;
        b0 = r0; b1 = r1;
      } else {
        if (r == LD_N - 2) {
          b0 = r0; b1 = r1;
        } else if (r == LD_N - 1) {
          a0 = r0; a1 = r1; a2 = r2; a3 = r3;
        }
      }
    }

    for (int m = 0; m < mma_iters; ++m) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + grid_stage_u4) & buf_mask;
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int NUM_WARPS, bool USE_CP_ASYNC>
static void dispatch_sync_loop_bytes(int stage_bytes, int grid, const uint4* d_buf,
                                     float* d_sink, size_t buf_mask, int steps, int mma_iters) {
  const int threads = NUM_WARPS * 32;
  const size_t dyn_smem = static_cast<size_t>(stage_bytes);
  switch (stage_bytes) {
    case 4096:
      sync_stage_loop_kernel<4096, NUM_WARPS, USE_CP_ASYNC>
          <<<grid, threads, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    case 8192:
      sync_stage_loop_kernel<8192, NUM_WARPS, USE_CP_ASYNC>
          <<<grid, threads, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    case 16384:
      sync_stage_loop_kernel<16384, NUM_WARPS, USE_CP_ASYNC>
          <<<grid, threads, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    default:
      break;
  }
}

static int run_sync_loop(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 4 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 25, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Calibrate solo MMA time per iter for 4-warp and 8-warp CTAs.
  auto cal_mma_ns = [&](int warps, int cal_iters) -> double {
    for (int w = 0; w < 2; ++w) mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 1.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return (static_cast<double>(ms) * 1e6) / static_cast<double>(cal_iters);
  };
  const double ns_per_iter_4w = cal_mma_ns(4, 20000);
  const double ns_per_iter_8w = cal_mma_ns(8, 10000);

  const std::vector<int> bytes_list = {4096, 8192, 16384};
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> cps_list = {1, 2, 3};
  // mma_iters in {0, 2, 8, 16} => 0 / 8 / 32 / 64 m16n8k16 MMAs per warp per stage.
  const std::vector<int> mma_iters_list = {0, 2, 8, 16};

  std::printf("{\n  \"samples\": %d,\n  \"sync_stage_loop\": [", samples);
  bool first = true;

  for (bool is_dram : {true, false}) {
    const char* level_name = is_dram ? "dram" : "l2";
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    for (bool use_cp : {false, true}) {
      const char* t_mode = use_cp ? "cp_async_s1" : "ld_st_sync";
      for (int warps : warp_list) {
        const double ns_per_iter = (warps == 4) ? ns_per_iter_4w : ns_per_iter_8w;
        const double flops_per_cta_iter = static_cast<double>(warps) * 4.0 * 4096.0;
        for (int sb : bytes_list) {
          for (int cps : cps_list) {
            const int active_ctas = sm_count * cps;
            for (int mma_iters : mma_iters_list) {
              const int mma_chain_len = mma_iters * 4;
              const double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_iter;
              int steps = 128;

              auto launch_once = [&]() {
                if (warps == 4) {
                  if (use_cp) dispatch_sync_loop_bytes<4, true>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
                  else dispatch_sync_loop_bytes<4, false>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
                } else {
                  if (use_cp) dispatch_sync_loop_bytes<8, true>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
                  else dispatch_sync_loop_bytes<8, false>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
                }
              };

              auto prepare = [&]() {
                if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
              };
              if (schedule) {
                std::printf("%s\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
                            "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, "
                            "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f",
                            first ? "" : ",", level_name, t_mode, cps, active_ctas, warps, sb, mma_chain_len, solo_tc_ns);
                first = false;
                auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, prepare,
                                                  [&](int length) { steps = length; launch_once(); });
                const double bytes = static_cast<double>(active_ctas) * sb * timed.length;
                const double flops = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * timed.length;
                const auto& ns = timed.launch_ns;
                std::printf(", \"bw_gbps_samples\": [");
                for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", bytes / ns[i]);
                std::printf("], \"turnaround_ns_samples\": [");
                for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i] / timed.length);
                std::printf("], \"mma_tflops_per_sm_samples\": [");
                for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", flops / 1e3 / ns[i]);
                std::printf("]}");
                continue;
              }
              const double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(sb) * steps;
              const double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * steps;

              for (int w = 0; w < warmup; ++w) {
                prepare();
                launch_once();
              }
              CUDA_CHECK(cudaDeviceSynchronize());

              std::vector<double> bw_s, turn_s, tf_s;
              bw_s.reserve(samples);
              turn_s.reserve(samples);
              tf_s.reserve(samples);

              for (int s = 0; s < samples; ++s) {
                if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
                CUDA_CHECK(cudaEventRecord(ev_start));
                launch_once();
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
                double sec = static_cast<double>(ms) / 1e3;
                bw_s.push_back((total_bytes / 1e9) / sec);
                turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
                tf_s.push_back((total_flops_per_sm / 1e12) / sec);
              }

              if (!first) std::printf(",");
              first = false;
              std::printf(
                  "\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
                  "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, "
                  "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
                  level_name, t_mode, cps, active_ctas, warps, sb, mma_chain_len, solo_tc_ns);
              for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
              std::printf("], \"turnaround_ns_samples\": [");
              for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
              std::printf("], \"mma_tflops_per_sm_samples\": [");
              for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
              std::printf("]}");
            }
          }
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 16. Post-hoc Group 1: Global load issue throughput by width (2/4/8/16 B)
// ---------------------------------------------------------------------------

template <int WIDTH_BYTES>
__global__ void gmem_load_issue_kernel(const uint8_t* __restrict__ src,
                                       uint32_t* __restrict__ sink,
                                       size_t elem_mask,
                                       int iters) {
  const uint32_t tid = threadIdx.x;
  const uint32_t threads = blockDim.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * threads * 8ULL + tid) & elem_mask;
  const size_t step = static_cast<size_t>(threads) * 8ULL;
  uint32_t acc = 0u;

  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
      size_t cur = (idx + static_cast<size_t>(u) * threads) & elem_mask;
      if constexpr (WIDTH_BYTES == 2) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src) + cur;
        uint16_t v = 0;
        asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
        acc ^= static_cast<uint32_t>(v);
      } else if constexpr (WIDTH_BYTES == 4) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(src) + cur;
        uint32_t v = 0;
        asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
        acc ^= v;
      } else if constexpr (WIDTH_BYTES == 8) {
        const uint2* p = reinterpret_cast<const uint2*>(src) + cur;
        uint32_t x = 0, y = 0;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n" : "=r"(x), "=r"(y) : "l"(p));
        acc ^= (x ^ y);
      } else {
        const uint4* p = reinterpret_cast<const uint4*>(src) + cur;
        uint32_t x = 0, y = 0, z = 0, w = 0;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(p));
        acc ^= (x ^ y ^ z ^ w);
      }
    }
    idx = (idx + step) & elem_mask;
  }

  if (acc == 0x0000beefu) {
    sink[blockIdx.x * threads + tid] = acc;
  }
}

static int run_gmem_issue(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // 4 MiB hot L2 buffer (fits comfortably in both 36 MiB RTX 4070 L2 and 24 MiB GB10 L2).
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  uint8_t* d_l2 = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 512 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_l2, 0x5a, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> grid_list = {1, sm_count};
  const std::vector<int> width_list = {2, 4, 8, 16};
  const std::vector<int> warp_list = {4, 8, 16};

  std::printf("{\n  \"samples\": %d,\n  \"gmem_load_issue_per_sm\": [", samples);
  bool first = true;

  for (int active_sms : grid_list) {
    for (int width : width_list) {
      const size_t elem_mask = (l2_bytes / static_cast<size_t>(width)) - 1ULL;
      for (int warps : warp_list) {
        const int threads = warps * 32;
        int iters = (active_sms == 1) ? 4096 : 2048;
        auto launch_once = [&]() {
          if (width == 2) gmem_load_issue_kernel<2><<<active_sms, threads>>>(d_l2, d_sink, elem_mask, iters);
          else if (width == 4) gmem_load_issue_kernel<4><<<active_sms, threads>>>(d_l2, d_sink, elem_mask, iters);
          else if (width == 8) gmem_load_issue_kernel<8><<<active_sms, threads>>>(d_l2, d_sink, elem_mask, iters);
          else gmem_load_issue_kernel<16><<<active_sms, threads>>>(d_l2, d_sink, elem_mask, iters);
        };
        if (schedule) {
          std::printf("%s\n    {\"active_sms\": %d, \"load_width_bytes\": %d, \"num_warps\": %d",
                      first ? "" : ",", active_sms, width, warps);
          first = false;
          auto timed = schedule->time_point(iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {},
                                            [&](int length) { iters = length; launch_once(); });
          const double insts = static_cast<double>(warps) * timed.length * 8.0;
          std::printf(", \"issue_rate_gwarp_inst_per_sm_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", insts / timed.launch_ns[i]);
          std::printf("], \"delivered_bw_per_sm_gbps_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i)
            std::printf("%s%.3f", i ? ", " : "", insts * 32.0 * width / timed.launch_ns[i]);
          std::printf("]}");
          continue;
        }
        const double warp_insts_per_sm = static_cast<double>(warps) * static_cast<double>(iters) * 8.0;
        const double bytes_per_sm = warp_insts_per_sm * 32.0 * static_cast<double>(width);

        for (int w = 0; w < warmup; ++w) launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ginst_s, bw_s;
        ginst_s.reserve(samples);
        bw_s.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          double sec = static_cast<double>(ms) / 1e3;
          ginst_s.push_back((warp_insts_per_sm / 1e9) / sec);
          bw_s.push_back((bytes_per_sm / 1e9) / sec);
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"active_sms\": %d, \"load_width_bytes\": %d, \"num_warps\": %d, "
            "\"issue_rate_gwarp_inst_per_sm_samples\": [",
            active_sms, width, warps);
        for (size_t i = 0; i < ginst_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", ginst_s[i]); }
        std::printf("], \"delivered_bw_per_sm_gbps_samples\": [");
        for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
        std::printf("]}");
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 17. Post-hoc Group 2: Spill reload latency in dependent compute loop
// ---------------------------------------------------------------------------

template <int FRAME_U32, bool SERIAL_CHAIN>
__global__ void spill_reload_latency_kernel(uint32_t* __restrict__ sink,
                                            int reload_count,
                                            int stride_u32,
                                            int steps) {
  // Runtime-indexed local array forces real .local frame allocation (LDL/STL).
  // All 32 lanes of a warp use the SAME frame offset (warp-uniform), matching
  // ptxas register spill reloads (LDL [R1 + imm]) which coalesce to 128 B/warp.
  uint32_t loc[FRAME_U32];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t mask = static_cast<uint32_t>(FRAME_U32 - 1);

  for (int i = 0; i < FRAME_U32; ++i) {
    loc[i] = static_cast<uint32_t>((i + stride_u32) & mask);
  }

  // Warp-uniform starting offset (identical across all 32 lanes of the warp).
  uint32_t idx = (warp_id * 7u) & mask;
  float dep = 1.0f + static_cast<float>(warp_id & 7u) * 0.01f;

  for (int s = 0; s < steps; ++s) {
    // Guarantee warp-uniform index at iteration boundary.
    idx = __shfl_sync(0xffffffffu, idx, 0);
    uint32_t r_acc = 0u;
    if constexpr (SERIAL_CHAIN) {
      // Each local reload feeds the next reload's warp-uniform frame offset.
      for (int k = 0; k < reload_count; ++k) {
        uint32_t v = loc[idx];
        r_acc ^= v;
        idx = v & mask;
      }
    } else {
      // All reloads issued as a coalesced warp-uniform batch before dependent compute.
      uint32_t base_idx = idx;
      for (int k = 0; k < reload_count; ++k) {
        r_acc ^= loc[(base_idx + static_cast<uint32_t>(k * stride_u32)) & mask];
      }
      idx = (base_idx + static_cast<uint32_t>(reload_count * stride_u32) + (r_acc & 1u)) & mask;
    }

    // Small dependent FP32 compute chain consuming warp-uniform r_acc.
    dep = dep * 0.9999f + static_cast<float>(r_acc & 15u) * 0.001f;
    #pragma unroll
    for (int f = 0; f < 8; ++f) {
      dep = fmaf(dep, 1.0001f, 0.0001f);
    }
    idx = (idx + (dep < -1e9f ? 1u : 0u)) & mask;
  }

  if (dep == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = static_cast<uint32_t>(dep);
  }
}

template <bool SERIAL_CHAIN>
static void dispatch_spill_frame(int frame_bytes, int sm_count, uint32_t* d_sink,
                                 int rc, int stride_u32, int steps) {
  switch (frame_bytes) {
    case 128:  spill_reload_latency_kernel<32, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    case 256:  spill_reload_latency_kernel<64, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    case 512:  spill_reload_latency_kernel<128, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    case 1024: spill_reload_latency_kernel<256, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    case 2048: spill_reload_latency_kernel<512, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    case 4096: spill_reload_latency_kernel<1024, SERIAL_CHAIN><<<sm_count, 128>>>(d_sink, rc, stride_u32, steps); break;
    default: break;
  }
}

constexpr int spill_frame_bytes[] = {128, 256, 512, 1024, 2048, 4096};
constexpr int spill_reload_counts[] = {0, 8, 32, 68, 128};

static int run_spill_counters(int device, bool probe_only) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int ctas = prop.multiProcessorCount;
  uint32_t* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(ctas) * 128 * sizeof(uint32_t)));
  for (int frame : spill_frame_bytes) {
    for (bool serial : {true, false}) {
      for (int reloads : spill_reload_counts) {
        for (int steps : {0, 128}) {
          if (serial) dispatch_spill_frame<true>(frame, ctas, sink, reloads, 1, steps);
          else dispatch_spill_frame<false>(frame, ctas, sink, reloads, 1, steps);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          std::printf("SPILL_COUNTER_LAUNCH %d %d %d %d %d\n",
                      frame, serial, reloads, steps, ctas);
          // NCU writes to the same pipe, so send each line whole.
          std::fflush(stdout);
          if (probe_only) {
            CUDA_CHECK(cudaFree(sink));
            return 0;
          }
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(sink));
  return 0;
}

// Local memory allocations a frozen spill-latency capture reads its reload points in
// (docs/results/analysis/v0.3-spill-allocation-design-registration.md).
constexpr int kSpillAllocations = 15;

// spill-latency under a frozen schedule: every point with reloads is read in
// kSpillAllocations local memory allocations, each in a new context (cudaDeviceReset) at the
// default stack limit, at its frozen length; the points without reloads and the launch-cost
// points are read once, in the first. A point's sample j is the mean over allocations of
// their j-th smallest sample, so the median of its samples is the mean of the per-allocation
// medians; every allocation's samples are kept in `allocation_launch_ns`.
static int run_spill_allocations(int device, int samples, int warmup, LaunchSchedule* schedule) {
  struct Point {
    int frame, reloads;
    bool serial;
    const char* regime;
    int length;
    std::vector<std::vector<double>> launch_ns;  // per allocation
    std::vector<double> mhz_before, mhz_after;
  };
  const int resident_threads_per_sm = 128;
  std::vector<Point> points, costs;
  for (int fb : spill_frame_bytes) {
    const int total = fb * resident_threads_per_sm;
    const char* regime = (total <= 65536) ? "l1_hit" : ((total >= 262144) ? "l2_spill" : "l1_boundary");
    for (bool serial : {true, false})
      for (int rc : spill_reload_counts) points.push_back({fb, rc, serial, regime, 0, {}, {}, {}});
  }
  for (Point& p : points) p.length = schedule->frozen_length(1, 1 << 24);
  for (int fb : spill_frame_bytes)
    for (bool serial : {true, false}) costs.push_back({fb, 0, serial, "", schedule->frozen_length(1, 1 << 20), {}, {}, {}});

  int sm_count = 0;
  uint32_t* d_sink = nullptr;
  cudaEvent_t ev_start, ev_stop, cal_start, cal_stop;
  auto open_context = [&]() {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    sm_count = prop.multiProcessorCount;
    CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * resident_threads_per_sm * sizeof(uint32_t)));
    for (cudaEvent_t* e : {&ev_start, &ev_stop, &cal_start, &cal_stop}) CUDA_CHECK(cudaEventCreate(e));
  };
  auto close_context = [&]() {
    for (cudaEvent_t e : {ev_start, ev_stop, cal_start, cal_stop}) CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaFree(d_sink));
  };
  // Time one point: warm-ups, then `samples` launches; records the SM clock around it.
  auto read = [&](Point& p, auto launch) {
    p.mhz_before.push_back(1e3 * measure_cycles_per_ns(cal_start, cal_stop));
    for (int w = 0; w < warmup; ++w) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> ns;
    for (int i = 0; i < samples; ++i) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      launch();
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
  auto frame_launch = [&](const Point& p, int reloads, int steps) {
    return [&, reloads, steps]() {
      if (p.serial) dispatch_spill_frame<true>(p.frame, sm_count, d_sink, reloads, 1, steps);
      else dispatch_spill_frame<false>(p.frame, sm_count, d_sink, reloads, 1, steps);
    };
  };

  open_context();
  for (int a = 0; a < kSpillAllocations; ++a) {
    if (a > 0) {
      close_context();
      CUDA_CHECK(cudaDeviceReset());
      open_context();
    }
    for (Point& p : points)
      if (a == 0 || p.reloads > 0) read(p, frame_launch(p, p.reloads, p.length));
    if (a == 0)
      for (Point& c : costs) {
        auto once = frame_launch(c, 0, 0);
        read(c, [&]() { for (int k = 0; k < c.length; ++k) once(); });
      }
  }
  close_context();

  auto mean = [](const std::vector<double>& v) {
    double t = 0.0;
    for (double x : v) t += x;
    return t / static_cast<double>(v.size());
  };
  auto list = [](const char* key, const std::vector<double>& v, double scale, const char* fmt) {
    std::printf(", \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) std::printf(", ");
      std::printf(fmt, v[i] / scale);
    }
    std::printf("]");
  };
  std::printf("{\n  \"samples\": %d,\n  \"resident_threads_per_sm\": %d,\n  \"allocations\": %d,\n"
              "  \"reload_latency_by_count\": [",
              samples, resident_threads_per_sm, kSpillAllocations);
  for (size_t i = 0; i < points.size(); ++i) {
    Point& p = points[i];
    // Rank means from sorted copies; allocation_launch_ns keeps launch order.
    std::vector<double> rank(samples, 0.0);
    for (std::vector<double> v : p.launch_ns) {
      std::sort(v.begin(), v.end());
      for (int j = 0; j < samples; ++j) rank[j] += v[j] / static_cast<double>(p.launch_ns.size());
    }
    std::printf("%s\n    {\"regime\": \"%s\", \"dependency_mode\": \"%s\", \"num_warps\": 4, "
                "\"resident_threads_per_sm\": %d, \"frame_bytes_per_thread\": %d, \"reload_count\": %d, "
                "\"sm_mhz_before\": %.1f, \"sm_mhz_after\": %.1f, \"steps\": %d",
                i ? "," : "", p.regime, p.serial ? "serial_chain" : "batch_before_dep", resident_threads_per_sm,
                p.frame, p.reloads, mean(p.mhz_before), mean(p.mhz_after), p.length);
    list("launch_ns_samples", rank, 1.0, "%.1f");
    list("iter_time_ns_samples", rank, p.length, "%.3f");
    if (p.reloads > 0) {
      std::printf(", \"allocation_launch_ns\": [");
      for (size_t a = 0; a < p.launch_ns.size(); ++a) {
        std::printf("%s[", a ? ", " : "");
        for (int j = 0; j < samples; ++j) std::printf("%s%.1f", j ? ", " : "", p.launch_ns[a][j]);
        std::printf("]");
      }
      std::printf("]");
    }
    std::printf("}");
  }
  std::printf("\n  ],\n  \"launch_cost_by_frame\": [");
  for (size_t i = 0; i < costs.size(); ++i) {
    const Point& c = costs[i];
    std::printf("%s\n    {\"dependency_mode\": \"%s\", \"frame_bytes_per_thread\": %d, "
                "\"sm_mhz_before\": %.1f, \"sm_mhz_after\": %.1f, \"steps\": %d",
                i ? "," : "", c.serial ? "serial_chain" : "batch_before_dep", c.frame, c.mhz_before[0],
                c.mhz_after[0], c.length);
    list("launch_ns_samples", c.launch_ns[0], 1.0, "%.1f");
    list("launch_cost_ns_samples", c.launch_ns[0], c.length, "%.1f");
    std::printf("}");
  }
  std::printf("\n  ]\n}\n");
  return 0;
}

static int run_spill_latency(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  if (schedule && !schedule->is_pilot()) return run_spill_allocations(device, samples, warmup, schedule);
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const int resident_threads_per_sm = 128;  // 4 warps/CTA x 1 CTA/SM

  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * resident_threads_per_sm * sizeof(uint32_t)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Sweep frame_bytes_per_thread from 128 B to 4096 B (16 KB to 512 KB/SM at 128 resident threads/SM),
  // crossing the 128 KB L1 boundary cleanly. stride_u32=1 visits every 128 B warp-coalesced line.

  std::printf("{\n  \"samples\": %d,\n  \"resident_threads_per_sm\": %d,\n  \"reload_latency_by_count\": [",
              samples, resident_threads_per_sm);
  bool first = true;

  for (int fb : spill_frame_bytes) {
    const int total_sm_bytes = fb * resident_threads_per_sm;
    const char* regime = (total_sm_bytes <= 65536) ? "l1_hit" : ((total_sm_bytes >= 262144) ? "l2_spill" : "l1_boundary");
    const int stride_u32 = 1;
    for (bool serial : {true, false}) {
      const char* dep_mode = serial ? "serial_chain" : "batch_before_dep";
      for (int rc : spill_reload_counts) {
        bool likely_l2 = (total_sm_bytes >= 131072);
        int est_ns = 30 + rc * (likely_l2 ? (serial ? 120 : 20) : (serial ? 20 : 3));
        int steps = std::max(128, std::min(4096, 400000 / est_ns));

        auto launch_once = [&]() {
          if (serial) dispatch_spill_frame<true>(fb, sm_count, d_sink, rc, stride_u32, steps);
          else dispatch_spill_frame<false>(fb, sm_count, d_sink, rc, stride_u32, steps);
        };
        if (schedule) {
          std::printf("%s\n    {\"regime\": \"%s\", \"dependency_mode\": \"%s\", \"num_warps\": 4, "
                      "\"resident_threads_per_sm\": %d, \"frame_bytes_per_thread\": %d, \"reload_count\": %d",
                      first ? "" : ",", regime, dep_mode, resident_threads_per_sm, fb, rc);
          first = false;
          auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, [] {},
                                            [&](int length) { steps = length; launch_once(); });
          std::printf(", \"iter_time_ns_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i)
            std::printf("%s%.3f", i ? ", " : "", timed.launch_ns[i] / timed.length);
          std::printf("]}");
          continue;
        }

        for (int w = 0; w < warmup; ++w) launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ns_samples;
        ns_samples.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          ns_samples.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"regime\": \"%s\", \"dependency_mode\": \"%s\", \"num_warps\": 4, "
            "\"resident_threads_per_sm\": %d, \"frame_bytes_per_thread\": %d, "
            "\"reload_count\": %d, \"steps\": %d, \"iter_time_ns_samples\": [",
            regime, dep_mode, resident_threads_per_sm, fb, rc, steps);
        for (size_t i = 0; i < ns_samples.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", ns_samples[i]); }
        std::printf("]}");
      }
    }
  }
  std::printf("\n  ],\n  \"launch_cost_by_frame\": [");

  // A launch with no steps still initializes the frame. Its time is the fixed cost every
  // point above carries once per launch, so collect_profile.py can take it out per step.
  // One launch is a few us, so each sample is the mean of a batch of launches. Under a
  // schedule the batch size is the point's length, sized like any other point.
  constexpr int cost_batch = 256;
  first = true;
  for (int fb : spill_frame_bytes) {
    for (bool serial : {true, false}) {
      auto launch_once = [&]() {
        if (serial) dispatch_spill_frame<true>(fb, sm_count, d_sink, 0, 1, 0);
        else dispatch_spill_frame<false>(fb, sm_count, d_sink, 0, 1, 0);
      };
      if (schedule) {
        std::printf("%s\n    {\"dependency_mode\": \"%s\", \"frame_bytes_per_thread\": %d",
                    first ? "" : ",", serial ? "serial_chain" : "batch_before_dep", fb);
        first = false;
        auto timed = schedule->time_point(cost_batch, 1, 1 << 20, warmup, samples, ev_start, ev_stop,
                                          [] {}, [&](int length) {
                                            for (int k = 0; k < length; ++k) launch_once();
                                          });
        std::printf(", \"launch_cost_ns_samples\": [");
        for (size_t i = 0; i < timed.launch_ns.size(); ++i)
          std::printf("%s%.1f", i ? ", " : "", timed.launch_ns[i] / timed.length);
        std::printf("]}");
        continue;
      }
      for (int w = 0; w < warmup; ++w) launch_once();
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::printf("%s\n    {\"dependency_mode\": \"%s\", \"frame_bytes_per_thread\": %d, "
                  "\"launches_per_sample\": %d, \"launch_cost_ns_samples\": [",
                  first ? "" : ",", serial ? "serial_chain" : "batch_before_dep", fb, cost_batch);
      first = false;
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int k = 0; k < cost_batch; ++k) launch_once();
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        std::printf("%s%.1f", s ? ", " : "", static_cast<double>(ms) * 1e6 / cost_batch);
      }
      std::printf("]}");
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 18. Post-hoc Group 3: Segmented / strided MLP variant (decode B-matrix style)
// ---------------------------------------------------------------------------

template <int STAGES, int SUB_PER_STAGE, int SEG_U4, int STRIDE_U4>
__global__ void strided_mlp_cp_async_kernel(const uint4* __restrict__ src,
                                            uint4* __restrict__ sink,
                                            size_t n_vec4_mask,
                                            int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t n_vec4 = n_vec4_mask + 1ULL;
  const size_t total_strides = n_vec4 / static_cast<size_t>(STRIDE_U4);
  const size_t total_strides_mask = total_strides - 1ULL;
  const size_t lin_per_stage = static_cast<size_t>(SUB_PER_STAGE) * 256ULL;
  const size_t segs_per_grid_stage = (static_cast<size_t>(gridDim.x) * lin_per_stage) / static_cast<size_t>(SEG_U4);
  size_t base_seg = (static_cast<size_t>(blockIdx.x) * lin_per_stage) / static_cast<size_t>(SEG_U4);

  auto issue_group = [&](int slot, size_t cur_base_seg) {
    #pragma unroll
    for (int sub = 0; sub < SUB_PER_STAGE; ++sub) {
      size_t lin = static_cast<size_t>(sub) * 256ULL + tid;
      size_t seg_idx = cur_base_seg + (lin / static_cast<size_t>(SEG_U4));
      size_t in_seg = lin & static_cast<size_t>(SEG_U4 - 1);
      // Advance in-stride segment phase on each buffer wrap so all 256 MiB are touched uniquely (no L2 reuse).
      size_t wrap = seg_idx / total_strides;
      size_t row = seg_idx & total_strides_mask;
      size_t phase = (wrap * static_cast<size_t>(SEG_U4)) & static_cast<size_t>(STRIDE_U4 - 1);
      size_t g_idx = (row * static_cast<size_t>(STRIDE_U4) + phase + in_seg) & n_vec4_mask;
      size_t smem_off = (static_cast<size_t>(slot) * SUB_PER_STAGE + sub) * 256ULL + tid;
      uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[smem_off]));
      const void* gptr = src + g_idx;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue_group(s, base_seg);
    base_seg += segs_per_grid_stage;
  }

  for (int step = STAGES; step < steps; ++step) {
    if constexpr (STAGES == 1) asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (STAGES == 2) asm volatile("cp.async.wait_group 1;\n" ::);
    else if constexpr (STAGES == 4) asm volatile("cp.async.wait_group 3;\n" ::);
    issue_group(step % STAGES, base_seg);
    base_seg += segs_per_grid_stage;
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

template <int SEG_U4, int STRIDE_U4>
static void dispatch_strided_mlp(int mult, int ctas, size_t dyn_smem,
                                 const uint4* d_dram, uint4* d_sink, size_t dram_mask, int steps) {
  if (mult == 1) strided_mlp_cp_async_kernel<1, 1, SEG_U4, STRIDE_U4><<<ctas, 256, dyn_smem>>>(d_dram, d_sink, dram_mask, steps);
  else if (mult == 2) strided_mlp_cp_async_kernel<2, 1, SEG_U4, STRIDE_U4><<<ctas, 256, dyn_smem>>>(d_dram, d_sink, dram_mask, steps);
  else if (mult == 4) strided_mlp_cp_async_kernel<4, 1, SEG_U4, STRIDE_U4><<<ctas, 256, dyn_smem>>>(d_dram, d_sink, dram_mask, steps);
  else strided_mlp_cp_async_kernel<4, 2, SEG_U4, STRIDE_U4><<<ctas, 256, dyn_smem>>>(d_dram, d_sink, dram_mask, steps);
}

static int run_strided_mlp(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t dram_bytes = (schedule ? 2048ULL : 256ULL) * LaunchSchedule::MiB;
  const size_t flush_bytes = 256ULL * LaunchSchedule::MiB;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 31, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 32, flush_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  struct StrideCfg { int seg_bytes; int stride_bytes; };
  const std::vector<StrideCfg> cfgs = {
      {128, 4096},
      {128, 16384},
      {512, 16384},
      {4096, 16384},
  };
  struct MlpPt { int ctas; int mult; };
  const std::vector<MlpPt> pts = {
      {4, 1},            // 16 KB
      {8, 1},            // 32 KB
      {16, 1},           // 64 KB
      {32, 1},           // 128 KB
      {sm_count, 1},     // 184-192 KB
      {sm_count, 2},     // 368-384 KB
      {sm_count, 4},     // 736-768 KB
      {sm_count * 2, 4}, // ~1.5 MB
      {sm_count * 3, 4}, // ~2.2 MB
      {sm_count * 3, 8}, // ~4.4 MB
  };

  std::printf("{\n  \"samples\": %d,\n  \"dram_bw_by_bytes_in_flight_strided\": [", samples);
  bool first = true;

  for (const auto& cfg : cfgs) {
    for (const auto& pt : pts) {
      const int ctas = pt.ctas;
      const int mult = pt.mult;
      const size_t bytes_per_cta = static_cast<size_t>(mult) * 4096ULL;
      const size_t bif = static_cast<size_t>(ctas) * bytes_per_cta;
      const int cp_sub = (mult <= 4) ? 1 : (mult / 4);
      const int cp_stages = (mult <= 4) ? mult : 4;
      const size_t dyn_smem = static_cast<size_t>(cp_stages) * static_cast<size_t>(cp_sub) * 4096ULL;
      const size_t target_bytes = std::min<size_t>(dram_bytes / 2ULL, std::max<size_t>(16ULL * 1024ULL * 1024ULL, bif * 64ULL));
      int steps = std::max(cp_stages * 4, static_cast<int>(target_bytes / (static_cast<size_t>(ctas) * 4096ULL * cp_sub)));
      const double actual_bytes = static_cast<double>(ctas) * 4096.0 * cp_sub * steps;

      auto launch_once = [&]() {
        if (cfg.seg_bytes == 128 && cfg.stride_bytes == 4096)
          dispatch_strided_mlp<8, 256>(mult, ctas, dyn_smem, d_dram, d_sink, dram_mask, steps);
        else if (cfg.seg_bytes == 128 && cfg.stride_bytes == 16384)
          dispatch_strided_mlp<8, 1024>(mult, ctas, dyn_smem, d_dram, d_sink, dram_mask, steps);
        else if (cfg.seg_bytes == 512 && cfg.stride_bytes == 16384)
          dispatch_strided_mlp<32, 1024>(mult, ctas, dyn_smem, d_dram, d_sink, dram_mask, steps);
        else
          dispatch_strided_mlp<256, 1024>(mult, ctas, dyn_smem, d_dram, d_sink, dram_mask, steps);
      };

      if (schedule) {
        auto prepare = [&]() {
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
        };
        auto launch = [&](int count) {
          steps = count;
          launch_once();
        };
        const size_t bytes_per_step = static_cast<size_t>(ctas) * 4096ULL * cp_sub;
        // Each CTA's stripe divides exactly into segments. Segment ordinals
        // map injectively to (stride row, segment phase) up to the source size.
        // Keep the existing half-buffer transfer cap, including the prologue.
        const int max_steps = static_cast<int>((dram_bytes / 2ULL) / bytes_per_step);
        const int minimum = cp_stages * 4;
        const int initial = std::max(minimum, static_cast<int>(16ULL * LaunchSchedule::MiB / bytes_per_step));
        std::printf("%s\n    {\"segment_bytes\": %d, \"stride_bytes\": %d, \"bytes_in_flight\": %zu, \"active_ctas\": %d, \"bytes_per_cta\": %zu",
                    first ? "" : ",", cfg.seg_bytes, cfg.stride_bytes, bif, ctas, bytes_per_cta);
        first = false;
        schedule->measure(initial, minimum, max_steps, bytes_per_step, dram_bytes,
                          warmup, samples, ev_start, ev_stop, prepare, launch);
        std::printf("}");
        continue;
      }

      for (int w = 0; w < warmup; ++w) {
        l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
        launch_once();
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> gbps;
      gbps.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch_once();
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        gbps.push_back((actual_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
      }

      if (!first) std::printf(",");
      first = false;
      std::printf(
          "\n    {\"segment_bytes\": %d, \"stride_bytes\": %d, \"bytes_in_flight\": %zu, "
          "\"active_ctas\": %d, \"bytes_per_cta\": %zu, \"gbps_samples\": [",
          cfg.seg_bytes, cfg.stride_bytes, bif, ctas, bytes_per_cta);
      for (size_t k = 0; k < gbps.size(); ++k) { if (k) std::printf(", "); std::printf("%.3f", gbps[k]); }
      std::printf("]}");
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 19. Post-hoc / Phase 4: SFU (ex2/rcp) and SHFL (bfly/idx) issue throughput
//     per SM by warps_per_sm, plus SHFL dependency chain latency
// ---------------------------------------------------------------------------

template <bool IS_RCP>
__global__ void sfu_issue_kernel(float* __restrict__ sink, int iters) {
  const uint32_t tid = threadIdx.x;
  float x0 = IS_RCP ? (1.5f + static_cast<float>(tid & 7u) * 0.01f) : (-0.5f - static_cast<float>(tid & 7u) * 0.01f);
  float x1 = x0 + 0.02f;
  float x2 = x0 + 0.04f;
  float x3 = x0 + 0.06f;

  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      if constexpr (IS_RCP) {
        asm volatile("rcp.approx.f32 %0, %1;\n" : "=f"(x0) : "f"(x0));
        asm volatile("rcp.approx.f32 %0, %1;\n" : "=f"(x1) : "f"(x1));
        asm volatile("rcp.approx.f32 %0, %1;\n" : "=f"(x2) : "f"(x2));
        asm volatile("rcp.approx.f32 %0, %1;\n" : "=f"(x3) : "f"(x3));
      } else {
        float y0, y1, y2, y3;
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(y0) : "f"(x0));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(y1) : "f"(x1));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(y2) : "f"(x2));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(y3) : "f"(x3));
        // Negate so ex2(-y) stays bounded near fixed point ~0.641 (no Inf/NaN/subnormal).
        x0 = -y0; x1 = -y1; x2 = -y2; x3 = -y3;
      }
    }
  }
  float sum = x0 + x1 + x2 + x3;
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <bool IS_IDX>
__global__ void shfl_issue_kernel(uint32_t* __restrict__ sink, int iters) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  uint32_t r0 = tid + 1u;
  uint32_t r1 = tid + 17u;
  uint32_t r2 = tid + 33u;
  uint32_t r3 = tid + 65u;
  const uint32_t src0 = lane ^ 1u;
  const uint32_t src1 = lane ^ 2u;
  const uint32_t src2 = lane ^ 4u;
  const uint32_t src3 = lane ^ 8u;

  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      if constexpr (IS_IDX) {
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n" : "=r"(r0) : "r"(r0), "r"(src0));
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n" : "=r"(r1) : "r"(r1), "r"(src1));
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n" : "=r"(r2) : "r"(r2), "r"(src2));
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;\n" : "=r"(r3) : "r"(r3), "r"(src3));
      } else {
        asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;\n" : "=r"(r0) : "r"(r0));
        asm volatile("shfl.sync.bfly.b32 %0, %1, 2, 0x1f, 0xffffffff;\n" : "=r"(r1) : "r"(r1));
        asm volatile("shfl.sync.bfly.b32 %0, %1, 4, 0x1f, 0xffffffff;\n" : "=r"(r2) : "r"(r2));
        asm volatile("shfl.sync.bfly.b32 %0, %1, 8, 0x1f, 0xffffffff;\n" : "=r"(r3) : "r"(r3));
      }
    }
  }
  uint32_t sum = r0 ^ r1 ^ r2 ^ r3;
  if (sum == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

// Independent fma.rn.f32 and lop3.b32 chains, 1:1: the issue rate of a mix of FP32
// and integer instructions. Counted as 128 per iteration; the loop counter and branch
// are not counted, so the rate reads about 2% low.
__global__ void alu_issue_kernel(uint32_t* __restrict__ sink, int iters) {
  const uint32_t tid = threadIdx.x;
  const float fa = 0.999f - static_cast<float>(tid & 7u) * 1e-4f;
  const float fb = 1e-3f;
  const uint32_t k1 = tid * 2654435761u;
  const uint32_t k2 = tid ^ 0x5bd1e995u;
  float f0 = 1.0f, f1 = 1.1f, f2 = 1.2f, f3 = 1.3f;
  uint32_t r0 = tid + 1u, r1 = tid + 17u, r2 = tid + 33u, r3 = tid + 65u;

  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(f0) : "f"(fa), "f"(fb));
      asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;\n" : "+r"(r0) : "r"(k1), "r"(k2));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(f1) : "f"(fa), "f"(fb));
      asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;\n" : "+r"(r1) : "r"(k1), "r"(k2));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(f2) : "f"(fa), "f"(fb));
      asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;\n" : "+r"(r2) : "r"(k1), "r"(k2));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;\n" : "+f"(f3) : "f"(fa), "f"(fb));
      asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;\n" : "+r"(r3) : "r"(k1), "r"(k2));
    }
  }
  uint32_t sum = r0 ^ r1 ^ r2 ^ r3 ^ __float_as_uint(f0 + f1 + f2 + f3);
  if (sum == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

__global__ void shfl_latency_chain_kernel(uint32_t* __restrict__ sink, int iters) {
  uint32_t r = threadIdx.x + 1u;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 64; ++u) {
      asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;\n" : "=r"(r) : "r"(r));
    }
  }
  if (r == 0xdeadbeefu) {
    sink[threadIdx.x] = r;
  }
}

static int run_sfu_shfl(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  float* d_sink_f = nullptr;
  uint32_t* d_sink_u = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 1024 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sink_u, static_cast<size_t>(sm_count) * 1024 * sizeof(uint32_t)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> warp_list = {1, 2, 4, 8, 16, 32};

  auto measure_curve = [&](const char* key, int kind, bool comma) {
    std::printf("  \"%s\": [", key);
    bool first = true;
    for (int warps : warp_list) {
      int threads = warps * 32;
      int iters = (warps <= 4) ? 16384 : ((warps <= 16) ? 8192 : 4096);
      const double insts_per_iter = (kind == 4) ? 128.0 : 16.0;
      double warp_insts_per_sm = static_cast<double>(warps) * static_cast<double>(iters) * insts_per_iter;

      auto launch_length = [&](int length) {
        if (kind == 0) sfu_issue_kernel<false><<<sm_count, threads>>>(d_sink_f, length);
        else if (kind == 1) sfu_issue_kernel<true><<<sm_count, threads>>>(d_sink_f, length);
        else if (kind == 2) shfl_issue_kernel<false><<<sm_count, threads>>>(d_sink_u, length);
        else if (kind == 3) shfl_issue_kernel<true><<<sm_count, threads>>>(d_sink_u, length);
        else alu_issue_kernel<<<sm_count, threads>>>(d_sink_u, length);
      };
      if (schedule) {
        std::printf("%s\n    {\"warps_per_sm\": %d", first ? "" : ",", warps);
        first = false;
        auto timed = schedule->time_point(iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop,
                                          [] {}, launch_length);
        const double insts = static_cast<double>(warps) * timed.length * insts_per_iter;
        std::printf(", \"rate_gwarp_inst_per_sm_samples\": [");
        for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
          std::printf("%s%.4f", k ? ", " : "", insts / timed.launch_ns[k]);
        }
        std::printf("]}");
        continue;
      }
      auto launch = [&]() { launch_length(iters); };

      for (int w = 0; w < warmup; ++w) launch();
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> rate_s;
      rate_s.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch();
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        rate_s.push_back((warp_insts_per_sm / 1e9) / (static_cast<double>(ms) / 1e3));
      }

      if (!first) std::printf(",");
      first = false;
      std::printf("\n    {\"warps_per_sm\": %d, \"rate_gwarp_inst_per_sm_samples\": [", warps);
      for (size_t i = 0; i < rate_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", rate_s[i]); }
      std::printf("]}");
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n", samples, sm_count);
  measure_curve("sfu_issue_rate_per_sm_gips", 0, true);
  measure_curve("sfu_rcp_issue_rate_per_sm_gips", 1, true);
  measure_curve("shfl_issue_rate_per_sm_gips", 2, true);
  measure_curve("shfl_idx_issue_rate_per_sm_gips", 3, true);
  measure_curve("alu_issue_rate_per_sm_gips", 4, true);

  // Measure single-warp dependent shfl.sync.bfly chain latency.
  const int lat_iters = 4096;
  const double total_shfls = static_cast<double>(lat_iters) * 64.0;
  if (schedule) {
    std::printf("  \"shfl_latency\": {\"warps\": 1");
    auto timed = schedule->time_point(
        lat_iters, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {},
        [&](int length) { shfl_latency_chain_kernel<<<1, 32>>>(d_sink_u, length); });
    std::printf("},\n  \"shfl_latency_ns_samples\": [");
    for (size_t k = 0; k < timed.launch_ns.size(); ++k) {
      std::printf("%s%.3f", k ? ", " : "", timed.launch_ns[k] / (timed.length * 64.0));
    }
    std::printf("]\n}\n");
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_sink_f));
    CUDA_CHECK(cudaFree(d_sink_u));
    return 0;
  }
  for (int w = 0; w < warmup; ++w) shfl_latency_chain_kernel<<<1, 32>>>(d_sink_u, lat_iters);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("  \"shfl_latency_ns_samples\": [");
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    shfl_latency_chain_kernel<<<1, 32>>>(d_sink_u, lat_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    if (s) std::printf(", ");
    std::printf("%.3f", (static_cast<double>(ms) * 1e6) / total_shfls);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_sink_f));
  CUDA_CHECK(cudaFree(d_sink_u));
  return 0;
}

// ---------------------------------------------------------------------------
// 20. Phase 4 Attention: Dependent MMA -> Softmax (ALU/SFU/SHFL) -> MMA chain
//     measuring serial vs overlapped execution across 1/2/3 CTAs/SM
// ---------------------------------------------------------------------------

template <int M_HALF, int K_SOFTMAX>
__global__ void mma_softmax_chain_kernel(float* __restrict__ sink, int steps) {
  const uint32_t tid = threadIdx.x;
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  float o00 = 0.f, o01 = 0.f, o02 = 0.f, o03 = 0.f;
  float o10 = 0.f, o11 = 0.f, o12 = 0.f, o13 = 0.f;
  float o20 = 0.f, o21 = 0.f, o22 = 0.f, o23 = 0.f;
  float o30 = 0.f, o31 = 0.f, o32 = 0.f, o33 = 0.f;
  float sf_acc = -0.5f - static_cast<float>(tid & 7u) * 0.01f;

  auto run_mma_block = [&](float& r00, float& r01, float& r02, float& r03,
                           float& r10, float& r11, float& r12, float& r13,
                           float& r20, float& r21, float& r22, float& r23,
                           float& r30, float& r31, float& r32, float& r33) {
    constexpr int GROUPS = M_HALF / 4;
    #pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(r00), "+f"(r01), "+f"(r02), "+f"(r03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(r10), "+f"(r11), "+f"(r12), "+f"(r13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(r20), "+f"(r21), "+f"(r22), "+f"(r23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(r30), "+f"(r31), "+f"(r32), "+f"(r33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  };

  for (int step = 0; step < steps; ++step) {
    // 1. QK^T MMA block.
    if constexpr (M_HALF > 0) {
      run_mma_block(c00, c01, c02, c03, c10, c11, c12, c13,
                    c20, c21, c22, c23, c30, c31, c32, c33);
    }

    // 2. Dependent softmax block (shfl.bfly max/sum reduction + ex2.approx + fmax/fmul).
    if constexpr (K_SOFTMAX > 0) {
      float x0 = -0.5f - static_cast<float>(__float_as_uint(c00) & 7u) * 0.01f + sf_acc * 0.01f;
      float x1 = -0.5f - static_cast<float>(__float_as_uint(c10) & 7u) * 0.01f;
      float x2 = -0.5f - static_cast<float>(__float_as_uint(c20) & 7u) * 0.01f;
      float x3 = -0.5f - static_cast<float>(__float_as_uint(c30) & 7u) * 0.01f;

      // Row-max butterfly reduction (2 shfl.bfly).
      float m_val = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
      uint32_t mb = __float_as_uint(m_val);
      uint32_t sh1 = 0u, sh2 = 0u;
      asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;\n" : "=r"(sh1) : "r"(mb));
      m_val = fmaxf(m_val, __uint_as_float(sh1));
      mb = __float_as_uint(m_val);
      asm volatile("shfl.sync.bfly.b32 %0, %1, 2, 0x1f, 0xffffffff;\n" : "=r"(sh2) : "r"(mb));
      m_val = fmaxf(m_val, __uint_as_float(sh2));

      // K_SOFTMAX/2 ex2.approx + K_SOFTMAX/2 ALU (fsub/fmul/fma), 4-way unrolled.
      constexpr int SF_ROUNDS = K_SOFTMAX / 8;
      float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
      #pragma unroll 4
      for (int r = 0; r < SF_ROUNDS; ++r) {
        float e0, e1, e2, e3;
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e0) : "f"(x0));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e1) : "f"(x1));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e2) : "f"(x2));
        asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e3) : "f"(x3));
        sum0 = fmaf(e0, 0.5f, sum0);
        sum1 = fmaf(e1, 0.5f, sum1);
        sum2 = fmaf(e2, 0.5f, sum2);
        sum3 = fmaf(e3, 0.5f, sum3);
        x0 = -e0; x1 = -e1; x2 = -e2; x3 = -e3;
      }

      // Row-sum butterfly reduction (2 shfl.bfly).
      float l_val = (sum0 + sum1) + (sum2 + sum3) + m_val * 0.001f;
      uint32_t lb = __float_as_uint(l_val);
      asm volatile("shfl.sync.bfly.b32 %0, %1, 1, 0x1f, 0xffffffff;\n" : "=r"(sh1) : "r"(lb));
      l_val += __uint_as_float(sh1);
      lb = __float_as_uint(l_val);
      asm volatile("shfl.sync.bfly.b32 %0, %1, 2, 0x1f, 0xffffffff;\n" : "=r"(sh2) : "r"(lb));
      l_val += __uint_as_float(sh2);
      sf_acc = l_val;

      // Feed softmax result into PV MMA operand a0 (keeps normal f16 values near 1.0).
      a0 = 0x3c003c00u ^ (__float_as_uint(l_val) & 1u);
    }

    // 3. PV MMA block depending on a0 from softmax.
    if constexpr (M_HALF > 0) {
      run_mma_block(o00, o01, o02, o03, o10, o11, o12, o13,
                    o20, o21, o22, o23, o30, o31, o32, o33);
      a1 = 0x3c003c00u ^ (__float_as_uint(o00) & 1u);
    }

    // 4. Stage barrier.
    asm volatile("bar.sync 0;\n" ::: "memory");
  }

  float sum = c00 + c10 + c20 + c30 + o00 + o10 + o20 + o30 + sf_acc;
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int M_HALF>
static void dispatch_mma_softmax_k(int k_sf, int grid, int threads, float* d_sink, int steps) {
  switch (k_sf) {
    case 0: mma_softmax_chain_kernel<M_HALF, 0><<<grid, threads>>>(d_sink, steps); break;
    case 16: mma_softmax_chain_kernel<M_HALF, 16><<<grid, threads>>>(d_sink, steps); break;
    case 64: mma_softmax_chain_kernel<M_HALF, 64><<<grid, threads>>>(d_sink, steps); break;
    case 256: mma_softmax_chain_kernel<M_HALF, 256><<<grid, threads>>>(d_sink, steps); break;
    default: break;
  }
}

static void dispatch_mma_softmax(int m_half, int k_sf, int grid, int threads, float* d_sink, int steps) {
  switch (m_half) {
    case 0: dispatch_mma_softmax_k<0>(k_sf, grid, threads, d_sink, steps); break;
    case 4: dispatch_mma_softmax_k<4>(k_sf, grid, threads, d_sink, steps); break;
    case 8: dispatch_mma_softmax_k<8>(k_sf, grid, threads, d_sink, steps); break;
    case 16: dispatch_mma_softmax_k<16>(k_sf, grid, threads, d_sink, steps); break;
    case 32: dispatch_mma_softmax_k<32>(k_sf, grid, threads, d_sink, steps); break;
    default: break;
  }
}

static int run_mma_softmax_chain(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 3 * 256 * sizeof(float)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> cps_list = {1, 2, 3};
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> m_list = {4, 8, 16, 32};
  const std::vector<int> k_list = {16, 64, 256};

  // With a schedule every point, the solo calibrations included, is a timed point whose
  // steps the pilot sizes; it prints its design fields and returns its per-step times.
  auto timed_one = [&](int cps, int warps, int m_half, int k_sf) -> std::vector<double> {
    int threads = warps * 32;
    int grid = sm_count * cps;
    std::printf("{\"ctas_per_sm\": %d, \"num_warps\": %d, \"mma_per_half\": %d, \"softmax_ops\": %d", cps,
                warps, m_half, k_sf);
    int steps = 128;
    auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, [] {},
                                      [&](int length) {
                                        steps = length;
                                        dispatch_mma_softmax(m_half, k_sf, grid, threads, d_sink, steps);
                                      });
    std::vector<double> out;
    for (double ns : timed.launch_ns) out.push_back(ns / timed.length);
    return out;
  };

  auto measure_one = [&](int cps, int warps, int m_half, int k_sf) -> std::vector<double> {
    int threads = warps * 32;
    int grid = sm_count * cps;
    int est_ns = std::max(30, cps * (warps / 4) * (24 * m_half + 2 * k_sf + 20));
    int steps = std::max(128, std::min(8192, 350000 / est_ns));
    for (int w = 0; w < warmup; ++w) dispatch_mma_softmax(m_half, k_sf, grid, threads, d_sink, steps);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> out;
    out.reserve(samples);
    for (int s = 0; s < samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      dispatch_mma_softmax(m_half, k_sf, grid, threads, d_sink, steps);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      out.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
    }
    return out;
  };

  auto vec_median = [](std::vector<double> v) -> double {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };

  auto print_ns = [](const std::vector<double>& ns) {
    std::printf(", \"iter_time_ns_samples\": [");
    for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i]);
    std::printf("]}");
  };

  // Calibrate solo MMA (k=0) and solo softmax (m=0) at ctas_per_sm=1.
  double solo_mma[2][4] = {};
  double solo_sf[2][3] = {};
  if (schedule) std::printf("{\n  \"samples\": %d,\n  \"solo_calibration\": [", samples);
  bool first_cal = true;
  auto calibrate = [&](int warps, int m_half, int k_sf) {
    if (!schedule) return vec_median(measure_one(1, warps, m_half, k_sf));
    std::printf("%s\n    ", first_cal ? "" : ",");
    first_cal = false;
    auto ns = timed_one(1, warps, m_half, k_sf);
    print_ns(ns);
    return vec_median(ns);
  };
  for (size_t wi = 0; wi < warp_list.size(); ++wi) {
    for (size_t mi = 0; mi < m_list.size(); ++mi) {
      solo_mma[wi][mi] = calibrate(warp_list[wi], m_list[mi], 0);
    }
    for (size_t ki = 0; ki < k_list.size(); ++ki) {
      solo_sf[wi][ki] = calibrate(warp_list[wi], 0, k_list[ki]);
    }
  }

  if (schedule) std::printf("\n  ],\n  \"mma_softmax_chain\": [");
  else std::printf("{\n  \"samples\": %d,\n  \"mma_softmax_chain\": [", samples);
  bool first = true;
  for (int cps : cps_list) {
    for (size_t wi = 0; wi < warp_list.size(); ++wi) {
      int warps = warp_list[wi];
      for (size_t mi = 0; mi < m_list.size(); ++mi) {
        int m_half = m_list[mi];
        for (size_t ki = 0; ki < k_list.size(); ++ki) {
          int k_sf = k_list[ki];
          if (schedule) {
            std::printf("%s\n    ", first ? "" : ",");
            first = false;
            auto ns = timed_one(cps, warps, m_half, k_sf);
            std::printf(", \"solo_mma_ns\": %.2f, \"solo_softmax_ns\": %.2f", solo_mma[wi][mi], solo_sf[wi][ki]);
            print_ns(ns);
            continue;
          }
          auto ns_s = measure_one(cps, warps, m_half, k_sf);
          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"ctas_per_sm\": %d, \"num_warps\": %d, \"mma_per_half\": %d, "
              "\"softmax_ops\": %d, \"solo_mma_ns\": %.2f, \"solo_softmax_ns\": %.2f, "
              "\"iter_time_ns_samples\": [",
              cps, warps, m_half, k_sf, solo_mma[wi][mi], solo_sf[wi][ki]);
          for (size_t i = 0; i < ns_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", ns_s[i]); }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 21. Post-hoc ablation: S=1 sync path (ld_discard_sync & cross-stage MMA dep)
// ---------------------------------------------------------------------------

// VARIANT: 0 = cp_async_s1, 1 = ld_st_sync, 2 = ld_discard_sync, 3 = ld_st_sync_cross_dep
template <int STAGE_BYTES, int NUM_WARPS, int VARIANT>
__global__ void sync_loop_ablation_kernel(const uint4* __restrict__ src,
                                          float* __restrict__ sink,
                                          size_t buf_mask,
                                          int steps,
                                          int mma_iters) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;

  if constexpr (VARIANT == 2) {
    uint4 ones = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
    for (size_t i = tid; i < STAGE_U4; i += blockDim.x) smem[i] = ones;
    __syncthreads();
  }

  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (VARIANT == 0) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else if constexpr (VARIANT == 2) {
      uint4 vals[VECS];
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        vals[v] = ld_cg_u4(src + ((base + off) & buf_mask));
      }
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        live_acc ^= (vals[v].x ^ vals[v].y ^ vals[v].z ^ vals[v].w);
      }
    } else {
      uint4 vals[VECS];
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        vals[v] = ld_cg_u4(src + ((base + off) & buf_mask));
      }
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(saddr), "r"(vals[v].x), "r"(vals[v].y), "r"(vals[v].z), "r"(vals[v].w) : "memory");
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if (r == LD_N - 2) { b0 = r0; b1 = r1; }
      else if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; }
    }

    for (int m = 0; m < mma_iters; ++m) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
    if constexpr (VARIANT == 3) {
      float dep_f = c00 + c10 + c20 + c30;
      size_t dep = (__float_as_uint(dep_f) == 0xdeadbeefu) ? 1ULL : 0ULL;
      base = (base + grid_stage_u4 + dep) & buf_mask;
    } else {
      base = (base + grid_stage_u4) & buf_mask;
    }
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int STAGE_BYTES, int NUM_WARPS>
static void dispatch_sync_ablation_v(int variant, int grid, const uint4* d_dram,
                                     float* d_sink, size_t mask, int steps, int mma_iters) {
  int threads = NUM_WARPS * 32;
  size_t smem = static_cast<size_t>(STAGE_BYTES);
  if (variant == 0) sync_loop_ablation_kernel<STAGE_BYTES, NUM_WARPS, 0><<<grid, threads, smem>>>(d_dram, d_sink, mask, steps, mma_iters);
  else if (variant == 1) sync_loop_ablation_kernel<STAGE_BYTES, NUM_WARPS, 1><<<grid, threads, smem>>>(d_dram, d_sink, mask, steps, mma_iters);
  else if (variant == 2) sync_loop_ablation_kernel<STAGE_BYTES, NUM_WARPS, 2><<<grid, threads, smem>>>(d_dram, d_sink, mask, steps, mma_iters);
  else sync_loop_ablation_kernel<STAGE_BYTES, NUM_WARPS, 3><<<grid, threads, smem>>>(d_dram, d_sink, mask, steps, mma_iters);
}

static int run_sync_loop_ablation(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 256 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 26, dram_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const char* v_names[4] = {"cp_async_s1", "ld_st_sync", "ld_discard_sync", "ld_st_sync_cross_dep"};
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> bytes_list = {8192, 16384};
  const std::vector<int> mma_iters_list = {0, 2, 8, 16};

  std::printf("{\n  \"samples\": %d,\n  \"sync_stage_loop_ablation\": [", samples);
  bool first = true;
  for (int v = 0; v < 4; ++v) {
    for (int warps : warp_list) {
      for (int sb : bytes_list) {
        for (int mma_iters : mma_iters_list) {
          int mma_chain_len = mma_iters * 4;
          int steps = 128;
          auto launch_once = [&]() {
            if (warps == 4 && sb == 8192) dispatch_sync_ablation_v<8192, 4>(v, sm_count, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else if (warps == 4 && sb == 16384) dispatch_sync_ablation_v<16384, 4>(v, sm_count, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else if (warps == 8 && sb == 8192) dispatch_sync_ablation_v<8192, 8>(v, sm_count, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else dispatch_sync_ablation_v<16384, 8>(v, sm_count, d_dram, d_sink_f, dram_mask, steps, mma_iters);
          };

          auto prepare = [&]() { l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1); };
          if (schedule) {
            std::printf("%s\n    {\"variant\": \"%s\", \"ctas_per_sm\": 1, \"active_ctas\": %d, \"num_warps\": %d, "
                        "\"stage_bytes\": %d, \"mma_chain_len\": %d",
                        first ? "" : ",", v_names[v], sm_count, warps, sb, mma_chain_len);
            first = false;
            auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, prepare,
                                              [&](int length) { steps = length; launch_once(); });
            const double bytes = static_cast<double>(sm_count) * sb * timed.length;
            const auto& ns = timed.launch_ns;
            std::printf(", \"bw_gbps_samples\": [");
            for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", bytes / ns[i]);
            std::printf("], \"turnaround_ns_samples\": [");
            for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i] / timed.length);
            std::printf("]}");
            continue;
          }
          double total_bytes = static_cast<double>(sm_count) * static_cast<double>(sb) * steps;

          for (int w = 0; w < warmup; ++w) {
            prepare();
            launch_once();
          }
          CUDA_CHECK(cudaDeviceSynchronize());

          std::vector<double> bw_s, turn_s;
          bw_s.reserve(samples);
          turn_s.reserve(samples);
          for (int s = 0; s < samples; ++s) {
            prepare();
            CUDA_CHECK(cudaEventRecord(ev_start));
            launch_once();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            bw_s.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
            turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
          }

          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"variant\": \"%s\", \"ctas_per_sm\": 1, \"active_ctas\": %d, \"num_warps\": %d, "
              "\"stage_bytes\": %d, \"mma_chain_len\": %d, \"bw_gbps_samples\": [",
              v_names[v], sm_count, warps, sb, mma_chain_len);
          for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
          std::printf("], \"turnaround_ns_samples\": [");
          for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 22. Post-hoc: Controlled-in-flight global load issue (1 SM, hot L2, inter-batch dep)
// ---------------------------------------------------------------------------

template <int WIDTH_BYTES, int BATCH_N>
__global__ void gmem_issue_controlled_kernel(const uint8_t* __restrict__ src,
                                             uint32_t* __restrict__ sink,
                                             size_t elem_mask,
                                             int batches) {
  const uint32_t tid = threadIdx.x;
  const uint32_t threads = blockDim.x;
  size_t idx = tid & elem_mask;
  const size_t step = static_cast<size_t>(threads) * static_cast<size_t>(BATCH_N);
  uint32_t acc = 0u;

  for (int b = 0; b < batches; ++b) {
    uint32_t round_xor = 0u;
    #pragma unroll
    for (int u = 0; u < BATCH_N; ++u) {
      size_t cur = (idx + static_cast<size_t>(u) * threads) & elem_mask;
      if constexpr (WIDTH_BYTES == 2) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(src) + cur;
        uint16_t v = 0;
        asm volatile("ld.global.cg.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
        round_xor ^= static_cast<uint32_t>(v);
      } else if constexpr (WIDTH_BYTES == 4) {
        const uint32_t* p = reinterpret_cast<const uint32_t*>(src) + cur;
        uint32_t v = 0;
        asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
        round_xor ^= v;
      } else {
        const uint4* p = reinterpret_cast<const uint4*>(src) + cur;
        uint32_t x = 0, y = 0, z = 0, w = 0;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(p));
        round_xor ^= (x ^ y ^ z ^ w);
      }
    }
    acc ^= round_xor;
    size_t dep = (round_xor == 0x0000beefu) ? 1ULL : 0ULL;
    idx = (idx + step + dep) & elem_mask;
  }

  if (acc == 0x0000beefu) {
    sink[tid] = acc;
  }
}

template <int WIDTH_BYTES>
static void dispatch_gmem_ctrl_n(int batch_n, int threads, const uint8_t* d_l2,
                                 uint32_t* d_sink, size_t elem_mask, int batches) {
  if (batch_n == 8) gmem_issue_controlled_kernel<WIDTH_BYTES, 8><<<1, threads>>>(d_l2, d_sink, elem_mask, batches);
  else if (batch_n == 16) gmem_issue_controlled_kernel<WIDTH_BYTES, 16><<<1, threads>>>(d_l2, d_sink, elem_mask, batches);
  else gmem_issue_controlled_kernel<WIDTH_BYTES, 32><<<1, threads>>>(d_l2, d_sink, elem_mask, batches);
}

static int run_gmem_issue_controlled(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  uint8_t* d_l2 = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_l2, 0x5a, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> width_list = {2, 4, 16};
  const std::vector<int> batch_n_list = {8, 16, 32};
  const std::vector<int> warp_list = {4, 8, 16, 32};

  std::printf("{\n  \"samples\": %d,\n  \"gmem_load_issue_controlled\": [", samples);
  bool first = true;

  for (int width : width_list) {
    size_t elem_mask = (l2_bytes / static_cast<size_t>(width)) - 1ULL;
    for (int bn : batch_n_list) {
      for (int warps : warp_list) {
        int threads = warps * 32;
        size_t bif = static_cast<size_t>(threads) * static_cast<size_t>(bn) * static_cast<size_t>(width);
        int batches = std::max(256, 32768 / (bn * (warps / 4)));
        auto launch_once = [&]() {
          if (width == 2) dispatch_gmem_ctrl_n<2>(bn, threads, d_l2, d_sink, elem_mask, batches);
          else if (width == 4) dispatch_gmem_ctrl_n<4>(bn, threads, d_l2, d_sink, elem_mask, batches);
          else dispatch_gmem_ctrl_n<16>(bn, threads, d_l2, d_sink, elem_mask, batches);
        };
        if (schedule) {
          std::printf("%s\n    {\"load_width_bytes\": %d, \"batch_loads_per_thread\": %d, \"num_warps\": %d, "
                      "\"bytes_in_flight\": %zu",
                      first ? "" : ",", width, bn, warps, bif);
          first = false;
          auto timed = schedule->time_point(batches, 1, 1 << 28, warmup, samples, ev_start, ev_stop, [] {},
                                            [&](int length) { batches = length; launch_once(); });
          const double insts = static_cast<double>(warps) * bn * timed.length;
          std::printf(", \"issue_rate_gwarp_inst_per_sm_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", insts / timed.launch_ns[i]);
          std::printf("], \"delivered_bw_per_sm_gbps_samples\": [");
          for (size_t i = 0; i < timed.launch_ns.size(); ++i)
            std::printf("%s%.3f", i ? ", " : "", static_cast<double>(bif) * timed.length / timed.launch_ns[i]);
          std::printf("]}");
          continue;
        }
        double warp_insts = static_cast<double>(warps) * static_cast<double>(bn) * static_cast<double>(batches);
        double total_bytes = static_cast<double>(bif) * static_cast<double>(batches);

        for (int w = 0; w < warmup; ++w) launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ginst_s, bw_s;
        ginst_s.reserve(samples);
        bw_s.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          double sec = static_cast<double>(ms) / 1e3;
          ginst_s.push_back((warp_insts / 1e9) / sec);
          bw_s.push_back((total_bytes / 1e9) / sec);
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"load_width_bytes\": %d, \"batch_loads_per_thread\": %d, \"num_warps\": %d, "
            "\"bytes_in_flight\": %zu, \"issue_rate_gwarp_inst_per_sm_samples\": [",
            width, bn, warps, bif);
        for (size_t i = 0; i < ginst_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", ginst_s[i]); }
        std::printf("], \"delivered_bw_per_sm_gbps_samples\": [");
        for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
        std::printf("]}");
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 23. Post-hoc: Usable L1 capacity vs shared memory carveout (ld.global.ca pchase)
// ---------------------------------------------------------------------------

template <int Tag>
__global__ void l1_carveout_pchase_kernel(const uint32_t* __restrict__ next_idx,
                                          uint32_t* __restrict__ sink,
                                          int smem_bytes,
                                          int warmup_steps,
                                          int timed_steps,
                                          double* __restrict__ out_cycles) {
  extern __shared__ uint32_t l1c_smem[];
  if (threadIdx.x == 0) {
    uint32_t smem_touch = 0u;
    if (smem_bytes >= 4) {
      l1c_smem[0] = 1u;
      smem_touch = l1c_smem[0];
    }
    uint32_t cur = 0u;
    for (int i = 0; i < warmup_steps; ++i) {
      asm volatile("ld.global.ca.u32 %0, [%1];\n" : "=r"(cur) : "l"(next_idx + cur));
    }
    uint64_t t0 = static_cast<uint64_t>(clock64());
    for (int i = 0; i < timed_steps; ++i) {
      asm volatile("ld.global.ca.u32 %0, [%1];\n" : "=r"(cur) : "l"(next_idx + cur));
    }
    uint64_t t1 = static_cast<uint64_t>(clock64());
    *out_cycles = static_cast<double>(t1 - t0) / static_cast<double>(timed_steps);
    if ((cur ^ smem_touch) == 0xdeadbeefu) {
      sink[0] = cur;
    }
  }
}

template <int Tag>
static void run_l1_carveout_one(int smem_req,
                                int samples,
                                int warmup,
                                double cycles_per_ns,
                                const std::vector<int>& ws_kb_list,
                                std::vector<uint32_t>& h_ring,
                                uint32_t* d_ring,
                                uint32_t* d_sink,
                                double* d_out,
                                LaunchSchedule* schedule = nullptr,
                                cudaEvent_t ev_start = nullptr,
                                cudaEvent_t ev_stop = nullptr) {
  if (smem_req > 49152) {
    CUDA_CHECK(cudaFuncSetAttribute(
        l1_carveout_pchase_kernel<Tag>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_req));
  }
  const int stride_u32 = 32;  // 128 B = 1 L1 cache line
  std::printf("\n    {\"smem_carveout_bytes\": %d, \"points\": [", smem_req);
  bool first_w = true;
  for (int ws_kb : ws_kb_list) {
    int lines = (ws_kb * 1024) / 128;
    for (int i = 0; i < lines; ++i) {
      int next_line = (i + 1) % lines;
      h_ring[static_cast<size_t>(i) * stride_u32] = static_cast<uint32_t>(next_line * stride_u32);
    }
    CUDA_CHECK(cudaMemcpy(d_ring, h_ring.data(), static_cast<size_t>(ws_kb) * 1024ULL, cudaMemcpyHostToDevice));

    int w_steps = lines * 3;
    int t_steps = std::max(lines * 4, 2048);
    if (schedule) {
      // Length = timed chase steps; window = cycles per step times the steps.
      int current = 0;
      std::printf("%s\n      {\"smem_carveout_bytes\": %d, \"ws_bytes\": %d", first_w ? "" : ",", smem_req,
                  ws_kb * 1024);
      first_w = false;
      auto timed = schedule->time_point(
          t_steps, 1, 1 << 26, std::max(warmup, 1), samples, ev_start, ev_stop, [] {},
          [&](int length) {
            current = length;
            l1_carveout_pchase_kernel<Tag><<<1, 1024, static_cast<size_t>(smem_req)>>>(
                d_ring, d_sink, smem_req, w_steps, length, d_out);
          },
          [&]() {
            double cyc = 0.0;
            CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            return cyc * current / cycles_per_ns;
          });
      std::printf(", \"window_ns_samples\": [");
      for (size_t k = 0; k < timed.observed.size(); ++k) std::printf("%s%.1f", k ? ", " : "", timed.observed[k]);
      std::printf("], \"latency_ns_samples\": [");
      for (size_t k = 0; k < timed.observed.size(); ++k) {
        std::printf("%s%.2f", k ? ", " : "", timed.observed[k] / timed.length);
      }
      std::printf("]}");
      continue;
    }
    // Use 1024 threads (1 CTA/SM max) so the driver selects the SM carveout
    // matching this single CTA's smem request rather than scaling for multi-CTA occupancy.
    for (int w = 0; w < warmup; ++w) {
      l1_carveout_pchase_kernel<Tag><<<1, 1024, static_cast<size_t>(smem_req)>>>(
          d_ring, d_sink, smem_req, w_steps, t_steps, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lat_ns;
    lat_ns.reserve(samples);
    for (int s = 0; s < samples; ++s) {
      l1_carveout_pchase_kernel<Tag><<<1, 1024, static_cast<size_t>(smem_req)>>>(
          d_ring, d_sink, smem_req, w_steps, t_steps, d_out);
      CUDA_CHECK(cudaDeviceSynchronize());
      double cyc = 0.0;
      CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
      lat_ns.push_back(cyc / cycles_per_ns);
    }

    if (!first_w) std::printf(",");
    first_w = false;
    std::printf("\n      {\"ws_bytes\": %d, \"latency_ns_samples\": [", ws_kb * 1024);
    for (size_t k = 0; k < lat_ns.size(); ++k) { if (k) std::printf(", "); std::printf("%.2f", lat_ns[k]); }
    std::printf("]}");
  }
  std::printf("\n    ]}");
}

static int run_l1_carveout(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int max_optin = static_cast<int>(prop.sharedMemPerBlockOptin);

  const size_t max_ws_bytes = 176ULL * 1024ULL;
  const size_t max_u32 = max_ws_bytes / sizeof(uint32_t);
  uint32_t* d_ring = nullptr;
  uint32_t* d_sink = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ring, max_ws_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  std::vector<int> ws_kb_list;
  for (int kb = 8; kb <= 160; kb += 4) ws_kb_list.push_back(kb);
  std::vector<uint32_t> h_ring(max_u32, 0u);

  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"l1_carveout_sweep\": [",
              samples, cycles_per_ns);
  run_l1_carveout_one<0>(0, samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf(",");
  run_l1_carveout_one<1>(16384, samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf(",");
  run_l1_carveout_one<2>(32768, samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf(",");
  run_l1_carveout_one<3>(49152, samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf(",");
  run_l1_carveout_one<4>(65536, samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf(",");
  run_l1_carveout_one<5>(std::min(102400, max_optin), samples, warmup, cycles_per_ns, ws_kb_list, h_ring, d_ring, d_sink, d_out,
                         schedule, ev_start, ev_stop);
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_ring));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

// ---------------------------------------------------------------------------
// 24. Post-hoc: Multi-CTA sync_stage_loop (ctas_per_sm = 1, 2, 4, 6, 8)
// ---------------------------------------------------------------------------

template <int STAGE_BYTES, bool USE_CP_ASYNC>
__global__ __launch_bounds__(128, 8)
void sync_stage_loop_multicta_kernel(const uint4* __restrict__ src,
                                     float* __restrict__ sink,
                                     size_t buf_mask,
                                     int steps,
                                     int mma_iters) {
  extern __shared__ uint4 smem[];
  constexpr int NUM_WARPS = 4;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;

  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (USE_CP_ASYNC) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      uint4 vals[VECS];
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        vals[v] = ld_cg_u4(src + ((base + off) & buf_mask));
      }
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                     :: "r"(saddr), "r"(vals[v].x), "r"(vals[v].y), "r"(vals[v].z), "r"(vals[v].w) : "memory");
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if constexpr (LD_N == 1) {
        a0 = r0; a1 = r1; a2 = r2; a3 = r3;
        b0 = r0; b1 = r1;
      } else {
        if (r == LD_N - 2) {
          b0 = r0; b1 = r1;
        } else if (r == LD_N - 1) {
          a0 = r0; a1 = r1; a2 = r2; a3 = r3;
        }
      }
    }

    for (int m = 0; m < mma_iters; ++m) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + grid_stage_u4) & buf_mask;
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <bool USE_CP_ASYNC>
static void dispatch_sync_loop_multicta(int stage_bytes, int grid, const uint4* d_buf,
                                        float* d_sink, size_t buf_mask, int steps, int mma_iters) {
  const size_t dyn_smem = static_cast<size_t>(stage_bytes);
  switch (stage_bytes) {
    case 2048:
      sync_stage_loop_multicta_kernel<2048, USE_CP_ASYNC>
          <<<grid, 128, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    case 4096:
      sync_stage_loop_multicta_kernel<4096, USE_CP_ASYNC>
          <<<grid, 128, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    case 8192:
      sync_stage_loop_multicta_kernel<8192, USE_CP_ASYNC>
          <<<grid, 128, dyn_smem>>>(d_buf, d_sink, buf_mask, steps, mma_iters);
      break;
    default:
      break;
  }
}

template <int STAGE_BYTES, bool USE_CP_ASYNC>
static void query_multicta_occ(int& max_blocks, int& num_regs, size_t& local_bytes) {
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, sync_stage_loop_multicta_kernel<STAGE_BYTES, USE_CP_ASYNC>));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, sync_stage_loop_multicta_kernel<STAGE_BYTES, USE_CP_ASYNC>, 128, STAGE_BYTES));
  num_regs = attr.numRegs;
  local_bytes = attr.localSizeBytes;
}

static int run_sync_loop_multicta(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 128 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 25, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  auto cal_mma_ns = [&](int warps, int cal_iters) -> double {
    for (int w = 0; w < 2; ++w) mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 1.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return (static_cast<double>(ms) * 1e6) / static_cast<double>(cal_iters);
  };
  const double ns_per_iter_4w = cal_mma_ns(4, 20000);
  const double flops_per_cta_iter = 4.0 * 4.0 * 4096.0;

  std::printf("{\n  \"samples\": %d,\n  \"occupancy_check\": [", samples);
  bool first_occ = true;
  auto emit_occ = [&](const char* mode, int sb, int mb, int regs, size_t lbytes) {
    if (!first_occ) std::printf(",");
    first_occ = false;
    std::printf("\n    {\"transfer_mode\": \"%s\", \"stage_bytes\": %d, \"max_active_ctas_per_sm\": %d, \"num_regs\": %d, \"local_bytes\": %zu}",
                mode, sb, mb, regs, lbytes);
  };
  {
    int mb, regs; size_t lb;
    query_multicta_occ<2048, false>(mb, regs, lb); emit_occ("ld_st_sync", 2048, mb, regs, lb);
    query_multicta_occ<4096, false>(mb, regs, lb); emit_occ("ld_st_sync", 4096, mb, regs, lb);
    query_multicta_occ<8192, false>(mb, regs, lb); emit_occ("ld_st_sync", 8192, mb, regs, lb);
    query_multicta_occ<2048, true>(mb, regs, lb);  emit_occ("cp_async_s1", 2048, mb, regs, lb);
    query_multicta_occ<4096, true>(mb, regs, lb);  emit_occ("cp_async_s1", 4096, mb, regs, lb);
    query_multicta_occ<8192, true>(mb, regs, lb);  emit_occ("cp_async_s1", 8192, mb, regs, lb);
  }
  std::printf("\n  ],\n  \"sync_stage_loop\": [");

  const std::vector<int> bytes_list = {2048, 4096, 8192};
  const std::vector<int> cps_list = {1, 2, 4, 6, 8};
  const std::vector<int> mma_iters_list = {0, 2, 8};  // 0, 8, 32 MMAs per warp per stage

  bool first = true;
  for (bool is_dram : {true, false}) {
    const char* level_name = is_dram ? "dram" : "l2";
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    const int legacy_steps = is_dram ? 128 : 512;
    for (bool use_cp : {false, true}) {
      const char* t_mode = use_cp ? "cp_async_s1" : "ld_st_sync";
      for (int sb : bytes_list) {
        for (int cps : cps_list) {
          const int active_ctas = sm_count * cps;
          for (int mma_iters : mma_iters_list) {
            const int mma_chain_len = mma_iters * 4;
            const double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_iter_4w;
            int steps = legacy_steps;

            auto launch_once = [&]() {
              if (use_cp) dispatch_sync_loop_multicta<true>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
              else dispatch_sync_loop_multicta<false>(sb, active_ctas, buf, d_sink_f, mask, steps, mma_iters);
            };
            auto prepare = [&]() {
              if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
            };
            if (schedule) {
              std::printf("%s\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
                          "\"active_ctas\": %d, \"num_warps\": 4, \"stage_bytes\": %d, "
                          "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f",
                          first ? "" : ",", level_name, t_mode, cps, active_ctas, sb, mma_chain_len, solo_tc_ns);
              first = false;
              auto timed = schedule->time_point(steps, 1, 1 << 24, warmup, samples, ev_start, ev_stop, prepare,
                                                [&](int length) { steps = length; launch_once(); });
              const double bytes = static_cast<double>(active_ctas) * sb * timed.length;
              const double flops = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * timed.length;
              const auto& ns = timed.launch_ns;
              std::printf(", \"bw_gbps_samples\": [");
              for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", bytes / ns[i]);
              std::printf("], \"turnaround_ns_samples\": [");
              for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i] / timed.length);
              std::printf("], \"mma_tflops_per_sm_samples\": [");
              for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", flops / 1e3 / ns[i]);
              std::printf("]}");
              continue;
            }
            const double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(sb) * steps;
            const double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * steps;

            for (int w = 0; w < warmup; ++w) {
              prepare();
              launch_once();
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> bw_s, turn_s, tf_s;
            bw_s.reserve(samples);
            turn_s.reserve(samples);
            tf_s.reserve(samples);
            for (int s = 0; s < samples; ++s) {
              prepare();
              CUDA_CHECK(cudaEventRecord(ev_start));
              launch_once();
              CUDA_CHECK(cudaEventRecord(ev_stop));
              CUDA_CHECK(cudaEventSynchronize(ev_stop));
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
              double sec = static_cast<double>(ms) / 1e3;
              bw_s.push_back((total_bytes / 1e9) / sec);
              turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
              tf_s.push_back((total_flops_per_sm / 1e12) / sec);
            }

            if (!first) std::printf(",");
            first = false;
            std::printf(
                "\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
                "\"active_ctas\": %d, \"num_warps\": 4, \"stage_bytes\": %d, "
                "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
                level_name, t_mode, cps, active_ctas, sb, mma_chain_len, solo_tc_ns);
            for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
            std::printf("], \"turnaround_ns_samples\": [");
            for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
            std::printf("], \"mma_tflops_per_sm_samples\": [");
            for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
            std::printf("]}");
          }
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 25. Post-hoc: fill-vs-loop ablation & compile-time-constant ~1 ms sync_stage_loop
// ---------------------------------------------------------------------------

__global__ void fvl_warmup_kernel(float* __restrict__ sink, int iters) {
  float x = static_cast<float>(threadIdx.x) * 0.001f;
  for (int i = 0; i < iters; ++i) x = x * 1.00001f + 0.00001f;
  if (x == -1.0f) sink[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

template <int STAGE_BYTES>
__global__ void fvl_serial_fill_kernel(const uint4* __restrict__ src, uint32_t* __restrict__ sink,
                                       size_t l2_mask, int steps, double* __restrict__ out_cyc) {
  extern __shared__ uint4 smem[];
  constexpr int NUM_WARPS = 4;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & l2_mask;
  uint32_t acc = 0u;

  uint64_t t0 = static_cast<uint64_t>(clock64());
  for (int step = 0; step < steps; ++step) {
    #pragma unroll
    for (int v = 0; v < VECS; ++v) {
      size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
      const void* gptr = src + ((base + off) & l2_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      acc ^= (r0 ^ r1 ^ r2 ^ r3);
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + grid_stage_u4) & l2_mask;
  }
  uint64_t t1 = static_cast<uint64_t>(clock64());
  if (blockIdx.x == 0 && tid == 0) *out_cyc = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  uint4 tail = smem[tid & (STAGE_U4 - 1ULL)];
  acc ^= (tail.x ^ tail.y ^ tail.z ^ tail.w);
  if (acc == 0xdeadbeefu) sink[blockIdx.x * blockDim.x + tid] = acc;
}

template <int STAGE_BYTES, bool HAS_MMA_LOOP, bool HAS_2ND_BAR, bool ADVANCE_BASE>
__global__ void fvl_sync_loop_kernel(const uint4* __restrict__ src, float* __restrict__ sink,
                                     size_t buf_mask, int steps, int mma_iters, double* __restrict__ out_cyc) {
  extern __shared__ uint4 smem[];
  constexpr int NUM_WARPS = 4;
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;

  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  uint64_t t0 = static_cast<uint64_t>(clock64());
  for (int step = 0; step < steps; ++step) {
    #pragma unroll
    for (int v = 0; v < VECS; ++v) {
      size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
      const void* gptr = src + ((base + off) & buf_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if constexpr (HAS_MMA_LOOP) {
        if (r == LD_N - 2) { b0 = r0; b1 = r1; }
        else if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; }
      }
    }

    if constexpr (HAS_MMA_LOOP) {
      for (int m = 0; m < mma_iters; ++m) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }

    if constexpr (HAS_2ND_BAR) asm volatile("bar.sync 0;\n" ::: "memory");
    if constexpr (ADVANCE_BASE) base = (base + grid_stage_u4) & buf_mask;
  }
  uint64_t t1 = static_cast<uint64_t>(clock64());
  if (blockIdx.x == 0 && tid == 0) *out_cyc = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

static int run_fill_vs_loop_ablation(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm = prop.multiProcessorCount;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  uint4* d_l2 = nullptr; uint32_t* d_su = nullptr; float* d_sf = nullptr; double* d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_su, static_cast<size_t>(sm) * 256 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sf, static_cast<size_t>(sm) * 1024 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(double)));
  cudaEvent_t ev_s, ev_e;
  CUDA_CHECK(cudaEventCreate(&ev_s));
  CUDA_CHECK(cudaEventCreate(&ev_e));
  fvl_warmup_kernel<<<sm * 4, 256>>>(d_sf, 200000);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("{\n  \"device\": \"%s\",\n  \"sm_count\": %d,\n  \"l2_buffer_bytes\": %zu,\n  \"samples\": %d,\n  \"ablation\": [",
              prop.name, sm, l2_bytes, samples);
  bool first = true;

  auto run_one = [&](int sb, const char* variant, int steps, auto fn) {
    fvl_warmup_kernel<<<sm * 4, 256>>>(d_sf, 20000);
    for (int w = 0; w < warmup; ++w) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> ns_v, cyc_v;
    ns_v.reserve(samples); cyc_v.reserve(samples);
    for (int i = 0; i < samples; ++i) {
      CUDA_CHECK(cudaEventRecord(ev_s));
      fn();
      CUDA_CHECK(cudaEventRecord(ev_e));
      CUDA_CHECK(cudaEventSynchronize(ev_e));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
      double cyc = 0.0;
      CUDA_CHECK(cudaMemcpy(&cyc, d_cyc, sizeof(double), cudaMemcpyDeviceToHost));
      ns_v.push_back(static_cast<double>(ms) * 1e6 / static_cast<double>(steps));
      cyc_v.push_back(cyc);
    }
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"stage_bytes\": %d, \"variant\": \"%s\", \"steps\": %d, \"event_ns_samples\": [",
                sb, variant, steps);
    for (size_t i = 0; i < ns_v.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", ns_v[i]); }
    std::printf("], \"in_kernel_cycles_samples\": [");
    for (size_t i = 0; i < cyc_v.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", cyc_v[i]); }
    std::printf("]}");
  };

  auto sweep_sb = [&](auto sb_tag) {
    constexpr int SB = decltype(sb_tag)::value;
    run_one(SB, "serial_fill_4096", 4096, [&]{ fvl_serial_fill_kernel<SB><<<sm, 128, SB>>>(d_l2, d_su, mask, 4096, d_cyc); });
    run_one(SB, "serial_fill_128", 128, [&]{ fvl_serial_fill_kernel<SB><<<sm, 128, SB>>>(d_l2, d_su, mask, 128, d_cyc); });
    run_one(SB, "sync_loop_128_runtime_mma0", 128, [&]{ fvl_sync_loop_kernel<SB, true, true, true><<<sm, 128, SB>>>(d_l2, d_sf, mask, 128, 0, d_cyc); });
    run_one(SB, "sync_loop_512_runtime_mma0", 512, [&]{ fvl_sync_loop_kernel<SB, true, true, true><<<sm, 128, SB>>>(d_l2, d_sf, mask, 512, 0, d_cyc); });
    run_one(SB, "sync_loop_4096_runtime_mma0", 4096, [&]{ fvl_sync_loop_kernel<SB, true, true, true><<<sm, 128, SB>>>(d_l2, d_sf, mask, 4096, 0, d_cyc); });
    run_one(SB, "sync_loop_4096_const_mma0", 4096, [&]{ fvl_sync_loop_kernel<SB, false, true, true><<<sm, 128, SB>>>(d_l2, d_sf, mask, 4096, 0, d_cyc); });
    run_one(SB, "sync_loop_4096_no_2nd_bar", 4096, [&]{ fvl_sync_loop_kernel<SB, true, false, true><<<sm, 128, SB>>>(d_l2, d_sf, mask, 4096, 0, d_cyc); });
    run_one(SB, "sync_loop_4096_fixed_addr", 4096, [&]{ fvl_sync_loop_kernel<SB, true, true, false><<<sm, 128, SB>>>(d_l2, d_sf, mask, 4096, 0, d_cyc); });
  };
  sweep_sb(std::integral_constant<int, 4096>{});
  sweep_sb(std::integral_constant<int, 8192>{});

  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaEventDestroy(ev_s));
  CUDA_CHECK(cudaEventDestroy(ev_e));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_su));
  CUDA_CHECK(cudaFree(d_sf));
  CUDA_CHECK(cudaFree(d_cyc));
  return 0;
}

// Compile-time-constant MMA_ITERS sync_stage_loop kernel supporting 2 KB..32 KB stages.
template <int STAGE_BYTES, int NUM_WARPS, bool USE_CP_ASYNC, int MMA_ITERS>
__global__ __launch_bounds__(NUM_WARPS * 32, (NUM_WARPS == 4 && STAGE_BYTES <= 8192) ? 8 : 3)
void sync_stage_loop_const_kernel(const uint4* __restrict__ src,
                                  float* __restrict__ sink,
                                  size_t buf_mask,
                                  int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;

  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (USE_CP_ASYNC) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      // Batch up to 4 uint4 loads per thread at a time to avoid artificial register spilling on 12-32 KB stages.
      constexpr int CHUNK = (VECS <= 4) ? VECS : 4;
      #pragma unroll
      for (int c = 0; c < VECS; c += CHUNK) {
        uint4 vals[CHUNK];
        #pragma unroll
        for (int j = 0; j < CHUNK; ++j) {
          if (c + j < VECS) {
            size_t off = static_cast<size_t>(c + j) * (NUM_WARPS * 32) + tid;
            vals[j] = ld_cg_u4(src + ((base + off) & buf_mask));
          }
        }
        #pragma unroll
        for (int j = 0; j < CHUNK; ++j) {
          if (c + j < VECS) {
            size_t off = static_cast<size_t>(c + j) * (NUM_WARPS * 32) + tid;
            uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                         :: "r"(saddr), "r"(vals[j].x), "r"(vals[j].y), "r"(vals[j].z), "r"(vals[j].w) : "memory");
          }
        }
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if constexpr (MMA_ITERS > 0) {
        if constexpr (LD_N == 1) {
          a0 = r0; a1 = r1; a2 = r2; a3 = r3;
          b0 = r0; b1 = r1;
        } else {
          if (r == LD_N - 2) { b0 = r0; b1 = r1; }
          else if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; }
        }
      }
    }

    if constexpr (MMA_ITERS > 0) {
      #pragma unroll 1
      for (int m = 0; m < MMA_ITERS; ++m) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + grid_stage_u4) & buf_mask;
  }

  float sum = static_cast<float>(live_acc & 1u);
  if constexpr (MMA_ITERS > 0) {
    sum += c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
           c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33;
  }
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

static int run_sync_loop_reconcile(int device, int samples, int warmup, LaunchSchedule* schedule = nullptr) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Scheduled launches run longer, so the DRAM buffer is a power of two of at least
  // max(256 MiB, 8 x L2); the legacy path keeps its v0.2 size.
  size_t dram_bytes = 256ULL * LaunchSchedule::MiB;
  while (schedule && dram_bytes < 8ULL * static_cast<size_t>(prop.l2CacheSize)) dram_bytes *= 2;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr; uint4* d_flush = nullptr; uint4* d_l2 = nullptr;
  uint4* d_sink_u4 = nullptr; float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 25, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  auto cal_mma_ns = [&](int warps, int cal_iters) -> double {
    for (int w = 0; w < 2; ++w) mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));
    mma_m16n8k16_kernel<false><<<sm_count, warps * 32>>>(d_sink_f, cal_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 1.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return (static_cast<double>(ms) * 1e6) / static_cast<double>(cal_iters);
  };
  const double ns_per_iter_4w = cal_mma_ns(4, 20000);
  const double ns_per_iter_8w = cal_mma_ns(8, 10000);

  std::printf("{\n  \"samples\": %d,\n  \"sync_stage_loop\": [", samples);
  bool first = true;

  auto measure_cfg = [&](bool is_dram, int sb, int warps, bool use_cp, int mma_iters,
                         const std::vector<int>& cps_list, auto launch_fn, int max_blocks) {
    const char* level_name = is_dram ? "dram" : "l2";
    const char* t_mode = use_cp ? "cp_async_s1" : "ld_st_sync";
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    const double ns_per_iter = (warps == 4) ? ns_per_iter_4w : ns_per_iter_8w;
    const double flops_per_cta_iter = static_cast<double>(warps) * 4.0 * 4096.0;
    const int mma_chain_len = mma_iters * 4;
    const double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_iter;

    for (int cps : cps_list) {
      if (cps > max_blocks) continue;
      const int active_ctas = sm_count * cps;
      auto prepare = [&]() {
        if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
      };

      // With a schedule the pilot sizes the steps instead of the 64-step probe below.
      if (schedule) {
        std::printf("%s\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
                    "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, "
                    "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f",
                    first ? "" : ",", level_name, t_mode, cps, active_ctas, warps, sb, mma_chain_len, solo_tc_ns);
        first = false;
        auto timed = schedule->time_point(128, 1, 1 << 24, warmup, samples, ev_start, ev_stop, prepare,
                                          [&](int length) { launch_fn(active_ctas, buf, mask, length); });
        const double bytes = static_cast<double>(active_ctas) * sb * timed.length;
        const double flops = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * timed.length;
        const auto& ns = timed.launch_ns;
        std::printf(", \"bw_gbps_samples\": [");
        for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.3f", i ? ", " : "", bytes / ns[i]);
        std::printf("], \"turnaround_ns_samples\": [");
        for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.2f", i ? ", " : "", ns[i] / timed.length);
        std::printf("], \"mma_tflops_per_sm_samples\": [");
        for (size_t i = 0; i < ns.size(); ++i) std::printf("%s%.4f", i ? ", " : "", flops / 1e3 / ns[i]);
        std::printf("]}");
        continue;
      }

      // Probe 64 steps to adapt steps so each timed kernel runs ~1.0 ms (clamped to [128, 8192]).
      prepare();
      CUDA_CHECK(cudaEventRecord(ev_start));
      launch_fn(active_ctas, buf, mask, 64);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float probe_ms = 0.05f;
      CUDA_CHECK(cudaEventElapsedTime(&probe_ms, ev_start, ev_stop));
      double est_ns_per_step = std::max(50.0, (static_cast<double>(probe_ms) * 1e6) / 64.0);
      int steps = static_cast<int>(1000000.0 / est_ns_per_step);
      steps = ((steps + 63) / 64) * 64;
      if (steps < 128) steps = 128;
      if (steps > 8192) steps = 8192;

      const double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(sb) * steps;
      const double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * steps;

      for (int w = 0; w < warmup; ++w) {
        prepare();
        launch_fn(active_ctas, buf, mask, steps);
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> bw_s, turn_s, tf_s;
      bw_s.reserve(samples); turn_s.reserve(samples); tf_s.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        prepare();
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch_fn(active_ctas, buf, mask, steps);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double sec = static_cast<double>(ms) / 1e3;
        bw_s.push_back((total_bytes / 1e9) / sec);
        turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
        tf_s.push_back((total_flops_per_sm / 1e12) / sec);
      }

      if (!first) std::printf(",");
      first = false;
      std::printf(
          "\n    {\"mem_level\": \"%s\", \"transfer_mode\": \"%s\", \"ctas_per_sm\": %d, "
          "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, "
          "\"mma_chain_len\": %d, \"solo_compute_ns\": %.2f, \"steps\": %d, \"bw_gbps_samples\": [",
          level_name, t_mode, cps, active_ctas, warps, sb, mma_chain_len, solo_tc_ns, steps);
      for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
      std::printf("], \"turnaround_ns_samples\": [");
      for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
      std::printf("], \"mma_tflops_per_sm_samples\": [");
      for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
      std::printf("]}");
    }
  };

  auto run_inst = [&](bool is_dram, auto sb_c, auto w_c, auto cp_c, auto mma_c, const std::vector<int>& cps_list) {
    constexpr int SB = decltype(sb_c)::value;
    constexpr int W = decltype(w_c)::value;
    constexpr bool CP = decltype(cp_c)::value;
    constexpr int MI = decltype(mma_c)::value;
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, sync_stage_loop_const_kernel<SB, W, CP, MI>, W * 32, SB));
    auto fn = [&](int grid, const uint4* b, size_t m, int st) {
      sync_stage_loop_const_kernel<SB, W, CP, MI><<<grid, W * 32, SB>>>(b, d_sink_f, m, st);
    };
    measure_cfg(is_dram, SB, W, CP, MI, cps_list, fn, max_blocks);
  };

  const std::vector<int> cps_123 = {1, 2, 3};
  const std::vector<int> cps_123468 = {1, 2, 3, 4, 6, 8};
  const std::vector<int> cps_12468 = {1, 2, 4, 6, 8};

  // Part A: Re-measure all L2 points (original 0.12.0 + multicta) with compile-time MMA_ITERS and ~1 ms duration.
  auto run_l2_all = [&](auto cp_c) {
    // 4-warp, 2048 B: cps 1,2,4,6,8; mma 0,8,32
    run_inst(false, std::integral_constant<int, 2048>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 0>{}, cps_12468);
    run_inst(false, std::integral_constant<int, 2048>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 2>{}, cps_12468);
    run_inst(false, std::integral_constant<int, 2048>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 8>{}, cps_12468);

    // 4-warp, 4096 & 8192 B: cps 1,2,3,4,6,8 for mma 0,8,32; cps 1,2,3 for mma 64
    run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 0>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 2>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 8>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);

    run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 0>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 2>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 8>{}, cps_123468);
    run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);

    // 4-warp, 16384 B: cps 1,2,3; mma 0,8,32,64
    run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 0>{}, cps_123);
    run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 2>{}, cps_123);
    run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 8>{}, cps_123);
    run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 4>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);

    // 8-warp, 4096, 8192, 16384 B: cps 1,2,3; mma 0,8,32,64
    for (int dummy = 0; dummy < 1; ++dummy) {
      run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 0>{}, cps_123);
      run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 2>{}, cps_123);
      run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 8>{}, cps_123);
      run_inst(false, std::integral_constant<int, 4096>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);

      run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 0>{}, cps_123);
      run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 2>{}, cps_123);
      run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 8>{}, cps_123);
      run_inst(false, std::integral_constant<int, 8192>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);

      run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 0>{}, cps_123);
      run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 2>{}, cps_123);
      run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 8>{}, cps_123);
      run_inst(false, std::integral_constant<int, 16384>{}, std::integral_constant<int, 8>{}, cp_c, std::integral_constant<int, 16>{}, cps_123);
    }
  };
  run_l2_all(std::false_type{});
  run_l2_all(std::true_type{});

  // Part B: Large stages (12, 24, 32 KB), 4 & 8 warps, cps 1/2/3 (subject to smem occupancy), L2 & DRAM, mma 0 & 32.
  auto run_large = [&](bool is_dram, auto cp_c, auto w_c) {
    run_inst(is_dram, std::integral_constant<int, 12288>{}, w_c, cp_c, std::integral_constant<int, 0>{}, cps_123);
    run_inst(is_dram, std::integral_constant<int, 12288>{}, w_c, cp_c, std::integral_constant<int, 8>{}, cps_123);
    run_inst(is_dram, std::integral_constant<int, 24576>{}, w_c, cp_c, std::integral_constant<int, 0>{}, cps_123);
    run_inst(is_dram, std::integral_constant<int, 24576>{}, w_c, cp_c, std::integral_constant<int, 8>{}, cps_123);
    run_inst(is_dram, std::integral_constant<int, 32768>{}, w_c, cp_c, std::integral_constant<int, 0>{}, cps_123);
    run_inst(is_dram, std::integral_constant<int, 32768>{}, w_c, cp_c, std::integral_constant<int, 8>{}, cps_123);
  };
  for (bool is_dram : {false, true}) {
    run_large(is_dram, std::false_type{}, std::integral_constant<int, 4>{});
    run_large(is_dram, std::false_type{}, std::integral_constant<int, 8>{});
    run_large(is_dram, std::true_type{}, std::integral_constant<int, 4>{});
    run_large(is_dram, std::true_type{}, std::integral_constant<int, 8>{});
  }

  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 26. Post-hoc: single-buffer loop overlap of the next loads with in-flight MMAs
// ---------------------------------------------------------------------------
// PATTERN: 0 = gemm_s2 (cp.async, wait 0, ldmatrix, MMA), 1 = attention (cp.async, wait 0,
// ldmatrix, QK MMA, dependent softmax, PV MMA that depends on it).
// DEP: 0 = free (next loads issued right after the last barrier, as compiled),
// 1 = dep (next load address depends on the last MMA results), 2 = noload, 3 = nocompute.
// MI: MMA groups of 4 independent m16n8k16 per warp, per MMA block.
#define OVL_MMA(acc0, acc1, acc2, acc3)                                                           \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
               "{%8,%9}, {%0,%1,%2,%3};\n"                                                         \
               : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                    \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <int PATTERN, int DEP, int STAGE_BYTES, int NUM_WARPS, int MI>
__global__ void ovl_kernel(const uint4* __restrict__ src, float* __restrict__ sink, size_t buf_mask,
                           int steps, double* __restrict__ out_cyc) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & buf_mask;
  if constexpr (DEP == 2) {
    for (size_t i = tid; i < STAGE_U4; i += blockDim.x) smem[i] = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
    __syncthreads();
  }
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  float o00 = 0.f, o01 = 0.f, o02 = 0.f, o03 = 0.f, o10 = 0.f, o11 = 0.f, o12 = 0.f, o13 = 0.f;
  float o20 = 0.f, o21 = 0.f, o22 = 0.f, o23 = 0.f, o30 = 0.f, o31 = 0.f, o32 = 0.f, o33 = 0.f;
  float sf_acc = 0.5f;
  uint64_t t0 = static_cast<uint64_t>(clock64());
  for (int step = 0; step < steps; ++step) {
    if constexpr (DEP == 1) {
      float d = c00 + c10 + c20 + c30 + o00 + o10 + o20 + o30;
      base = (base + ((__float_as_uint(d) == 0xdeadbeefu) ? 1ULL : 0ULL)) & buf_mask;
    }
    if constexpr (DEP != 2) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if constexpr (DEP != 3) {
        if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
      }
    }
    if constexpr (DEP != 3) {
      #pragma unroll
      for (int m = 0; m < MI; ++m) { OVL_MMA(c00, c01, c02, c03); OVL_MMA(c10, c11, c12, c13); OVL_MMA(c20, c21, c22, c23); OVL_MMA(c30, c31, c32, c33); }
      if constexpr (PATTERN == 1) {
        // Softmax on the QK results: 32 dependent ex2 and a 4-step shuffle reduction, then PV.
        float x = -0.5f - static_cast<float>(__float_as_uint(c00 + c10 + c20 + c30) & 7u) * 0.01f + sf_acc * 0.001f;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) { float e; asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e) : "f"(x)); s = fmaf(e, 0.5f, s); x = -e; }
        #pragma unroll
        for (int sh = 1; sh <= 8; sh <<= 1) s += __shfl_xor_sync(0xffffffffu, s, sh);
        sf_acc = s;
        a0 = 0x3c003c00u ^ (__float_as_uint(s) & 1u);
        #pragma unroll
        for (int m = 0; m < MI; ++m) { OVL_MMA(o00, o01, o02, o03); OVL_MMA(o10, o11, o12, o13); OVL_MMA(o20, o21, o22, o23); OVL_MMA(o30, o31, o32, o33); }
      }
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + grid_stage_u4) & buf_mask;
  }
  uint64_t t1 = static_cast<uint64_t>(clock64());
  if (blockIdx.x == 0 && tid == 0) *out_cyc = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              o00 + o01 + o02 + o03 + o10 + o11 + o12 + o13 + o20 + o21 + o22 + o23 + o30 + o31 + o32 + o33 +
              sf_acc + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

// Attention-shaped single-buffer loop with G CTAs reading the same stage (the query-block CTAs of one
// head share K and V) and, with SPLIT, K and V as two commit groups waited together at the loop top.
// DEP: 0 = free, 2 = noload, 4 = empty (barriers and loop only). MI groups of 4 MMAs per warp in QK and in PV.
template <int DEP, int STAGE_BYTES, int NUM_WARPS, int MI, int G, bool SPLIT>
__global__ void ovl_share_kernel(const uint4* __restrict__ src, float* __restrict__ sink, size_t buf_mask,
                                 int steps, double* __restrict__ out_cyc) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t stride_u4 = static_cast<size_t>(gridDim.x / G) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x / G) * STAGE_U4) & buf_mask;
  if constexpr (DEP != 0) {
    for (size_t i = tid; i < STAGE_U4; i += blockDim.x) smem[i] = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
    __syncthreads();
  }
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  float o00 = 0.f, o01 = 0.f, o02 = 0.f, o03 = 0.f, o10 = 0.f, o11 = 0.f, o12 = 0.f, o13 = 0.f;
  float o20 = 0.f, o21 = 0.f, o22 = 0.f, o23 = 0.f, o30 = 0.f, o31 = 0.f, o32 = 0.f, o33 = 0.f;
  float sf_acc = 0.5f;
  uint64_t t0 = static_cast<uint64_t>(clock64());
  for (int step = 0; step < steps; ++step) {
    if constexpr (DEP == 0) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = src + ((base + off) & buf_mask);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
        if constexpr (SPLIT) {
          if (v == VECS / 2 - 1) asm volatile("cp.async.commit_group;\n" ::: "memory");
        }
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    if constexpr (DEP != 4) {
      #pragma unroll
      for (int r = 0; r < LD_N; ++r) {
        size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
        uint32_t r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
        live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
        if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
      }
      #pragma unroll
      for (int m = 0; m < MI; ++m) { OVL_MMA(c00, c01, c02, c03); OVL_MMA(c10, c11, c12, c13); OVL_MMA(c20, c21, c22, c23); OVL_MMA(c30, c31, c32, c33); }
      float x = -0.5f - static_cast<float>(__float_as_uint(c00 + c10 + c20 + c30) & 7u) * 0.01f + sf_acc * 0.001f;
      float sm = 0.f;
      #pragma unroll
      for (int i = 0; i < 32; ++i) { float e; asm volatile("ex2.approx.f32 %0, %1;\n" : "=f"(e) : "f"(x)); sm = fmaf(e, 0.5f, sm); x = -e; }
      #pragma unroll
      for (int sh = 1; sh <= 8; sh <<= 1) sm += __shfl_xor_sync(0xffffffffu, sm, sh);
      sf_acc = sm;
      a0 = 0x3c003c00u ^ (__float_as_uint(sm) & 1u);
      #pragma unroll
      for (int m = 0; m < MI; ++m) { OVL_MMA(o00, o01, o02, o03); OVL_MMA(o10, o11, o12, o13); OVL_MMA(o20, o21, o22, o23); OVL_MMA(o30, o31, o32, o33); }
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + stride_u4) & buf_mask;
  }
  uint64_t t1 = static_cast<uint64_t>(clock64());
  if (blockIdx.x == 0 && tid == 0) *out_cyc = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              o00 + o01 + o02 + o03 + o10 + o11 + o12 + o13 + o20 + o21 + o22 + o23 + o30 + o31 + o32 + o33 +
              sf_acc + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

// Cycles from issuing the last of MI groups of MMAs until a dependent read of their results completes.
template <int NUM_WARPS, int MI>
__global__ void ovl_drain_kernel(float* __restrict__ sink, int steps, double* __restrict__ out) {
  const uint32_t tid = threadIdx.x;
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  uint64_t issue = 0, drain = 0;
  float carry = 0.f;
  for (int step = 0; step < steps; ++step) {
    asm volatile("bar.sync 0;\n" ::: "memory");
    uint64_t s0 = static_cast<uint64_t>(clock64());
    #pragma unroll
    for (int m = 0; m < MI; ++m) { OVL_MMA(c00, c01, c02, c03); OVL_MMA(c10, c11, c12, c13); OVL_MMA(c20, c21, c22, c23); OVL_MMA(c30, c31, c32, c33); }
    uint64_t s1 = static_cast<uint64_t>(clock64());
    float d;
    asm volatile("add.f32 %0, %1, %2;\n" : "=f"(d) : "f"(c00 + c10), "f"(c20 + c30));
    asm volatile("add.f32 %0, %0, %1;\n" : "+f"(carry) : "f"(d));
    uint64_t s2 = static_cast<uint64_t>(clock64());
    issue += s1 - s0;
    drain += s2 - s1;
    a0 = 0x3c003c00u ^ (__float_as_uint(carry) & 1u);
  }
  if (blockIdx.x == 0 && tid == 0) { out[0] = static_cast<double>(issue) / steps; out[1] = static_cast<double>(drain) / steps; }
  if (carry + c01 + c11 + c21 + c31 == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = carry;
}
#undef OVL_MMA

static int run_sync_loop_overlap(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);
  uint4 *d_dram = nullptr, *d_flush = nullptr, *d_l2 = nullptr, *d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  double* d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cyc, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 25, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const char* pattern_names[2] = {"gemm_s2", "attention"};
  const char* dep_names[4] = {"free", "dep", "noload", "nocompute"};

  auto time_ms = [&](auto launch, bool flush) {
    if (flush) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
    CUDA_CHECK(cudaEventRecord(ev_start));
    launch();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return static_cast<double>(ms);
  };

  std::printf("{\n  \"samples\": %d,\n  \"sync_loop_overlap\": [", samples);
  bool first = true;
  auto run_point = [&](auto pat_c, auto dep_c, auto sb_c, auto w_c, auto mi_c) {
    constexpr int PAT = decltype(pat_c)::value, DEP = decltype(dep_c)::value;
    constexpr int SB = decltype(sb_c)::value, W = decltype(w_c)::value, MI = decltype(mi_c)::value;
    auto kernel = ovl_kernel<PAT, DEP, SB, W, MI>;
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, W * 32, SB));
    for (int level = 0; level < 2; ++level) {
      const bool is_dram = level == 1;
      const uint4* buf = is_dram ? d_dram : d_l2;
      const size_t mask = is_dram ? dram_mask : l2_mask;
      for (int cps : {1, 2}) {
        if (cps > max_blocks) continue;
        const int grid = sm_count * cps;
        auto launch_steps = [&](int st) { return [&, st]() { kernel<<<grid, W * 32, SB>>>(buf, d_sink_f, mask, st, d_cyc); }; };
        double probe = time_ms(launch_steps(64), is_dram);
        int steps = static_cast<int>(1.0 / std::max(probe / 64.0, 1e-5));
        steps = std::min(65536, std::max(256, ((steps + 63) / 64) * 64));
        for (int w = 0; w < warmup; ++w) time_ms(launch_steps(steps), is_dram);
        std::vector<double> turn, cyc;
        for (int s = 0; s < samples; ++s) {
          double ms = time_ms(launch_steps(steps), is_dram);
          double c = 0.0;
          CUDA_CHECK(cudaMemcpy(&c, d_cyc, sizeof(double), cudaMemcpyDeviceToHost));
          turn.push_back(ms * 1e6 / steps);
          cyc.push_back(c);
        }
        if (!first) std::printf(",");
        first = false;
        std::printf("\n    {\"pattern\": \"%s\", \"dep\": \"%s\", \"mem_level\": \"%s\", \"ctas_per_sm\": %d, "
                    "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, \"mma_per_warp_per_block\": %d, "
                    "\"steps\": %d, \"turnaround_ns_samples\": [",
                    pattern_names[PAT], dep_names[DEP], is_dram ? "dram" : "l2", cps, grid, W, SB, MI * 4, steps);
        for (size_t i = 0; i < turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", turn[i]);
        std::printf("], \"cycles_per_step_samples\": [");
        for (size_t i = 0; i < cyc.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", cyc[i]);
        std::printf("]}");
      }
    }
  };
  auto each_dep = [&](auto pat_c, auto sb_c, auto w_c, auto mi_c) {
    run_point(pat_c, std::integral_constant<int, 0>{}, sb_c, w_c, mi_c);
    run_point(pat_c, std::integral_constant<int, 1>{}, sb_c, w_c, mi_c);
    run_point(pat_c, std::integral_constant<int, 2>{}, sb_c, w_c, mi_c);
    run_point(pat_c, std::integral_constant<int, 3>{}, sb_c, w_c, mi_c);
  };
  auto each_mi = [&](auto pat_c, auto sb_c, auto w_c) {
    each_dep(pat_c, sb_c, w_c, std::integral_constant<int, 2>{});
    each_dep(pat_c, sb_c, w_c, std::integral_constant<int, 8>{});
    each_dep(pat_c, sb_c, w_c, std::integral_constant<int, 32>{});
  };
  auto each_shape = [&](auto pat_c) {
    each_mi(pat_c, std::integral_constant<int, 4096>{}, std::integral_constant<int, 4>{});
    each_mi(pat_c, std::integral_constant<int, 4096>{}, std::integral_constant<int, 8>{});
    each_mi(pat_c, std::integral_constant<int, 16384>{}, std::integral_constant<int, 4>{});
    each_mi(pat_c, std::integral_constant<int, 16384>{}, std::integral_constant<int, 8>{});
  };
  each_shape(std::integral_constant<int, 0>{});
  each_shape(std::integral_constant<int, 1>{});
  std::printf("\n  ],\n  \"mma_drain\": [");

  first = true;
  auto drain = [&](auto w_c, auto mi_c) {
    constexpr int W = decltype(w_c)::value, MI = decltype(mi_c)::value;
    for (int cps : {1, 2}) {
      double out[2] = {0.0, 0.0};
      const int steps = 4096;
      for (int w = 0; w < warmup; ++w) ovl_drain_kernel<W, MI><<<sm_count * cps, W * 32>>>(d_sink_f, steps, d_cyc);
      std::vector<double> iss, drn;
      for (int s = 0; s < samples; ++s) {
        ovl_drain_kernel<W, MI><<<sm_count * cps, W * 32>>>(d_sink_f, steps, d_cyc);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out, d_cyc, 2 * sizeof(double), cudaMemcpyDeviceToHost));
        iss.push_back(out[0]);
        drn.push_back(out[1]);
      }
      if (!first) std::printf(",");
      first = false;
      std::printf("\n    {\"ctas_per_sm\": %d, \"num_warps\": %d, \"mma_per_warp\": %d, \"issue_cycles_samples\": [", cps, W, MI * 4);
      for (size_t i = 0; i < iss.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", iss[i]);
      std::printf("], \"drain_cycles_samples\": [");
      for (size_t i = 0; i < drn.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", drn[i]);
      std::printf("]}");
    }
  };
  drain(std::integral_constant<int, 4>{}, std::integral_constant<int, 2>{});
  drain(std::integral_constant<int, 4>{}, std::integral_constant<int, 8>{});
  drain(std::integral_constant<int, 4>{}, std::integral_constant<int, 32>{});
  drain(std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});
  drain(std::integral_constant<int, 8>{}, std::integral_constant<int, 8>{});
  drain(std::integral_constant<int, 8>{}, std::integral_constant<int, 32>{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_dram)); CUDA_CHECK(cudaFree(d_flush)); CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink_u4)); CUDA_CHECK(cudaFree(d_sink_f)); CUDA_CHECK(cudaFree(d_cyc));
  return 0;
}

static int run_sync_loop_share(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);
  uint4 *d_dram = nullptr, *d_flush = nullptr, *d_l2 = nullptr, *d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  double* d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 25, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const char* dep_names[5] = {"free", "", "noload", "", "empty"};
  auto time_ms = [&](auto launch, bool flush) {
    if (flush) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
    CUDA_CHECK(cudaEventRecord(ev_start));
    launch();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    return static_cast<double>(ms);
  };
  std::printf("{\n  \"samples\": %d,\n  \"sync_loop_share\": [", samples);
  bool first = true;
  auto run_point = [&](auto dep_c, auto w_c, auto g_c, auto split_c) {
    constexpr int DEP = decltype(dep_c)::value, W = decltype(w_c)::value, G = decltype(g_c)::value;
    constexpr bool SPLIT = decltype(split_c)::value;
    constexpr int SB = 16384, MI = 16;
    auto kernel = ovl_share_kernel<DEP, SB, W, MI, G, SPLIT>;
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, W * 32, SB));
    for (int level = 0; level < 2; ++level) {
      const bool is_dram = level == 1;
      const uint4* buf = is_dram ? d_dram : d_l2;
      const size_t mask = is_dram ? dram_mask : l2_mask;
      for (int cps : {1, 2}) {
        if (cps > max_blocks) continue;
        const int grid = sm_count * cps;
        auto launch_steps = [&](int st) { return [&, st]() { kernel<<<grid, W * 32, SB>>>(buf, d_sink_f, mask, st, d_cyc); }; };
        double probe = time_ms(launch_steps(64), is_dram);
        int steps = static_cast<int>(1.0 / std::max(probe / 64.0, 1e-5));
        steps = std::min(65536, std::max(256, ((steps + 63) / 64) * 64));
        for (int w = 0; w < warmup; ++w) time_ms(launch_steps(steps), is_dram);
        std::vector<double> turn, cyc;
        for (int s = 0; s < samples; ++s) {
          double ms = time_ms(launch_steps(steps), is_dram);
          double c = 0.0;
          CUDA_CHECK(cudaMemcpy(&c, d_cyc, sizeof(double), cudaMemcpyDeviceToHost));
          turn.push_back(ms * 1e6 / steps);
          cyc.push_back(c);
        }
        if (!first) std::printf(",");
        first = false;
        std::printf("\n    {\"dep\": \"%s\", \"share_ctas\": %d, \"split_kv\": %s, \"mem_level\": \"%s\", \"ctas_per_sm\": %d, "
                    "\"active_ctas\": %d, \"num_warps\": %d, \"stage_bytes\": %d, \"mma_per_warp_per_block\": %d, "
                    "\"steps\": %d, \"turnaround_ns_samples\": [",
                    dep_names[DEP], G, SPLIT ? "true" : "false", is_dram ? "dram" : "l2", cps, grid, W, SB, MI * 4, steps);
        for (size_t i = 0; i < turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", turn[i]);
        std::printf("], \"cycles_per_step_samples\": [");
        for (size_t i = 0; i < cyc.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", cyc[i]);
        std::printf("]}");
      }
    }
  };
  auto each_g = [&](auto w_c, auto split_c) {
    run_point(std::integral_constant<int, 0>{}, w_c, std::integral_constant<int, 1>{}, split_c);
    run_point(std::integral_constant<int, 0>{}, w_c, std::integral_constant<int, 4>{}, split_c);
    run_point(std::integral_constant<int, 0>{}, w_c, std::integral_constant<int, 16>{}, split_c);
    run_point(std::integral_constant<int, 0>{}, w_c, std::integral_constant<int, 32>{}, split_c);
  };
  for (int rep = 0; rep < 1; ++rep) {
    each_g(std::integral_constant<int, 4>{}, std::false_type{});
    each_g(std::integral_constant<int, 4>{}, std::true_type{});
    each_g(std::integral_constant<int, 8>{}, std::false_type{});
    each_g(std::integral_constant<int, 8>{}, std::true_type{});
  }
  run_point(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}, std::false_type{});
  run_point(std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{}, std::false_type{});
  run_point(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{}, std::false_type{});
  run_point(std::integral_constant<int, 4>{}, std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{}, std::false_type{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_dram)); CUDA_CHECK(cudaFree(d_flush)); CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink_u4)); CUDA_CHECK(cudaFree(d_sink_f)); CUDA_CHECK(cudaFree(d_cyc));
  return 0;
}

// Exposed round trip of the single-buffer loop against stage size: ovl_kernel (gemm_s2 shape, MI = 8)
// free and noload on warm L2, one CTA per SM, grid 1 or one per SM.
static int run_sync_loop_bytes(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  uint4* d_l2 = nullptr;
  float* d_sink_f = nullptr;
  double* d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cyc, 2 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const char* dep_names[3] = {"free", "dep", "noload"};

  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"sync_loop_bytes\": [", samples, sm_count);
  bool first = true;
  auto run_point = [&](auto dep_c, auto sb_c, auto w_c) {
    constexpr int DEP = decltype(dep_c)::value, SB = decltype(sb_c)::value, W = decltype(w_c)::value;
    auto kernel = ovl_kernel<0, DEP, SB, W, 8>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SB));
    for (int grid : {1, sm_count}) {
      auto launch = [&](int st) { kernel<<<grid, W * 32, SB>>>(d_l2, d_sink_f, l2_mask, st, d_cyc); };
      auto time_ms = [&](int st) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        launch(st);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        return static_cast<double>(ms);
      };
      double probe = time_ms(64);
      int steps = static_cast<int>(1.0 / std::max(probe / 64.0, 1e-5));
      steps = std::min(65536, std::max(256, ((steps + 63) / 64) * 64));
      for (int w = 0; w < warmup; ++w) time_ms(steps);
      std::vector<double> turn, cyc;
      for (int s = 0; s < samples; ++s) {
        double ms = time_ms(steps);
        double c = 0.0;
        CUDA_CHECK(cudaMemcpy(&c, d_cyc, sizeof(double), cudaMemcpyDeviceToHost));
        turn.push_back(ms * 1e6 / steps);
        cyc.push_back(c);
      }
      if (!first) std::printf(",");
      first = false;
      std::printf("\n    {\"dep\": \"%s\", \"grid\": %d, \"num_warps\": %d, \"stage_bytes\": %d, \"steps\": %d, "
                  "\"turnaround_ns_samples\": [", dep_names[DEP], grid, W, SB, steps);
      for (size_t i = 0; i < turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", turn[i]);
      std::printf("], \"cycles_per_step_samples\": [");
      for (size_t i = 0; i < cyc.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", cyc[i]);
      std::printf("]}");
    }
  };
  auto each_dep = [&](auto sb_c, auto w_c) {
    run_point(std::integral_constant<int, 0>{}, sb_c, w_c);
    run_point(std::integral_constant<int, 2>{}, sb_c, w_c);
  };
  auto each_sb = [&](auto w_c) {
    each_dep(std::integral_constant<int, 4096>{}, w_c);
    each_dep(std::integral_constant<int, 8192>{}, w_c);
    each_dep(std::integral_constant<int, 16384>{}, w_c);
    each_dep(std::integral_constant<int, 32768>{}, w_c);
    each_dep(std::integral_constant<int, 49152>{}, w_c);
    each_dep(std::integral_constant<int, 65536>{}, w_c);
  };
  each_sb(std::integral_constant<int, 4>{});
  each_sb(std::integral_constant<int, 8>{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_l2)); CUDA_CHECK(cudaFree(d_sink_f)); CUDA_CHECK(cudaFree(d_cyc));
  return 0;
}

// Single-buffer loop (gemm_s2 shape) where the first 1/FD of each stage's 16 B vectors come from a cold
// DRAM region that advances every step, and the rest from a warm 4 MiB L2 region. Groups of G CTAs read
// the same cold addresses in a step, so all but the first request of a group can merge into its fill.
// The cold region starts at cold_start, which the host advances after every launch so no launch rereads
// cold data within 256 MiB of its last read. FD = 0: no cold vectors. NOLOAD: no copies in the loop. The cold vectors are vector 0 of the lowest
// threads, so each thread issues its cold copy first.
#define MISS_MMA(acc0, acc1, acc2, acc3)                                                          \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
               "{%8,%9}, {%0,%1,%2,%3};\n"                                                         \
               : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                    \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <bool NOLOAD, int STAGE_BYTES, int NUM_WARPS, int MI, int FD, int G>
__global__ void miss_kernel(const uint4* __restrict__ hot, const uint4* __restrict__ cold, float* __restrict__ sink,
                            size_t hot_mask, size_t cold_mask, size_t cold_start, int steps,
                            double* __restrict__ out_cyc) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr size_t COLD_U4 = FD == 0 ? 0 : STAGE_U4 / FD;
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t groups = gridDim.x / G;
  const size_t group = blockIdx.x / G;
  size_t hot_base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & hot_mask;
  const size_t hot_step = static_cast<size_t>(gridDim.x) * STAGE_U4;
  if constexpr (NOLOAD) {
    for (size_t i = tid; i < STAGE_U4; i += blockDim.x) smem[i] = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
    __syncthreads();
  }
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  uint64_t t0 = static_cast<uint64_t>(clock64());
  for (int step = 0; step < steps; ++step) {
    if constexpr (!NOLOAD) {
      const size_t cold_base = (cold_start + (static_cast<size_t>(step) * groups + group) * COLD_U4) & cold_mask;
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        const void* gptr = off < COLD_U4 ? static_cast<const void*>(cold + ((cold_base + off) & cold_mask))
                                         : static_cast<const void*>(hot + ((hot_base + off) & hot_mask));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
    }
    #pragma unroll
    for (int m = 0; m < MI; ++m) { MISS_MMA(c00, c01, c02, c03); MISS_MMA(c10, c11, c12, c13); MISS_MMA(c20, c21, c22, c23); MISS_MMA(c30, c31, c32, c33); }
    asm volatile("bar.sync 0;\n" ::: "memory");
    hot_base = (hot_base + hot_step) & hot_mask;
  }
  uint64_t t1 = static_cast<uint64_t>(clock64());
  if (blockIdx.x == 0 && tid == 0) *out_cyc = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}
#undef MISS_MMA

// sync-loop-miss: free and noload points of miss_kernel (8 warps, MI = 8, one CTA per SM). Configs:
// grid 1; grid = SMs with G = 1; grid = SMs with G = 16.
static int run_sync_loop_miss(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t hot_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t cold_bytes = 256ULL * 1024ULL * 1024ULL;
  uint4 *d_hot = nullptr, *d_cold = nullptr;
  float* d_sink_f = nullptr;
  double* d_cyc = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hot, hot_bytes));
  CUDA_CHECK(cudaMalloc(&d_cold, cold_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 8 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cyc, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_hot, 0x3c, hot_bytes));
  CUDA_CHECK(cudaMemset(d_cold, 0x3c, cold_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const size_t hot_mask = hot_bytes / sizeof(uint4) - 1, cold_mask = cold_bytes / sizeof(uint4) - 1;

  size_t cold_start = 0;
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"sync_loop_miss\": [", samples, sm_count);
  bool first = true;
  auto run_point = [&](auto noload_c, auto sb_c, auto fd_c, auto g_c, int grid) {
    constexpr bool NOLOAD = decltype(noload_c)::value;
    constexpr int SB = decltype(sb_c)::value, FD = decltype(fd_c)::value, G = decltype(g_c)::value;
    auto kernel = miss_kernel<NOLOAD, SB, 8, 8, FD, G>;
    // At least 52 KiB of shared memory per CTA keeps one CTA per SM at every stage size.
    constexpr int SMEM = SB > 53248 ? SB : 53248;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, 256, SMEM));
    if (max_blocks != 1) { std::fprintf(stderr, "miss_kernel: %d CTAs per SM\n", max_blocks); std::exit(1); }
    auto time_ms = [&](int st) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      kernel<<<grid, 256, SMEM>>>(d_hot, d_cold, d_sink_f, hot_mask, cold_mask, cold_start, st, d_cyc);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      constexpr size_t COLD_U4 = FD == 0 ? 0 : static_cast<size_t>(SB) / sizeof(uint4) / (FD == 0 ? 1 : FD);
      cold_start = (cold_start + static_cast<size_t>(st) * static_cast<size_t>(grid / G) * COLD_U4) & cold_mask;
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      return static_cast<double>(ms);
    };
    double probe = time_ms(64);
    int steps = static_cast<int>(1.0 / std::max(probe / 64.0, 1e-5));
    steps = std::min(65536, std::max(256, ((steps + 63) / 64) * 64));
    for (int w = 0; w < warmup; ++w) time_ms(steps);
    std::vector<double> turn, cyc;
    for (int s = 0; s < samples; ++s) {
      double ms = time_ms(steps);
      double c = 0.0;
      CUDA_CHECK(cudaMemcpy(&c, d_cyc, sizeof(double), cudaMemcpyDeviceToHost));
      turn.push_back(ms * 1e6 / steps);
      cyc.push_back(c);
    }
    // Self-check: with every CTA reading its own cold stage, the aggregate cold rate must be of DRAM
    // order (GB10: 264 GB/s), far below L2 (over 2 TB/s); otherwise the cold region is cached.
    if (!NOLOAD && FD == 1 && G == 1 && grid == sm_count) {
      std::vector<double> sorted = turn;
      std::sort(sorted.begin(), sorted.end());
      const double gbps = static_cast<double>(grid) * SB / sorted[sorted.size() / 2];
      if (gbps > 600.0) { std::fprintf(stderr, "sync-loop-miss: cold rate %.0f GB/s is not DRAM\n", gbps); std::exit(1); }
    }
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"dep\": \"%s\", \"grid\": %d, \"group\": %d, \"num_warps\": 8, \"stage_bytes\": %d, "
                "\"cold_fraction_den\": %d, \"steps\": %d, \"turnaround_ns_samples\": [",
                NOLOAD ? "noload" : "free", grid, G, SB, FD, steps);
    for (size_t i = 0; i < turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", turn[i]);
    std::printf("], \"cycles_per_step_samples\": [");
    for (size_t i = 0; i < cyc.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", cyc[i]);
    std::printf("]}");
  };
  using T = std::true_type;
  using F = std::false_type;
  auto each_config = [&](auto noload_c, auto sb_c, auto fd_c) {
    run_point(noload_c, sb_c, fd_c, std::integral_constant<int, 1>{}, 1);
    run_point(noload_c, sb_c, fd_c, std::integral_constant<int, 1>{}, sm_count);
    run_point(noload_c, sb_c, fd_c, std::integral_constant<int, 16>{}, sm_count);
  };
  auto each_sb = [&](auto sb_c) {
    each_config(T{}, sb_c, std::integral_constant<int, 0>{});
    each_config(F{}, sb_c, std::integral_constant<int, 0>{});
    each_config(F{}, sb_c, std::integral_constant<int, 16>{});
    each_config(F{}, sb_c, std::integral_constant<int, 4>{});
    each_config(F{}, sb_c, std::integral_constant<int, 1>{});
  };
  each_sb(std::integral_constant<int, 8192>{});
  each_sb(std::integral_constant<int, 16384>{});
  each_sb(std::integral_constant<int, 32768>{});
  each_sb(std::integral_constant<int, 65536>{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_hot)); CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_sink_f)); CUDA_CHECK(cudaFree(d_cyc));
  return 0;
}

// sync-loop-delta: groups of G CTAs read the same stage in a single-buffer loop. A flag between SMs takes
// about one DRAM round trip on GB10, so the offset is scheduled on %globaltimer (32 ns steps) instead of
// signalled: once all followers have finished the previous step, CTA 0 of a group (the lead) publishes a
// start time `lead_ns` ahead, and issues its copies at that time; each follower issues the same copies
// delay_ns later. Every CTA records its stage latency (cycles from a barrier before its copies to the
// barrier after wait_group 0), its issue time after the start time (so the real offset is measured),
// and how many steps the start time had already passed when it learned of it. The
// region is addressed with `mask` from `start`: the host advances `start` after every launch on a
// 256 MiB ring (cold), or passes a 4 MiB mask and start 0 (warm L2 baseline).
#define DELTA_MMA(acc0, acc1, acc2, acc3)                                                         \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
               "{%8,%9}, {%0,%1,%2,%3};\n"                                                         \
               : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                    \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

static __device__ __forceinline__ void delta_publish(unsigned* flag, unsigned value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;\n" :: "l"(flag), "r"(value) : "memory");
}

static __device__ __forceinline__ unsigned delta_observe(const unsigned* flag) {
  unsigned value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(value) : "l"(flag) : "memory");
  return value;
}

template <int STAGE_BYTES, int NUM_WARPS, int MI, int G>
__global__ void delta_kernel(const uint4* __restrict__ cold, float* __restrict__ sink, size_t cold_mask,
                             size_t cold_start, int steps, unsigned long long lead_ns, unsigned long long delay_ns,
                             unsigned* __restrict__ progress, unsigned long long* __restrict__ start,
                             unsigned* __restrict__ done, double* __restrict__ out_latency,
                             double* __restrict__ out_issue_ns, unsigned* __restrict__ out_late,
                             unsigned long long* __restrict__ out_clock) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t groups = gridDim.x / G;
  const size_t group = blockIdx.x / G;
  const bool lead = blockIdx.x % G == 0;
  unsigned* flag = progress + group;
  unsigned* group_done = done + group;
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  long long latency = 0;
  unsigned long long issue_ns = 0;
  unsigned late = 0;
  const unsigned long long g0 = read_globaltimer();
  const long long k0 = clock64();
  for (int step = 0; step < steps; ++step) {
    unsigned long long when = 0;
    if (tid == 0) {
      if (lead) {
        while (delta_observe(group_done) < static_cast<unsigned>(step * (G - 1))) {}
        when = read_globaltimer() + lead_ns;
        start[group] = when;
        delta_publish(flag, static_cast<unsigned>(step + 1));
      } else {
        while (delta_observe(flag) < static_cast<unsigned>(step + 1)) {}
        when = *reinterpret_cast<volatile unsigned long long*>(start + group) + delay_ns;
      }
      late += read_globaltimer() > when;
      while (read_globaltimer() < when) {}
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    const long long t0 = clock64();
    if (tid == 0) issue_ns += read_globaltimer() - (lead ? when : when - delay_ns);
    const size_t base = (cold_start + (static_cast<size_t>(step) * groups + group) * STAGE_U4) & cold_mask;
    #pragma unroll
    for (int v = 0; v < VECS; ++v) {
      size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(cold + ((base + off) & cold_mask)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync 0;\n" ::: "memory");
    latency += clock64() - t0;
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
    }
    #pragma unroll
    for (int m = 0; m < MI; ++m) { DELTA_MMA(c00, c01, c02, c03); DELTA_MMA(c10, c11, c12, c13); DELTA_MMA(c20, c21, c22, c23); DELTA_MMA(c30, c31, c32, c33); }
    asm volatile("bar.sync 0;\n" ::: "memory");
    if (!lead && tid == 0) atomicAdd(group_done, 1u);
  }
  if (tid == 0) {
    out_latency[blockIdx.x] = static_cast<double>(latency) / steps;
    out_issue_ns[blockIdx.x] = static_cast<double>(issue_ns) / steps;
    out_late[blockIdx.x] = late;
    if (blockIdx.x == 0) {
      out_clock[0] = static_cast<unsigned long long>(clock64() - k0);
      out_clock[1] = read_globaltimer() - g0;
    }
  }
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}
#undef DELTA_MMA

// Smallest observed %globaltimer increments, one thread.
__global__ void delta_timer_step_kernel(unsigned long long* out, int count) {
  unsigned long long prev = read_globaltimer();
  for (int n = 0; n < count;) {
    const unsigned long long t = read_globaltimer();
    if (t != prev) { out[n++] = t - prev; prev = t; }
  }
}

// Flag ping-pong between two CTAs on different SMs: block 0 records the cycles of `rounds` round trips.
__global__ void delta_pingpong_kernel(unsigned* flags, int rounds, unsigned long long* out) {
  if (threadIdx.x != 0) return;
  if (blockIdx.x == 0) {
    const long long t0 = clock64();
    for (int i = 1; i <= rounds; ++i) {
      delta_publish(flags, static_cast<unsigned>(i));
      while (delta_observe(flags + 1) < static_cast<unsigned>(i)) {}
    }
    out[0] = static_cast<unsigned long long>(clock64() - t0);
  } else {
    for (int i = 1; i <= rounds; ++i) {
      while (delta_observe(flags) < static_cast<unsigned>(i)) {}
      delta_publish(flags + 1, static_cast<unsigned>(i));
    }
  }
}

static int run_sync_loop_delta(int device, int samples, int warmup, bool flag_only) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  constexpr int G = 16;
  // Whole groups only, one CTA per SM: 48 CTAs on GB10, 32 on the RTX 4070 (46 SMs).
  const int grid = sm_count / G * G;
  if (grid == 0) { std::fprintf(stderr, "sync-loop-delta: fewer than %d SMs\n", G); return 1; }
  const size_t cold_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t cold_mask = cold_bytes / sizeof(uint4) - 1;
  uint4* d_cold = nullptr;
  float* d_sink = nullptr;
  unsigned *d_progress = nullptr, *d_done = nullptr, *d_late = nullptr;
  double *d_latency = nullptr, *d_issue = nullptr;
  unsigned long long* d_start = nullptr;
  unsigned long long* d_clock = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cold, cold_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_progress, 64 * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_done, 64 * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_late, sm_count * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&d_latency, sm_count * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_issue, sm_count * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_start, 64 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_cold, 0x3c, cold_bytes));
  size_t cold_start = 0;
  // The lead schedules each start this far ahead; the flag takes about 400 ns on GB10.
  constexpr unsigned long long LEAD_NS = 1000;

  // Flag latency: half a ping-pong round trip, in cycles of block 0's SM. Shared memory forces two SMs.
  constexpr int ROUNDS = 1000;
  CUDA_CHECK(cudaFuncSetAttribute(delta_pingpong_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
  std::vector<double> pingpong;
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaMemset(d_progress, 0, 64 * sizeof(unsigned)));
    delta_pingpong_kernel<<<2, 32, 65536>>>(d_progress, ROUNDS, d_clock);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long cycles = 0;
    CUDA_CHECK(cudaMemcpy(&cycles, d_clock, sizeof(cycles), cudaMemcpyDeviceToHost));
    pingpong.push_back(static_cast<double>(cycles) / ROUNDS / 2.0);
  }
  constexpr int TICKS = 64;
  unsigned long long* d_ticks = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ticks, TICKS * sizeof(unsigned long long)));
  delta_timer_step_kernel<<<1, 1>>>(d_ticks, TICKS);
  std::vector<unsigned long long> ticks(TICKS);
  CUDA_CHECK(cudaMemcpy(ticks.data(), d_ticks, TICKS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_ticks));
  std::sort(ticks.begin(), ticks.end());
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"grid\": %d,\n  \"group\": %d,\n  \"lead_ns\": %llu,\n"
              "  \"globaltimer_step_ns\": {\"min\": %llu, \"median\": %llu, \"max\": %llu},\n  \"flag_one_way_cycles_samples\": [",
              samples, sm_count, grid, G, LEAD_NS, ticks[0], ticks[TICKS / 2], ticks[TICKS - 1]);
  for (size_t i = 0; i < pingpong.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", pingpong[i]);
  std::printf("],\n  \"sync_loop_delta\": [");
  if (flag_only) {
    std::printf("]\n}\n");
    return 0;
  }

  bool first = true;
  double ghz = 0.0;
  const size_t warm_mask = 4ULL * 1024ULL * 1024ULL / sizeof(uint4) - 1;
  auto run_point = [&](auto sb_c, int delta_ns, bool warm) {
    constexpr int SB = decltype(sb_c)::value;
    constexpr int SMEM = SB > 53248 ? SB : 53248;
    auto kernel = delta_kernel<SB, 8, 8, G>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, 256, SMEM));
    if (max_blocks != 1) { std::fprintf(stderr, "delta_kernel: %d CTAs per SM\n", max_blocks); std::exit(1); }
    std::vector<double> lat(sm_count), issue(sm_count);
    std::vector<unsigned> late(sm_count);
    unsigned long long clk[2] = {0, 0};
    auto launch = [&](int steps) {
      CUDA_CHECK(cudaMemset(d_progress, 0, 64 * sizeof(unsigned)));
      CUDA_CHECK(cudaMemset(d_done, 0, 64 * sizeof(unsigned)));
      kernel<<<grid, 256, SMEM>>>(d_cold, d_sink, warm ? warm_mask : cold_mask, warm ? 0 : cold_start, steps,
                                      LEAD_NS, delta_ns, d_progress, d_start, d_done, d_latency, d_issue, d_late,
                                      d_clock);
      CUDA_CHECK(cudaDeviceSynchronize());
      if (!warm) cold_start = (cold_start + static_cast<size_t>(steps) * (grid / G) * (SB / sizeof(uint4))) & cold_mask;
      CUDA_CHECK(cudaMemcpy(lat.data(), d_latency, sm_count * sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(issue.data(), d_issue, sm_count * sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(late.data(), d_late, sm_count * sizeof(unsigned), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(clk, d_clock, sizeof(clk), cudaMemcpyDeviceToHost));
      ghz = static_cast<double>(clk[0]) / static_cast<double>(clk[1]);
    };
    launch(64);
    const double step_ns = static_cast<double>(clk[1]) / 64.0;
    int steps = static_cast<int>(1e6 / std::max(step_ns, 1.0));
    steps = std::min(65536, std::max(256, ((steps + 63) / 64) * 64));
    for (int w = 0; w < warmup; ++w) launch(steps);
    std::vector<double> lead_s, follow_s, offset_s, ghz_s;
    unsigned late_steps = 0;
    const int leads = grid / G, followers = grid - leads;
    for (int s = 0; s < samples; ++s) {
      launch(steps);
      double lead_sum = 0.0, follow_sum = 0.0, lead_issue = 0.0, follow_issue = 0.0;
      for (int b = 0; b < grid; ++b) {
        if (b % G == 0) { lead_sum += lat[b]; lead_issue += issue[b]; }
        else { follow_sum += lat[b]; follow_issue += issue[b]; }
        late_steps += late[b];
      }
      lead_s.push_back(lead_sum / leads);
      follow_s.push_back(follow_sum / followers);
      offset_s.push_back(follow_issue / followers - lead_issue / leads);
      ghz_s.push_back(ghz);
    }
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"region\": \"%s\", \"stage_bytes\": %d, \"delta_ns\": %d, \"num_warps\": 8, \"steps\": %d, "
                "\"late_steps\": %u, \"follower_latency_cycles_samples\": [", warm ? "warm" : "cold", SB, delta_ns, steps, late_steps);
    for (size_t i = 0; i < follow_s.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", follow_s[i]);
    std::printf("], \"lead_latency_cycles_samples\": [");
    for (size_t i = 0; i < lead_s.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", lead_s[i]);
    std::printf("], \"measured_offset_ns_samples\": [");
    for (size_t i = 0; i < offset_s.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", offset_s[i]);
    std::printf("], \"sm_ghz_samples\": [");
    for (size_t i = 0; i < ghz_s.size(); ++i) std::printf(i ? ", %.4f" : "%.4f", ghz_s[i]);
    std::printf("]}");
  };
  // Warm L2 baselines (no cold miss) first: the shared calibration point is warm, 16 KiB, delay 0.
  run_point(std::integral_constant<int, 16384>{}, 0, true);
  run_point(std::integral_constant<int, 32768>{}, 0, true);
  // The delay grid comes from VGPU_DELTA_NS (comma separated ns) when set; the default is the GB10 grid.
  std::vector<int> deltas = {0, 50, 100, 150, 200, 250, 300, 400, 700, 1500};
  if (const char* env = std::getenv("VGPU_DELTA_NS")) {
    deltas.clear();
    for (const char* c = env; *c;) {
      char* end = nullptr;
      deltas.push_back(static_cast<int>(std::strtol(c, &end, 10)));
      if (end == c) { std::fprintf(stderr, "sync-loop-delta: bad VGPU_DELTA_NS\n"); return 1; }
      c = *end == ',' ? end + 1 : end;
    }
  }
  for (int d : deltas) run_point(std::integral_constant<int, 16384>{}, d, false);
  for (int d : deltas) run_point(std::integral_constant<int, 32768>{}, d, false);
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_cold)); CUDA_CHECK(cudaFree(d_sink)); CUDA_CHECK(cudaFree(d_progress)); CUDA_CHECK(cudaFree(d_done));
  CUDA_CHECK(cudaFree(d_late)); CUDA_CHECK(cudaFree(d_latency)); CUDA_CHECK(cudaFree(d_issue));
  CUDA_CHECK(cudaFree(d_start)); CUDA_CHECK(cudaFree(d_clock));
  return 0;
}

// sync-loop-free: groups of G CTAs read the same stage in a free-running single-buffer loop (no schedule,
// no step barrier), so the followers' natural offset from the first issuer can be measured. With STAMP,
// thread 0 records %globaltimer just before each step's copies. Every CTA records its stage latency
// (cycles from a barrier before its copies to the barrier after wait_group 0) and its finish time; the
// output is the mean over all CTAs, since the first issuer of a step changes from step to step. The
// region is addressed with `mask` from `start`, as in sync-loop-delta (cold ring or warm 4 MiB).
#define FREE_MMA(acc0, acc1, acc2, acc3)                                                          \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
               "{%8,%9}, {%0,%1,%2,%3};\n"                                                         \
               : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                    \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

template <int STAGE_BYTES, int NUM_WARPS, int MI, int G, bool STAMP>
__global__ void free_kernel(const uint4* __restrict__ src, float* __restrict__ sink, size_t mask, size_t start,
                            int steps, unsigned long long* __restrict__ stamps, double* __restrict__ out_latency,
                            unsigned long long* __restrict__ out_finish, unsigned long long* __restrict__ out_clock) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  const size_t groups = gridDim.x / G;
  const size_t group = blockIdx.x / G;
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  long long latency = 0;
  const unsigned long long g0 = read_globaltimer();
  const long long k0 = clock64();
  for (int step = 0; step < steps; ++step) {
    asm volatile("bar.sync 0;\n" ::: "memory");
    const long long t0 = clock64();
    if (STAMP && tid == 0) stamps[static_cast<size_t>(blockIdx.x) * steps + step] = read_globaltimer();
    const size_t base = (start + (static_cast<size_t>(step) * groups + group) * STAGE_U4) & mask;
    #pragma unroll
    for (int v = 0; v < VECS; ++v) {
      size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(src + ((base + off) & mask)) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    asm volatile("bar.sync 0;\n" ::: "memory");
    latency += clock64() - t0;
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
    }
    #pragma unroll
    for (int m = 0; m < MI; ++m) { FREE_MMA(c00, c01, c02, c03); FREE_MMA(c10, c11, c12, c13); FREE_MMA(c20, c21, c22, c23); FREE_MMA(c30, c31, c32, c33); }
  }
  asm volatile("bar.sync 0;\n" ::: "memory");
  if (tid == 0) {
    out_latency[blockIdx.x] = static_cast<double>(latency) / steps;
    out_finish[blockIdx.x] = read_globaltimer();
    if (blockIdx.x == 0) {
      out_clock[0] = static_cast<unsigned long long>(clock64() - k0);
      out_clock[1] = read_globaltimer() - g0;
    }
  }
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}
#undef FREE_MMA

static int run_sync_loop_free(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  constexpr int STEPS = 1024;
  // The same number of groups at every G (three, or fewer on GPUs with under 48 SMs), so every group
  // size puts the same cold traffic on DRAM.
  const int GROUPS = std::min(3, sm_count / 16);
  if (GROUPS == 0) { std::fprintf(stderr, "sync-loop-free: needs at least 16 SMs\n"); return 1; }
  constexpr int BINS = 256, BIN_NS = 32;
  const size_t cold_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t cold_mask = cold_bytes / sizeof(uint4) - 1, warm_mask = 4ULL * 1024ULL * 1024ULL / sizeof(uint4) - 1;
  uint4* d_src = nullptr;
  float* d_sink = nullptr;
  unsigned long long *d_stamps = nullptr, *d_finish = nullptr, *d_clock = nullptr;
  double* d_latency = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, cold_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_stamps, static_cast<size_t>(sm_count) * STEPS * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_finish, sm_count * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_latency, sm_count * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_src, 0x3c, cold_bytes));
  size_t cold_start = 0;
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"groups\": %d,\n  \"steps\": %d,\n  \"offset_bin_ns\": %d,\n  \"sync_loop_free\": [",
              samples, sm_count, GROUPS, STEPS, BIN_NS);
  bool first = true;
  auto run_point = [&](auto sb_c, auto w_c, auto mi_c, auto g_c, auto stamp_c, bool warm) {
    constexpr int SB = decltype(sb_c)::value, W = decltype(w_c)::value, MI = decltype(mi_c)::value;
    constexpr int G = decltype(g_c)::value;
    constexpr bool STAMP = decltype(stamp_c)::value;
    constexpr int SMEM = SB > 53248 ? SB : 53248;
    const int grid = GROUPS * G;
    auto kernel = free_kernel<SB, W, MI, G, STAMP>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, W * 32, SMEM));
    if (max_blocks != 1) { std::fprintf(stderr, "free_kernel: %d CTAs per SM\n", max_blocks); std::exit(1); }
    std::vector<double> lat(grid);
    std::vector<unsigned long long> finish(grid), stamps(STAMP ? static_cast<size_t>(grid) * STEPS : 0);
    unsigned long long clk[2] = {0, 0};
    auto launch = [&]() {
      kernel<<<grid, W * 32, SMEM>>>(d_src, d_sink, warm ? warm_mask : cold_mask, warm ? 0 : cold_start, STEPS, d_stamps,
                                     d_latency, d_finish, d_clock);
      CUDA_CHECK(cudaDeviceSynchronize());
      if (!warm) cold_start = (cold_start + static_cast<size_t>(STEPS) * (grid / G) * (SB / sizeof(uint4))) & cold_mask;
      CUDA_CHECK(cudaMemcpy(lat.data(), d_latency, grid * sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(finish.data(), d_finish, grid * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(clk, d_clock, sizeof(clk), cudaMemcpyDeviceToHost));
      if (STAMP) CUDA_CHECK(cudaMemcpy(stamps.data(), d_stamps, stamps.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    };
    for (int w = 0; w < warmup; ++w) launch();
    std::vector<double> latency_s, step_s, spread_s;
    std::vector<unsigned long long> hist(BINS + 1, 0);
    std::vector<double> offsets;
    for (int s = 0; s < samples; ++s) {
      launch();
      const double ghz = static_cast<double>(clk[0]) / static_cast<double>(clk[1]);
      double latency_sum = 0.0, spread_sum = 0.0;
      for (int g = 0; g < grid / G; ++g) {
        double mean = 0.0, sq = 0.0;
        auto rel = [&](int i) {
          return static_cast<double>(static_cast<long long>(finish[g * G + i] - finish[g * G]));
        };
        for (int i = 0; i < G; ++i) mean += rel(i) / G;
        for (int i = 0; i < G; ++i) {
          const double x = rel(i) - mean;
          sq += x * x / G;
        }
        spread_sum += std::sqrt(sq);
        for (int i = 0; i < G; ++i) latency_sum += lat[g * G + i] / ghz;
      }
      const int groups = grid / G;
      latency_s.push_back(latency_sum / grid);
      spread_s.push_back(spread_sum / groups);
      step_s.push_back(static_cast<double>(clk[1]) / STEPS);
      if (STAMP) {
        for (int g = 0; g < groups; ++g) {
          for (int step = 0; step < STEPS; ++step) {
            unsigned long long lo = ~0ULL;
            int lead = 0;
            for (int i = 0; i < G; ++i) {
              const unsigned long long t = stamps[static_cast<size_t>(g * G + i) * STEPS + step];
              if (t < lo) { lo = t; lead = i; }
            }
            for (int i = 0; i < G; ++i) {
              if (i == lead) continue;
              const double d = static_cast<double>(stamps[static_cast<size_t>(g * G + i) * STEPS + step] - lo);
              offsets.push_back(d);
              ++hist[std::min<size_t>(BINS, static_cast<size_t>(d / BIN_NS))];
            }
          }
        }
      }
    }
    auto list = [](const char* name, const std::vector<double>& v) {
      std::printf(", \"%s\": [", name);
      for (size_t i = 0; i < v.size(); ++i) std::printf(i ? ", %.1f" : "%.1f", v[i]);
      std::printf("]");
    };
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"region\": \"%s\", \"stamp\": %s, \"stage_bytes\": %d, \"num_warps\": %d, \"mma_groups\": %d, "
                "\"group\": %d, \"grid\": %d", warm ? "warm" : "cold", STAMP ? "true" : "false", SB, W, MI, G, grid);
    list("stage_latency_ns_samples", latency_s);
    list("step_ns_samples", step_s);
    list("finish_spread_ns_samples", spread_s);
    if (STAMP) {
      std::sort(offsets.begin(), offsets.end());
      auto q = [&](double p) { return offsets[static_cast<size_t>(p * (offsets.size() - 1))]; };
      std::printf(", \"follower_offset_ns_quantiles\": {\"p10\": %.0f, \"p25\": %.0f, \"p50\": %.0f, \"p75\": %.0f, \"p90\": %.0f, \"p99\": %.0f}",
                  q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), q(0.99));
      std::printf(", \"follower_offset_histogram\": [");
      for (int b = 0; b <= BINS; ++b) std::printf(b ? ", %llu" : "%llu", hist[b]);
      std::printf("]");
    }
    std::printf("}");
  };
  using T = std::true_type;
  using F = std::false_type;
  auto each = [&](auto sb_c, auto w_c, auto mi_c, auto g_c) {
    run_point(sb_c, w_c, mi_c, g_c, F{}, true);
    run_point(sb_c, w_c, mi_c, g_c, F{}, false);
    run_point(sb_c, w_c, mi_c, g_c, T{}, false);
  };
  auto each_g = [&](auto sb_c, auto w_c, auto mi_c) {
    each(sb_c, w_c, mi_c, std::integral_constant<int, 4>{});
    each(sb_c, w_c, mi_c, std::integral_constant<int, 16>{});
  };
  auto each_mi = [&](auto sb_c, auto w_c) {
    each_g(sb_c, w_c, std::integral_constant<int, 2>{});
    each_g(sb_c, w_c, std::integral_constant<int, 8>{});
    each_g(sb_c, w_c, std::integral_constant<int, 32>{});
  };
  auto each_w = [&](auto sb_c) {
    each_mi(sb_c, std::integral_constant<int, 4>{});
    each_mi(sb_c, std::integral_constant<int, 8>{});
  };
  each_w(std::integral_constant<int, 16384>{});
  each_w(std::integral_constant<int, 32768>{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_sink)); CUDA_CHECK(cudaFree(d_stamps));
  CUDA_CHECK(cudaFree(d_finish)); CUDA_CHECK(cudaFree(d_clock)); CUDA_CHECK(cudaFree(d_latency));
  return 0;
}

// sync-loop-burst: the gemm_s2 shaped single-buffer loop on warm L2 with n CTAs, either started together
// (lock) or with CTA c held back c * stagger_ns (stag), to see whether the CTAs' simultaneous stages
// meet the L2 bandwidth. Each CTA records %globaltimer at the start of steps 0, 16, 64, 256, S/2 and
// S - 1, and its cycles per step after the start offset; block 0 records the SM clock.
#define BURST_MMA(acc0, acc1, acc2, acc3)                                                         \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
               "{%8,%9}, {%0,%1,%2,%3};\n"                                                         \
               : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)                                    \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

constexpr int BURST_STAMPS = 6;

template <bool NOLOAD, int STAGE_BYTES, int NUM_WARPS, int MI>
__global__ void burst_kernel(const uint4* __restrict__ src, float* __restrict__ sink, size_t mask, int steps,
                             unsigned long long stagger_ns, unsigned long long* __restrict__ out_stamps,
                             double* __restrict__ out_cycles, unsigned long long* __restrict__ out_clock) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
  constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & mask;
  const size_t step_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  if constexpr (NOLOAD) {
    for (size_t i = tid; i < STAGE_U4; i += blockDim.x) smem[i] = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
  }
  if (tid == 0 && stagger_ns > 0) {
    const unsigned long long until = read_globaltimer() + stagger_ns * blockIdx.x;
    while (read_globaltimer() < until) {}
  }
  __syncthreads();
  const long long k0 = clock64();
  const unsigned long long g0 = read_globaltimer();
  const int marks[BURST_STAMPS] = {0, 16, 64, 256, steps / 2, steps - 1};
  uint32_t a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f, c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;
  for (int step = 0; step < steps; ++step) {
    if (tid == 0) {
      #pragma unroll
      for (int m = 0; m < BURST_STAMPS; ++m) {
        if (step == marks[m]) out_stamps[blockIdx.x * BURST_STAMPS + m] = read_globaltimer();
      }
    }
    if constexpr (!NOLOAD) {
      #pragma unroll
      for (int v = 0; v < VECS; ++v) {
        size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(src + ((base + off) & mask)) : "memory");
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    #pragma unroll
    for (int r = 0; r < LD_N; ++r) {
      size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
      live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
      if (r == LD_N - 1) { a0 = r0; a1 = r1; a2 = r2; a3 = r3; b0 = r0; b1 = r1; }
    }
    #pragma unroll
    for (int m = 0; m < MI; ++m) { BURST_MMA(c00, c01, c02, c03); BURST_MMA(c10, c11, c12, c13); BURST_MMA(c20, c21, c22, c23); BURST_MMA(c30, c31, c32, c33); }
    asm volatile("bar.sync 0;\n" ::: "memory");
    base = (base + step_u4) & mask;
  }
  if (tid == 0) {
    out_cycles[blockIdx.x] = static_cast<double>(clock64() - k0) / steps;
    if (blockIdx.x == 0) {
      out_clock[0] = static_cast<unsigned long long>(clock64() - k0);
      out_clock[1] = read_globaltimer() - g0;
    }
  }
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 + c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}
#undef BURST_MMA

static int run_sync_loop_burst(int device, int samples, int warmup, bool check) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t hot_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t mask = hot_bytes / sizeof(uint4) - 1;
  uint4* d_hot = nullptr;
  float* d_sink = nullptr;
  unsigned long long *d_stamps = nullptr, *d_clock = nullptr;
  double* d_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hot, hot_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_stamps, static_cast<size_t>(sm_count) * BURST_STAMPS * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_cycles, sm_count * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_clock, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_hot, 0x3c, hot_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"sync_loop_burst\": [", samples, sm_count);
  bool first = true;
  // Mean resultant length of the CTAs' phases (time modulo the step) at one stamp: 1 when they move
  // together, near 0 when they are spread evenly over the step.
  auto coherence = [](const std::vector<unsigned long long>& st, int grid, int which, double step_ns) {
    double c = 0.0, s = 0.0;
    for (int b = 0; b < grid; ++b) {
      const double rel = static_cast<double>(static_cast<long long>(st[b * BURST_STAMPS + which] - st[which]));
      const double phase = 2.0 * 3.14159265358979323846 * std::fmod(std::fmod(rel, step_ns) + step_ns, step_ns) / step_ns;
      c += std::cos(phase);
      s += std::sin(phase);
    }
    return std::sqrt(c * c + s * s) / grid;
  };
  auto run_point = [&](auto noload_c, auto sb_c, int grid, double stagger_ns) {
    constexpr bool NOLOAD = decltype(noload_c)::value;
    constexpr int SB = decltype(sb_c)::value;
    constexpr int SMEM = SB > 53248 ? SB : 53248;
    auto kernel = burst_kernel<NOLOAD, SB, 8, 8>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, 256, SMEM));
    if (max_blocks != 1) { std::fprintf(stderr, "burst_kernel: %d CTAs per SM\n", max_blocks); std::exit(1); }
    const unsigned long long stagger = static_cast<unsigned long long>(stagger_ns);
    auto time_ms = [&](int st) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      kernel<<<grid, 256, SMEM>>>(d_hot, d_sink, mask, st, stagger, d_stamps, d_cycles, d_clock);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      return static_cast<double>(ms);
    };
    // At least 1024 steps so the coherence marks at 16, 64 and 256 come before the middle one.
    double probe = time_ms(64);
    int steps = static_cast<int>(1.0 / std::max(probe / 64.0, 1e-5));
    steps = std::min(65536, std::max(1024, ((steps + 63) / 64) * 64));
    for (int w = 0; w < warmup; ++w) time_ms(steps);
    std::vector<double> turn, event_turn;
    std::vector<std::vector<double>> coh(BURST_STAMPS);
    std::vector<unsigned long long> st(static_cast<size_t>(grid) * BURST_STAMPS);
    std::vector<double> cyc(grid);
    unsigned long long clk[2] = {0, 0};
    for (int s = 0; s < samples; ++s) {
      const double ms = time_ms(steps);
      event_turn.push_back(ms * 1e6 / steps);
      CUDA_CHECK(cudaMemcpy(st.data(), d_stamps, st.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cyc.data(), d_cycles, grid * sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(clk, d_clock, sizeof(clk), cudaMemcpyDeviceToHost));
      // Per-CTA steps after the start offset, so the stag offsets add no tail.
      const double ghz = static_cast<double>(clk[0]) / static_cast<double>(clk[1]);
      double mean = 0.0;
      for (int b = 0; b < grid; ++b) mean += cyc[b] / ghz / grid;
      turn.push_back(mean);
      const double step_ns = static_cast<double>(st[5] - st[4]) / (steps - 1 - steps / 2);
      for (int m = 0; m < BURST_STAMPS; ++m) coh[m].push_back(coherence(st, grid, m, step_ns));
    }
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"dep\": \"%s\", \"start\": \"%s\", \"grid\": %d, \"num_warps\": 8, \"stage_bytes\": %d, \"steps\": %d, "
                "\"stagger_ns\": %.1f, \"turnaround_ns_samples\": [", NOLOAD ? "noload" : "free", stagger > 0 ? "stag" : "lock",
                grid, SB, steps, stagger_ns);
    for (size_t i = 0; i < turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", turn[i]);
    std::printf("], \"event_turnaround_ns_samples\": [");
    for (size_t i = 0; i < event_turn.size(); ++i) std::printf(i ? ", %.2f" : "%.2f", event_turn[i]);
    std::printf("], \"coherence_steps\": [0, 16, 64, 256, %d, %d], \"coherence_samples\": [", steps / 2, steps - 1);
    for (int m = 0; m < BURST_STAMPS; ++m) {
      std::printf(m ? ", [" : "[");
      for (size_t i = 0; i < coh[m].size(); ++i) std::printf(i ? ", %.3f" : "%.3f", coh[m][i]);
      std::printf("]");
    }
    std::printf("]}");
    std::vector<double> sorted = turn;
    std::sort(sorted.begin(), sorted.end());
    return sorted[sorted.size() / 2];
  };
  auto each = [&](auto sb_c) {
    for (int n : {12, 36, 40, 44, 48}) {
      if (n > sm_count) continue;
      run_point(std::true_type{}, sb_c, n, 0.0);
      const double step = run_point(std::false_type{}, sb_c, n, 0.0);
      run_point(std::false_type{}, sb_c, n, step / n);
    }
  };
  // The check grid: stage sizes and CTA counts the burst sweep did not measure, simultaneous starts only.
  auto each_check = [&](auto sb_c) {
    for (int n : {1, 16, 28, 34, 42, 46}) {
      if (n > sm_count) continue;
      run_point(std::true_type{}, sb_c, n, 0.0);
      run_point(std::false_type{}, sb_c, n, 0.0);
    }
  };
  if (check) {
    each_check(std::integral_constant<int, 24576>{});
    each_check(std::integral_constant<int, 40960>{});
    each_check(std::integral_constant<int, 57344>{});
  } else {
    each(std::integral_constant<int, 16384>{});
    each(std::integral_constant<int, 32768>{});
    each(std::integral_constant<int, 49152>{});
    each(std::integral_constant<int, 65536>{});
  }
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_hot)); CUDA_CHECK(cudaFree(d_sink)); CUDA_CHECK(cudaFree(d_stamps));
  CUDA_CHECK(cudaFree(d_cycles)); CUDA_CHECK(cudaFree(d_clock));
  return 0;
}

// l2-paths: the L2 read bandwidth against active SMs on warm L2 through two load paths in one session:
// the ld.global.cg ILP-8 kernel of run_l2 (16 warps per CTA) and a 4-stage cp.async ring as in
// VirtualGPU issue #8 (16 warps per CTA, two 16 B copies per thread per stage), one CTA per SM at
// every point, plus the cp.async kernel at three CTAs per SM on the full GPU as in #8 (8 warps per CTA
// there, so three CTAs' rings fit in shared memory).
__global__ void l2_cp_async_kernel(const uint4* __restrict__ src, uint4* __restrict__ sink, size_t mask, int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + tid) & mask;
  auto issue = [&](int slot) {
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const size_t off = (static_cast<size_t>(slot) * 2 + sub) * blockDim.x + tid;
      const uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[off]));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(src + idx) : "memory");
      idx = (idx + stride) & mask;
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };
  #pragma unroll
  for (int s = 0; s < 4; ++s) issue(s);
  for (int step = 4; step < steps; ++step) {
    asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    issue(step & 3);
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  if (smem[tid].x == 0xdeadbeefu) sink[tid & 1023u] = smem[tid];
}

static int run_l2_paths(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t ws_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t mask = ws_bytes / sizeof(uint4) - 1ULL;
  uint4 *d_buf = nullptr, *d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, ws_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_buf, 3, ws_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const int threads = 512;
  constexpr int CP_SMEM = 4 * 2 * 512 * 16;
  CUDA_CHECK(cudaFuncSetAttribute(l2_cp_async_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, CP_SMEM));
  auto time_gbps = [&](auto launch, double bytes) {
    for (int w = 0; w < warmup; ++w) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> g;
    for (int s = 0; s < samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      launch();
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      g.push_back((bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }
    return g;
  };
  auto print = [](const char* path, int sms, int grid, const std::vector<double>& g, bool& first) {
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"path\": \"%s\", \"active_sms\": %d, \"grid\": %d, \"gbps_samples\": [", path, sms, grid);
    for (size_t i = 0; i < g.size(); ++i) std::printf(i ? ", %.3f" : "%.3f", g[i]);
    std::printf("]}");
  };
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"working_set_bytes\": %zu,\n  \"l2_paths\": [", samples, sm_count, ws_bytes);
  bool first = true;
  for (int sms : {1, 12, 24, 36, 40, 44, 48}) {
    if (sms > sm_count) continue;
    const int iters = sms <= 4 ? 4096 : 2048;
    const double ld_bytes = static_cast<double>(sms) * threads * 8.0 * 16.0 * iters;
    print("ld_global_cg", sms, sms,
          time_gbps([&] { l2_read_cg_ilp8_kernel<<<sms, threads>>>(d_buf, d_sink, mask, iters); }, ld_bytes), first);
    const int steps = 2 * iters * 4;
    const double cp_bytes = static_cast<double>(sms) * threads * 2.0 * 16.0 * steps;
    print("cp_async", sms, sms,
          time_gbps([&] { l2_cp_async_kernel<<<sms, threads, CP_SMEM>>>(d_buf, d_sink, mask, steps); }, cp_bytes), first);
  }
  {
    const int grid = sm_count * 3, steps = 8192, half = threads / 2;
    const double cp_bytes = static_cast<double>(grid) * half * 2.0 * 16.0 * steps;
    print("cp_async", sm_count, grid,
          time_gbps([&] { l2_cp_async_kernel<<<grid, half, CP_SMEM / 2>>>(d_buf, d_sink, mask, steps); }, cp_bytes), first);
  }
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_buf)); CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// mma-ilp: each warp keeps ILP independent m16n8k16 f16 -> f32 accumulators and runs a chain of N
// dependent MMAs on each per step, rotating over the accumulators, with no memory traffic. Every
// accumulator feeds the sink so nothing is eliminated. CTAs per SM are set by dynamic shared memory.
template <int ILP, int N>
__global__ void mma_ilp_kernel(float* __restrict__ sink, int steps, uint32_t seed) {
  const uint32_t a0 = 0x3c003c00u ^ (seed & 1u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  const uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u ^ (seed & 2u);
  float acc[ILP][4];
  #pragma unroll
  for (int i = 0; i < ILP; ++i) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }
  for (int step = 0; step < steps; ++step) {
    #pragma unroll
    for (int n = 0; n < N; ++n) {
      #pragma unroll
      for (int i = 0; i < ILP; ++i) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
  }
  float sum = 0.f;
  #pragma unroll
  for (int i = 0; i < ILP; ++i) sum += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

static int run_mma_ilp(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 3 * 512 * sizeof(float)));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"chain_lengths\": [16, 64],\n  \"mma_ilp\": [", samples, sm_count);
  bool first = true;
  auto run_point = [&](auto ilp_c, int warps, int cps) {
    constexpr int ILP = decltype(ilp_c)::value;
    // Shared memory that leaves room for exactly cps CTAs per SM.
    const int smem = static_cast<int>(prop.sharedMemPerMultiprocessor / cps) - static_cast<int>(prop.reservedSharedMemPerBlock) - 128;
    auto k16 = mma_ilp_kernel<ILP, 16>;
    auto k64 = mma_ilp_kernel<ILP, 64>;
    for (auto k : {k16, k64}) CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k64, warps * 32, smem));
    if (blocks != cps) { std::fprintf(stderr, "mma_ilp: %d CTAs per SM, wanted %d\n", blocks, cps); std::exit(1); }
    const int grid = sm_count * cps;
    auto time_ms = [&](auto k, int steps) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      k<<<grid, warps * 32, smem>>>(d_sink, steps, 0u);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      return static_cast<double>(ms);
    };
    // Steps so that the N = 64 launch takes about 2 ms; the same steps for N = 16.
    const double probe = time_ms(k64, 64);
    int steps = static_cast<int>(64 * 2.0 / std::max(probe, 1e-4));
    steps = std::min(1 << 20, std::max(256, steps));
    std::vector<double> slope;
    for (int w = 0; w < warmup; ++w) { time_ms(k16, steps); time_ms(k64, steps); }
    for (int s = 0; s < samples; ++s) {
      const double t16 = time_ms(k16, steps), t64 = time_ms(k64, steps);
      slope.push_back((t64 - t16) * 1e6 / (static_cast<double>(steps) * 48.0 * ILP));
    }
    if (!first) std::printf(",");
    first = false;
    std::printf("\n    {\"ilp\": %d, \"num_warps\": %d, \"ctas_per_sm\": %d, \"steps\": %d, \"ns_per_mma_per_warp_samples\": [",
                ILP, warps, cps, steps);
    for (size_t i = 0; i < slope.size(); ++i) std::printf(i ? ", %.4f" : "%.4f", slope[i]);
    std::printf("]}");
  };
  auto each = [&](auto ilp_c) {
    for (int warps : {1, 2, 3, 4, 8})
      for (int cps : {1, 2, 3})
        if (warps * cps <= 16) run_point(ilp_c, warps, cps);
  };
  each(std::integral_constant<int, 1>{});
  each(std::integral_constant<int, 2>{});
  each(std::integral_constant<int, 4>{});
  each(std::integral_constant<int, 8>{});
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
