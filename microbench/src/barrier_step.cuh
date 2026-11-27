// barrier-step: the cost of barriers per loop step at fixed work (diagnostic, not part of a
// profile). Registration: docs/results/analysis/v0.3-barrier-sync-plan.md.
//
// One loop iteration per step: W warp instructions of independent FP32 FMA work per warp
// (8 chains per thread), then B `bar.sync 0`. The step loop is never unrolled, so the
// assembler output has exactly B BAR.SYNC and W FFMA per iteration
// (microbench/barrier_step.py sass checks it per instance).

template <int B, int W>
__global__ void barrier_step_kernel(float* __restrict__ sink, int steps) {
  static_assert(W % 8 == 0, "work comes in rounds of 8 chains");
  const float a = 1.0001f + 1e-7f * static_cast<float>(threadIdx.x);
  const float b = 0.9999f;
  float r0 = 0.1f, r1 = 0.2f, r2 = 0.3f, r3 = 0.4f;
  float r4 = 0.5f, r5 = 0.6f, r6 = 0.7f, r7 = 0.8f;
  int count = 0;
  #pragma unroll 1
  for (int s = 0; s < steps; ++s) {
    #pragma unroll
    for (int i = 0; i < W / 8; ++i) {
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r0) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r1) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r2) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r3) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r4) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r5) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r6) : "f"(a), "f"(b));
      asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r7) : "f"(a), "f"(b));
    }
    #pragma unroll
    for (int j = 0; j < B; ++j) {
      asm volatile("bar.sync 0;" ::: "memory");
    }
    asm volatile("add.s32 %0, %0, 1;" : "+r"(count));
  }
  const float sum = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + static_cast<float>(count);
  if (sum == -1.0f) sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

template <int B>
static void launch_barrier_step(int work, int grid, int threads, float* sink, int steps) {
  if (work == 0) barrier_step_kernel<B, 0><<<grid, threads>>>(sink, steps);
  else if (work == 64) barrier_step_kernel<B, 64><<<grid, threads>>>(sink, steps);
  else barrier_step_kernel<B, 512><<<grid, threads>>>(sink, steps);
}

static void launch_barrier_step(int barriers, int work, int grid, int threads, float* sink,
                                int steps) {
  switch (barriers) {
    case 0: launch_barrier_step<0>(work, grid, threads, sink, steps); break;
    case 1: launch_barrier_step<1>(work, grid, threads, sink, steps); break;
    case 2: launch_barrier_step<2>(work, grid, threads, sink, steps); break;
    case 4: launch_barrier_step<4>(work, grid, threads, sink, steps); break;
    case 8: launch_barrier_step<8>(work, grid, threads, sink, steps); break;
    default: launch_barrier_step<14>(work, grid, threads, sink, steps); break;
  }
}

static int run_barrier_step(int device, int samples, int warmup, LaunchSchedule* schedule) {
  if (!schedule) {
    std::fprintf(stderr, "barrier-step runs only under a launch schedule (pilot or frozen)\n");
    return 2;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  float* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::printf("{\n  \"samples\": %d,\n  \"sm_count\": %d,\n  \"points\": [", samples, sm_count);
  bool first = true;
  for (int barriers : {0, 1, 2, 4, 8, 14}) {
    for (int work : {0, 64, 512}) {
      for (int warps : {4, 8}) {
        for (int grid : {1, sm_count}) {
          std::printf("%s\n    {\"barriers_per_step\": %d, \"work_per_step\": %d, "
                      "\"num_warps\": %d, \"grid\": %d",
                      first ? "" : ",", barriers, work, warps, grid);
          first = false;
          auto timed = schedule->time_point(
              4096, 1, 1 << 30, warmup, samples, start, stop, [] {},
              [&](int steps) { launch_barrier_step(barriers, work, grid, warps * 32, sink, steps); });
          std::printf(", \"ns_per_step_samples\": [");
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
  CUDA_CHECK(cudaFree(sink));
  return 0;
}
