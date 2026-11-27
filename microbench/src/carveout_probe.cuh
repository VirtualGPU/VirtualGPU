#pragma once

// Diagnostic, not a profile mode: which carveout a kernel with no shared memory request
// gets on GB10 (docs/results/analysis/v0.3-gb10-carveout-probe-registration.md). Step 1
// runs the L1 carveout pointer chase in four launch shapes; step 2 locates the
// spill-latency kernel's own L1 to L2 transition from the warps per CTA, in five new
// contexts. Runs only under a launch schedule.

template <int Tag>
__global__ void carveout_probe_chase_kernel(const uint32_t* __restrict__ next_idx,
                                            uint32_t* __restrict__ sink, int smem_bytes,
                                            int warmup_steps, int timed_steps,
                                            double* __restrict__ out_cycles) {
  // The pointer chase of l1_carveout_pchase_kernel, one instantiation per launch shape.
  extern __shared__ uint32_t cp_smem[];
  if (threadIdx.x == 0) {
    uint32_t smem_touch = 0u;
    if (smem_bytes >= 4) {
      cp_smem[0] = 1u;
      smem_touch = cp_smem[0];
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
    if ((cur ^ smem_touch) == 0xdeadbeefu) sink[0] = cur;
  }
}

struct ProbeShape {
  const char* name;
  int threads;
  int smem_request;
  int tag;
};

static void carveout_probe_launch(const ProbeShape& s, const uint32_t* ring, uint32_t* sink,
                                  int w_steps, int t_steps, double* out) {
  const size_t smem = static_cast<size_t>(s.smem_request);
  switch (s.tag) {
    case 0: carveout_probe_chase_kernel<0><<<1, s.threads, smem>>>(ring, sink, s.smem_request, w_steps, t_steps, out); break;
    case 1: carveout_probe_chase_kernel<1><<<1, s.threads, smem>>>(ring, sink, s.smem_request, w_steps, t_steps, out); break;
    case 2: carveout_probe_chase_kernel<2><<<1, s.threads, smem>>>(ring, sink, s.smem_request, w_steps, t_steps, out); break;
    default: carveout_probe_chase_kernel<3><<<1, s.threads, smem>>>(ring, sink, s.smem_request, w_steps, t_steps, out); break;
  }
}

static void carveout_probe_spill(int frame_bytes, int sm_count, int threads, uint32_t* sink,
                                 int steps) {
  constexpr int kReloads = 128, kStride = 1;
  switch (frame_bytes) {
    case 128: spill_reload_latency_kernel<32, false><<<sm_count, threads>>>(sink, kReloads, kStride, steps); break;
    case 512: spill_reload_latency_kernel<128, false><<<sm_count, threads>>>(sink, kReloads, kStride, steps); break;
    default: spill_reload_latency_kernel<1024, false><<<sm_count, threads>>>(sink, kReloads, kStride, steps); break;
  }
}

static int run_carveout_probe(int device, int samples, int warmup, LaunchSchedule* schedule) {
  if (!schedule) {
    std::fprintf(stderr, "carveout-probe runs only under a launch schedule\n");
    return 2;
  }
  const bool probe_clock = static_cast<bool>(schedule->clock_probe);
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Step 1: the pointer chase in four launch shapes, interleaved per working set.
  const ProbeShape shapes[] = {
      {"R0", 1024, 0, 0}, {"R8", 1024, 8192, 1}, {"R16", 1024, 16384, 2}, {"S", 128, 0, 3}};
  std::vector<int> ws_kb = {16, 32, 64};
  for (int kb = 96; kb <= 160; kb += 4) ws_kb.push_back(kb);
  for (int kb : {192, 224, 256, 320}) ws_kb.push_back(kb);
  const int stride_u32 = 32;  // one 128 B line per step
  const size_t ring_bytes = static_cast<size_t>(ws_kb.back()) * 1024;
  std::vector<uint32_t> h_ring(ring_bytes / sizeof(uint32_t), 0u);
  uint32_t *d_ring = nullptr, *d_sink = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ring, ring_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  if (probe_clock) schedule->clock_probe = [&]() { return 1e3 * measure_cycles_per_ns(e0, e1); };
  const double cycles_per_ns = measure_cycles_per_ns(e0, e1);
  auto list = [](const char* key, const std::vector<double>& v, double scale) {
    std::printf(", \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%.4f", i ? ", " : "", v[i] * scale);
    std::printf("]");
  };

  // Every timed window is sized to twice the pilot floor (second registration,
  // v0.3-gb10-carveout-probe-reregistration.md), with no exception.
  const double target = schedule->observed_target_ns;
  schedule->observed_target_ns = 2.0 * LaunchSchedule::pilot_min_ns;
  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"l1_cliff\": [", samples, cycles_per_ns);
  bool first = true;
  for (int kb : ws_kb) {
    const int lines = kb * 1024 / 128;
    for (int i = 0; i < lines; ++i)
      h_ring[static_cast<size_t>(i) * stride_u32] = static_cast<uint32_t>(((i + 1) % lines) * stride_u32);
    CUDA_CHECK(cudaMemcpy(d_ring, h_ring.data(), static_cast<size_t>(kb) * 1024, cudaMemcpyHostToDevice));
    const int w_steps = lines * 3;
    for (const auto& s : shapes) {
      int current = 0;
      std::printf("%s\n    {\"config\": \"%s\", \"threads\": %d, \"smem_request_bytes\": %d, \"ws_bytes\": %d",
                  first ? "" : ",", s.name, s.threads, s.smem_request, kb * 1024);
      first = false;
      auto timed = schedule->time_point(
          std::max(lines * 4, 2048), 1, 1 << 26, std::max(warmup, 1), samples, e0, e1, [] {},
          [&](int length) {
            current = length;
            carveout_probe_launch(s, d_ring, d_sink, w_steps, length, d_out);
          },
          [&]() {
            double cyc = 0.0;
            CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
            return cyc * current / cycles_per_ns;
          });
      list("window_ns_samples", timed.observed, 1.0);
      list("latency_ns_samples", timed.observed, 1.0 / timed.length);
      std::printf("}");
    }
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaFree(d_ring));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out));

  // Step 2: the spill-latency kernel, batched, 128 reloads, in five new contexts.
  std::printf("\n  ],\n  \"spill_transition\": [");
  first = true;
  for (int context = 1; context <= 5; ++context) {
    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaSetDevice(device));
    uint32_t* sink = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(sm_count) * 1024 * sizeof(uint32_t)));
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    if (probe_clock) schedule->clock_probe = [&]() { return 1e3 * measure_cycles_per_ns(e0, e1); };
    for (int warps = 6; warps <= 10; ++warps) {
      for (int frame : {4096, 512, 128}) {
        std::printf("%s\n    {\"context\": %d, \"warps\": %d, \"frame_bytes_per_thread\": %d, "
                    "\"reload_count\": 128, \"footprint_bytes_per_sm\": %d",
                    first ? "" : ",", context, warps, frame, frame * warps * 32);
        first = false;
        // The timed window of a spill point is its launch, reported to the same overload.
        auto timed = schedule->time_point(
            256, 1, 1 << 24, std::max(warmup, 1), samples, e0, e1, [] {},
            [&](int length) { carveout_probe_spill(frame, sm_count, warps * 32, sink, length); },
            [&]() {
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
              return static_cast<double>(ms) * 1e6;
            });
        list("step_ns_samples", timed.launch_ns, 1.0 / timed.length);
        std::printf("}");
      }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(sink));
  }
  schedule->clock_probe = nullptr;  // its events are gone with the last context
  schedule->observed_target_ns = target;
  std::printf("\n  ]\n}\n");
  return 0;
}
