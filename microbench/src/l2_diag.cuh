#pragma once

// Diagnostic, not a profile mode: GB10's L2 capacity sweep near 1.5 times L2 per buffer and
// fill (docs/results/analysis/v0.3-gb10-first-capture-l2-diagnosis.md, section 3). Six
// separate buffers per process, each read at the sweep's sizes with its kernel and the
// capture's frozen passes, then refilled in place with incompressible data and read again.

__global__ void splitmix_fill_kernel(uint64_t* __restrict__ dst, size_t n_words, uint64_t seed) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_words;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint64_t z = seed + (i + 1) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    dst[i] = z ^ (z >> 31);
  }
}

static int run_l2_buffer_diag(int device, int samples, bool clock_probe) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  constexpr int kBuffers = 6;
  constexpr size_t kBufferBytes = 64ULL * 1024ULL * 1024ULL;
  // Sizes and passes per launch as frozen for GB10's capacity sweep in the 42e039d capture.
  const std::vector<std::pair<size_t, int>> sizes = {
      {20ULL << 20, 125}, {28ULL << 20, 80}, {36ULL << 20, 43}, {48ULL << 20, 10}, {60ULL << 20, 8}};
  const size_t one_pass_ws = 36ULL << 20;
  const size_t flush_bytes = std::max<size_t>(3ULL * prop.l2CacheSize, 128ULL << 20);
  const int grid = sm_count * 4;
  const int threads = 512;

  std::vector<uint4*> bufs(kBuffers, nullptr);
  for (auto& b : bufs) {
    CUDA_CHECK(cudaMalloc(&b, kBufferBytes));
    CUDA_CHECK(cudaMemset(b, 3, kBufferBytes));
  }
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_flush, 12, flush_bytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1, e2;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  auto mhz = [&]() { return clock_probe ? 1e3 * measure_cycles_per_ns(e0, e1) : 0.0; };
  auto list = [](const char* key, const std::vector<double>& v) {
    std::printf(", \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%.3f", i ? ", " : "", v[i]);
    std::printf("]");
  };
  auto elapsed_ns = [](cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return static_cast<double>(ms) * 1e6;
  };

  std::printf("{\n  \"samples\": %d,\n  \"buffers\": %d,\n  \"buffer_bytes\": %zu,\n  \"l2_bytes\": %d,\n",
              samples, kBuffers, kBufferBytes, prop.l2CacheSize);
  bool first_steady = true;
  std::printf("  \"steady\": [");
  auto steady_visit = [&](const char* fill, int visit, int b) {
    for (const auto& [ws, passes] : sizes) {
      const size_t n_vec4 = ws / sizeof(uint4);
      for (int w = 0; w < 2; ++w) l2_read_cg_kernel<<<grid, threads>>>(bufs[b], d_sink, n_vec4, 2);
      CUDA_CHECK(cudaDeviceSynchronize());
      const double before = mhz();
      std::vector<double> ns, gbps;
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(e0));
        l2_read_cg_kernel<<<grid, threads>>>(bufs[b], d_sink, n_vec4, passes);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        ns.push_back(elapsed_ns(e0, e1));
        gbps.push_back(static_cast<double>(ws) * passes / ns.back());
      }
      const double after = mhz();
      std::printf("%s\n    {\"fill\": \"%s\", \"visit\": %d, \"buffer\": %d, \"working_set_bytes\": %zu, "
                  "\"passes\": %d, \"sm_mhz_before\": %.1f, \"sm_mhz_after\": %.1f",
                  first_steady ? "" : ",", fill, visit, b + 1, ws, passes, before, after);
      first_steady = false;
      list("launch_ns_samples", ns);
      list("gbps_samples", gbps);
      std::printf("}");
    }
  };
  std::vector<std::string> once_rows;
  auto one_pass = [&](const char* fill, int b) {
    const size_t n_vec4 = one_pass_ws / sizeof(uint4);
    std::vector<double> p1, p2;
    for (int s = 0; s < samples; ++s) {
      l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_bytes / sizeof(uint4), 1);
      CUDA_CHECK(cudaEventRecord(e0));
      l2_read_cg_kernel<<<grid, threads>>>(bufs[b], d_sink, n_vec4, 1);
      CUDA_CHECK(cudaEventRecord(e1));
      l2_read_cg_kernel<<<grid, threads>>>(bufs[b], d_sink, n_vec4, 1);
      CUDA_CHECK(cudaEventRecord(e2));
      CUDA_CHECK(cudaEventSynchronize(e2));
      p1.push_back(static_cast<double>(one_pass_ws) / elapsed_ns(e0, e1));
      p2.push_back(static_cast<double>(one_pass_ws) / elapsed_ns(e1, e2));
    }
    char head[160];
    std::snprintf(head, sizeof(head), "\n    {\"fill\": \"%s\", \"buffer\": %d, \"working_set_bytes\": %zu",
                  fill, b + 1, one_pass_ws);
    std::string row = head;
    for (auto [key, v] : {std::pair<const char*, std::vector<double>*>{"pass1_gbps_samples", &p1},
                          {"pass2_gbps_samples", &p2}}) {
      row += std::string(", \"") + key + "\": [";
      for (size_t i = 0; i < v->size(); ++i) {
        char x[32];
        std::snprintf(x, sizeof(x), "%s%.3f", i ? ", " : "", (*v)[i]);
        row += x;
      }
      row += "]";
    }
    once_rows.push_back(row + "}");
  };

  for (const char* fill : {"constant", "random"}) {
    if (std::string(fill) == "random") {
      for (int b = 0; b < kBuffers; ++b)
        splitmix_fill_kernel<<<sm_count * 4, 256>>>(reinterpret_cast<uint64_t*>(bufs[b]),
                                                     kBufferBytes / sizeof(uint64_t), 0x5EEDULL + b);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    for (int b = 0; b < kBuffers; ++b) steady_visit(fill, 1, b);
    for (int b = kBuffers - 1; b >= 0; --b) steady_visit(fill, 2, b);
    for (int b = 0; b < kBuffers; ++b) one_pass(fill, b);
  }
  std::printf("\n  ],\n  \"one_pass\": [");
  for (size_t i = 0; i < once_rows.size(); ++i) std::printf("%s%s", i ? "," : "", once_rows[i].c_str());
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
  for (auto b : bufs) CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
