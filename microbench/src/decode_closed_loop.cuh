// Diagnostic, not a profile mode (docs/results/analysis/v0.3-gb10-decode-compression.md):
// the DRAM closed loop at decode-16's grids with decode's strided chunk access, against
// contiguous stages in the same process. Strided points run strided_mlp_cp_async_kernel,
// which already is a closed loop over STAGES buffers (wait for the oldest, refill its
// slot, no compute) with the phase advanced on wrap; contiguous points run
// closed_loop_cp_async_kernel with no compute. Both read with cp.async.cg, 256 threads.

template <int TOKENS>
static void launch_decode_contiguous(int vecs, int ctas, const uint4* src, uint4* sink,
                                     size_t mask, int steps) {
  const size_t smem = static_cast<size_t>(TOKENS) * vecs * 4096ULL;
  if (vecs == 1) closed_loop_cp_async_kernel<TOKENS, 1><<<ctas, 256, smem>>>(src, sink, mask, steps, 0);
  else closed_loop_cp_async_kernel<TOKENS, 2><<<ctas, 256, smem>>>(src, sink, mask, steps, 0);
}

template <int TOKENS, int SEG_U4>
static void launch_decode_strided(int vecs, int ctas, const uint4* src, uint4* sink,
                                  size_t mask, int steps) {
  const size_t smem = static_cast<size_t>(TOKENS) * vecs * 4096ULL;
  if (vecs == 1) strided_mlp_cp_async_kernel<TOKENS, 1, SEG_U4, 1024><<<ctas, 256, smem>>>(src, sink, mask, steps);
  else strided_mlp_cp_async_kernel<TOKENS, 2, SEG_U4, 1024><<<ctas, 256, smem>>>(src, sink, mask, steps);
}

template <int TOKENS>
static void launch_decode_point(int segment, int vecs, int ctas, const uint4* src, uint4* sink,
                                size_t mask, int steps) {
  if (segment == 0) launch_decode_contiguous<TOKENS>(vecs, ctas, src, sink, mask, steps);
  else if (segment == 128) launch_decode_strided<TOKENS, 8>(vecs, ctas, src, sink, mask, steps);
  else launch_decode_strided<TOKENS, 16>(vecs, ctas, src, sink, mask, steps);
}

static int run_decode_closed_loop(int device, int samples, int warmup, LaunchSchedule* schedule) {
  if (!schedule) {
    std::fprintf(stderr, "decode-closed-loop runs only under a launch schedule\n");
    return 2;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  // Registered windows: at least twice the pilot floor.
  schedule->launch_target_ns = 2.0 * LaunchSchedule::pilot_min_ns;

  const size_t dram_bytes = 4096ULL * LaunchSchedule::MiB;
  const size_t flush_bytes = 256ULL * LaunchSchedule::MiB;
  const size_t mask = dram_bytes / sizeof(uint4) - 1ULL;
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);
  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 41, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 42, flush_bytes));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Every point must be one concurrent wave.
  int resident = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &resident, closed_loop_cp_async_kernel<4, 2>, 256, 4 * 2 * 4096));
  if (sm_count * resident < 128) {
    std::fprintf(stderr, "decode-closed-loop: 128 CTAs do not fit one wave\n");
    return 2;
  }

  // Fixed order: grid, stage, access (contiguous, 128 B, 256 B), buffers ascending.
  std::printf("{\n  \"samples\": %d,\n  \"decode_closed_loop\": [", samples);
  bool first = true;
  for (int ctas : {96, 128}) {
    for (int stage : {4096, 8192}) {
      for (int segment : {0, 128, 256}) {
        for (int tokens : {1, 2, 4}) {
          const int vecs = stage / 4096;
          int steps = 0;
          auto launch_once = [&]() {
            if (tokens == 1) launch_decode_point<1>(segment, vecs, ctas, d_dram, d_sink, mask, steps);
            else if (tokens == 2) launch_decode_point<2>(segment, vecs, ctas, d_dram, d_sink, mask, steps);
            else launch_decode_point<4>(segment, vecs, ctas, d_dram, d_sink, mask, steps);
          };
          auto prepare = [&]() {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          };
          auto launch = [&](int count) {
            steps = count;
            launch_once();
          };
          const size_t bytes_per_step = static_cast<size_t>(ctas) * stage;
          // At most half the buffer per launch, as strided-mlp, so no line is read twice.
          const int max_steps = static_cast<int>((dram_bytes / 2ULL) / bytes_per_step);
          const int minimum = tokens * 4;
          const int initial = std::max(minimum, static_cast<int>(64ULL * LaunchSchedule::MiB / bytes_per_step));
          std::printf("%s\n    {\"access\": \"%s\", \"segment_bytes\": %d, \"stride_bytes\": %d, "
                      "\"active_ctas\": %d, \"stage_bytes\": %d, \"tokens\": %d",
                      first ? "" : ",", segment ? "strided" : "contiguous", segment,
                      segment ? 16384 : 0, ctas, stage, tokens);
          first = false;
          schedule->measure(initial, minimum, max_steps, bytes_per_step, dram_bytes, warmup, samples,
                            ev_start, ev_stop, prepare, launch);
          std::printf("}");
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
  return 0;
}
