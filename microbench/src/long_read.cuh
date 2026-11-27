// Optional long-launch protocol for existing read sweeps. A schedule lists point
// lengths in emission order, one per line. Pilot output carries all sizing trials.
struct LongReadSchedule {
  static constexpr size_t MiB = 1024ULL * 1024ULL;
  static constexpr double pilot_min_ns = 1250000.0;

  explicit LongReadSchedule(const std::string& path) : pilot(path == "pilot") {
    if (pilot) return;
    std::ifstream input(path);
    int length;
    while (input >> length) lengths.push_back(length);
    if (!input.eof() || lengths.empty()) fail("Invalid long-read schedule");
  }

  void finish() const {
    if (!pilot && next != lengths.size()) fail("Unused lengths in long-read schedule");
  }

  struct Timed {
    int length;
    std::vector<double> launch_ns;
  };

  // Size (pilot) or look up (frozen schedule) one point's length and time it. Prints
  // the pilot's sizing trials, then `"steps"` and `"launch_ns_samples"`; the caller
  // prints its own fields from the returned durations.
  template <class Prepare, class Launch>
  Timed time_point(int initial, int minimum, int maximum, int warmup, int samples,
                   cudaEvent_t start, cudaEvent_t stop, Prepare prepare, Launch launch) {
    int length = std::min(initial, maximum);
    if (!pilot) {
      if (next == lengths.size()) fail("Missing point in launch schedule");
      length = lengths[next++];
    }
    std::vector<double> durations;
    if (pilot) {
      std::printf(", \"sizing_trials\": [");
      for (int attempt = 0; ; ++attempt) {
        durations = collect(length, minimum, maximum, warmup, 3, start, stop, prepare, launch);
        if (attempt) std::printf(", ");
        std::printf("{\"steps\": %d, \"launch_ns_samples\": ", length);
        print_list(durations, 1.0, false);
        std::printf("}");
        const double fastest = *std::min_element(durations.begin(), durations.end());
        if (fastest >= pilot_min_ns || length == maximum || attempt == 7) break;
        const double scaled = std::ceil(length * (pilot_min_ns + 50000.0) / std::max(fastest, 1.0));
        length = static_cast<int>(std::min<double>(maximum, std::max(length + 1.0, scaled)));
      }
      std::printf("]");
    } else {
      durations = collect(length, minimum, maximum, warmup, samples, start, stop, prepare, launch);
    }
    std::printf(", \"steps\": %d, \"launch_ns_samples\": ", length);
    print_list(durations, 1.0, false);
    return {length, durations};
  }

  template <class Prepare, class Launch>
  void measure(int initial, int minimum, int maximum, size_t bytes_per_step,
               size_t buffer_bytes, int warmup, int samples, cudaEvent_t start,
               cudaEvent_t stop, Prepare prepare, Launch launch) {
    int length = std::min(initial, maximum);
    if (!pilot) {
      if (next == lengths.size()) fail("Missing point in long-read schedule");
      length = lengths[next++];
    }
    auto collect = [&](int count) {
      if (length < minimum || length > maximum) fail("Long-read length outside address bounds");
      for (int w = 0; w < warmup; ++w) {
        prepare();
        launch(length);
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<double> durations;
      for (int sample = 0; sample < count; ++sample) {
        prepare();
        CUDA_CHECK(cudaEventRecord(start));
        launch(length);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        durations.push_back(static_cast<double>(ms) * 1e6);
      }
      return durations;
    };

    std::vector<double> durations;
    if (pilot) {
      std::printf(", \"sizing_trials\": [");
      for (int attempt = 0; ; ++attempt) {
        durations = collect(3);
        if (attempt) std::printf(", ");
        std::printf("{\"steps\": %d, \"launch_ns_samples\": ", length);
        print_list(durations, 1.0, false);
        std::printf("}");
        const double fastest = *std::min_element(durations.begin(), durations.end());
        if (fastest >= pilot_min_ns || length == maximum || attempt == 7) break;
        const double scaled = std::ceil(length * (pilot_min_ns + 50000.0) / std::max(fastest, 1.0));
        length = static_cast<int>(std::min<double>(maximum, std::max(length + 1.0, scaled)));
      }
      std::printf("]");
    } else {
      durations = collect(samples);
    }
    const size_t transferred = static_cast<size_t>(length) * bytes_per_step;
    std::printf(", \"steps\": %d, \"transfer_bytes\": %zu, \"source_buffer_bytes\": %zu, "
                "\"launch_ns_samples\": ", length, transferred, buffer_bytes);
    print_list(durations, 1.0, false);
    std::printf(", \"gbps_samples\": ");
    print_list(durations, static_cast<double>(transferred), true);
  }

 private:
  template <class Prepare, class Launch>
  static std::vector<double> collect(int length, int minimum, int maximum, int warmup, int count,
                                     cudaEvent_t start, cudaEvent_t stop, Prepare prepare,
                                     Launch launch) {
    if (length < minimum || length > maximum) fail("Launch length outside bounds");
    for (int w = 0; w < warmup; ++w) {
      prepare();
      launch(length);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> durations;
    for (int sample = 0; sample < count; ++sample) {
      prepare();
      CUDA_CHECK(cudaEventRecord(start));
      launch(length);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms = 0;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      durations.push_back(static_cast<double>(ms) * 1e6);
    }
    return durations;
  }

  bool pilot;
  size_t next = 0;
  std::vector<int> lengths;

  [[noreturn]] static void fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(2);
  }

  static void print_list(const std::vector<double>& values, double numerator, bool inverse) {
    std::printf("[");
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) std::printf(", ");
      double value = values[i];
      if (inverse && value > 0) value = numerator / value;
      std::printf("%.6f", value);
    }
    std::printf("]");
  }
};
