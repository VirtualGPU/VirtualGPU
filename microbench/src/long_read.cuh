#include <functional>

// Optional long-launch protocol for any timed mode. A schedule lists point
// lengths in emission order, one per line. Pilot output carries all sizing trials.
// With `clock_probe` set, each point (time_point or measure) also records the SM
// clock (MHz) measured just before its first launch and just after its last.
struct LaunchSchedule {
  static constexpr size_t MiB = 1024ULL * 1024ULL;
  static constexpr double pilot_min_ns = 1250000.0;
  // What the pilot sizes the observed window to in the overload with `observe`. A mode may
  // raise it for points whose window varies between processes by a known factor (GB10's
  // L2 bistable band), so the frozen length still clears the capture floor in either state.
  double observed_target_ns = pilot_min_ns;
  // A mode may repeat its measured unit: with group k above 1, a capture sample is the median
  // of k consecutive launches (prepare and launch each time) and every launch is printed as
  // `launch_ns_each`; the pilot still sizes and prints single launches.
  int group = 1;
  std::function<double()> clock_probe;

  explicit LaunchSchedule(const std::string& path) : pilot(path == "pilot") {
    if (pilot) return;
    std::ifstream input(path);
    int length;
    while (input >> length) lengths.push_back(length);
    if (!input.eof() || lengths.empty()) fail("Invalid launch schedule");
  }

  // A point parameter derived at run time (from a clock calibration, say): the pilot uses
  // its own value and prints it with the point, a frozen schedule supplies the pilot's,
  // on the line before the point's length, so every round runs the same work.
  int parameter(int derived) {
    if (pilot) return derived;
    if (next == lengths.size()) fail("Missing parameter in launch schedule");
    return lengths[next++];
  }

  void finish() const {
    if (!pilot && next != lengths.size()) fail("Unused lengths in launch schedule");
  }

  struct Timed {
    int length;
    std::vector<double> launch_ns;
    std::vector<double> observed;
    std::vector<double> launch_each;  // every capture launch when group > 1, else empty
  };

  // The medians of consecutive groups of `k` values.
  static std::vector<double> group_medians(const std::vector<double>& v, int k) {
    std::vector<double> out;
    for (size_t i = 0; i + k <= v.size(); i += k) {
      std::vector<double> g(v.begin() + i, v.begin() + i + k);
      std::sort(g.begin(), g.end());
      out.push_back(k % 2 ? g[k / 2] : 0.5 * (g[k / 2 - 1] + g[k / 2]));
    }
    return out;
  }

  // Size (pilot) or look up (frozen schedule) one point's length and time it. Prints
  // the pilot's sizing trials, then `"steps"` and `"launch_ns_samples"`; the caller
  // prints its own fields from the returned durations.
  template <class Prepare, class Launch>
  Timed time_point(int initial, int minimum, int maximum, int warmup, int samples,
                   cudaEvent_t start, cudaEvent_t stop, Prepare prepare, Launch launch) {
    return run_point(initial, minimum, maximum, warmup, samples, start, stop, prepare, launch,
                     [] { return 0.0; }, false);
  }

  // As above for probes timed inside the kernel: `observe` runs after each timed launch,
  // outside the events, and returns the kernel's own timed window in ns. The capture
  // samples' windows come back in `observed`. The pilot sizes on the shorter of the launch
  // and the window (a probe's untimed warm-up makes the window shorter) and prints each
  // trial's windows.
  template <class Prepare, class Launch, class Observe>
  Timed time_point(int initial, int minimum, int maximum, int warmup, int samples,
                   cudaEvent_t start, cudaEvent_t stop, Prepare prepare, Launch launch,
                   Observe observe) {
    return run_point(initial, minimum, maximum, warmup, samples, start, stop, prepare, launch,
                     observe, true);
  }

  template <class Prepare, class Launch>
  Timed measure(int initial, int minimum, int maximum, size_t bytes_per_step,
                size_t buffer_bytes, int warmup, int samples, cudaEvent_t start,
                cudaEvent_t stop, Prepare prepare, Launch launch) {
    auto timed = time_point(initial, minimum, maximum, warmup, samples, start, stop,
                            prepare, launch);
    const size_t transferred = static_cast<size_t>(timed.length) * bytes_per_step;
    std::printf(", \"transfer_bytes\": %zu, \"source_buffer_bytes\": %zu, \"gbps_samples\": ",
                transferred, buffer_bytes);
    print_bandwidth_samples(timed, bytes_per_step);
    return timed;
  }

  static void print_bandwidth_samples(const Timed& timed, size_t bytes_per_step) {
    const double transferred = static_cast<double>(timed.length) * bytes_per_step;
    print_list(timed.launch_ns, transferred, true);
  }

 private:
  template <class Prepare, class Launch, class Observe>
  Timed run_point(int initial, int minimum, int maximum, int warmup, int samples,
                  cudaEvent_t start, cudaEvent_t stop, Prepare prepare, Launch launch,
                  Observe observe, bool windowed) {
    if (clock_probe) std::printf(", \"sm_mhz_before\": %.1f", clock_probe());
    int length = std::min(initial, maximum);
    if (!pilot) {
      if (next == lengths.size()) fail("Missing point in launch schedule");
      length = lengths[next++];
    }
    std::vector<double> durations, observed;
    if (pilot) {
      const double target = windowed ? std::max(pilot_min_ns, observed_target_ns) : pilot_min_ns;
      std::printf(", \"sizing_trials\": [");
      for (int attempt = 0; ; ++attempt) {
        observed.clear();
        durations = collect(length, minimum, maximum, warmup, 3, start, stop, prepare, launch,
                            observe, observed);
        if (attempt) std::printf(", ");
        std::printf("{\"steps\": %d, \"launch_ns_samples\": ", length);
        print_list(durations, 1.0, false);
        double fastest = *std::min_element(durations.begin(), durations.end());
        if (windowed) {
          std::printf(", \"window_ns_samples\": ");
          print_list(observed, 1.0, false);
          fastest = std::min(fastest, *std::min_element(observed.begin(), observed.end()));
        }
        std::printf("}");
        if (fastest >= target || length == maximum || attempt == 7) break;
        const double scaled = std::ceil(length * (target + 50000.0) / std::max(fastest, 1.0));
        length = static_cast<int>(std::min<double>(maximum, std::max(length + 1.0, scaled)));
      }
      std::printf("]");
    } else {
      durations = collect(length, minimum, maximum, warmup, samples * group, start, stop, prepare,
                          launch, observe, observed);
    }
    std::vector<double> each;
    if (!pilot && group > 1) {
      each = durations;
      durations = group_medians(each, group);
    }
    std::printf(", \"steps\": %d, \"launch_ns_samples\": ", length);
    print_list(durations, 1.0, false);
    if (!each.empty()) {
      std::printf(", \"launch_ns_each\": ");
      print_list(each, 1.0, false);
    }
    if (clock_probe) std::printf(", \"sm_mhz_after\": %.1f", clock_probe());
    return {length, durations, observed, each};
  }

  template <class Prepare, class Launch, class Observe>
  static std::vector<double> collect(int length, int minimum, int maximum, int warmup, int count,
                                     cudaEvent_t start, cudaEvent_t stop, Prepare prepare,
                                     Launch launch, Observe observe, std::vector<double>& observed) {
    if (length < minimum || length > maximum) fail("Launch length outside bounds");
    for (int w = 0; w < warmup; ++w) {
      prepare();
      launch(length);
    }
    CUDA_CHECK(cudaGetLastError());
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
      observed.push_back(observe());
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

// The name used by the read sweeps' protocol (microbench/long_mlp.py).
using LongReadSchedule = LaunchSchedule;
