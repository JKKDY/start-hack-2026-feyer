#pragma once

// Minimal phase profiler for performance work. Enabled with the environment
// variable DFM_PROFILE=1 (checked once); otherwise every scope costs one
// predictable branch. Accumulated times are printed to stderr by report()
// (after each recorded forward pass and each backward pass) and reset; the
// recorded forward pass also prints CoDiPack's tape statistics.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace dfm::profile {

enum Phase : int {
  ResolveParameters = 0,
  CarrierAssembly,
  CarrierSolve,
  SignalAssembly,
  NoiseAssembly,
  SignalFactor,
  SignalSolve,
  Selection,
  NoiseSolves,
  NoiseProduct,
  NoiseReduction,
  ReverseSolve,
  ReverseProduct,
  TapeEvaluate,
  kPhaseCount
};

inline const char* name(int p) {
  static const char* names[] = {"resolve parameters", "carrier assembly", "carrier solve", "signal assembly",
                                "noise assembly", "signal factor", "signal solve (no noise)",
                                "selection vectors", "signal + noise chain", "noise product", "noise reduction",
                                "reverse: solve rules", "reverse: product rules", "tape evaluate (total)"};
  return names[p];
}

inline bool enabled() {
  static const bool on = [] {
    const char* v = std::getenv("DFM_PROFILE");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

inline std::array<double, kPhaseCount>& totals() {
  static std::array<double, kPhaseCount> t{};
  return t;
}

class Scope {
 public:
  explicit Scope(Phase p) : phase_(p), on_(enabled()) {
    if (on_) start_ = std::chrono::steady_clock::now();
  }
  ~Scope() {
    if (on_) totals()[phase_] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  Phase phase_;
  bool on_;
  std::chrono::steady_clock::time_point start_;
};

inline void report(const char* title) {
  if (!enabled()) return;
  std::fprintf(stderr, "[dfm profile] %s\n", title);
  for (int p = 0; p < kPhaseCount; ++p) {
    if (totals()[p] > 0) std::fprintf(stderr, "  %-26s %9.3f ms\n", name(p), totals()[p] * 1e3);
  }
  totals().fill(0.0);
}

}  // namespace dfm::profile
