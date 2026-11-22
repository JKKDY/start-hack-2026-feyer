#pragma once

// Native simulation engine: forward evaluation and reverse-mode gradients.
//
// Ownership model
// ---------------
//   SetupSpec       immutable, shared_ptr; owned by every plan built from it
//   SimulationPlan  immutable, shared_ptr; shared by simulators and states
//   Simulator       owns reusable workspaces sized from its plan; NOT
//                   thread-safe (use one simulator per thread)
//   SimulationState result of forward_with_state; owns everything backward
//                   needs (tape segment, LU factors) and keeps the plan alive
//
// Python garbage-collection order therefore never matters: every object
// keeps alive what it points at.
//
// CoDiPack's reverse tape is one process-wide object. Only the most recent
// recorded state can be replayed (from any thread); an older state raises
// StateError instead of returning wrong gradients. Recording and replay are
// serialised by tape_mutex().
//
// Complex cotangent convention
// ----------------------------
// For a real loss L and a complex output z, the cotangent is
//   zbar = dL/dRe(z) + i * dL/dIm(z).
// JAX's convention is the complex conjugate of this; the Python layer
// documents the difference and converts where it offers JAX interop.

#include <memory>
#include <mutex>
#include <utility>

#include "dfm/plan.hpp"
#include "dfm/types.hpp"

namespace dfm {

/// Results in upstream layout (see Dimensions). `noise` is always 2-D here;
/// the Python layer applies upstream's squeeze.
struct ForwardOutputs {
  MatrixC carrier;
  MatrixC signal;
  MatrixR noise;
};

/// Cotangents for the three outputs. An empty (0x0) matrix means "zero".
struct OutputCotangents {
  MatrixC carrier;
  MatrixC signal;
  MatrixR noise;
};

class SimulationState;

/// Call after any code resets CoDiPack's global tape outside the Simulator
/// (e.g. assembly_vjp, test hooks): makes existing states report stale
/// instead of replaying a tape that no longer belongs to them.
void invalidate_recorded_states() noexcept;

/// Serialises use of CoDiPack's process-wide tape (recording and replay).
std::mutex& tape_mutex();

class Simulator {
 public:
  explicit Simulator(SimulationPlanPtr plan);
  ~Simulator();
  Simulator(Simulator&&) noexcept;
  Simulator& operator=(Simulator&&) noexcept;
  Simulator(const Simulator&) = delete;
  Simulator& operator=(const Simulator&) = delete;

  const SimulationPlan& plan() const noexcept { return *plan_; }
  const SimulationPlanPtr& plan_ptr() const noexcept { return plan_; }

  /// Forward pass only (no tape). `optimized_parameters` has length
  /// dimensions().optimized_values.
  ForwardOutputs forward(const Eigen::Ref<const VectorR>& optimized_parameters);

  /// Forward pass that also records what backward() needs.
  std::pair<ForwardOutputs, std::unique_ptr<SimulationState>> forward_with_state(
      const Eigen::Ref<const VectorR>& optimized_parameters);

  /// Vector-Jacobian product: gradient of <cotangents, outputs> with respect
  /// to the optimized parameters (length optimized_values).
  VectorR backward(const SimulationState& state, const OutputCotangents& cotangents);

 private:
  struct Workspace;
  SimulationPlanPtr plan_;
  std::unique_ptr<Workspace> workspace_;
};

/// Opaque evaluation-local state. Created only by forward_with_state.
class SimulationState {
 public:
  ~SimulationState();
  SimulationState(const SimulationState&) = delete;
  SimulationState& operator=(const SimulationState&) = delete;

  const SimulationPlanPtr& plan_ptr() const noexcept { return plan_; }
  /// False once a newer state was recorded on the same thread.
  bool is_current() const noexcept;

  struct Data;

 private:
  friend class Simulator;
  SimulationState(SimulationPlanPtr plan, std::unique_ptr<Data> data);

  SimulationPlanPtr plan_;
  std::unique_ptr<Data> data_;
};

}  // namespace dfm
