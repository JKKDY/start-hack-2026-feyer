#pragma once

// Runtime planner: Setup snapshot + sweep/optimisation request -> immutable
// SimulationPlan.
//
// A plan is the native counterpart of upstream `run_build_step`: everything
// that depends only on geometry and on *which* parameters are swept or
// optimised (port numbering, component instructions, sparsity patterns,
// sweep values, detector mappings). It is built once per setup and shared,
// read-only, by any number of simulators. All sizes are runtime values;
// nothing is specialised at compile time, so one binary serves every
// supported geometry.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "dfm/setup_spec.hpp"
#include "dfm/types.hpp"

namespace dfm {

/// One optimisation variable. A single pair, or several tied pairs that all
/// receive the same value (upstream's grouped list entries).
using OptimizationTarget = std::vector<ParameterPair>;

enum class SignalSolver : int {
  Packed = 0,  ///< sparse signal solve (upstream updated default "packed")
  Dense,       ///< dense reference path (upstream original behaviour)
};

struct PlanRequest {
  std::vector<ParameterPair> changing_pairs;         ///< e.g. {("f", "frequency")}
  MatrixR changing_values;                           ///< (len(changing_pairs), num_values)
  std::vector<OptimizationTarget> optimization_targets;
  SignalSolver signal_solver = SignalSolver::Packed;
  std::optional<bool> pack_noise;                    ///< None = planner decides
};

/// Runtime array dimensions of a plan. Output shapes follow upstream:
///   carrier (carrier_ports, carrier_values)
///   signal  (signal_variables, signal_values)
///   noise   (noise_detectors, signal_values) before the Python-side squeeze
struct Dimensions {
  Index parameters = 0;         ///< P, length of the expanded parameter vector
  Index optimized_values = 0;   ///< OPV, length of `optimized_parameters`
  Index carrier_ports = 0;
  Index carrier_values = 0;
  Index signal_variables = 0;
  Index signal_values = 0;
  Index noise_detectors = 0;
};

/// Upstream `build` metadata, in upstream order (upstream `run` returns
/// these four index arrays followed by `port_to_index`).
struct PortMetadata {
  VectorI detector_indices;
  VectorI mirror_indices;
  VectorI beamsplitter_indices;
  VectorI isolator_indices;
  /// "node.port.in" / "node.port.out" -> carrier index, in upstream order.
  std::vector<std::pair<std::string, Index>> port_to_index;
};

using MatrixI = Eigen::Matrix<std::int64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// One named array of the upstream `simulation_arrays` contract.
/// std::monostate stands for Python None. A non-empty `shape` reshapes the
/// array on export (e.g. noise_selection_vectors is (Q, 1, N)).
using ArrayValue = std::variant<std::monostate, VectorI, VectorR, VectorC, MatrixI, MatrixR, MatrixC>;
struct NamedArray {
  std::string name;
  ArrayValue value;
  std::vector<Index> shape;
};

class SimulationPlan {
 public:
  ~SimulationPlan();
  SimulationPlan(const SimulationPlan&) = delete;
  SimulationPlan& operator=(const SimulationPlan&) = delete;

  const SetupSpec& setup() const noexcept { return *setup_; }
  const PlanRequest& request() const noexcept { return request_; }
  const Dimensions& dimensions() const noexcept { return dimensions_; }
  const PortMetadata& metadata() const noexcept { return metadata_; }

  /// Upstream-compatible `simulation_arrays` (names as `simulate` arguments).
  /// Used only for compatibility and inspection, never by the native engine.
  std::vector<NamedArray> export_simulation_arrays() const;

  /// Planner internals (instructions, sparsity patterns, ...). Opaque to
  /// users; the simulator reads it directly (no virtual dispatch).
  struct Internals;
  const Internals& internals() const noexcept { return *internals_; }

 private:
  friend std::shared_ptr<const SimulationPlan> build_plan(SetupSpecPtr, PlanRequest);
  SimulationPlan(SetupSpecPtr setup, PlanRequest request);
  friend struct PlanBuilder;

  SetupSpecPtr setup_;
  PlanRequest request_;
  Dimensions dimensions_;
  PortMetadata metadata_;
  std::unique_ptr<Internals> internals_;
};

using SimulationPlanPtr = std::shared_ptr<const SimulationPlan>;

/// Validates the request against the setup, then builds the plan.
/// Throws ValidationError for bad input (unknown pairs, wrong shapes).
SimulationPlanPtr build_plan(SetupSpecPtr setup, PlanRequest request);

/// Request validation on its own (also called by build_plan).
void validate_request(const SetupSpec& setup, const PlanRequest& request);

}  // namespace dfm
