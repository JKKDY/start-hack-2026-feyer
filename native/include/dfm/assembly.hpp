#pragma once

// Sparse matrix assembly from a SimulationPlan (native counterpart of the
// matrix-building part of upstream `simulate`).
//
// Semantics, per batch element b (upstream order):
//   1. parameters: base values, optimized values written in, links applied
//      (resolve_parameters), then the sweep values of element b;
//   2. kernels evaluated for every kernel row of the table;
//   3. values start from the plan's constant template, then each placement
//      SETS its slot (later placements overwrite earlier ones);
//   4. signal only: carrier couplings multiply their slot by -carrier;
//   5. signal only: lower-sideband blocks are complex-conjugated.
//
// Batch sizes: carrier = carrier sweep length CV; signal and noise =
// max(CV, SV). The carrier fields for the signal are an input
// (CV x carrier_ports), so signal assembly does not need the solver.
//
// Matrices come out in CSC form: a shared SparsePattern plus slot-aligned
// values, one row per batch element, and a dense right-hand side.

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/plan.hpp"
#include "dfm/sparse.hpp"
#include "dfm/types.hpp"

#include <vector>

namespace dfm {

enum class SystemKind : int { Carrier = 0, Signal = 1, Noise = 2 };

struct AssembledSystem {
  SparsePatternPtr pattern;
  int n = 0;
  MatrixC values;  ///< (batch, nnz)
  MatrixC rhs;     ///< (batch, n)
};

/// Upstream simulate preamble: parameters[optimized] = x[value_indices],
/// then parameter links. Returns the full parameter vector (length P).
VectorR resolve_parameters(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& optimized_values);

AssembledSystem assemble_carrier(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved);
/// `carrier` is (carrier_values, carrier_ports).
AssembledSystem assemble_signal(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved,
                                const Eigen::Ref<const MatrixC>& carrier);
AssembledSystem assemble_noise(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved);

// ------------------------------------------------------------ recorded
// Same assembly on CoDiPack types. Values and right-hand sides are
// interleaved active reals: element b, slot s -> values[b * 2 nnz + 2 s (+1)].

struct RecordedSystem {
  SparsePatternPtr pattern;
  int n = 0;
  int batch = 0;
  std::vector<ActiveReal> values;
  std::vector<ActiveReal> rhs;
};

std::vector<ActiveReal> resolve_parameters_recorded(const SimulationPlan& plan, const ActiveReal* optimized_values);

RecordedSystem record_carrier(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved);
/// `carrier` holds carrier_values * carrier_ports complex values, interleaved,
/// row-major over (value, port).
RecordedSystem record_signal(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved,
                             const ActiveReal* carrier);
RecordedSystem record_noise(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved);

/// Gradient of L = sum Re(conj(values_cot) * values) + sum Re(conj(rhs_cot) * rhs)
/// with respect to the optimized values and (signal only) the carrier fields.
struct AssemblyGradient {
  VectorR optimized;
  MatrixC carrier;  ///< (carrier_values, carrier_ports); empty unless kind == Signal
};
AssemblyGradient assembly_vjp(const SimulationPlan& plan, SystemKind kind,
                              const Eigen::Ref<const VectorR>& optimized_values,
                              const Eigen::Ref<const MatrixC>& carrier, const Eigen::Ref<const MatrixC>& values_cot,
                              const Eigen::Ref<const MatrixC>& rhs_cot);

}  // namespace dfm
