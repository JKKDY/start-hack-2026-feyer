#pragma once

// Internal data of a SimulationPlan, shared by the planner (which fills it)
// and the assembler (which reads it). Not part of the public API.
//
// Everything mirrors upstream build.py / pairs_to_arrays / prepare_arrays:
// - `InstructionTable` is one of upstream's carrier / signal / noise array
//   sets: kernel rows (function + 7 parameter indices) and placements
//   (kernel output -> matrix position; column -1 is the right-hand side).
// - `MatrixLayout` is the derived sparse structure: pattern, constant
//   template values, the slot of every placement, carrier couplings and the
//   slots that are complex-conjugated.

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "dfm/components.hpp"
#include "dfm/link.hpp"
#include "dfm/plan.hpp"
#include "dfm/sparse.hpp"
#include "dfm/types.hpp"

namespace dfm {

inline constexpr int kRhsColumn = -1;

struct InstructionTable {
  // Kernel rows.
  std::vector<ComponentFunction> functions;
  std::vector<std::array<int, kComponentInputWidth>> inputs;
  // Placements: value = output[output_rows[p]][output_cols[p]] at (rows[p], cols[p]).
  std::vector<int> output_rows;
  std::vector<int> output_cols;
  std::vector<int> rows;
  std::vector<int> cols;
  // Signal only: after placement, value(coupling_rows[c], coupling_cols[c]) *= -carrier[carrier_indices[c]].
  std::vector<int> carrier_indices;
  std::vector<int> coupling_rows;
  std::vector<int> coupling_cols;
  bool is_dummy = false;  ///< upstream's placeholder arrays for an empty table
};

struct MatrixLayout {
  int n = 0;
  SparsePatternPtr pattern;
  std::vector<Complex> template_values;  ///< slot-aligned constant values
  std::vector<Complex> template_rhs;     ///< length n
  std::vector<int> placement_slots;      ///< per placement: slot, or -1 when it targets the RHS
  std::vector<int> coupling_slots;       ///< per coupling: slot, or -1 when it targets the RHS
  std::vector<int> conjugate_slots;      ///< signal lower-sideband blocks
  std::vector<int> conjugate_rhs_rows;
  /// Kernel rows of the table grouped by function, for batched evaluation.
  std::array<std::vector<int>, kNumComponentFunctions> rows_by_function;
};

struct SimulationPlan::Internals {
  // Parameters (upstream `parameters[0]`, real here) and names.
  std::vector<double> parameters;
  std::vector<std::string> parameter_names;

  // Links: parameters[indices_to_link[i]] = f_i(parameters[linked_indices[i]]).
  // Upstream adds an identity placeholder when there are no links.
  std::vector<int> linked_indices;
  std::vector<int> indices_to_link;
  std::vector<LinkFunction> link_functions;

  // Optimisation: parameters[optimized_parameter_indices[i]] = x[optimized_value_indices[i]].
  std::vector<int> optimized_parameter_indices;
  std::vector<int> optimized_value_indices;
  bool default_optimization = false;  ///< no optimisation pairs: upstream maps x[0] -> parameter 0

  // Sweeps (upstream prepare_arrays, including its defaults).
  std::vector<int> carrier_changing_indices;
  MatrixR carrier_changing_values;  ///< (len(indices), CV)
  std::vector<int> signal_changing_indices;
  MatrixR signal_changing_values;   ///< (len(indices), SV)

  int system_size = 0;  ///< S: carrier unknowns
  int signal_size = 0;  ///< explicit signal unknowns (>= 1)

  InstructionTable carrier;
  InstructionTable signal;
  InstructionTable noise;

  MatrixLayout carrier_layout;
  MatrixLayout signal_layout;
  MatrixLayout noise_layout;

  // Quantum-noise readout (used by the simulation pipeline).
  MatrixC noise_selection;  ///< (Q, N)
  std::vector<int> qhd_parameter_indices;
  std::vector<std::array<int, 3>> qhd_placing_indices;

  std::vector<std::pair<std::string, int>> port_to_index;
};

}  // namespace dfm
