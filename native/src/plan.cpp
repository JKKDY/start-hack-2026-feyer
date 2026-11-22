#include "dfm/plan.hpp"

#include <string>
#include <utility>

#include "dfm/errors.hpp"
#include "dfm/plan_internals.hpp"

namespace dfm {

SimulationPlan::SimulationPlan(SetupSpecPtr setup, PlanRequest request)
    : setup_(std::move(setup)), request_(std::move(request)), internals_(std::make_unique<Internals>()) {}

SimulationPlan::~SimulationPlan() = default;

namespace {

VectorI to_vector(const std::vector<int>& v) {
  VectorI out(static_cast<Index>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) out(static_cast<Index>(i)) = v[i];
  return out;
}

MatrixI two_rows(const std::vector<int>& a, const std::vector<int>& b) {
  MatrixI out(2, static_cast<Index>(a.size()));
  for (std::size_t i = 0; i < a.size(); ++i) {
    out(0, static_cast<Index>(i)) = a[i];
    out(1, static_cast<Index>(i)) = b[i];
  }
  return out;
}

// Dense upstream matrix (n x (n + cols_extra)) from a sparse template.
MatrixC dense(const MatrixLayout& l, bool with_rhs) {
  MatrixC m = MatrixC::Zero(l.n, l.n + (with_rhs ? 1 : 0));
  const SparsePattern& p = *l.pattern;
  const auto cols = p.slot_columns();
  for (int s = 0; s < p.nnz(); ++s) {
    m(p.row_idx()[static_cast<std::size_t>(s)], cols[static_cast<std::size_t>(s)]) = l.template_values[static_cast<std::size_t>(s)];
  }
  if (with_rhs) {
    for (int r = 0; r < l.n; ++r) m(r, l.n) = l.template_rhs[static_cast<std::size_t>(r)];
  }
  return m;
}

void export_table(std::vector<NamedArray>& out, const std::string& prefix, const InstructionTable& t, bool signal) {
  MatrixI inputs;
  if (t.is_dummy) {
    inputs = MatrixI::Zero(1, 3);  // upstream placeholder [[0, 0, 0]]
  } else {
    inputs.resize(static_cast<Index>(t.inputs.size()), kComponentInputWidth);
    for (std::size_t r = 0; r < t.inputs.size(); ++r) {
      for (int k = 0; k < kComponentInputWidth; ++k) inputs(static_cast<Index>(r), k) = t.inputs[r][k];
    }
  }
  std::vector<int> functions;
  for (ComponentFunction f : t.functions) functions.push_back(static_cast<int>(f));
  out.push_back({prefix + "_function_input_indices", inputs, {}});
  out.push_back({prefix + "_function_indices", to_vector(functions), {}});
  out.push_back({prefix + "_output_indices", two_rows(t.output_rows, t.output_cols), {}});
  out.push_back({prefix + (prefix == "noise" ? "_system_matrix_indices" : "_matrix_indices"),
                 two_rows(t.rows, t.cols), {}});
  if (signal) {
    out.push_back({"signal_carrier_indices", to_vector(t.carrier_indices), {}});
    out.push_back({"signal_carrier_placing_indices", two_rows(t.coupling_rows, t.coupling_cols), {}});
  }
}

}  // namespace

std::vector<NamedArray> SimulationPlan::export_simulation_arrays() const {
  const Internals& in = *internals_;
  std::vector<NamedArray> out;

  if (in.default_optimization) {
    VectorR x(1);
    x(0) = in.parameters[0];
    out.push_back({"optimized_parameters", x, {}});
  } else {
    out.push_back({"optimized_parameters", std::monostate{}, {}});  // upstream: None, set by the caller
  }
  out.push_back({"optimized_parameter_indices", to_vector(in.optimized_parameter_indices), {}});
  out.push_back({"optimized_value_indices", to_vector(in.optimized_value_indices), {}});
  out.push_back({"carrier_changing_parameter_indices", to_vector(in.carrier_changing_indices), {}});
  out.push_back({"carrier_changing_values", in.carrier_changing_values, {}});
  out.push_back({"signal_changing_parameter_indices", to_vector(in.signal_changing_indices), {}});
  out.push_back({"signal_changing_values", in.signal_changing_values, {}});

  MatrixC parameters(1, static_cast<Index>(in.parameters.size()));
  for (std::size_t i = 0; i < in.parameters.size(); ++i) parameters(0, static_cast<Index>(i)) = in.parameters[i];
  out.push_back({"parameters", parameters, {}});
  out.push_back({"carrier_matrix", dense(in.carrier_layout, true), {}});
  out.push_back({"signal_matrix", dense(in.signal_layout, true), {}});
  out.push_back({"noise_matrix", MatrixC(MatrixC::Zero(in.noise_layout.n, in.noise_layout.n)), {}});
  out.push_back({"noise_selection_vectors", in.noise_selection, {in.noise_selection.rows(), 1, in.noise_selection.cols()}});
  out.push_back({"qhd_parameter_indices", to_vector(in.qhd_parameter_indices), {}});
  if (in.qhd_placing_indices.empty()) {
    out.push_back({"qhd_placing_indices", VectorI(0), {}});
  } else {
    MatrixI placing(static_cast<Index>(in.qhd_placing_indices.size()), 3);
    for (std::size_t i = 0; i < in.qhd_placing_indices.size(); ++i) {
      for (int k = 0; k < 3; ++k) placing(static_cast<Index>(i), k) = in.qhd_placing_indices[i][k];
    }
    out.push_back({"qhd_placing_indices", placing, {}});
  }
  std::vector<int> link_functions;
  for (LinkFunction f : in.link_functions) link_functions.push_back(static_cast<int>(f));
  out.push_back({"linked_indices", to_vector(in.linked_indices), {}});
  out.push_back({"linking_function_indices", to_vector(link_functions), {}});
  out.push_back({"indices_to_link", to_vector(in.indices_to_link), {}});
  export_table(out, "carrier", in.carrier, false);
  export_table(out, "signal", in.signal, true);
  export_table(out, "noise", in.noise, false);
  return out;
}

namespace {

std::string pair_text(const ParameterPair& p) { return "('" + p.first + "', '" + p.second + "')"; }

}  // namespace

void validate_request(const SetupSpec& setup, const PlanRequest& request) {
  for (const ParameterPair& p : request.changing_pairs) {
    if (!setup.has_property(p)) {
      throw ValidationError("Changing pair " + pair_text(p) + " does not name a property of the setup.");
    }
  }
  const Index n_pairs = static_cast<Index>(request.changing_pairs.size());
  if (n_pairs > 0) {
    if (request.changing_values.rows() != n_pairs) {
      throw ValidationError("changing_values must have shape (len(changing_pairs), num_values) = (" +
                            std::to_string(n_pairs) + ", N); got (" + std::to_string(request.changing_values.rows()) +
                            ", " + std::to_string(request.changing_values.cols()) + ").");
    }
    if (request.changing_values.cols() == 0) {
      throw ValidationError("changing_values must contain at least one sweep value.");
    }
  } else if (request.changing_values.size() != 0) {
    throw ValidationError("changing_values was given without changing_pairs.");
  }
  for (const OptimizationTarget& target : request.optimization_targets) {
    if (target.empty()) {
      throw ValidationError("An optimization entry must contain at least one (name, property) pair.");
    }
  }
}

}  // namespace dfm
