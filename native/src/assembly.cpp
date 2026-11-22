#include "dfm/assembly.hpp"

#include <codi.hpp>

#include <string>
#include <utility>
#include <vector>

#include "dfm/component_ad.hpp"
#include "dfm/component_ops.hpp"
#include "dfm/errors.hpp"
#include "dfm/link.hpp"
#include "dfm/plan_internals.hpp"
#include "dfm/simulator.hpp"

namespace dfm {
namespace {

using Internals = SimulationPlan::Internals;

const InstructionTable& table_of(const Internals& in, SystemKind kind) {
  return kind == SystemKind::Carrier ? in.carrier : kind == SystemKind::Signal ? in.signal : in.noise;
}
const MatrixLayout& layout_of(const Internals& in, SystemKind kind) {
  return kind == SystemKind::Carrier ? in.carrier_layout : kind == SystemKind::Signal ? in.signal_layout : in.noise_layout;
}

Index batch_size(const SimulationPlan& plan, SystemKind kind) {
  return kind == SystemKind::Carrier ? plan.dimensions().carrier_values : plan.dimensions().signal_values;
}

// Sweep overrides (parameter index, value) for batch element b. Carrier
// assembly uses the carrier sweep; signal and noise use the carrier sweep
// followed by the signal sweep (upstream expands twice).
std::vector<std::pair<int, double>> overrides(const Internals& in, SystemKind kind, Index b) {
  std::vector<std::pair<int, double>> out;
  const Index cv = in.carrier_changing_values.cols();
  const Index sv = in.signal_changing_values.cols();
  const Index cb = kind == SystemKind::Carrier ? b : (cv > 1 ? b : 0);
  for (std::size_t i = 0; i < in.carrier_changing_indices.size(); ++i) {
    out.emplace_back(in.carrier_changing_indices[i], in.carrier_changing_values(static_cast<Index>(i), cb));
  }
  if (kind != SystemKind::Carrier) {
    const Index sb = sv > 1 ? b : 0;
    for (std::size_t i = 0; i < in.signal_changing_indices.size(); ++i) {
      out.emplace_back(in.signal_changing_indices[i], in.signal_changing_values(static_cast<Index>(i), sb));
    }
  }
  return out;
}

void check_resolved(const SimulationPlan& plan, Index size) {
  if (size != plan.dimensions().parameters) {
    throw ValidationError("resolved parameter vector has length " + std::to_string(size) + ", expected " +
                          std::to_string(plan.dimensions().parameters) + ".");
  }
}

void check_carrier(const SimulationPlan& plan, Index rows, Index cols) {
  if (rows != plan.dimensions().carrier_values || cols != plan.dimensions().carrier_ports) {
    throw ValidationError("carrier fields must have shape (carrier_values, carrier_ports) = (" +
                          std::to_string(plan.dimensions().carrier_values) + ", " +
                          std::to_string(plan.dimensions().carrier_ports) + ").");
  }
}

// ------------------------------------------------------------- passive
AssembledSystem assemble(const SimulationPlan& plan, SystemKind kind, const Eigen::Ref<const VectorR>& resolved,
                         const MatrixC* carrier) {
  check_resolved(plan, resolved.size());
  const Internals& in = plan.internals();
  const InstructionTable& t = table_of(in, kind);
  const MatrixLayout& l = layout_of(in, kind);
  const Index batch = batch_size(plan, kind);
  const int nnz = l.pattern->nnz();
  const Index cv = in.carrier_changing_values.cols();

  AssembledSystem out;
  out.pattern = l.pattern;
  out.n = l.n;
  out.values.resize(batch, nnz);
  out.rhs.resize(batch, l.n);

  std::vector<double> params(resolved.data(), resolved.data() + resolved.size());
  std::vector<double> inputs;
  std::vector<Complex> outputs;
  std::vector<Complex> kernel_out(t.functions.size() * kComponentOutputs);

  for (Index b = 0; b < batch; ++b) {
    std::copy(resolved.data(), resolved.data() + resolved.size(), params.begin());
    for (const auto& [idx, v] : overrides(in, kind, b)) params[static_cast<std::size_t>(idx)] = v;

    // Kernels, one batched call per function.
    for (int f = 0; f < kNumComponentFunctions; ++f) {
      const auto& rows = l.rows_by_function[static_cast<std::size_t>(f)];
      if (rows.empty()) continue;
      inputs.resize(rows.size() * kComponentInputWidth);
      outputs.resize(rows.size() * kComponentOutputs);
      for (std::size_t e = 0; e < rows.size(); ++e) {
        const auto& idx = t.inputs[static_cast<std::size_t>(rows[e])];
        for (int k = 0; k < kComponentInputWidth; ++k) inputs[e * kComponentInputWidth + k] = params[static_cast<std::size_t>(idx[k])];
      }
      evaluate_components(static_cast<ComponentFunction>(f), static_cast<Index>(rows.size()), inputs.data(),
                          kComponentInputWidth, outputs.data());
      for (std::size_t e = 0; e < rows.size(); ++e) {
        for (int j = 0; j < kComponentOutputs; ++j) {
          kernel_out[static_cast<std::size_t>(rows[e]) * kComponentOutputs + j] = outputs[e * kComponentOutputs + j];
        }
      }
    }

    Complex* values = out.values.row(b).data();
    Complex* rhs = out.rhs.row(b).data();
    std::copy(l.template_values.begin(), l.template_values.end(), values);
    std::copy(l.template_rhs.begin(), l.template_rhs.end(), rhs);
    for (std::size_t p = 0; p < t.rows.size(); ++p) {
      const Complex v = kernel_out[static_cast<std::size_t>(t.output_rows[p]) * kComponentOutputs + t.output_cols[p]];
      const int slot = l.placement_slots[p];
      if (slot >= 0) values[slot] = v;
      else rhs[t.rows[p]] = v;
    }
    if (kind == SystemKind::Signal) {
      const Index cb = cv > 1 ? b : 0;
      for (std::size_t c = 0; c < t.carrier_indices.size(); ++c) {
        const Complex field = (*carrier)(cb, t.carrier_indices[c]);
        const int slot = l.coupling_slots[c];
        Complex& target = slot >= 0 ? values[slot] : rhs[t.coupling_rows[c]];
        target = -target * field;
      }
      for (int slot : l.conjugate_slots) values[slot] = std::conj(values[slot]);
      for (int r : l.conjugate_rhs_rows) rhs[r] = std::conj(rhs[r]);
    }
  }
  return out;
}

// ------------------------------------------------------------ recorded
RecordedSystem record(const SimulationPlan& plan, SystemKind kind, const std::vector<ActiveReal>& resolved,
                      const ActiveReal* carrier) {
  check_resolved(plan, static_cast<Index>(resolved.size()));
  const Internals& in = plan.internals();
  const InstructionTable& t = table_of(in, kind);
  const MatrixLayout& l = layout_of(in, kind);
  const Index batch = batch_size(plan, kind);
  const int nnz = l.pattern->nnz();
  const int n = l.n;
  const int s = in.system_size;
  const Index cv = in.carrier_changing_values.cols();

  RecordedSystem out;
  out.pattern = l.pattern;
  out.n = n;
  out.batch = static_cast<int>(batch);
  out.values.resize(static_cast<std::size_t>(batch) * 2 * nnz);
  out.rhs.resize(static_cast<std::size_t>(batch) * 2 * n);

  // Per-batch sweep values are passive constants; kernels read either them
  // or the shared active parameters through a pointer table (no copies).
  std::vector<ActiveReal> sweep(resolved.size());
  std::vector<int> sweep_slot(resolved.size(), -1);
  std::vector<const ActiveReal*> pointers;
  std::vector<ActiveReal> kernel_out(t.functions.size() * 2 * kComponentOutputs);
  std::vector<ActiveReal*> out_pointers;

  for (Index b = 0; b < batch; ++b) {
    std::fill(sweep_slot.begin(), sweep_slot.end(), -1);
    for (const auto& [idx, v] : overrides(in, kind, b)) {
      sweep[static_cast<std::size_t>(idx)] = v;
      sweep_slot[static_cast<std::size_t>(idx)] = idx;
    }
    auto param = [&](int idx) -> const ActiveReal* {
      return sweep_slot[static_cast<std::size_t>(idx)] >= 0 ? &sweep[static_cast<std::size_t>(idx)]
                                                            : &resolved[static_cast<std::size_t>(idx)];
    };

    for (int f = 0; f < kNumComponentFunctions; ++f) {
      const auto& rows = l.rows_by_function[static_cast<std::size_t>(f)];
      if (rows.empty()) continue;
      pointers.resize(rows.size() * kComponentInputWidth);
      out_pointers.resize(rows.size());
      for (std::size_t e = 0; e < rows.size(); ++e) {
        const auto& idx = t.inputs[static_cast<std::size_t>(rows[e])];
        for (int k = 0; k < kComponentInputWidth; ++k) pointers[e * kComponentInputWidth + k] = param(idx[k]);
      }
      for (std::size_t e = 0; e < rows.size(); ++e) {
        out_pointers[e] = kernel_out.data() + static_cast<std::size_t>(rows[e]) * 2 * kComponentOutputs;
      }
      record_components_indirect(static_cast<ComponentFunction>(f), static_cast<Index>(rows.size()), pointers.data(),
                                 out_pointers.data());
    }

    ActiveReal* values = out.values.data() + b * 2 * nnz;
    ActiveReal* rhs = out.rhs.data() + b * 2 * n;
    for (int slot = 0; slot < nnz; ++slot) {
      values[2 * slot] = l.template_values[static_cast<std::size_t>(slot)].real();
      values[2 * slot + 1] = l.template_values[static_cast<std::size_t>(slot)].imag();
    }
    for (int r = 0; r < n; ++r) {
      rhs[2 * r] = l.template_rhs[static_cast<std::size_t>(r)].real();
      rhs[2 * r + 1] = l.template_rhs[static_cast<std::size_t>(r)].imag();
    }
    for (std::size_t p = 0; p < t.rows.size(); ++p) {
      const std::size_t o = static_cast<std::size_t>(t.output_rows[p]) * 2 * kComponentOutputs + 2 * t.output_cols[p];
      const int slot = l.placement_slots[p];
      ActiveReal* target = slot >= 0 ? values + 2 * slot : rhs + 2 * t.rows[p];
      target[0] = kernel_out[o];
      target[1] = kernel_out[o + 1];
    }
    if (kind == SystemKind::Signal) {
      const Index cb = cv > 1 ? b : 0;
      for (std::size_t c = 0; c < t.carrier_indices.size(); ++c) {
        const ActiveReal* field = carrier + 2 * (cb * s + t.carrier_indices[c]);
        const int slot = l.coupling_slots[c];
        ActiveReal* target = slot >= 0 ? values + 2 * slot : rhs + 2 * t.coupling_rows[c];
        const ActiveReal re = -(target[0] * field[0] - target[1] * field[1]);
        const ActiveReal im = -(target[0] * field[1] + target[1] * field[0]);
        target[0] = re;
        target[1] = im;
      }
      for (int slot : l.conjugate_slots) values[2 * slot + 1] = -values[2 * slot + 1];
      for (int r : l.conjugate_rhs_rows) rhs[2 * r + 1] = -rhs[2 * r + 1];
    }
  }
  return out;
}

}  // namespace

VectorR resolve_parameters(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& x) {
  const Internals& in = plan.internals();
  if (x.size() != plan.dimensions().optimized_values) {
    throw ValidationError("optimized_parameters has length " + std::to_string(x.size()) + ", expected " +
                          std::to_string(plan.dimensions().optimized_values) + ".");
  }
  VectorR p = Eigen::Map<const VectorR>(in.parameters.data(), static_cast<Index>(in.parameters.size()));
  for (std::size_t i = 0; i < in.optimized_parameter_indices.size(); ++i) {
    p(in.optimized_parameter_indices[i]) = x(in.optimized_value_indices[i]);
  }
  std::vector<double> linked(in.linked_indices.size());
  for (std::size_t i = 0; i < linked.size(); ++i) linked[i] = apply_link(in.link_functions[i], p(in.linked_indices[i]));
  for (std::size_t i = 0; i < linked.size(); ++i) p(in.indices_to_link[i]) = linked[i];
  return p;
}

std::vector<ActiveReal> resolve_parameters_recorded(const SimulationPlan& plan, const ActiveReal* x) {
  const Internals& in = plan.internals();
  std::vector<ActiveReal> p(in.parameters.begin(), in.parameters.end());
  for (std::size_t i = 0; i < in.optimized_parameter_indices.size(); ++i) {
    p[static_cast<std::size_t>(in.optimized_parameter_indices[i])] = x[in.optimized_value_indices[i]];
  }
  std::vector<ActiveReal> linked(in.linked_indices.size());
  for (std::size_t i = 0; i < linked.size(); ++i) {
    linked[i] = apply_link(in.link_functions[i], p[static_cast<std::size_t>(in.linked_indices[i])]);
  }
  for (std::size_t i = 0; i < linked.size(); ++i) p[static_cast<std::size_t>(in.indices_to_link[i])] = linked[i];
  return p;
}

AssembledSystem assemble_carrier(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved) {
  return assemble(plan, SystemKind::Carrier, resolved, nullptr);
}

AssembledSystem assemble_signal(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved,
                                const Eigen::Ref<const MatrixC>& carrier) {
  check_carrier(plan, carrier.rows(), carrier.cols());
  const MatrixC fields = carrier;
  return assemble(plan, SystemKind::Signal, resolved, &fields);
}

AssembledSystem assemble_noise(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& resolved) {
  return assemble(plan, SystemKind::Noise, resolved, nullptr);
}

RecordedSystem record_carrier(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved) {
  return record(plan, SystemKind::Carrier, resolved, nullptr);
}

RecordedSystem record_signal(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved,
                             const ActiveReal* carrier) {
  return record(plan, SystemKind::Signal, resolved, carrier);
}

RecordedSystem record_noise(const SimulationPlan& plan, const std::vector<ActiveReal>& resolved) {
  return record(plan, SystemKind::Noise, resolved, nullptr);
}

AssemblyGradient assembly_vjp(const SimulationPlan& plan, SystemKind kind, const Eigen::Ref<const VectorR>& x,
                              const Eigen::Ref<const MatrixC>& carrier, const Eigen::Ref<const MatrixC>& values_cot,
                              const Eigen::Ref<const MatrixC>& rhs_cot) {
  const Index opv = plan.dimensions().optimized_values;
  if (x.size() != opv) throw ValidationError("optimized_parameters has the wrong length.");
  if (kind == SystemKind::Signal) check_carrier(plan, carrier.rows(), carrier.cols());

  std::lock_guard<std::mutex> lock(tape_mutex());
  auto& tape = ActiveReal::getTape();
  invalidate_recorded_states();
  tape.reset();
  tape.setActive();
  std::vector<ActiveReal> xa(x.data(), x.data() + x.size());
  for (auto& v : xa) tape.registerInput(v);
  std::vector<ActiveReal> fields;
  if (kind == SystemKind::Signal) {
    fields.resize(static_cast<std::size_t>(carrier.size()) * 2);
    for (Index k = 0; k < carrier.size(); ++k) {
      fields[static_cast<std::size_t>(2 * k)] = carrier.data()[k].real();
      fields[static_cast<std::size_t>(2 * k + 1)] = carrier.data()[k].imag();
      tape.registerInput(fields[static_cast<std::size_t>(2 * k)]);
      tape.registerInput(fields[static_cast<std::size_t>(2 * k + 1)]);
    }
  }
  const std::vector<ActiveReal> resolved = resolve_parameters_recorded(plan, xa.data());
  RecordedSystem sys = kind == SystemKind::Carrier ? record_carrier(plan, resolved)
                       : kind == SystemKind::Signal ? record_signal(plan, resolved, fields.data())
                                                    : record_noise(plan, resolved);
  const int nnz = sys.pattern->nnz();
  if (values_cot.rows() != sys.batch || values_cot.cols() != nnz || rhs_cot.rows() != sys.batch ||
      rhs_cot.cols() != sys.n) {
    tape.setPassive();
    tape.reset();
    throw ValidationError("cotangent shapes must be (batch, nnz) and (batch, n).");
  }
  for (auto& v : sys.values) tape.registerOutput(v);
  for (auto& v : sys.rhs) tape.registerOutput(v);
  tape.setPassive();
  for (int b = 0; b < sys.batch; ++b) {
    for (int k = 0; k < nnz; ++k) {
      sys.values[static_cast<std::size_t>(b * 2 * nnz + 2 * k)].setGradient(values_cot(b, k).real());
      sys.values[static_cast<std::size_t>(b * 2 * nnz + 2 * k + 1)].setGradient(values_cot(b, k).imag());
    }
    for (int k = 0; k < sys.n; ++k) {
      sys.rhs[static_cast<std::size_t>(b * 2 * sys.n + 2 * k)].setGradient(rhs_cot(b, k).real());
      sys.rhs[static_cast<std::size_t>(b * 2 * sys.n + 2 * k + 1)].setGradient(rhs_cot(b, k).imag());
    }
  }
  tape.evaluate();

  AssemblyGradient g;
  g.optimized.resize(opv);
  for (Index i = 0; i < opv; ++i) g.optimized(i) = xa[static_cast<std::size_t>(i)].getGradient();
  if (kind == SystemKind::Signal) {
    g.carrier.resize(carrier.rows(), carrier.cols());
    for (Index k = 0; k < carrier.size(); ++k) {
      g.carrier.data()[k] = Complex(fields[static_cast<std::size_t>(2 * k)].getGradient(),
                                    fields[static_cast<std::size_t>(2 * k + 1)].getGradient());
    }
  }
  tape.reset();
  return g;
}

}  // namespace dfm
