#include "dfm/pipeline.hpp"

#include <codi.hpp>

#include <cmath>
#include <utility>
#include <vector>

#include "dfm/assembly.hpp"
#include "dfm/constants.hpp"
#include "dfm/errors.hpp"
#include "dfm/plan_internals.hpp"
#include "dfm/sparse.hpp"
#include "dfm/sparse_ad.hpp"
#include "dfm/sparse_lu.hpp"

#ifndef DFM_CARRIER_REFINEMENT_STEPS
#define DFM_CARRIER_REFINEMENT_STEPS 2
#endif

namespace dfm {
namespace {

using Internals = SimulationPlan::Internals;

constexpr double kNoiseScale = 2.0 * constants::UNIT_VACUUM * constants::H_PLANCK * constants::F0 * 0.25;
constexpr double kDegree = 3.141592653589793 / 180.0;  // jnp.radians factor

// Carrier solves use the natural column order: with partial pivoting this is
// dense LAPACK's pivot sequence, which upstream uses. The carrier enters the
// signal readout through carrier x signal products that cancel almost
// completely at dark-fringe / near-null operating points (voyager_fm nominal:
// detected signal ~1e-21), so its rounding has to track the reference's;
// with COLAMD the carrier differed from it by 4.9e-13 (cond ~3e5), which was
// enough to change that readout by a factor ~2,400. Carrier matrices are
// small, so the extra fill costs little. Signal solves keep COLAMD.
constexpr LUOptions kCarrierLU{/*natural_ordering=*/true, /*refinement_steps=*/DFM_CARRIER_REFINEMENT_STEPS};

// Non-zero positions of the selection vectors (static per plan): only these
// entries of s are non-zero, so s^H c only needs them.
std::vector<std::vector<int>> selection_support(const Internals& in) {
  std::vector<std::vector<int>> support(static_cast<std::size_t>(in.noise_selection.rows()));
  for (Index q = 0; q < in.noise_selection.rows(); ++q) {
    for (Index i = 0; i < in.noise_selection.cols(); ++i) {
      if (in.noise_selection(q, i) != Complex(0.0, 0.0)) support[static_cast<std::size_t>(q)].push_back(static_cast<int>(i));
    }
  }
  return support;
}

}  // namespace

// ------------------------------------------------------------------ passive
ForwardOutputs simulate_passive(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& x) {
  const Internals& in = plan.internals();
  const Dimensions& d = plan.dimensions();
  const int s = in.system_size;
  const Index cv = d.carrier_values;
  const Index batch = d.signal_values;
  const Index q_count = d.noise_detectors;
  const int n = static_cast<int>(d.signal_variables);

  const VectorR resolved = resolve_parameters(plan, x);

  // 1. Carrier.
  const AssembledSystem carrier_sys = assemble_carrier(plan, resolved);
  MatrixC carrier(cv, s);  // (value, port), the layout signal assembly expects
  for (Index c = 0; c < cv; ++c) {
    ComplexSparseLU lu(*carrier_sys.pattern, carrier_sys.values.row(c).data(), kCarrierLU);
    VectorC x_c = carrier_sys.rhs.row(c).transpose();
    lu.solve_inplace(x_c);
    carrier.row(c) = x_c.transpose();
  }

  // 2. Signal and 3. noise.
  const AssembledSystem signal_sys = assemble_signal(plan, resolved, carrier);
  const AssembledSystem noise_sys = q_count > 0 ? assemble_noise(plan, resolved) : AssembledSystem{};

  // Selection vectors with QHD phases (upstream uses the unswept parameters).
  MatrixC selection = in.noise_selection;
  for (std::size_t k = 0; k < in.qhd_placing_indices.size(); ++k) {
    const double phase = resolved(in.qhd_parameter_indices[k]) * kDegree;
    selection(in.qhd_placing_indices[k][0], in.qhd_placing_indices[k][2]) *= std::exp(Complex(0.0, phase));
  }
  const auto support = selection_support(in);

  ForwardOutputs out;
  out.carrier = carrier.transpose();
  out.signal.resize(n, batch);
  out.noise.resize(q_count, batch);

  Eigen::MatrixXcd sel(n, q_count), work(n, q_count), sources(n, q_count);
  for (Index b = 0; b < batch; ++b) {
    ComplexSparseLU lu(*signal_sys.pattern, signal_sys.values.row(b).data());
    VectorC y = signal_sys.rhs.row(b).transpose();
    lu.solve_inplace(y);
    out.signal.col(b) = y;
    if (q_count == 0) continue;

    const Index cb = cv > 1 ? b : 0;
    for (Index q = 0; q < q_count; ++q) {
      for (int i = 0; i < n; ++i) {
        Complex v = selection(q, i);
        if (i < s) v *= carrier(cb, i);
        else if (i < 2 * s) v = std::conj(v * carrier(cb, i - s));
        sel(i, q) = v;
      }
    }
    work = sel;
    lu.solve_conjugate_transpose_inplace(work.data(), static_cast<int>(q_count));               // w = M^{-H} s
    sparse_matmul(*noise_sys.pattern, noise_sys.values.row(b).data(), work.data(), static_cast<int>(q_count),
                  sources.data());                                                               // u = V w
    lu.solve_inplace(sources.data(), static_cast<int>(q_count));                                 // c = M^{-1} u
    for (Index q = 0; q < q_count; ++q) {
      double total = 0.0;
      for (int i : support[static_cast<std::size_t>(q)]) total += std::real(sources(i, q) * std::conj(sel(i, q)));
      out.noise(q, b) = std::sqrt(kNoiseScale * total);
    }
  }
  return out;
}

// ----------------------------------------------------------------- recorded
RecordedOutputs simulate_recorded(const SimulationPlan& plan, const ActiveReal* x) {
  const Internals& in = plan.internals();
  const Dimensions& d = plan.dimensions();
  const int s = in.system_size;
  const Index cv = d.carrier_values;
  const Index batch = d.signal_values;
  const Index q_count = d.noise_detectors;
  const int n = static_cast<int>(d.signal_variables);
  const int qn = static_cast<int>(q_count);

  const std::vector<ActiveReal> resolved = resolve_parameters_recorded(plan, x);

  // 1. Carrier: one recorded solve per carrier value. carrier_fields is
  // (value, port) interleaved, as record_signal expects.
  const RecordedSystem carrier_sys = record_carrier(plan, resolved);
  const int carrier_nnz = carrier_sys.pattern->nnz();
  std::vector<ActiveReal> carrier_fields(static_cast<std::size_t>(cv) * 2 * s);
  for (Index c = 0; c < cv; ++c) {
    RecordedLinearSystem system(carrier_sys.pattern, carrier_sys.values.data() + c * 2 * carrier_nnz, kCarrierLU);
    system.solve(carrier_sys.rhs.data() + c * 2 * s, 1, Op::Normal, carrier_fields.data() + c * 2 * s);
  }

  const RecordedSystem signal_sys = record_signal(plan, resolved, carrier_fields.data());
  const int signal_nnz = signal_sys.pattern->nnz();
  RecordedSystem noise_sys;
  if (q_count > 0) noise_sys = record_noise(plan, resolved);
  const int noise_nnz = q_count > 0 ? noise_sys.pattern->nnz() : 0;

  // QHD phase factors exp(i * radians(phase)) on the unswept parameters.
  std::vector<ActiveReal> sel_base_re(static_cast<std::size_t>(q_count * n)), sel_base_im(static_cast<std::size_t>(q_count * n));
  for (Index q = 0; q < q_count; ++q) {
    for (int i = 0; i < n; ++i) {
      sel_base_re[static_cast<std::size_t>(q * n + i)] = in.noise_selection(q, i).real();
      sel_base_im[static_cast<std::size_t>(q * n + i)] = in.noise_selection(q, i).imag();
    }
  }
  for (std::size_t k = 0; k < in.qhd_placing_indices.size(); ++k) {
    const ActiveReal phase = resolved[static_cast<std::size_t>(in.qhd_parameter_indices[k])] * kDegree;
    const ActiveReal c = cos(phase);
    const ActiveReal sn = sin(phase);
    const std::size_t at = static_cast<std::size_t>(in.qhd_placing_indices[k][0] * n + in.qhd_placing_indices[k][2]);
    const ActiveReal re = sel_base_re[at] * c - sel_base_im[at] * sn;
    const ActiveReal im = sel_base_re[at] * sn + sel_base_im[at] * c;
    sel_base_re[at] = re;
    sel_base_im[at] = im;
  }
  const auto support = selection_support(in);

  RecordedOutputs out;
  out.carrier.resize(static_cast<std::size_t>(s) * cv * 2);
  for (int i = 0; i < s; ++i) {
    for (Index c = 0; c < cv; ++c) {
      out.carrier[static_cast<std::size_t>(2 * (i * cv + c))] = carrier_fields[static_cast<std::size_t>(c * 2 * s + 2 * i)];
      out.carrier[static_cast<std::size_t>(2 * (i * cv + c) + 1)] = carrier_fields[static_cast<std::size_t>(c * 2 * s + 2 * i + 1)];
    }
  }
  out.signal.resize(static_cast<std::size_t>(n) * batch * 2);
  out.noise.resize(static_cast<std::size_t>(q_count * batch));

  std::vector<ActiveReal> y(static_cast<std::size_t>(2 * n));
  std::vector<ActiveReal> sel(static_cast<std::size_t>(2 * n * qn)), w(sel.size()), u(sel.size()), cov(sel.size());
  for (Index b = 0; b < batch; ++b) {
    RecordedLinearSystem system(signal_sys.pattern, signal_sys.values.data() + b * 2 * signal_nnz);
    system.solve(signal_sys.rhs.data() + b * 2 * n, 1, Op::Normal, y.data());
    for (int r = 0; r < n; ++r) {
      out.signal[static_cast<std::size_t>(2 * (r * batch + b))] = y[static_cast<std::size_t>(2 * r)];
      out.signal[static_cast<std::size_t>(2 * (r * batch + b) + 1)] = y[static_cast<std::size_t>(2 * r + 1)];
    }
    if (q_count == 0) continue;

    // Carrier selection vectors s_q (column q, interleaved, column-major).
    const Index cb = cv > 1 ? b : 0;
    const ActiveReal* field = carrier_fields.data() + cb * 2 * s;
    for (int q = 0; q < qn; ++q) {
      for (int i = 0; i < n; ++i) {
        ActiveReal* t = sel.data() + 2 * (q * n + i);
        const std::size_t base = static_cast<std::size_t>(q * n + i);
        if (in.noise_selection(q, i) == Complex(0.0, 0.0) && sel_base_re[base].getIdentifier() == 0 &&
            sel_base_im[base].getIdentifier() == 0) {
          t[0] = 0.0;
          t[1] = 0.0;
          continue;
        }
        const ActiveReal& ar = sel_base_re[base];
        const ActiveReal& ai = sel_base_im[base];
        if (i < 2 * s) {
          const int port = i < s ? i : i - s;
          const ActiveReal& cr = field[2 * port];
          const ActiveReal& ci = field[2 * port + 1];
          const ActiveReal re = ar * cr - ai * ci;
          const ActiveReal im = ar * ci + ai * cr;
          t[0] = re;
          t[1] = i < s ? im : -im;  // lower sideband: conj(selection * carrier)
        } else {
          t[0] = ar;
          t[1] = ai;
        }
      }
    }
    system.solve(sel.data(), qn, Op::ConjugateTranspose, w.data());                      // w = M^{-H} s
    sparse_matmul_recorded(noise_sys.pattern, noise_sys.values.data() + b * 2 * noise_nnz, w.data(), qn, Op::Normal,
                           u.data());                                                     // u = V w
    system.solve(u.data(), qn, Op::Normal, cov.data());                                  // c = M^{-1} u
    for (int q = 0; q < qn; ++q) {
      ActiveReal total = 0.0;
      for (int i : support[static_cast<std::size_t>(q)]) {
        const std::size_t k = static_cast<std::size_t>(2 * (q * n + i));
        total += cov[k] * sel[k] + cov[k + 1] * sel[k + 1];  // Re(c * conj(s))
      }
      out.noise[static_cast<std::size_t>(q * batch + b)] = sqrt(kNoiseScale * total);
    }
  }
  return out;
}

}  // namespace dfm
