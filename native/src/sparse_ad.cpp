#include "dfm/sparse_ad.hpp"

#include <codi.hpp>

#include <stdexcept>
#include <string>
#include <utility>

#include "dfm/errors.hpp"
#include "dfm/profile.hpp"

namespace dfm {
namespace {

using Helper = codi::ExternalFunctionHelper<ActiveReal>;
using UserData = codi::ExternalFunctionUserData;

// Everything a reverse sweep needs, captured by value. shared_ptrs keep the
// factors and the pattern alive for as long as the tape holds the entry.
struct SolveData {
  std::shared_ptr<const ComplexSparseLU> lu;
  SparsePatternPtr pattern;
  int m = 0;
  Op op = Op::Normal;
};

struct MatmulData {
  SparsePatternPtr pattern;
  int m = 0;
  Op op = Op::Normal;
};

inline const Complex* as_complex(const double* p) { return reinterpret_cast<const Complex*>(p); }
inline Complex* as_complex(double* p) { return reinterpret_cast<Complex*>(p); }

// ---------------------------------------------------------------- solve
// Input layout: [A values (2 nnz) | B (2 n m)]. Output: X (2 n m).
void solve_primal(const double* x, size_t /*n_in*/, double* y, size_t n_out, UserData* d) {
  const SolveData& data = d->getDataByIndex<SolveData>(0);
  const int nnz = data.pattern->nnz();
  std::copy(x + 2 * nnz, x + 2 * nnz + n_out, y);
  Complex* out = as_complex(y);
  if (data.op == Op::Normal) {
    data.lu->solve_inplace(out, data.m);
  } else {
    data.lu->solve_conjugate_transpose_inplace(out, data.m);
  }
}

void solve_reverse(const double* /*x*/, double* x_b, size_t /*n_in*/, const double* y, const double* y_b,
                   size_t n_out, UserData* d) {
  profile::Scope sc(profile::ReverseSolve);
  const SolveData& data = d->getDataByIndex<SolveData>(0);
  const SparsePattern& p = *data.pattern;
  const int n = p.size();
  const int nnz = p.nnz();

  // Bbar = op(A)^{-H} Xbar, written straight into the B part of x_b.
  double* b_bar_raw = x_b + 2 * nnz;
  std::copy(y_b, y_b + n_out, b_bar_raw);
  Complex* b_bar = as_complex(b_bar_raw);
  if (data.op == Op::Normal) {
    data.lu->solve_conjugate_transpose_inplace(b_bar, data.m);
  } else {
    data.lu->solve_inplace(b_bar, data.m);
  }

  // Abar on occupied slots only.
  const Complex* xs = as_complex(y);
  Complex* a_bar = as_complex(x_b);
  const auto& cp = p.col_ptr();
  const auto& ri = p.row_idx();
  for (int j = 0; j < n; ++j) {
    for (int s = cp[j]; s < cp[j + 1]; ++s) {
      const int i = ri[s];
      Complex acc(0.0, 0.0);
      for (int c = 0; c < data.m; ++c) {
        const Index off = static_cast<Index>(c) * n;
        if (data.op == Op::Normal) {
          acc += b_bar[off + i] * std::conj(xs[off + j]);
        } else {
          acc += xs[off + i] * std::conj(b_bar[off + j]);
        }
      }
      a_bar[s] = -acc;
    }
  }
}

// --------------------------------------------------------------- matmul
void matmul_primal(const double* x, size_t /*n_in*/, double* y, size_t /*n_out*/, UserData* d) {
  const MatmulData& data = d->getDataByIndex<MatmulData>(0);
  const int nnz = data.pattern->nnz();
  const Complex* a = as_complex(x);
  const Complex* b = as_complex(x + 2 * nnz);
  if (data.op == Op::Normal) {
    sparse_matmul(*data.pattern, a, b, data.m, as_complex(y));
  } else {
    sparse_matmul_adjoint(*data.pattern, a, b, data.m, as_complex(y));
  }
}

void matmul_reverse(const double* x, double* x_b, size_t /*n_in*/, const double* /*y*/, const double* y_b,
                    size_t /*n_out*/, UserData* d) {
  profile::Scope sc(profile::ReverseProduct);
  const MatmulData& data = d->getDataByIndex<MatmulData>(0);
  const SparsePattern& p = *data.pattern;
  const int n = p.size();
  const int nnz = p.nnz();
  const Complex* a = as_complex(x);
  const Complex* b = as_complex(x + 2 * nnz);
  const Complex* y_bar = as_complex(y_b);
  Complex* a_bar = as_complex(x_b);
  Complex* b_bar = as_complex(x_b + 2 * nnz);

  // Bbar = op(A)^H Ybar
  if (data.op == Op::Normal) {
    sparse_matmul_adjoint(p, a, y_bar, data.m, b_bar);
  } else {
    sparse_matmul(p, a, y_bar, data.m, b_bar);
  }

  const auto& cp = p.col_ptr();
  const auto& ri = p.row_idx();
  for (int j = 0; j < n; ++j) {
    for (int s = cp[j]; s < cp[j + 1]; ++s) {
      const int i = ri[s];
      Complex acc(0.0, 0.0);
      for (int c = 0; c < data.m; ++c) {
        const Index off = static_cast<Index>(c) * n;
        if (data.op == Op::Normal) {
          acc += y_bar[off + i] * std::conj(b[off + j]);
        } else {
          acc += b[off + i] * std::conj(y_bar[off + j]);
        }
      }
      a_bar[s] = acc;
    }
  }
}

// ------------------------------------------------- signal + noise chain
// Input layout:  [A (2 nnzA) | b (2 n) | s (2 n m) | V (2 nnzV)]
// Output layout: [y (2 n) | c (2 n m)]
struct ChainWork {
  std::vector<Complex> w;  ///< A^{-H} s, n x m (set by the primal call)
  std::vector<Complex> v;  ///< V values (set by the primal call)
};

struct ChainData {
  std::shared_ptr<const ComplexSparseLU> lu;
  SparsePatternPtr a_pattern;
  SparsePatternPtr v_pattern;
  int m = 0;
  std::shared_ptr<ChainWork> work;
};

void chain_primal(const double* x, size_t /*n_in*/, double* y, size_t /*n_out*/, UserData* d) {
  const ChainData& data = d->getDataByIndex<ChainData>(0);
  const int n = data.a_pattern->size();
  const Index nnz_a = data.a_pattern->nnz();
  const Index nnz_v = data.v_pattern->nnz();
  const Index len_b = 2 * static_cast<Index>(n);
  const Index len_s = len_b * data.m;
  const double* b = x + 2 * nnz_a;
  const double* s = b + len_b;
  const double* v = s + len_s;
  ChainWork& work = *data.work;
  work.v.assign(as_complex(v), as_complex(v) + nnz_v);
  // y = A^{-1} b
  std::copy(b, b + len_b, y);
  data.lu->solve_inplace(as_complex(y), 1);
  // w = A^{-H} s;  u = V w (written to the c part);  c = A^{-1} u
  work.w.assign(as_complex(s), as_complex(s) + static_cast<Index>(n) * data.m);
  data.lu->solve_conjugate_transpose_inplace(work.w.data(), data.m);
  Complex* c = as_complex(y + len_b);
  sparse_matmul(*data.v_pattern, work.v.data(), work.w.data(), data.m, c);
  data.lu->solve_inplace(c, data.m);
}

void chain_reverse(const double* /*x*/, double* x_b, size_t /*n_in*/, const double* y, const double* y_b,
                   size_t /*n_out*/, UserData* d) {
  profile::Scope sc(profile::ReverseSolve);
  const ChainData& data = d->getDataByIndex<ChainData>(0);
  const SparsePattern& pa = *data.a_pattern;
  const SparsePattern& pv = *data.v_pattern;
  const int n = pa.size();
  const int m = data.m;
  const Index nnz_a = pa.nnz();
  const ChainWork& work = *data.work;

  // [bbar, ubar] = A^{-H} [ybar, cbar]: one (1 + m)-column solve.
  std::vector<Complex> t(as_complex(y_b), as_complex(y_b) + static_cast<Index>(n) * (1 + m));
  data.lu->solve_conjugate_transpose_inplace(t.data(), 1 + m);
  const Complex* b_bar = t.data();
  const Complex* u_bar = t.data() + n;

  Complex* a_bar = as_complex(x_b);
  Complex* b_bar_out = as_complex(x_b + 2 * nnz_a);
  Complex* s_bar = b_bar_out + n;
  Complex* v_bar = s_bar + static_cast<Index>(n) * m;
  std::copy(b_bar, b_bar + n, b_bar_out);

  // wbar = V^H ubar, then sbar = A^{-1} wbar (in place).
  sparse_matmul_adjoint(pv, work.v.data(), u_bar, m, s_bar);
  data.lu->solve_inplace(s_bar, m);

  // Vbar on V's slots: ubar w^H.
  {
    const auto& cp = pv.col_ptr();
    const auto& ri = pv.row_idx();
    for (int j = 0; j < n; ++j) {
      for (int sl = cp[j]; sl < cp[j + 1]; ++sl) {
        const int i = ri[sl];
        Complex acc(0.0, 0.0);
        for (int col = 0; col < m; ++col) {
          const Index off = static_cast<Index>(col) * n;
          acc += u_bar[off + i] * std::conj(work.w[off + j]);
        }
        v_bar[sl] = acc;
      }
    }
  }
  // Abar on A's slots: -(bbar y^H + ubar c^H + w sbar^H).
  const Complex* ys = as_complex(y);
  const Complex* cs = ys + n;
  const auto& cp = pa.col_ptr();
  const auto& ri = pa.row_idx();
  for (int j = 0; j < n; ++j) {
    for (int sl = cp[j]; sl < cp[j + 1]; ++sl) {
      const int i = ri[sl];
      Complex acc = b_bar[i] * std::conj(ys[j]);
      for (int col = 0; col < m; ++col) {
        const Index off = static_cast<Index>(col) * n;
        acc += u_bar[off + i] * std::conj(cs[off + j]) + work.w[off + i] * std::conj(s_bar[off + j]);
      }
      a_bar[sl] = -acc;
    }
  }
}

void check_sizes(int m) {
  if (m < 0) throw ValidationError("number of right-hand sides must be non-negative");
}

}  // namespace

RecordedLinearSystem::RecordedLinearSystem(SparsePatternPtr pattern, const ActiveReal* values, LUOptions options)
    : pattern_(std::move(pattern)) {
  if (!pattern_) throw ValidationError("RecordedLinearSystem: pattern must not be null");
  const int nnz = pattern_->nnz();
  values_ = values;
  std::vector<Complex> primal(static_cast<std::size_t>(nnz));
  for (int s = 0; s < nnz; ++s) {
    primal[static_cast<std::size_t>(s)] = Complex(values[2 * s].getValue(), values[2 * s + 1].getValue());
  }
  lu_ = std::make_shared<const ComplexSparseLU>(*pattern_, primal.data(), options);
}

void RecordedLinearSystem::solve(const ActiveReal* b, int m, Op op, ActiveReal* x) const {
  check_sizes(m);
  const Index len = 2 * static_cast<Index>(pattern_->size()) * m;
  Helper eh;
  const Index nnz2 = 2 * static_cast<Index>(pattern_->nnz());
  for (Index k = 0; k < nnz2; ++k) eh.addInput(values_[k]);
  for (Index k = 0; k < len; ++k) eh.addInput(b[k]);
  for (Index k = 0; k < len; ++k) eh.addOutput(x[k]);
  // The reverse rule needs X (outputs) and the factors, not the inputs.
  eh.disableInputPrimalStore();
  eh.addUserData(SolveData{lu_, pattern_, m, op});
  eh.callPrimalFunc(solve_primal);
  eh.addToTape(solve_reverse);
}

void sparse_matmul_recorded(const SparsePatternPtr& pattern, const ActiveReal* values, const ActiveReal* b,
                            int m, Op op, ActiveReal* y) {
  if (!pattern) throw ValidationError("sparse_matmul_recorded: pattern must not be null");
  check_sizes(m);
  const Index nnz2 = 2 * static_cast<Index>(pattern->nnz());
  const Index len = 2 * static_cast<Index>(pattern->size()) * m;
  Helper eh;
  for (Index k = 0; k < nnz2; ++k) eh.addInput(values[k]);
  for (Index k = 0; k < len; ++k) eh.addInput(b[k]);
  for (Index k = 0; k < len; ++k) eh.addOutput(y[k]);
  // Outputs are not needed by the reverse rule (it uses A and B).
  eh.disableOutputPrimalStore();
  eh.addUserData(MatmulData{pattern, m, op});
  eh.callPrimalFunc(matmul_primal);
  eh.addToTape(matmul_reverse);
}

void record_signal_noise(const RecordedLinearSystem& system, const ActiveReal* b, const SparsePatternPtr& v_pattern,
                         const ActiveReal* v_values, const ActiveReal* s, int m, ActiveReal* y, ActiveReal* c) {
  if (!v_pattern) throw ValidationError("record_signal_noise: noise pattern must not be null");
  check_sizes(m);
  const SparsePatternPtr& a_pattern = system.shared_pattern();
  if (v_pattern->size() != a_pattern->size()) throw ValidationError("record_signal_noise: size mismatch");
  const Index len_b = 2 * static_cast<Index>(a_pattern->size());
  const Index len_s = len_b * m;
  const ActiveReal* a_values = system.values();
  Helper eh;
  for (Index k = 0, e = 2 * static_cast<Index>(a_pattern->nnz()); k < e; ++k) eh.addInput(a_values[k]);
  for (Index k = 0; k < len_b; ++k) eh.addInput(b[k]);
  for (Index k = 0; k < len_s; ++k) eh.addInput(s[k]);
  for (Index k = 0, e = 2 * static_cast<Index>(v_pattern->nnz()); k < e; ++k) eh.addInput(v_values[k]);
  for (Index k = 0; k < len_b; ++k) eh.addOutput(y[k]);
  for (Index k = 0; k < len_s; ++k) eh.addOutput(c[k]);
  // The reverse rule needs y and c (outputs), w and V (kept in ChainWork) and
  // the factors, not the inputs.
  eh.disableInputPrimalStore();
  eh.addUserData(ChainData{system.shared_lu(), a_pattern, v_pattern, m, std::make_shared<ChainWork>()});
  eh.callPrimalFunc(chain_primal);
  eh.addToTape(chain_reverse);
}

}  // namespace dfm
