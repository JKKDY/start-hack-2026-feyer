#include "dfm/sparse_ad.hpp"

#include <codi.hpp>

#include <stdexcept>
#include <string>
#include <utility>

#include "dfm/errors.hpp"

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

void check_sizes(int m) {
  if (m < 0) throw ValidationError("number of right-hand sides must be non-negative");
}

}  // namespace

RecordedLinearSystem::RecordedLinearSystem(SparsePatternPtr pattern, const ActiveReal* values, LUOptions options)
    : pattern_(std::move(pattern)) {
  if (!pattern_) throw ValidationError("RecordedLinearSystem: pattern must not be null");
  const int nnz = pattern_->nnz();
  values_.assign(values, values + 2 * nnz);
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
  for (const ActiveReal& v : values_) eh.addInput(v);
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

}  // namespace dfm
