#include "dfm/ad_sparse_solve.hpp"

#include <codi.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace dfm {
namespace {

using Helper = codi::ExternalFunctionHelper<ActiveReal>;
using UserData = codi::ExternalFunctionUserData;

// Primal: y = A^{-1} b, computed with plain doubles (no tape involvement).
// Mode 1 of the helper: the caller provides the primal explicitly.
void primal(double const* x, size_t m, double* y, size_t n, UserData* d) {
  const ComplexSparseLU* lu = nullptr;
  d->getDataByIndex(lu, 0);

  VectorC v(static_cast<Eigen::Index>(m / 2));
  for (size_t i = 0; i < m / 2; ++i) {
    v(static_cast<Eigen::Index>(i)) = Complex(x[2 * i], x[2 * i + 1]);
  }
  lu->solve_inplace(v);

  for (size_t i = 0; i < n / 2; ++i) {
    const Complex yi = v(static_cast<Eigen::Index>(i));
    y[2 * i] = yi.real();
    y[2 * i + 1] = yi.imag();
  }
}

// Reverse: bbar = A^{-H} ybar. The helper accumulates x_b into the tape, so
// every entry is written here (x_b is not zeroed beforehand).
void reverse(double const* /*x*/, double* x_b, size_t m, double const* /*y*/,
             double const* y_b, size_t n, UserData* d) {
  const ComplexSparseLU* lu = nullptr;
  d->getDataByIndex(lu, 0);

  VectorC adj(static_cast<Eigen::Index>(n / 2));
  for (size_t i = 0; i < n / 2; ++i) {
    adj(static_cast<Eigen::Index>(i)) = Complex(y_b[2 * i], y_b[2 * i + 1]);
  }
  lu->solve_conjugate_transpose_inplace(adj);

  for (size_t i = 0; i < m / 2; ++i) {
    const Complex bi = adj(static_cast<Eigen::Index>(i));
    x_b[2 * i] = bi.real();
    x_b[2 * i + 1] = bi.imag();
  }
}

}  // namespace

ActiveVectorC solve_recorded(const ComplexSparseLU& lu, const ActiveVectorC& b) {
  const Eigen::Index n = lu.size();
  if (b.size() != n) {
    throw std::invalid_argument("solve_recorded: right-hand side has size " +
                                std::to_string(b.size()) + ", expected " + std::to_string(n));
  }

  // Interleaved real inputs and outputs. The helper keeps identifiers into
  // these vectors, so they must stay alive until addToTape returns.
  std::vector<ActiveReal> xin(static_cast<std::size_t>(2 * n));
  std::vector<ActiveReal> yout(static_cast<std::size_t>(2 * n));
  for (Eigen::Index i = 0; i < n; ++i) {
    xin[static_cast<std::size_t>(2 * i)] = b(i).real();
    xin[static_cast<std::size_t>(2 * i + 1)] = b(i).imag();
  }

  Helper eh;
  for (ActiveReal& v : xin) {
    eh.addInput(v);
  }
  for (ActiveReal& v : yout) {
    eh.addOutput(v);
  }
  // The LU is referenced by pointer, not copied: one entry, no per-entry data.
  const ComplexSparseLU* lu_ptr = &lu;
  eh.addUserData(lu_ptr);
  eh.callPrimalFunc(primal);
  eh.addToTape(reverse);

  ActiveVectorC y(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    y(i) = ActiveComplex(yout[static_cast<std::size_t>(2 * i)],
                         yout[static_cast<std::size_t>(2 * i + 1)]);
  }
  return y;
}

}  // namespace dfm
