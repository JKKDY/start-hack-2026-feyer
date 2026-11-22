#pragma once

// Reverse-mode sparse linear algebra as CoDiPack external functions.
//
// Complex values are stored as interleaved active reals: value v at slot s
// is (re, im) = (a[2s], a[2s+1]). Dense n x m blocks are column-major with
// the same interleaving: entry (i, c) is at 2 * (c * n + i).
//
// Every operation here is ONE tape entry, however large the matrix. The
// arithmetic inside (SuperLU factorisation, triangular solves, products) is
// never recorded. Reverse rules (implicit adjoint method), for a real loss L
// and cotangent zbar = dL/dRe z + i dL/dIm z:
//
//   X = A^{-1} B    :  Bbar = A^{-H} Xbar,   Abar_ij = -sum_c Bbar_ic conj(X_jc)
//   X = A^{-H} B    :  Bbar = A^{-1} Xbar,   Abar_ij = -sum_c X_ic conj(Bbar_jc)
//   Y = A B         :  Bbar = A^H Ybar,      Abar_ij =  sum_c Ybar_ic conj(B_jc)
//   Y = A^H B       :  Bbar = A Ybar,        Abar_ij =  sum_c B_ic conj(Ybar_jc)
//
// Abar is only formed on the pattern's slots (occupied entries).

#include <memory>
#include <vector>

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/sparse.hpp"
#include "dfm/sparse_lu.hpp"

namespace dfm {

enum class Op : int {
  Normal = 0,              ///< A
  ConjugateTranspose = 1,  ///< A^H
};

/// A sparse matrix with active values, factorised once (from the primal
/// values) and reusable for any number of recorded solves. The factors and
/// pattern are shared with the tape entries through shared_ptr, so the tape
/// stays valid even after this object is destroyed.
class RecordedLinearSystem {
 public:
  /// `values` points at 2 * nnz interleaved active reals. They are copied
  /// (as active handles), so the caller's storage may be released.
  RecordedLinearSystem(SparsePatternPtr pattern, const ActiveReal* values, LUOptions options = {});

  const SparsePattern& pattern() const noexcept { return *pattern_; }
  const ComplexSparseLU& lu() const noexcept { return *lu_; }

  /// Records X = op(A)^{-1} B. `b` and `x` hold 2 * n * m active reals
  /// (column-major, interleaved). `x` may not alias `b`.
  void solve(const ActiveReal* b, int m, Op op, ActiveReal* x) const;

 private:
  SparsePatternPtr pattern_;
  std::shared_ptr<const ComplexSparseLU> lu_;
  std::vector<ActiveReal> values_;
};

/// Records Y = op(A) B with active A values and active B (same layouts).
void sparse_matmul_recorded(const SparsePatternPtr& pattern, const ActiveReal* values, const ActiveReal* b,
                            int m, Op op, ActiveReal* y);

}  // namespace dfm
