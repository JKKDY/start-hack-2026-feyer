#pragma once

// Native left-looking sparse LU (private to sparse_lu.cpp).
//
// Gilbert-Peierls LU with partial pivoting, as in CSparse's cs_lu or KLU
// without the block triangular form: column k of L and U comes from one
// sparse triangular solve against the columns already factored, and a
// depth-first search finds that column's non-zero structure in time
// proportional to the arithmetic. Pivot: the largest magnitude among the
// rows not pivotal yet, the rule SuperLU uses with diag_pivot_thresh = 1 (an
// exact tie goes to the original diagonal). The column order is COLAMD,
// computed once per structure and cached.
//
// At the sizes the simulator produces (n up to a few thousand, a handful of
// non-zeros per column) SuperLU's supernodal set-up dominates its run time;
// this avoids it (scored UIFO signal matrices, n = 751: 0.2 ms per
// factorisation instead of 0.5 ms, and cheaper triangular solves).
//
// Not done: reusing one frequency's pivot sequence for the next (KLU-style
// refactorisation). With strict partial pivoting the pivot sequence changes
// at every frequency of the scored setup, so exact reuse never applies, and
// threshold pivoting would change the numerics.

#include <cmath>
#include <memory>
#include <vector>

#include "dfm/types.hpp"

namespace dfm::native_lu {

/// P A Q = L U. L is unit lower with its diagonal stored first in each
/// column, U upper with its diagonal last; both CSC in pivot order.
struct Factors {
  int n = 0;
  std::shared_ptr<const std::vector<int>> q;  ///< column factorised at step k (shared per structure)
  std::vector<int> pinv;                      ///< pivot step of original row i
  std::vector<int> lp, li, up, ui;
  std::vector<Complex> lx, ux;
  std::vector<Complex> udiag_inv;  ///< 1 / U(k, k): solves multiply instead of dividing
};

/// Pivot comparison key: |z|^2 (monotone in |z|, without hypot's cost); falls
/// back to |z| squared where |z|^2 would under- or overflow.
inline double pivot_key(Complex z) {
  const double k = std::norm(z);
  if (k > 1e-290 && k < 1e290) return k;
  const double a = std::abs(z);
  return a * a;
}

/// COLAMD column order of this structure (cached per structure; the cache
/// compares the full structure, so a hit is never stale).
std::shared_ptr<const std::vector<int>> column_order(int n, int nnz, const int* colptr, const int* rowind);

/// Full factorisation with partial pivoting along the column order `q`.
/// Throws std::runtime_error on a singular matrix.
void factor(int n, const int* colptr, const int* rowind, const Complex* values,
            std::shared_ptr<const std::vector<int>> q, Factors& f);

/// x <- op(A)^{-1} x for nrhs column-major columns, with `work` of size n.
void solve(const Factors& f, Complex* x, int nrhs, bool conjugate_transpose, Complex* work);

}  // namespace dfm::native_lu
