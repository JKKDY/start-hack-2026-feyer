#include "dfm/sparse_lu.hpp"

#include <slu_zdefs.h>

#include "dfm/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace dfm {

// Accuracy on badly scaled optical matrices (condition numbers up to ~1e31):
//   - SuperLU factorisation, COLAMD ordering, partial pivoting
//     (diag_pivot_thresh = 1), the same scheme as scipy.sparse.linalg.splu;
//   - two steps of iterative refinement in working precision for every
//     solve, for both A and A^H, with the same factors.
// This is the scheme the repository's packed JAX solver validated
// (differometor/sparse_signal.py: Voyager noise off by 4.8e-3 in ln without
// refinement, 3e-13 with it), and it passes all 13 harness setups.
//
// Optional equilibration (DFM_EQUILIBRATE=1 at build time): A~ = R A C with
// exact power-of-two scalings before factorisation. It is off by default: it
// changes the round-off at near-null operating points (voyager_fm nominal,
// detected signal ~1e-21) so that log-sensitivity gradients there no longer
// track the reference, and refinement alone already reaches the accuracy the
// harness requires everywhere else.
#ifndef DFM_EQUILIBRATE
#define DFM_EQUILIBRATE 0
#endif

// SuperLU owns L and U (allocated by zgssv) and must release them with its
// own destructors. The scaled matrix is owned here (needed for refinement).
struct ComplexSparseLU::Impl {
  int n = 0;
  bool factors_allocated = false;
  SuperMatrix L{};
  SuperMatrix U{};
  std::vector<int> perm_c;
  std::vector<int> perm_r;
  std::vector<int> colptr;
  std::vector<int> rowind;
  std::vector<Complex> scaled;  ///< R A C, CSC aligned with colptr/rowind
  std::vector<double> row_scale;
  std::vector<double> col_scale;
  std::vector<Complex> rhs_copy, residual;  ///< refinement work buffers
  int refinement_steps = 2;

  ~Impl() {
    if (factors_allocated) {
      Destroy_SuperNode_Matrix(&L);
      Destroy_CompCol_Matrix(&U);
    }
  }

  void triangular_solve(Complex* x, int nrhs, trans_t trans) {
    SuperMatrix bmat;
    zCreate_Dense_Matrix(&bmat, n, nrhs, reinterpret_cast<doublecomplex*>(x), n, SLU_DN, SLU_Z, SLU_GE);
    SuperLUStat_t stat;
    StatInit(&stat);
    int info = 0;
    zgstrs(trans, &L, &U, perm_c.data(), perm_r.data(), &bmat, &stat, &info);
    StatFree(&stat);
    Destroy_SuperMatrix_Store(&bmat);
    if (info != 0) {
      throw std::runtime_error("ComplexSparseLU: zgstrs failed with info " + std::to_string(info));
    }
  }

  // residual = b - op(A~) y for every column.
  void compute_residual(const Complex* b, const Complex* y, int nrhs, trans_t trans) {
    std::copy(b, b + static_cast<std::size_t>(n) * nrhs, residual.begin());
    for (int c = 0; c < nrhs; ++c) {
      const Complex* yc = y + static_cast<std::size_t>(c) * n;
      Complex* rc = residual.data() + static_cast<std::size_t>(c) * n;
      for (int j = 0; j < n; ++j) {
        if (trans == NOTRANS) {
          const Complex yj = yc[j];
          for (int s = colptr[j]; s < colptr[j + 1]; ++s) rc[rowind[s]] -= scaled[s] * yj;
        } else {
          Complex acc(0.0, 0.0);
          for (int s = colptr[j]; s < colptr[j + 1]; ++s) acc += std::conj(scaled[s]) * yc[rowind[s]];
          rc[j] -= acc;
        }
      }
    }
  }

  void solve_in_place(VectorC& x, trans_t trans) {
    if (x.size() != n) {
      throw std::invalid_argument("ComplexSparseLU: right-hand side has size " +
                                  std::to_string(x.size()) + ", expected " + std::to_string(n));
    }
    solve_columns(x.data(), 1, trans);
  }

  // x is n x nrhs column-major; overwritten with the solution.
  void solve_columns(Complex* x, int nrhs, trans_t trans) {
    if (n == 0 || nrhs == 0) {
      return;
    }
    if (nrhs < 0) {
      throw std::invalid_argument("ComplexSparseLU: negative number of right-hand sides");
    }
    const std::vector<double>& pre = trans == NOTRANS ? row_scale : col_scale;
    const std::vector<double>& post = trans == NOTRANS ? col_scale : row_scale;
    const std::size_t total = static_cast<std::size_t>(n) * nrhs;
    for (int c = 0; c < nrhs; ++c) {
      for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(c) * n + i] *= pre[static_cast<std::size_t>(i)];
    }
    rhs_copy.assign(x, x + total);
    residual.resize(total);
    triangular_solve(x, nrhs, trans);
    for (int step = 0; step < refinement_steps; ++step) {
      compute_residual(rhs_copy.data(), x, nrhs, trans);
      triangular_solve(residual.data(), nrhs, trans);
      for (std::size_t k = 0; k < total; ++k) x[k] += residual[k];
    }
    for (int c = 0; c < nrhs; ++c) {
      for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(c) * n + i] *= post[static_cast<std::size_t>(i)];
    }
  }
};

namespace {

// 2^-e with max in [2^e, 2^(e+1)): exact scaling to [1, 2).
double power_of_two_reciprocal(double max_abs) {
  if (!(max_abs > 0.0) || !std::isfinite(max_abs)) return 1.0;
  return std::ldexp(1.0, -std::ilogb(max_abs));
}

}  // namespace

ComplexSparseLU::ComplexSparseLU(const SparseMatrixC& matrix) : impl_(std::make_unique<Impl>()) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("ComplexSparseLU: matrix must be square");
  }
  if (matrix.isCompressed()) {
    factorize(static_cast<int>(matrix.rows()), static_cast<int>(matrix.nonZeros()),
              matrix.outerIndexPtr(), matrix.innerIndexPtr(), matrix.valuePtr());
    return;
  }
  SparseMatrixC owned = matrix;
  owned.makeCompressed();
  factorize(static_cast<int>(owned.rows()), static_cast<int>(owned.nonZeros()),
            owned.outerIndexPtr(), owned.innerIndexPtr(), owned.valuePtr());
}

ComplexSparseLU::ComplexSparseLU(Eigen::Map<const SparseMatrixC> matrix)
    : impl_(std::make_unique<Impl>()) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("ComplexSparseLU: matrix must be square");
  }
  factorize(static_cast<int>(matrix.rows()), static_cast<int>(matrix.nonZeros()),
            matrix.outerIndexPtr(), matrix.innerIndexPtr(), matrix.valuePtr());
}

void ComplexSparseLU::factorize(int n, int nnz, const int* colptr, const int* rowind,
                                const Complex* values, LUOptions lu_options) {
  Impl& m = *impl_;
  m.n = n;
  m.refinement_steps = lu_options.refinement_steps;
  if (n == 0) {
    return;
  }

  // Own the structure and the equilibrated values (refinement needs them).
  m.colptr.assign(colptr, colptr + n + 1);
  m.rowind.assign(rowind, rowind + nnz);
  m.scaled.assign(values, values + nnz);
  std::vector<double> row_max(static_cast<std::size_t>(n), 0.0);
  for (int s = 0; s < nnz; ++s) {
    double& r = row_max[static_cast<std::size_t>(m.rowind[s])];
    r = std::max(r, std::abs(m.scaled[s]));
  }
  m.row_scale.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) m.row_scale[i] = DFM_EQUILIBRATE ? power_of_two_reciprocal(row_max[i]) : 1.0;
  for (int s = 0; s < nnz; ++s) m.scaled[s] *= m.row_scale[static_cast<std::size_t>(m.rowind[s])];
  m.col_scale.resize(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    double col_max = 0.0;
    for (int s = m.colptr[j]; s < m.colptr[j + 1]; ++s) col_max = std::max(col_max, std::abs(m.scaled[s]));
    m.col_scale[j] = DFM_EQUILIBRATE ? power_of_two_reciprocal(col_max) : 1.0;
    for (int s = m.colptr[j]; s < m.colptr[j + 1]; ++s) m.scaled[s] *= m.col_scale[j];
  }

  SuperMatrix amat;
  zCreate_CompCol_Matrix(&amat, n, n, nnz, reinterpret_cast<doublecomplex*>(m.scaled.data()), m.rowind.data(),
                         m.colptr.data(), SLU_NC, SLU_Z, SLU_GE);

  // zgssv factorises and also overwrites B with a solution. We only need the
  // factors, so B is a throwaway zero right-hand side.
  std::vector<Complex> scratch(static_cast<std::size_t>(n), Complex(0.0, 0.0));
  SuperMatrix bmat;
  zCreate_Dense_Matrix(&bmat, n, 1, reinterpret_cast<doublecomplex*>(scratch.data()), n, SLU_DN, SLU_Z, SLU_GE);

  superlu_options_t options;
  set_default_options(&options);
  // Partial pivoting (diag_pivot_thresh = 1) with either a fill-reducing
  // COLAMD ordering or the natural order (dense LAPACK's pivot sequence).
  options.ColPerm = lu_options.natural_ordering ? NATURAL : COLAMD;

  m.perm_c.assign(static_cast<std::size_t>(n), 0);
  m.perm_r.assign(static_cast<std::size_t>(n), 0);

  SuperLUStat_t stat;
  StatInit(&stat);
  int info = 0;
  zgssv(&options, &amat, m.perm_c.data(), m.perm_r.data(), &m.L, &m.U, &bmat, &stat, &info);
  StatFree(&stat);
  Destroy_SuperMatrix_Store(&bmat);
  Destroy_SuperMatrix_Store(&amat);

  // On a singular matrix SuperLU still returns L and U, so they are marked
  // for release before throwing (the Impl destructor runs while unwinding).
  m.factors_allocated = (info >= 0);
  if (info > 0) {
    throw std::runtime_error("ComplexSparseLU: matrix is singular (zero pivot at column " + std::to_string(info) +
                             ")");
  }
  if (info < 0) {
    throw std::runtime_error("ComplexSparseLU: SuperLU zgssv argument " + std::to_string(-info) +
                             " had an illegal value");
  }
}

ComplexSparseLU::~ComplexSparseLU() = default;
ComplexSparseLU::ComplexSparseLU(ComplexSparseLU&&) noexcept = default;
ComplexSparseLU& ComplexSparseLU::operator=(ComplexSparseLU&&) noexcept = default;

int ComplexSparseLU::size() const noexcept { return impl_->n; }

VectorC ComplexSparseLU::solve(const Eigen::Ref<const VectorC>& rhs) const {
  VectorC x = rhs;
  impl_->solve_in_place(x, NOTRANS);
  return x;
}

VectorC ComplexSparseLU::solve_conjugate_transpose(const Eigen::Ref<const VectorC>& rhs) const {
  VectorC x = rhs;
  impl_->solve_in_place(x, CONJ);
  return x;
}

ComplexSparseLU::ComplexSparseLU(const SparsePattern& pattern, const Complex* values, LUOptions options)
    : impl_(std::make_unique<Impl>()) {
  factorize(pattern.size(), pattern.nnz(), pattern.col_ptr().data(), pattern.row_idx().data(), values, options);
}

void ComplexSparseLU::solve_inplace(Complex* x, int nrhs) const { impl_->solve_columns(x, nrhs, NOTRANS); }

void ComplexSparseLU::solve_conjugate_transpose_inplace(Complex* x, int nrhs) const {
  impl_->solve_columns(x, nrhs, CONJ);
}

void ComplexSparseLU::solve_inplace(VectorC& x) const { impl_->solve_in_place(x, NOTRANS); }

void ComplexSparseLU::solve_conjugate_transpose_inplace(VectorC& x) const {
  // CONJ solves A^H x = b with the same factors, i.e. no refactorisation.
  impl_->solve_in_place(x, CONJ);
}

}  // namespace dfm
