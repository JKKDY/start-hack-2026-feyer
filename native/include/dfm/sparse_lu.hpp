#pragma once

#include <Eigen/Sparse>

#include <memory>

#include "dfm/types.hpp"

namespace dfm {

class SparsePattern;

/// Factorisation options.
struct LUOptions {
  /// false: COLAMD fill-reducing column ordering (default).
  /// true: natural column order. With partial pivoting this reproduces the
  /// pivot sequence of dense LAPACK LU (zgetrf), which upstream uses; useful
  /// where results must track the dense reference's rounding (carrier).
  bool natural_ordering = false;
  /// Iterative-refinement steps per solve (working precision).
  int refinement_steps = 2;
};

using SparseMatrixC = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;

/// Sparse LU factorisation of a complex square matrix, backed by SuperLU.
///
/// The matrix is factorised once (pivoted LU with a fill-reducing column
/// ordering). The factors are then reused for every solve:
///   * solve(b)                     -> x with A x = b
///   * solve_conjugate_transpose(b) -> x with A^H x = b
/// The second one is what the adjoint pass needs, so the adjoint never
/// refactorises.
///
/// Accuracy: the matrix is equilibrated with exact power-of-two row and
/// column scalings before factorisation, and every solve does two steps of
/// iterative refinement (see sparse_lu.cpp). Optical matrices are badly
/// scaled (condition numbers up to ~1e31); this keeps backward errors small.
///
/// Thread safety: solves reuse internal work buffers, so one factorisation
/// must not be used by concurrent solves (use one per thread).
class ComplexSparseLU {
 public:
  /// Factorises `matrix`. Only a square matrix is accepted. A compressed
  /// matrix is used without copying its arrays. Throws std::invalid_argument
  /// if it is not square and std::runtime_error if it is singular.
  explicit ComplexSparseLU(const SparseMatrixC& matrix);
  /// Same, for a read-only view (e.g. over NumPy memory). Never copies.
  explicit ComplexSparseLU(Eigen::Map<const SparseMatrixC> matrix);
  /// From a SparsePattern and its slot-aligned values. Never copies.
  ComplexSparseLU(const SparsePattern& pattern, const Complex* values, LUOptions options = {});
  ~ComplexSparseLU();

  ComplexSparseLU(const ComplexSparseLU&) = delete;
  ComplexSparseLU& operator=(const ComplexSparseLU&) = delete;
  ComplexSparseLU(ComplexSparseLU&&) noexcept;
  ComplexSparseLU& operator=(ComplexSparseLU&&) noexcept;

  int size() const noexcept;

  /// Returns x with A x = rhs.
  VectorC solve(const Eigen::Ref<const VectorC>& rhs) const;
  /// Returns x with A^H x = rhs.
  VectorC solve_conjugate_transpose(const Eigen::Ref<const VectorC>& rhs) const;

  /// In-place variants: `x` holds the right-hand side on entry and the
  /// solution on exit. These avoid allocating a result vector.
  void solve_inplace(VectorC& x) const;
  void solve_conjugate_transpose_inplace(VectorC& x) const;

  /// Multiple right-hand sides at once: `x` is n x nrhs, column-major
  /// (column c starts at x + c * n). One SuperLU call for all columns.
  void solve_inplace(Complex* x, int nrhs) const;
  void solve_conjugate_transpose_inplace(Complex* x, int nrhs) const;

 private:
  struct Impl;
  void factorize(int n, int nnz, const int* colptr, const int* rowind, const Complex* values,
                 LUOptions options = {});
  std::unique_ptr<Impl> impl_;
};

}  // namespace dfm
