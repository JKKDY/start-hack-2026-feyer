#pragma once

#include <Eigen/Sparse>

#include <complex>
#include <memory>

namespace dfm {

using Complex = std::complex<double>;
using SparseMatrixC = Eigen::SparseMatrix<Complex, Eigen::ColMajor, int>;
using VectorC = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;

/// Sparse LU factorisation of a complex square matrix, backed by SuperLU.
///
/// The matrix is factorised once (pivoted LU with a fill-reducing column
/// ordering). The factors are then reused for:
///   * solve(b)                 -> x with A x = b
///   * solve_conjugate_transpose(b) -> x with A^H x = b
/// The second one is what the adjoint (reverse-mode) pass needs, so no
/// second factorisation is ever required.
class ComplexSparseLU {
 public:
  /// Factorises `matrix`. Throws std::invalid_argument if it is not square
  /// and std::runtime_error if it is singular.
  explicit ComplexSparseLU(const SparseMatrixC& matrix);
  ~ComplexSparseLU();

  ComplexSparseLU(const ComplexSparseLU&) = delete;
  ComplexSparseLU& operator=(const ComplexSparseLU&) = delete;
  ComplexSparseLU(ComplexSparseLU&&) noexcept;
  ComplexSparseLU& operator=(ComplexSparseLU&&) noexcept;

  int size() const noexcept;

  VectorC solve(const VectorC& rhs) const;
  VectorC solve_conjugate_transpose(const VectorC& rhs) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dfm
