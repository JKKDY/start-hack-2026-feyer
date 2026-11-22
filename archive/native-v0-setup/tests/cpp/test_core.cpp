// Minimal C++ smoke tests for dfm_core. No test framework: each check prints
// its result and the process exits non-zero if any check failed.

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <complex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "dfm/ad_eigen.hpp"
#include "dfm/sparse_lu.hpp"
#include "dfm/version.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& name) {
  std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name.c_str());
  if (!condition) {
    ++failures;
  }
}

using Complex = std::complex<double>;

// Deterministic, non-symmetric, complex sparse matrix with a dominant
// diagonal (so it is well conditioned) and a few off-diagonal couplings.
dfm::SparseMatrixC make_test_matrix(int n) {
  std::vector<Eigen::Triplet<Complex>> triplets;
  for (int i = 0; i < n; ++i) {
    triplets.emplace_back(i, i, Complex(4.0 + 0.1 * i, 0.5 - 0.05 * i));
    if (i + 1 < n) {
      triplets.emplace_back(i, i + 1, Complex(-1.0, 0.3 * (i % 3)));
      triplets.emplace_back(i + 1, i, Complex(0.2, -0.7));
    }
    if (i + 3 < n) {
      triplets.emplace_back(i, i + 3, Complex(0.0, 0.4));
    }
  }
  dfm::SparseMatrixC a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  a.makeCompressed();
  return a;
}

dfm::VectorC make_rhs(int n) {
  dfm::VectorC b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = Complex(1.0 + i, -0.5 * i);
  }
  return b;
}

void test_version() {
  const std::string v = dfm::version();
  check(!v.empty(), "version() returns a non-empty string (" + v + ")");
}

void test_sparse_solve_matches_dense() {
  const int n = 12;
  const dfm::SparseMatrixC a = make_test_matrix(n);
  const dfm::VectorC b = make_rhs(n);

  dfm::ComplexSparseLU lu(a);
  const dfm::VectorC x = lu.solve(b);

  const Eigen::MatrixXcd dense = Eigen::MatrixXcd(a);
  const dfm::VectorC x_ref = dense.fullPivLu().solve(b);

  const double residual = (a * x - b).norm() / b.norm();
  const double diff = (x - x_ref).norm() / x_ref.norm();
  check(residual < 1e-12, "complex sparse solve residual ||Ax-b||/||b|| = " + std::to_string(residual));
  check(diff < 1e-12, "complex sparse solve matches dense LU = " + std::to_string(diff));
}

void test_adjoint_solve_reuses_factors() {
  const int n = 12;
  const dfm::SparseMatrixC a = make_test_matrix(n);
  const dfm::VectorC b = make_rhs(n);

  dfm::ComplexSparseLU lu(a);
  const dfm::VectorC y = lu.solve_conjugate_transpose(b);

  // A^H y = b, where A^H is the conjugate transpose of A.
  const dfm::SparseMatrixC ah = a.adjoint();
  const double residual = (ah * y - b).norm() / b.norm();
  check(residual < 1e-12,
        "adjoint solve residual ||A^H y - b||/||b|| = " + std::to_string(residual));
}

void test_singular_matrix_throws() {
  dfm::SparseMatrixC a(3, 3);
  a.insert(0, 0) = Complex(1.0, 0.0);
  a.insert(1, 1) = Complex(0.0, 0.0);  // structurally present, numerically zero
  a.insert(2, 2) = Complex(1.0, 1.0);
  a.makeCompressed();

  bool threw = false;
  try {
    dfm::ComplexSparseLU lu(a);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "singular matrix raises std::runtime_error");
}

void test_empty_matrix() {
  dfm::SparseMatrixC a(0, 0);
  dfm::ComplexSparseLU lu(a);
  const dfm::VectorC x = lu.solve(dfm::VectorC(0));
  check(lu.size() == 0 && x.size() == 0, "empty (0x0) matrix factorises and solves");
}

void test_gradient_smoke() {
  const dfm::GradientSmokeResult r = dfm::complex_gradient_smoke({0.3, -0.2, 0.7});
  check(r.max_abs_error < 1e-10,
        "CoDiPack complex Eigen gradient matches analytic (max error " +
            std::to_string(r.max_abs_error) + ")");
}

}  // namespace

int main() {
  test_version();
  test_sparse_solve_matches_dense();
  test_adjoint_solve_reuses_factors();
  test_singular_matrix_throws();
  test_empty_matrix();
  test_gradient_smoke();

  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
