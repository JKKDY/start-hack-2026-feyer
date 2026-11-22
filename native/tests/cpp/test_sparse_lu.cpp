#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <stdexcept>

#include "dfm/sparse_lu.hpp"
#include "test_matrices.hpp"

namespace {

using dfm::ComplexSparseLU;
using dfm::testing::make_rhs;
using dfm::testing::make_test_matrix;

TEST(SparseLU, SolveMatchesDenseLU) {
  const int n = 12;
  const auto a = make_test_matrix(n);
  const auto b = make_rhs(n);

  ComplexSparseLU lu(a);
  const dfm::VectorC x = lu.solve(b);
  const dfm::VectorC x_ref = Eigen::MatrixXcd(a).fullPivLu().solve(b);

  EXPECT_LT((a * x - b).norm() / b.norm(), 1e-12);
  EXPECT_LT((x - x_ref).norm() / x_ref.norm(), 1e-12);
}

TEST(SparseLU, ConjugateTransposeSolveReusesFactors) {
  const int n = 12;
  const auto a = make_test_matrix(n);
  const auto b = make_rhs(n);

  ComplexSparseLU lu(a);
  const dfm::VectorC y = lu.solve_conjugate_transpose(b);

  const dfm::SparseMatrixC ah = a.adjoint();
  EXPECT_LT((ah * y - b).norm() / b.norm(), 1e-12);
}

TEST(SparseLU, InPlaceVariantsMatchAllocatingVariants) {
  const int n = 9;
  const auto a = make_test_matrix(n);
  const auto b = make_rhs(n);

  ComplexSparseLU lu(a);

  dfm::VectorC x = b;
  lu.solve_inplace(x);
  EXPECT_LT((x - lu.solve(b)).norm(), 1e-14 * x.norm());

  dfm::VectorC y = b;
  lu.solve_conjugate_transpose_inplace(y);
  EXPECT_LT((y - lu.solve_conjugate_transpose(b)).norm(), 1e-14 * y.norm());
}

TEST(SparseLU, AcceptsUncompressedMatrix) {
  const int n = 6;
  dfm::SparseMatrixC a(n, n);
  for (int i = 0; i < n; ++i) {
    a.insert(i, i) = dfm::Complex(2.0 + i, 0.25);
  }
  // Not compressed: the constructor must compress a private copy.
  ASSERT_FALSE(a.isCompressed());

  ComplexSparseLU lu(a);
  const auto b = make_rhs(n);
  EXPECT_LT((a * lu.solve(b) - b).norm(), 1e-12);
}

TEST(SparseLU, SingularMatrixThrows) {
  dfm::SparseMatrixC a(3, 3);
  a.insert(0, 0) = dfm::Complex(1.0, 0.0);
  a.insert(1, 1) = dfm::Complex(0.0, 0.0);  // structurally present, numerically zero
  a.insert(2, 2) = dfm::Complex(1.0, 1.0);
  a.makeCompressed();

  EXPECT_THROW(ComplexSparseLU lu(a), std::runtime_error);
}

TEST(SparseLU, NonSquareMatrixThrows) {
  dfm::SparseMatrixC a(3, 4);
  a.setIdentity();
  EXPECT_THROW(ComplexSparseLU lu(a), std::invalid_argument);
}

TEST(SparseLU, WrongRhsSizeThrows) {
  const auto a = make_test_matrix(4);
  ComplexSparseLU lu(a);
  EXPECT_THROW(lu.solve(make_rhs(3)), std::invalid_argument);
}

TEST(SparseLU, EmptyMatrix) {
  dfm::SparseMatrixC a(0, 0);
  ComplexSparseLU lu(a);
  EXPECT_EQ(lu.size(), 0);
  EXPECT_EQ(lu.solve(dfm::VectorC(0)).size(), 0);
}

}  // namespace
