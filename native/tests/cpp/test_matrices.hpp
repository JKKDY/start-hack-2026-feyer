#pragma once

// Shared deterministic test data for the C++ suites.

#include <Eigen/Sparse>

#include <complex>
#include <vector>

#include "dfm/sparse_lu.hpp"

namespace dfm::testing {

// Non-symmetric complex sparse matrix with a dominant diagonal (well
// conditioned) and a few off-diagonal couplings, so conjugation mistakes
// show up in the results.
inline SparseMatrixC make_test_matrix(int n) {
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
  SparseMatrixC a(n, n);
  a.setFromTriplets(triplets.begin(), triplets.end());
  a.makeCompressed();
  return a;
}

inline VectorC make_rhs(int n) {
  VectorC b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = Complex(1.0 + i, -0.5 * i);
  }
  return b;
}

}  // namespace dfm::testing
