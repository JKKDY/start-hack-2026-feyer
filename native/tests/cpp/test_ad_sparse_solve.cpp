#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/sparse_lu.hpp"
#include "test_matrices.hpp"

namespace {

using dfm::ActiveComplex;
using dfm::ActiveReal;
using dfm::ActiveVectorC;
using dfm::ComplexSparseLU;
using dfm::Complex;
using dfm::VectorC;

// L(b) = sum_i w_i |y_i|^2 with y = A^{-1} b, computed in plain doubles.
double loss_passive(const ComplexSparseLU& lu, const std::vector<double>& w,
                    const std::vector<double>& b_ri) {
  const int n = lu.size();
  VectorC b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = Complex(b_ri[2 * i], b_ri[2 * i + 1]);
  }
  const VectorC y = lu.solve(b);
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    total += w[i] * std::norm(y(i));
  }
  return total;
}

// Same loss, but recorded on the CoDiPack tape. Returns the reverse-mode
// gradient with respect to the interleaved real inputs.
std::vector<double> gradient_ad(const ComplexSparseLU& lu, const std::vector<double>& w,
                                const std::vector<double>& b_ri, double* value_out) {
  using Tape = ActiveReal::Tape;
  Tape& tape = ActiveReal::getTape();
  tape.reset();
  tape.setActive();

  const int n = lu.size();
  std::vector<ActiveReal> inputs(b_ri.begin(), b_ri.end());
  for (ActiveReal& v : inputs) {
    tape.registerInput(v);
  }

  ActiveVectorC b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = ActiveComplex(inputs[2 * i], inputs[2 * i + 1]);
  }

  const ActiveVectorC y = dfm::solve_recorded(lu, b);

  ActiveReal loss = 0.0;
  for (int i = 0; i < n; ++i) {
    const ActiveReal re = y(i).real();
    const ActiveReal im = y(i).imag();
    loss += w[i] * (re * re + im * im);
  }

  tape.registerOutput(loss);
  loss.setGradient(1.0);
  tape.setPassive();
  tape.evaluate();

  if (value_out != nullptr) {
    *value_out = loss.getValue();
  }
  std::vector<double> grad(inputs.size());
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    grad[k] = inputs[k].getGradient();
  }
  tape.reset();
  return grad;
}

std::vector<double> gradient_fd(const ComplexSparseLU& lu, const std::vector<double>& w,
                                std::vector<double> b_ri) {
  std::vector<double> grad(b_ri.size());
  for (std::size_t k = 0; k < b_ri.size(); ++k) {
    const double h = 1e-6 * std::max(1.0, std::abs(b_ri[k]));
    const double saved = b_ri[k];
    b_ri[k] = saved + h;
    const double up = loss_passive(lu, w, b_ri);
    b_ri[k] = saved - h;
    const double down = loss_passive(lu, w, b_ri);
    b_ri[k] = saved;
    grad[k] = (up - down) / (2.0 * h);
  }
  return grad;
}

std::vector<double> make_weights(int n) {
  std::vector<double> w(n);
  for (int i = 0; i < n; ++i) {
    w[i] = 0.5 + 0.1 * i;
  }
  return w;
}

std::vector<double> make_inputs(int n) {
  std::vector<double> b(2 * n);
  for (int i = 0; i < n; ++i) {
    b[2 * i] = 1.0 + 0.3 * i;
    b[2 * i + 1] = -0.5 + 0.2 * i;
  }
  return b;
}

TEST(AdSparseSolve, ValueMatchesPlainSolve) {
  const int n = 10;
  const auto a = dfm::testing::make_test_matrix(n);
  ComplexSparseLU lu(a);

  const std::vector<double> b_ri = make_inputs(n);
  const std::vector<double> w = make_weights(n);
  double value = 0.0;
  gradient_ad(lu, w, b_ri, &value);
  EXPECT_NEAR(value, loss_passive(lu, w, b_ri), 1e-12 * std::abs(value));
}

TEST(AdSparseSolve, GradientMatchesFiniteDifferences) {
  const int n = 10;
  const auto a = dfm::testing::make_test_matrix(n);
  ComplexSparseLU lu(a);

  const std::vector<double> b_ri = make_inputs(n);
  const std::vector<double> w = make_weights(n);
  const std::vector<double> ad = gradient_ad(lu, w, b_ri, nullptr);
  const std::vector<double> fd = gradient_fd(lu, w, b_ri);

  ASSERT_EQ(ad.size(), fd.size());
  for (std::size_t k = 0; k < ad.size(); ++k) {
    EXPECT_NEAR(ad[k], fd[k], 1e-6 * (1.0 + std::abs(fd[k]))) << "input component " << k;
  }
}

TEST(AdSparseSolve, NonHermitianMatrixAdjointIsConjugateTranspose) {
  // A deliberately asymmetric matrix: if the reverse rule used A instead of
  // A^H, the gradient would disagree with finite differences here.
  const int n = 7;
  const auto a = dfm::testing::make_test_matrix(n);
  ComplexSparseLU lu(a);

  const std::vector<double> b_ri = make_inputs(n);
  std::vector<double> w(n, 0.0);
  w[0] = 1.0;  // only the first output contributes

  const std::vector<double> ad = gradient_ad(lu, w, b_ri, nullptr);
  const std::vector<double> fd = gradient_fd(lu, w, b_ri);
  for (std::size_t k = 0; k < ad.size(); ++k) {
    EXPECT_NEAR(ad[k], fd[k], 1e-6 * (1.0 + std::abs(fd[k]))) << "input component " << k;
  }
}

TEST(AdSparseSolve, ChainedSolvesDifferentiate) {
  // Two solves in sequence: the second consumes the first's output. This
  // checks that outputs of one external function feed the next correctly.
  const int n = 8;
  const auto a = dfm::testing::make_test_matrix(n);
  ComplexSparseLU lu(a);

  using Tape = ActiveReal::Tape;
  Tape& tape = ActiveReal::getTape();
  tape.reset();
  tape.setActive();

  const std::vector<double> b_ri = make_inputs(n);
  std::vector<ActiveReal> inputs(b_ri.begin(), b_ri.end());
  for (ActiveReal& v : inputs) {
    tape.registerInput(v);
  }
  ActiveVectorC b(n);
  for (int i = 0; i < n; ++i) {
    b(i) = ActiveComplex(inputs[2 * i], inputs[2 * i + 1]);
  }
  const ActiveVectorC y1 = dfm::solve_recorded(lu, b);
  const ActiveVectorC y2 = dfm::solve_recorded(lu, y1);

  ActiveReal loss = 0.0;
  for (int i = 0; i < n; ++i) {
    loss += y2(i).real() * y2(i).real() + y2(i).imag() * y2(i).imag();
  }
  tape.registerOutput(loss);
  loss.setGradient(1.0);
  tape.setPassive();
  tape.evaluate();

  std::vector<double> ad(inputs.size());
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    ad[k] = inputs[k].getGradient();
  }
  tape.reset();

  // Finite differences on the composed map b -> A^{-1}(A^{-1} b).
  auto composed = [&](std::vector<double> ri) {
    VectorC v(n);
    for (int i = 0; i < n; ++i) {
      v(i) = Complex(ri[2 * i], ri[2 * i + 1]);
    }
    const VectorC out = lu.solve(lu.solve(v));
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
      total += std::norm(out(i));
    }
    return total;
  };
  for (std::size_t k = 0; k < ad.size(); ++k) {
    std::vector<double> up = b_ri;
    std::vector<double> down = b_ri;
    const double h = 1e-6 * std::max(1.0, std::abs(b_ri[k]));
    up[k] += h;
    down[k] -= h;
    const double fd = (composed(up) - composed(down)) / (2.0 * h);
    EXPECT_NEAR(ad[k], fd, 1e-6 * (1.0 + std::abs(fd))) << "input component " << k;
  }
}

TEST(AdSparseSolve, RejectsWrongSize) {
  const auto a = dfm::testing::make_test_matrix(4);
  ComplexSparseLU lu(a);
  ActiveVectorC b(3);
  EXPECT_THROW(dfm::solve_recorded(lu, b), std::invalid_argument);
}

}  // namespace
