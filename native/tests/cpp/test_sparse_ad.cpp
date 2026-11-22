// Sparse storage and recorded sparse operations (implicit adjoints) against
// finite differences, with arbitrary complex cotangents and active matrices.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "dfm/sparse.hpp"
#include "dfm/sparse_ad.hpp"
#include "dfm/sparse_lu.hpp"

namespace {

using dfm::ActiveReal;
using dfm::Complex;
using dfm::Op;

struct Problem {
  dfm::SparsePatternPtr pattern;
  std::vector<double> a;  // 2 nnz interleaved
  std::vector<double> b;  // 2 n m interleaved, column-major
  std::vector<Complex> cot;
  int n = 0;
  int m = 0;
};

Problem make_problem(int n, int m, unsigned seed) {
  std::vector<int> rows, cols;
  for (int i = 0; i < n; ++i) {
    rows.push_back(i);
    cols.push_back(i);
    if (i + 1 < n) {
      rows.push_back(i);
      cols.push_back(i + 1);
      rows.push_back(i + 1);
      cols.push_back(i);
    }
    if (i + 3 < n) {
      rows.push_back(i + 3);
      cols.push_back(i);
    }
  }
  rows.push_back(0);  // a duplicate coordinate: shares a slot
  cols.push_back(0);
  auto [pattern, slots] = dfm::SparsePattern::from_coordinates(n, rows, cols);
  Problem p;
  p.n = n;
  p.m = m;
  p.pattern = std::make_shared<const dfm::SparsePattern>(std::move(pattern));
  std::mt19937 rng(seed);
  std::normal_distribution<double> g;
  const auto slot_cols = p.pattern->slot_columns();
  for (int s = 0; s < p.pattern->nnz(); ++s) {
    const bool diagonal = p.pattern->row_idx()[s] == slot_cols[s];
    p.a.push_back((diagonal ? 4.0 : 0.0) + g(rng));
    p.a.push_back(g(rng));
  }
  for (int k = 0; k < 2 * n * m; ++k) p.b.push_back(g(rng));
  for (int k = 0; k < n * m; ++k) p.cot.emplace_back(g(rng), g(rng));
  return p;
}

// L = sum_k Re(conj(cot_k) * out_k), computed passively.
using Forward = std::function<std::vector<Complex>(const Problem&, const std::vector<double>&,
                                                   const std::vector<double>&)>;

double loss(const Problem& p, const Forward& fwd, const std::vector<double>& a, const std::vector<double>& b) {
  const auto out = fwd(p, a, b);
  double total = 0.0;
  for (std::size_t k = 0; k < out.size(); ++k) total += std::real(std::conj(p.cot[k]) * out[k]);
  return total;
}

std::vector<Complex> passive_solve(const Problem& p, const std::vector<double>& a, const std::vector<double>& b,
                                   Op op) {
  dfm::ComplexSparseLU lu(*p.pattern, reinterpret_cast<const Complex*>(a.data()));
  std::vector<Complex> x(reinterpret_cast<const Complex*>(b.data()),
                         reinterpret_cast<const Complex*>(b.data()) + p.n * p.m);
  if (op == Op::Normal) {
    lu.solve_inplace(x.data(), p.m);
  } else {
    lu.solve_conjugate_transpose_inplace(x.data(), p.m);
  }
  return x;
}

std::vector<Complex> passive_matmul(const Problem& p, const std::vector<double>& a, const std::vector<double>& b,
                                    Op op) {
  std::vector<Complex> y(static_cast<std::size_t>(p.n * p.m));
  const auto* av = reinterpret_cast<const Complex*>(a.data());
  const auto* bv = reinterpret_cast<const Complex*>(b.data());
  if (op == Op::Normal) {
    dfm::sparse_matmul(*p.pattern, av, bv, p.m, y.data());
  } else {
    dfm::sparse_matmul_adjoint(*p.pattern, av, bv, p.m, y.data());
  }
  return y;
}

using Recorder = std::function<void(const Problem&, const ActiveReal*, const ActiveReal*, ActiveReal*)>;

void check_against_fd(const Problem& p, const Forward& fwd, const Recorder& record) {
  auto& tape = ActiveReal::getTape();
  tape.reset();
  tape.setActive();
  std::vector<ActiveReal> a(p.a.begin(), p.a.end()), b(p.b.begin(), p.b.end());
  for (auto& v : a) tape.registerInput(v);
  for (auto& v : b) tape.registerInput(v);
  std::vector<ActiveReal> out(2 * p.n * p.m);
  record(p, a.data(), b.data(), out.data());
  for (auto& v : out) tape.registerOutput(v);
  tape.setPassive();
  for (int k = 0; k < p.n * p.m; ++k) {
    out[2 * k].setGradient(p.cot[k].real());
    out[2 * k + 1].setGradient(p.cot[k].imag());
  }
  tape.evaluate();

  // Primal values match the passive operation.
  const auto ref = fwd(p, p.a, p.b);
  for (int k = 0; k < p.n * p.m; ++k) {
    EXPECT_NEAR(out[2 * k].getValue(), ref[k].real(), 1e-12 * (1.0 + std::abs(ref[k])));
    EXPECT_NEAR(out[2 * k + 1].getValue(), ref[k].imag(), 1e-12 * (1.0 + std::abs(ref[k])));
  }

  const double h = 1e-6;
  for (std::size_t k = 0; k < p.a.size(); ++k) {
    std::vector<double> ac = p.a;
    ac[k] += h;
    const double up = loss(p, fwd, ac, p.b);
    ac[k] -= 2 * h;
    const double down = loss(p, fwd, ac, p.b);
    const double g = (up - down) / (2 * h);
    EXPECT_NEAR(a[k].getGradient(), g, 1e-6 * (1.0 + std::abs(g))) << "A component " << k;
  }
  for (std::size_t k = 0; k < p.b.size(); ++k) {
    std::vector<double> bc = p.b;
    bc[k] += h;
    const double up = loss(p, fwd, p.a, bc);
    bc[k] -= 2 * h;
    const double down = loss(p, fwd, p.a, bc);
    const double g = (up - down) / (2 * h);
    EXPECT_NEAR(b[k].getGradient(), g, 1e-6 * (1.0 + std::abs(g))) << "B component " << k;
  }
  tape.reset();
}

class RecordedSolve : public ::testing::TestWithParam<Op> {};

TEST_P(RecordedSolve, GradientsMatchFiniteDifferences) {
  const Op op = GetParam();
  const Problem p = make_problem(9, 3, 7);
  check_against_fd(
      p, [op](const Problem& q, const auto& a, const auto& b) { return passive_solve(q, a, b, op); },
      [op](const Problem& q, const ActiveReal* a, const ActiveReal* b, ActiveReal* x) {
        dfm::RecordedLinearSystem system(q.pattern, a);
        system.solve(b, q.m, op, x);
      });
}

TEST_P(RecordedSolve, MatmulGradientsMatchFiniteDifferences) {
  const Op op = GetParam();
  const Problem p = make_problem(8, 2, 11);
  check_against_fd(
      p, [op](const Problem& q, const auto& a, const auto& b) { return passive_matmul(q, a, b, op); },
      [op](const Problem& q, const ActiveReal* a, const ActiveReal* b, ActiveReal* y) {
        dfm::sparse_matmul_recorded(q.pattern, a, b, q.m, op, y);
      });
}

INSTANTIATE_TEST_SUITE_P(Ops, RecordedSolve, ::testing::Values(Op::Normal, Op::ConjugateTranspose),
                         [](const ::testing::TestParamInfo<Op>& info) {
                           return info.param == Op::Normal ? std::string("Normal") : std::string("ConjTranspose");
                         });

TEST(RecordedSolve, OneFactorisationServesSeveralRecordedSolves) {
  // Signal solve + adjoint (noise weight) solve on the same system: both
  // records share the factors; the gradient is the sum of both paths.
  const Problem p = make_problem(7, 1, 3);
  auto fwd = [](const Problem& q, const std::vector<double>& a, const std::vector<double>& b) {
    auto x = passive_solve(q, a, b, Op::Normal);
    std::vector<double> xi(2 * x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      xi[2 * k] = x[k].real();
      xi[2 * k + 1] = x[k].imag();
    }
    return passive_solve(q, a, xi, Op::ConjugateTranspose);
  };
  check_against_fd(p, fwd, [](const Problem& q, const ActiveReal* a, const ActiveReal* b, ActiveReal* y) {
    dfm::RecordedLinearSystem system(q.pattern, a);
    std::vector<ActiveReal> x(2 * q.n * q.m);
    system.solve(b, q.m, Op::Normal, x.data());
    system.solve(x.data(), q.m, Op::ConjugateTranspose, y);
  });
}

TEST(SparsePattern, CoordinatesMergeDuplicatesAndFindSlots) {
  auto [pattern, slots] = dfm::SparsePattern::from_coordinates(3, {2, 0, 0, 1, 2}, {0, 0, 0, 1, 2});
  EXPECT_EQ(pattern.nnz(), 4);
  EXPECT_EQ(slots[1], slots[2]);
  EXPECT_EQ(pattern.find(2, 0), slots[0]);
  EXPECT_EQ(pattern.find(1, 0), -1);
  EXPECT_THROW(dfm::SparsePattern(2, {0, 2, 2}, {1, 0}), std::invalid_argument);
}

TEST(SparseLU, MultipleRightHandSidesMatchSingleSolves) {
  const Problem p = make_problem(10, 4, 5);
  const auto* av = reinterpret_cast<const Complex*>(p.a.data());
  dfm::ComplexSparseLU lu(*p.pattern, av);
  std::vector<Complex> all(reinterpret_cast<const Complex*>(p.b.data()),
                           reinterpret_cast<const Complex*>(p.b.data()) + p.n * p.m);
  lu.solve_conjugate_transpose_inplace(all.data(), p.m);
  for (int c = 0; c < p.m; ++c) {
    dfm::VectorC one = Eigen::Map<const dfm::VectorC>(reinterpret_cast<const Complex*>(p.b.data()) + c * p.n, p.n);
    lu.solve_conjugate_transpose_inplace(one);
    for (int i = 0; i < p.n; ++i) EXPECT_NEAR(std::abs(one(i) - all[c * p.n + i]), 0.0, 1e-13);
  }
}

}  // namespace
