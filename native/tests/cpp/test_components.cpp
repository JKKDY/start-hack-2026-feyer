// Component kernels: Jacobians (jet) vs central finite differences, and the
// CoDiPack recording path vs the jet VJP. Values against JAX are checked in
// tests/python/test_kernels_vs_jax.py.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "dfm/component_ad.hpp"
#include "dfm/component_ops.hpp"
#include "dfm/components.hpp"
#include "dfm/link.hpp"

namespace {

using dfm::Complex;
using dfm::ComponentFunction;

// Physically valid input ranges per kernel (upstream parameter meaning).
std::vector<std::pair<double, double>> ranges(ComponentFunction f) {
  using CF = ComponentFunction;
  const std::pair<double, double> freq{100.0, 3000.0}, tuning{0.0, 90.0}, refl{0.1, 0.9}, loss{0.0, 0.1},
      n{1.0, 1.4}, alpha{-40.0, 40.0}, angle{-180.0, 180.0}, length{1.0, 4000.0};
  switch (f) {
    case CF::ForceCalculationRight: return {alpha, n, n};
    case CF::Laser: return {{0.1, 10.0}, angle};
    case CF::Surface: return {loss, refl, tuning, {-3000.0, 3000.0}, n, n, alpha};
    case CF::Space:
    case CF::SpaceLower: return {{-3000.0, 3000.0}, length, n};
    case CF::SpaceModulation:
    case CF::SpaceModulationLower: return {freq, length, n};
    case CF::SignalFunction: return {{0.1, 5.0}, angle};
    case CF::LossQuantumNoise: return {{0.0, 1.0}};
    case CF::LaserFrequencyModulation:
    case CF::LaserFrequencyModulationLower: return {freq};
    case CF::Susceptibility: return {freq, {0.1, 100.0}};
    case CF::ForceCalculationLeft: return {alpha};
    case CF::CorrectedOptomechanicalPhaseLeft: return {freq, tuning, refl, loss, n, alpha};
    case CF::CorrectedOptomechanicalPhaseRight: return {freq, tuning, refl, loss, n, n, alpha};
    case CF::Squeezer: return {{0.0, 15.0}, angle};
    default: return {};
  }
}

std::vector<double> sample(ComponentFunction f, std::mt19937& rng) {
  std::vector<double> p;
  for (auto [lo, hi] : ranges(f)) p.push_back(std::uniform_real_distribution<double>(lo, hi)(rng));
  p.resize(dfm::kComponentInputWidth, 0.0);  // upstream pads rows to 7
  return p;
}

class KernelTest : public ::testing::TestWithParam<int> {};

TEST_P(KernelTest, JacobianMatchesFiniteDifferences) {
  const auto f = static_cast<ComponentFunction>(GetParam());
  const int k = dfm::arity(f);
  ASSERT_EQ(static_cast<int>(ranges(f).size()), k) << dfm::name(f);
  std::mt19937 rng(1234 + GetParam());

  for (int trial = 0; trial < 3; ++trial) {
    std::vector<double> p = sample(f, rng);
    std::array<Complex, 4> y{};
    std::vector<Complex> jac(static_cast<std::size_t>(4 * std::max(k, 1)));
    dfm::component_jacobian(f, p.data(), y.data(), jac.data());

    std::array<Complex, 4> y0{};
    dfm::evaluate_component(f, p.data(), y0.data());
    for (int j = 0; j < 4; ++j) EXPECT_EQ(y[j], y0[j]) << dfm::name(f);

    for (int c = 0; c < k; ++c) {
      const double h = 1e-6 * std::max(1.0, std::abs(p[c]));
      std::vector<double> up = p, down = p;
      up[c] += h;
      down[c] -= h;
      std::array<Complex, 4> yu{}, yd{};
      dfm::evaluate_component(f, up.data(), yu.data());
      dfm::evaluate_component(f, down.data(), yd.data());
      for (int j = 0; j < 4; ++j) {
        const Complex fd = (yu[j] - yd[j]) / (2.0 * h);
        const Complex an = jac[static_cast<std::size_t>(j * k + c)];
        EXPECT_LE(std::abs(fd - an), 1e-5 * (1.0 + std::abs(an)))
            << dfm::name(f) << " output " << j << " input " << c;
      }
    }
  }
}

TEST_P(KernelTest, RecordedTapeMatchesJetVjp) {
  const auto f = static_cast<ComponentFunction>(GetParam());
  std::mt19937 rng(99 + GetParam());
  const int rows = 3;
  const int width = dfm::kComponentInputWidth;
  std::vector<double> params;
  for (int r = 0; r < rows; ++r) {
    auto p = sample(f, rng);
    params.insert(params.end(), p.begin(), p.end());
  }
  std::vector<Complex> cot(rows * 4);
  std::normal_distribution<double> g;
  for (auto& c : cot) c = Complex(g(rng), g(rng));

  // Expected: jet VJP accumulated per row.
  std::vector<double> expected(params.size(), 0.0);
  dfm::component_vjp_batch(f, rows, params.data(), width, cot.data(), expected.data());

  auto& tape = dfm::ActiveReal::getTape();
  tape.reset();
  tape.setActive();
  std::vector<dfm::ActiveReal> in(params.begin(), params.end());
  for (auto& v : in) tape.registerInput(v);
  std::vector<dfm::ActiveReal> out(rows * 8);
  dfm::record_components(f, rows, in.data(), width, out.data());
  for (auto& v : out) tape.registerOutput(v);
  tape.setPassive();
  for (int i = 0; i < rows * 4; ++i) {
    out[2 * i].setGradient(cot[i].real());
    out[2 * i + 1].setGradient(cot[i].imag());
  }
  tape.evaluate();
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_NEAR(in[i].getGradient(), expected[i], 1e-9 * (1.0 + std::abs(expected[i]))) << dfm::name(f);
  }
  // Primal values pass through the tape unchanged.
  std::vector<Complex> y(rows * 4);
  dfm::evaluate_components(f, rows, params.data(), width, y.data());
  for (int i = 0; i < rows * 4; ++i) {
    EXPECT_EQ(out[2 * i].getValue(), y[i].real());
    EXPECT_EQ(out[2 * i + 1].getValue(), y[i].imag());
  }
  tape.reset();
}

INSTANTIATE_TEST_SUITE_P(AllRegisteredFunctions, KernelTest, ::testing::Range(0, dfm::kNumComponentFunctions),
                         [](const ::testing::TestParamInfo<int>& info) {
                           return std::string(dfm::name(static_cast<ComponentFunction>(info.param)));
                         });

// Parameter links (upstream LINKING_FUNCTIONS) on CoDiPack scalars: value
// and reverse-mode derivative against the analytic derivative.
TEST(Links, ReverseModeDerivativesMatchAnalytic) {
  using dfm::LinkFunction;
  const double x0 = 0.7;
  const std::vector<std::pair<LinkFunction, double>> expected = {
      {LinkFunction::Identity, 1.0},           {LinkFunction::Sqrt, 0.5 / std::sqrt(x0)},
      {LinkFunction::Square, 2.0 * x0},        {LinkFunction::Exp, std::exp(x0)},
      {LinkFunction::Log, 1.0 / x0},           {LinkFunction::Sin, std::cos(x0)},
      {LinkFunction::Cos, -std::sin(x0)},      {LinkFunction::Negative, -1.0},
      {LinkFunction::Abs, 1.0},
  };
  auto& tape = dfm::ActiveReal::getTape();
  for (const auto& [f, derivative] : expected) {
    tape.reset();
    tape.setActive();
    dfm::ActiveReal x = x0;
    tape.registerInput(x);
    dfm::ActiveReal y = dfm::apply_link(f, x);
    tape.registerOutput(y);
    tape.setPassive();
    y.setGradient(1.0);
    tape.evaluate();
    EXPECT_DOUBLE_EQ(y.getValue(), dfm::apply_link(f, x0)) << dfm::link_function_name(f);
    EXPECT_NEAR(x.getGradient(), derivative, 1e-14) << dfm::link_function_name(f);
  }
  tape.reset();
}

TEST(Kernels, PassiveInputsDoNotGrowTheTape) {
  auto& tape = dfm::ActiveReal::getTape();
  tape.reset();
  tape.setActive();
  std::vector<dfm::ActiveReal> in(7, 0.0);
  in[0] = 1.0;
  in[1] = 30.0;
  std::vector<dfm::ActiveReal> out(8);
  // The linear index manager hands out one new identifier per statement.
  const auto before = tape.getParameter(codi::TapeParameters::LargestIdentifier);
  dfm::record_components(ComponentFunction::Laser, 1, in.data(), 7, out.data());
  EXPECT_EQ(tape.getParameter(codi::TapeParameters::LargestIdentifier), before);
  tape.setPassive();
  tape.reset();
}

TEST(Kernels, OneStatementPerNonZeroRealOutput) {
  // surface has three non-zero complex outputs -> 6 statements per entry,
  // independent of the number of operations inside the kernel.
  auto& tape = dfm::ActiveReal::getTape();
  tape.reset();
  tape.setActive();
  const int rows = 10;
  std::vector<dfm::ActiveReal> in(rows * 7);
  for (int r = 0; r < rows; ++r) {
    const double p[7] = {1e-5, 0.5, 10.0 + r, 100.0, 1.0, 1.2, 15.0};
    for (int k = 0; k < 7; ++k) {
      in[r * 7 + k] = p[k];
      tape.registerInput(in[r * 7 + k]);
    }
  }
  std::vector<dfm::ActiveReal> out(rows * 8);
  const auto before = tape.getParameter(codi::TapeParameters::LargestIdentifier);
  dfm::record_components(ComponentFunction::Surface, rows, in.data(), 7, out.data());
  EXPECT_EQ(tape.getParameter(codi::TapeParameters::LargestIdentifier) - before, static_cast<std::size_t>(rows * 6));
  tape.setPassive();
  tape.reset();
}

}  // namespace
