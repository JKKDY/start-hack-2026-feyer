#include "dfm/ad_eigen.hpp"

// Eigen must be included before codi.hpp: CoDiPack's Eigen traits build on it.
#include <Eigen/Dense>
#include <codi.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

namespace dfm {
namespace {

using Real = codi::RealReverse;
using ActiveComplex = std::complex<Real>;

using ComplexMatrix = Eigen::Matrix<std::complex<double>, 2, 2>;
using ComplexVector = Eigen::Matrix<std::complex<double>, 2, 1>;
using ActiveMatrix = Eigen::Matrix<ActiveComplex, 2, 2>;
using ActiveVector = Eigen::Matrix<ActiveComplex, 2, 1>;

constexpr int kNumParams = 3;

// Z(p) = Z0 + sum_k p_k Z_k and fixed u.  Values are arbitrary but
// non-degenerate, with genuinely complex entries so that conjugation
// errors would show up in the result.
const std::array<ComplexMatrix, kNumParams + 1> kTerms = {{
    // Z0
    (ComplexMatrix() << std::complex<double>(1.0, 0.5), std::complex<double>(0.2, 0.0),
     std::complex<double>(0.0, 0.1), std::complex<double>(2.0, -1.0))
        .finished(),
    // Z_0 (coefficient of p_0)
    (ComplexMatrix() << std::complex<double>(0.3, 0.0), std::complex<double>(0.0, -0.7),
     std::complex<double>(1.1, 0.0), std::complex<double>(0.4, 0.2))
        .finished(),
    // Z_1
    (ComplexMatrix() << std::complex<double>(0.0, -0.5), std::complex<double>(0.25, 0.0),
     std::complex<double>(0.6, 0.0), std::complex<double>(0.0, 0.1))
        .finished(),
    // Z_2
    (ComplexMatrix() << std::complex<double>(0.8, 0.0), std::complex<double>(0.0, 0.0),
     std::complex<double>(0.0, 0.2), std::complex<double>(-0.3, 0.0))
        .finished(),
}};

const ComplexVector kU = (ComplexVector() << std::complex<double>(1.0, 1.0),
                          std::complex<double>(0.5, -0.25))
                             .finished();

ActiveComplex to_active(const std::complex<double>& c) {
  return ActiveComplex(Real(c.real()), Real(c.imag()));
}

}  // namespace

GradientSmokeResult complex_gradient_smoke(const std::vector<double>& p) {
  if (static_cast<int>(p.size()) != kNumParams) {
    throw std::invalid_argument("complex_gradient_smoke: expected " + std::to_string(kNumParams) +
                                " parameters");
  }

  // ---- analytic reference in plain complex double -----------------------
  ComplexMatrix z = kTerms[0];
  for (int k = 0; k < kNumParams; ++k) {
    z += p[k] * kTerms[k + 1];
  }
  const ComplexVector w = z * kU;
  const double expected_value = w.squaredNorm();

  std::vector<double> expected_gradient(kNumParams);
  for (int k = 0; k < kNumParams; ++k) {
    const ComplexVector dw = kTerms[k + 1] * kU;
    expected_gradient[k] = 2.0 * std::real(w.dot(dw));  // dot conjugates its first argument
  }

  // ---- CoDiPack evaluation with reverse sweep ---------------------------
  auto& tape = Real::getTape();
  tape.setActive();

  std::array<Real, kNumParams> params;
  for (int k = 0; k < kNumParams; ++k) {
    params[k] = p[k];
    tape.registerInput(params[k]);
  }

  ActiveMatrix za;
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      ActiveComplex acc = to_active(kTerms[0](r, c));
      for (int k = 0; k < kNumParams; ++k) {
        acc += ActiveComplex(params[k]) * to_active(kTerms[k + 1](r, c));
      }
      za(r, c) = acc;
    }
  }

  ActiveVector ua;
  for (int r = 0; r < 2; ++r) {
    ua(r) = to_active(kU(r));
  }

  const ActiveVector wa = za * ua;

  // |w|^2 written out in real/imaginary parts. Eigen's squaredNorm() routes
  // through its abs2 helpers, which do not accept CoDiPack expression types.
  Real value = 0.0;
  for (int r = 0; r < 2; ++r) {
    const Real re = wa(r).real();
    const Real im = wa(r).imag();
    value += re * re + im * im;
  }

  tape.registerOutput(value);
  value.setGradient(1.0);
  tape.setPassive();
  tape.evaluate();

  GradientSmokeResult result{};
  result.value = value.getValue();
  result.gradient.resize(kNumParams);
  result.max_abs_error = 0.0;
  for (int k = 0; k < kNumParams; ++k) {
    result.gradient[k] = params[k].getGradient();
    result.max_abs_error =
        std::max(result.max_abs_error, std::abs(result.gradient[k] - expected_gradient[k]));
  }
  result.max_abs_error =
      std::max(result.max_abs_error, std::abs(result.value - expected_value));

  tape.reset();
  return result;
}

}  // namespace dfm
