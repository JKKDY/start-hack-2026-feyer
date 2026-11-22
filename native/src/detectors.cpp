#include "dfm/detectors.hpp"

#include <cmath>
#include <string>

#include "dfm/constants.hpp"
#include "dfm/errors.hpp"

namespace dfm {

void amplitude_detector(const Complex* in, Real* out, Index n) {
  const Eigen::Map<const Eigen::Array<Complex, Eigen::Dynamic, 1>> x(in, n);
  Eigen::Map<Eigen::Array<Real, Eigen::Dynamic, 1>> y(out, n);
  y = std::sqrt(0.5 * constants::EPSILON_0C) * x.abs();
}

void power_detector(const Complex* in, Real* out, Index n) {
  const Eigen::Map<const Eigen::Array<Complex, Eigen::Dynamic, 1>> x(in, n);
  Eigen::Map<Eigen::Array<Real, Eigen::Dynamic, 1>> y(out, n);
  y = (0.5 * constants::EPSILON_0C) * x.abs2();
}

MatrixC signal_detector(const Eigen::Ref<const MatrixC>& carrier,
                        const Eigen::Ref<const MatrixC>& signal) {
  const Index n = carrier.rows();
  const Index kc = carrier.cols();
  const Index ks = signal.cols();
  if (signal.rows() < 2 * n) {
    throw ValidationError("signal_detector: signal has " + std::to_string(signal.rows()) +
                          " rows, needs at least 2 * " + std::to_string(n) + ".");
  }
  const Index k = kc > ks ? kc : ks;
  if ((kc != 1 && kc != k) || (ks != 1 && ks != k)) {
    throw ValidationError("signal_detector: column counts " + std::to_string(kc) + " and " +
                          std::to_string(ks) + " do not broadcast.");
  }

  MatrixC out(n, k);
  const auto upper = signal.topRows(n);
  const auto lower = signal.middleRows(n, n);
  if (kc == k && ks == k) {
    out.array() = carrier.array().conjugate() * upper.array() + carrier.array() * lower.array();
  } else if (kc == 1) {
    // One carrier column against K signal columns (the usual sweep case).
    out.array() = upper.array().colwise() * carrier.col(0).array().conjugate() +
                  lower.array().colwise() * carrier.col(0).array();
  } else {
    // K carrier columns against one signal column.
    out.array() = carrier.array().conjugate().colwise() * upper.col(0).array() +
                  carrier.array().colwise() * lower.col(0).array();
  }
  return out;
}

}  // namespace dfm
