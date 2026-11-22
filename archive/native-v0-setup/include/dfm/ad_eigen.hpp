#pragma once

#include <vector>

namespace dfm {

/// Result of the CoDiPack + Eigen reverse-mode smoke test.
struct GradientSmokeResult {
  double value;                  ///< f(p) computed with CoDiPack types.
  std::vector<double> gradient;  ///< df/dp_k from one reverse sweep.
  double max_abs_error;          ///< max_k |gradient_k - analytic_k|.
};

/// Evaluates f(p) = || Z(p) u ||^2 where Z(p) is a complex 2x2 matrix that
/// depends affinely on three real parameters p, and u is a fixed complex
/// vector. Both the complex Eigen arithmetic and the reverse sweep run on
/// CoDiPack types (std::complex<codi::RealReverse> inside Eigen).
/// The gradient is compared against the analytic result
///   df/dp_k = 2 Re( (Z u)^H (Z_k u) ).
GradientSmokeResult complex_gradient_smoke(const std::vector<double>& p);

}  // namespace dfm
