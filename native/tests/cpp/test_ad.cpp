#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dfm/ad_eigen.hpp"

namespace {

// Complex Eigen arithmetic on CoDiPack types, checked against the analytic
// gradient computed in plain complex doubles.
TEST(AdEigen, ComplexGradientMatchesAnalytic) {
  const dfm::GradientSmokeResult r = dfm::complex_gradient_smoke({0.3, -0.2, 0.7});
  EXPECT_LT(r.max_abs_error, 1e-10);
  EXPECT_EQ(r.gradient.size(), 3u);
}

TEST(AdEigen, RejectsWrongParameterCount) {
  EXPECT_THROW(dfm::complex_gradient_smoke({1.0, 2.0}), std::invalid_argument);
}

}  // namespace
