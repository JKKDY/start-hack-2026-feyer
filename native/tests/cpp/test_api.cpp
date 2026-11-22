// API-contract tests for the native planner/simulator interface. The
// numerical engine is a later milestone, so these check validation,
// error types and the detector kernels only.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "dfm/detectors.hpp"
#include "dfm/errors.hpp"
#include "dfm/link.hpp"
#include "dfm/plan.hpp"
#include "dfm/setup_spec.hpp"

namespace {

std::shared_ptr<const dfm::SetupSpec> tiny_setup() {
  auto s = std::make_shared<dfm::SetupSpec>();
  s->nodes.push_back({"l0", "laser", {{"power", 1.0}, {"phase", 0.0}}});
  s->nodes.push_back({"m0", "mirror", {{"loss", 5e-6}, {"reflectivity", 0.5}, {"tuning", 0.0}}});
  s->nodes.push_back({"f", "frequency", {{"frequency", 1.0}}});
  s->edges.push_back({"l0", "m0", {{"length", 1.0}, {"refractive_index", 1.0}}});
  s->parameters = {{"l0", "power"}, {"m0", "reflectivity"}, {"l0_m0", "length"}};
  return s;
}

TEST(SetupSpec, ValidSetupPassesValidation) {
  EXPECT_NO_THROW(tiny_setup()->validate());
}

TEST(SetupSpec, UnderscoreInNodeNameIsRejected) {
  dfm::SetupSpec s;
  s.nodes.push_back({"bad_name", "laser", {}});
  EXPECT_THROW(s.validate(), dfm::ValidationError);
}

TEST(SetupSpec, EdgeToUnknownNodeIsRejected) {
  auto s = std::make_shared<dfm::SetupSpec>(*tiny_setup());
  s->edges.push_back({"m0", "missing", {}});
  EXPECT_THROW(s->validate(), dfm::ValidationError);
}

TEST(Planner, RejectsUnknownChangingPair) {
  dfm::PlanRequest r;
  r.changing_pairs = {{"nope", "frequency"}};
  r.changing_values = dfm::MatrixR::Ones(1, 3);
  EXPECT_THROW(dfm::build_plan(tiny_setup(), r), dfm::ValidationError);
}

TEST(Planner, RejectsChangingValuesWithWrongRowCount) {
  dfm::PlanRequest r;
  r.changing_pairs = {{"f", "frequency"}};
  r.changing_values = dfm::MatrixR::Ones(2, 3);
  EXPECT_THROW(dfm::build_plan(tiny_setup(), r), dfm::ValidationError);
}

TEST(Planner, ValidRequestBuildsAPlan) {
  dfm::PlanRequest r;
  r.changing_pairs = {{"f", "frequency"}};
  r.changing_values = dfm::MatrixR::Ones(1, 5);
  r.optimization_targets = {{{"l0", "power"}}, {{"m0", "tuning"}, {"m0", "reflectivity"}}};
  const auto plan = dfm::build_plan(tiny_setup(), r);
  EXPECT_EQ(plan->dimensions().optimized_values, 2);
  EXPECT_EQ(plan->dimensions().signal_values, 5);
}

TEST(Links, NamesRoundTripAndEvaluate) {
  for (int i = 0; i <= static_cast<int>(dfm::LinkFunction::Abs); ++i) {
    const auto f = static_cast<dfm::LinkFunction>(i);
    EXPECT_EQ(dfm::link_function_from_name(dfm::link_function_name(f)), f);
  }
  EXPECT_DOUBLE_EQ(dfm::apply_link(dfm::LinkFunction::Sqrt, 4.0), 2.0);
  EXPECT_FALSE(dfm::link_function_from_name("tanh").has_value());
}

TEST(Detectors, SignalDetectorBroadcastsCarrierColumn) {
  dfm::MatrixC carrier(2, 1);
  carrier << dfm::Complex(1.0, 1.0), dfm::Complex(0.0, 2.0);
  dfm::MatrixC signal = dfm::MatrixC::Constant(4, 3, dfm::Complex(0.5, -0.5));
  const dfm::MatrixC out = dfm::signal_detector(carrier, signal);
  ASSERT_EQ(out.rows(), 2);
  ASSERT_EQ(out.cols(), 3);
  const dfm::Complex expected = std::conj(carrier(1, 0)) * signal(1, 2) + carrier(1, 0) * signal(3, 2);
  EXPECT_NEAR(std::abs(out(1, 2) - expected), 0.0, 1e-15);
}

}  // namespace
