// Planner and assembly smoke tests on a small cavity with a signal sweep.
// The thorough comparison against upstream (cavity, aLIGO, Voyager, UIFO)
// is in tests/python/test_planner_vs_upstream.py.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>

#include "dfm/assembly.hpp"
#include "dfm/plan.hpp"

namespace {

std::shared_ptr<const dfm::SetupSpec> cavity() {
  auto s = std::make_shared<dfm::SetupSpec>();
  s->nodes.push_back({"f", "frequency", {{"frequency", 1.0}}});
  s->nodes.push_back({"l0", "laser", {{"power", 1.0}, {"phase", 0.0}}});
  s->nodes.push_back({"m0", "mirror", {{"loss", 0.0}, {"reflectivity", 0.99}, {"tuning", 0.0}}});
  s->nodes.push_back({"m1", "mirror", {{"loss", 0.0}, {"reflectivity", 0.991}, {"tuning", 0.0}}});
  s->nodes.push_back({"fm", "free_mass", {{"mass", 40.0}}, std::string("m1")});
  s->nodes.push_back({"sig", "signal", {{"amplitude", 1.0}, {"phase", 0.0}}, std::string("m0_m1")});
  dfm::NodeSpec det{"trns", "detector", {}};
  det.target = "m1";
  det.port = "right";
  det.direction = "out";
  s->nodes.push_back(det);
  s->edges.push_back({"l0", "m0", {{"length", 1.0}, {"refractive_index", 1.0}}});
  s->edges.push_back({"m0", "m1", {{"length", 1.0}, {"refractive_index", 1.0}}});
  s->parameters = {{"m0", "tuning"}, {"m1", "reflectivity"}, {"l0", "power"}};
  return s;
}

dfm::SimulationPlanPtr plan_for(std::vector<dfm::OptimizationTarget> targets) {
  dfm::PlanRequest r;
  r.changing_pairs = {{"f", "frequency"}};
  r.changing_values = dfm::MatrixR(1, 3);
  r.changing_values << 100.0, 500.0, 2000.0;
  r.optimization_targets = std::move(targets);
  return dfm::build_plan(cavity(), r);
}

TEST(Planner, DimensionsAndMetadata) {
  const auto plan = plan_for({{{"m0", "tuning"}}});
  const auto& d = plan->dimensions();
  // laser edge endpoint (2) + two mirrors (4 each)
  EXPECT_EQ(d.carrier_ports, 10);
  // free mass (2) + one signal -> signal block size 3
  EXPECT_EQ(d.signal_variables, 2 * 10 + 3);
  EXPECT_EQ(d.carrier_values, 1);
  EXPECT_EQ(d.signal_values, 3);
  EXPECT_EQ(d.optimized_values, 1);
  EXPECT_EQ(plan->metadata().detector_indices.size(), 1);
  EXPECT_EQ(plan->metadata().mirror_indices.size(), 8);
  EXPECT_EQ(plan->metadata().port_to_index.size(), 8u);  // 2 mirrors x 2 ports x in/out
}

TEST(Assembly, CarrierHasIdentityDiagonalAndLaserSource) {
  const auto plan = plan_for({});
  const dfm::VectorR x = dfm::VectorR::Zero(1);
  const auto sys = dfm::assemble_carrier(*plan, dfm::resolve_parameters(*plan, x));
  ASSERT_EQ(sys.values.rows(), 1);
  for (int i = 0; i < sys.n; ++i) {
    const int slot = sys.pattern->find(i, i);
    ASSERT_GE(slot, 0);
    EXPECT_EQ(sys.values(0, slot), dfm::Complex(1.0, 0.0));
  }
  // sqrt(2 * 1 W) on the laser row.
  EXPECT_NEAR(sys.rhs.row(0).cwiseAbs().maxCoeff(), std::sqrt(2.0), 1e-15);
}

// Gradient of a random linear functional of the assembled entries against
// central finite differences of the passive assembly.
void check_gradient(dfm::SystemKind kind) {
  const auto plan = plan_for({{{"m0", "tuning"}}, {{"m1", "reflectivity"}}, {{"fm", "mass"}}, {{"l0", "power"}}});
  const auto& d = plan->dimensions();
  dfm::VectorR x(4);
  x << 10.0, 0.95, 30.0, 2.0;
  std::mt19937 rng(5);
  std::normal_distribution<double> g;
  dfm::MatrixC carrier(d.carrier_values, d.carrier_ports);
  for (dfm::Index k = 0; k < carrier.size(); ++k) carrier.data()[k] = {g(rng), g(rng)};

  auto assemble = [&](const dfm::VectorR& xv, const dfm::MatrixC& c) {
    const auto r = dfm::resolve_parameters(*plan, xv);
    return kind == dfm::SystemKind::Carrier  ? dfm::assemble_carrier(*plan, r)
           : kind == dfm::SystemKind::Signal ? dfm::assemble_signal(*plan, r, c)
                                             : dfm::assemble_noise(*plan, r);
  };
  const auto base = assemble(x, carrier);
  dfm::MatrixC vcot(base.values.rows(), base.values.cols()), rcot(base.rhs.rows(), base.rhs.cols());
  // Entries span ~1e-9 .. ~1e11 (space modulation scales with w0 / w_g).
  // Scaling each cotangent by 1 / max(1, |entry|) keeps every slot's
  // contribution O(1), so central differences stay accurate for all slots.
  for (dfm::Index k = 0; k < vcot.size(); ++k) {
    vcot.data()[k] = dfm::Complex(g(rng), g(rng)) / std::max(1.0, std::abs(base.values.data()[k]));
  }
  for (dfm::Index k = 0; k < rcot.size(); ++k) {
    rcot.data()[k] = dfm::Complex(g(rng), g(rng)) / std::max(1.0, std::abs(base.rhs.data()[k]));
  }
  auto loss = [&](const dfm::AssembledSystem& s) {
    return (vcot.conjugate().cwiseProduct(s.values)).real().sum() + (rcot.conjugate().cwiseProduct(s.rhs)).real().sum();
  };

  const auto grad = dfm::assembly_vjp(*plan, kind, x, carrier, vcot, rcot);
  for (int i = 0; i < 4; ++i) {
    const double h = 1e-6 * std::max(1.0, std::abs(x(i)));
    dfm::VectorR up = x, down = x;
    up(i) += h;
    down(i) -= h;
    const double fd = (loss(assemble(up, carrier)) - loss(assemble(down, carrier))) / (2 * h);
    EXPECT_NEAR(grad.optimized(i), fd, 1e-5 * (1.0 + std::abs(fd))) << "parameter " << i;
  }
  if (kind == dfm::SystemKind::Signal) {
    for (dfm::Index k = 0; k < carrier.size(); ++k) {
      for (int part = 0; part < 2; ++part) {
        const dfm::Complex step = part == 0 ? dfm::Complex(1e-6, 0) : dfm::Complex(0, 1e-6);
        dfm::MatrixC up = carrier, down = carrier;
        up.data()[k] += step;
        down.data()[k] -= step;
        const double fd = (loss(assemble(x, up)) - loss(assemble(x, down))) / 2e-6;
        const double got = part == 0 ? grad.carrier.data()[k].real() : grad.carrier.data()[k].imag();
        EXPECT_NEAR(got, fd, 1e-6 * (1.0 + std::abs(fd))) << "carrier " << k << " part " << part;
      }
    }
  }
}

TEST(Assembly, CarrierGradientMatchesFiniteDifferences) { check_gradient(dfm::SystemKind::Carrier); }
TEST(Assembly, SignalGradientMatchesFiniteDifferences) { check_gradient(dfm::SystemKind::Signal); }
TEST(Assembly, NoiseGradientMatchesFiniteDifferences) { check_gradient(dfm::SystemKind::Noise); }

}  // namespace
