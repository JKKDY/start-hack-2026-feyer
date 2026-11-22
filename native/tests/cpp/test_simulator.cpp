// Full pipeline on a cavity with optomechanics, a space signal and a quantum
// noise detector: passive vs recorded forward, gradient vs finite
// differences for arbitrary output cotangents, and state lifetime rules.
// Validation against upstream references: tests/python/test_simulator_vs_upstream.py.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>

#include "dfm/errors.hpp"
#include "dfm/plan.hpp"
#include "dfm/simulator.hpp"

namespace {

std::shared_ptr<const dfm::SetupSpec> cavity() {
  auto s = std::make_shared<dfm::SetupSpec>();
  s->nodes.push_back({"f", "frequency", {{"frequency", 1.0}}});
  s->nodes.push_back({"l0", "laser", {{"power", 1.0}, {"phase", 0.0}}});
  s->nodes.push_back({"m0", "mirror", {{"loss", 1e-5}, {"reflectivity", 0.9}, {"tuning", 0.0}}});
  s->nodes.push_back({"m1", "mirror", {{"loss", 1e-5}, {"reflectivity", 0.95}, {"tuning", 0.0}}});
  s->nodes.push_back({"fm", "free_mass", {{"mass", 1.0}}, std::string("m1")});
  s->nodes.push_back({"sig", "signal", {{"amplitude", 1.0}, {"phase", 0.0}}, std::string("m0_m1")});
  dfm::NodeSpec qn{"qn", "qnoised", {}};
  qn.target = "m1";
  qn.port = "right";
  qn.direction = "out";
  s->nodes.push_back(qn);
  s->edges.push_back({"l0", "m0", {{"length", 1.0}, {"refractive_index", 1.0}}});
  s->edges.push_back({"m0", "m1", {{"length", 1.0}, {"refractive_index", 1.0}}});
  return s;
}

dfm::SimulationPlanPtr make_plan() {
  dfm::PlanRequest r;
  r.changing_pairs = {{"f", "frequency"}};
  r.changing_values = dfm::MatrixR(1, 2);
  r.changing_values << 200.0, 1500.0;
  r.optimization_targets = {{{"m0", "tuning"}}, {{"m1", "reflectivity"}}, {{"fm", "mass"}}, {{"l0", "power"}}};
  return dfm::build_plan(cavity(), r);
}

TEST(Simulator, RecordedForwardMatchesPassiveForward) {
  dfm::Simulator sim(make_plan());
  dfm::VectorR x(4);
  x << 3.0, 0.95, 1.0, 2.0;
  const auto passive = sim.forward(x);
  auto [recorded, state] = sim.forward_with_state(x);
  EXPECT_LT((passive.carrier - recorded.carrier).cwiseAbs().maxCoeff(), 1e-12 * passive.carrier.cwiseAbs().maxCoeff());
  EXPECT_LT((passive.signal - recorded.signal).cwiseAbs().maxCoeff(), 1e-10 * passive.signal.cwiseAbs().maxCoeff());
  EXPECT_LT((passive.noise - recorded.noise).cwiseAbs().maxCoeff(), 1e-12 * passive.noise.cwiseAbs().maxCoeff());
  EXPECT_TRUE((passive.noise.array() > 0).all());
  EXPECT_TRUE(state->is_current());
}

TEST(Simulator, GradientMatchesFiniteDifferencesForArbitraryCotangents) {
  dfm::Simulator sim(make_plan());
  dfm::VectorR x(4);
  x << 3.0, 0.95, 1.0, 2.0;
  auto [out, state] = sim.forward_with_state(x);

  std::mt19937 rng(11);
  std::normal_distribution<double> g;
  dfm::OutputCotangents ct;
  ct.carrier.resize(out.carrier.rows(), out.carrier.cols());
  ct.signal.resize(out.signal.rows(), out.signal.cols());
  ct.noise.resize(out.noise.rows(), out.noise.cols());
  // Scale by 1 / |value| so every output contributes O(1) to the functional,
  // with a floor so that numerically-zero entries (round-off) do not get
  // huge cotangents that only amplify finite-difference noise.
  const double cf = 1e-6 * out.carrier.cwiseAbs().maxCoeff();
  const double sf = 1e-6 * out.signal.cwiseAbs().maxCoeff();
  for (dfm::Index k = 0; k < ct.carrier.size(); ++k)
    ct.carrier.data()[k] = dfm::Complex(g(rng), g(rng)) / std::max(cf, std::abs(out.carrier.data()[k]));
  for (dfm::Index k = 0; k < ct.signal.size(); ++k)
    ct.signal.data()[k] = dfm::Complex(g(rng), g(rng)) / std::max(sf, std::abs(out.signal.data()[k]));
  for (dfm::Index k = 0; k < ct.noise.size(); ++k) ct.noise.data()[k] = g(rng) / out.noise.data()[k];

  const dfm::VectorR grad = sim.backward(*state, ct);
  auto functional = [&](const dfm::VectorR& xv) {
    const auto o = sim.forward(xv);
    return (ct.carrier.conjugate().cwiseProduct(o.carrier)).real().sum() +
           (ct.signal.conjugate().cwiseProduct(o.signal)).real().sum() + ct.noise.cwiseProduct(o.noise).sum();
  };
  for (int i = 0; i < 4; ++i) {
    const double h = 1e-6 * std::max(1.0, std::abs(x(i)));
    dfm::VectorR up = x, down = x;
    up(i) += h;
    down(i) -= h;
    const double fd = (functional(up) - functional(down)) / (2 * h);
    EXPECT_NEAR(grad(i), fd, 2e-5 * (1.0 + std::abs(fd))) << "parameter " << i;
  }
  // backward can be called again with the same state.
  EXPECT_NEAR((sim.backward(*state, ct) - grad).norm(), 0.0, 1e-12 * grad.norm());
}

TEST(Simulator, StaleStateIsRejected) {
  dfm::Simulator sim(make_plan());
  dfm::VectorR x(4);
  x << 3.0, 0.95, 1.0, 2.0;
  auto first = sim.forward_with_state(x);
  auto second = sim.forward_with_state(x);
  EXPECT_FALSE(first.second->is_current());
  EXPECT_THROW(sim.backward(*first.second, dfm::OutputCotangents{}), dfm::StateError);
  EXPECT_NO_THROW(sim.backward(*second.second, dfm::OutputCotangents{}));
}

}  // namespace
