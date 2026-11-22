#include "dfm/simulator.hpp"

#include <codi.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "dfm/errors.hpp"
#include "dfm/pipeline.hpp"

namespace dfm {
namespace {

// CoDiPack's RealReverse tape is one process-wide object. Every recording
// resets it, so only the newest state can be replayed. The generation
// counter identifies that state; replaying an older one is refused.
std::atomic<std::uint64_t> g_generation{0};

}  // namespace

std::mutex& tape_mutex() {
  static std::mutex m;
  return m;
}

void invalidate_recorded_states() noexcept { ++g_generation; }

// ------------------------------------------------------------------ state
struct SimulationState::Data {
  std::uint64_t generation = 0;
  std::thread::id thread;
  std::vector<ActiveReal> inputs;  ///< registered optimized values
  RecordedOutputs outputs;
  Index carrier_ports = 0, carrier_values = 0, signal_variables = 0, signal_values = 0, noise_detectors = 0;
};

SimulationState::SimulationState(SimulationPlanPtr plan, std::unique_ptr<Data> data)
    : plan_(std::move(plan)), data_(std::move(data)) {}

SimulationState::~SimulationState() = default;

bool SimulationState::is_current() const noexcept {
  // CoDiPack's tape is one process-wide static object (not thread-local), so
  // a state may be replayed from any thread; only the generation matters.
  return data_->generation == g_generation.load();
}

// -------------------------------------------------------------- simulator
struct Simulator::Workspace {};

Simulator::Simulator(SimulationPlanPtr plan) : plan_(std::move(plan)) {
  if (!plan_) {
    throw ValidationError("Simulator: plan must not be null.");
  }
  workspace_ = std::make_unique<Workspace>();
}

Simulator::~Simulator() = default;
Simulator::Simulator(Simulator&&) noexcept = default;
Simulator& Simulator::operator=(Simulator&&) noexcept = default;

namespace {

void check_parameters(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& p) {
  const Index expected = plan.dimensions().optimized_values;
  if (p.size() != expected) {
    throw ValidationError("optimized_parameters has length " + std::to_string(p.size()) + ", expected " +
                          std::to_string(expected) + ".");
  }
}

void check_cotangent(const char* name, Index rows, Index cols, Index er, Index ec) {
  if (rows == 0 && cols == 0) return;  // missing cotangent = zero
  if (rows != er || cols != ec) {
    throw ValidationError(std::string(name) + " cotangent has shape (" + std::to_string(rows) + ", " +
                          std::to_string(cols) + "), expected (" + std::to_string(er) + ", " + std::to_string(ec) +
                          ").");
  }
}

}  // namespace

ForwardOutputs Simulator::forward(const Eigen::Ref<const VectorR>& optimized_parameters) {
  check_parameters(*plan_, optimized_parameters);
  return simulate_passive(*plan_, optimized_parameters);
}

std::pair<ForwardOutputs, std::unique_ptr<SimulationState>> Simulator::forward_with_state(
    const Eigen::Ref<const VectorR>& optimized_parameters) {
  check_parameters(*plan_, optimized_parameters);
  const Dimensions& d = plan_->dimensions();

  std::lock_guard<std::mutex> lock(tape_mutex());
  auto& tape = ActiveReal::getTape();
  tape.setPassive();
  tape.reset();
  const std::uint64_t generation = ++g_generation;

  auto data = std::make_unique<SimulationState::Data>();
  data->generation = generation;
  data->thread = std::this_thread::get_id();
  data->carrier_ports = d.carrier_ports;
  data->carrier_values = d.carrier_values;
  data->signal_variables = d.signal_variables;
  data->signal_values = d.signal_values;
  data->noise_detectors = d.noise_detectors;

  tape.setActive();
  data->inputs.assign(optimized_parameters.data(), optimized_parameters.data() + optimized_parameters.size());
  for (ActiveReal& v : data->inputs) tape.registerInput(v);
  try {
    data->outputs = simulate_recorded(*plan_, data->inputs.data());
  } catch (...) {
    tape.setPassive();
    tape.reset();
    throw;
  }
  for (ActiveReal& v : data->outputs.carrier) tape.registerOutput(v);
  for (ActiveReal& v : data->outputs.signal) tape.registerOutput(v);
  for (ActiveReal& v : data->outputs.noise) tape.registerOutput(v);
  tape.setPassive();

  ForwardOutputs out;
  out.carrier.resize(d.carrier_ports, d.carrier_values);
  for (Index k = 0; k < out.carrier.size(); ++k) {
    out.carrier.data()[k] = Complex(data->outputs.carrier[static_cast<std::size_t>(2 * k)].getValue(),
                                    data->outputs.carrier[static_cast<std::size_t>(2 * k + 1)].getValue());
  }
  out.signal.resize(d.signal_variables, d.signal_values);
  for (Index k = 0; k < out.signal.size(); ++k) {
    out.signal.data()[k] = Complex(data->outputs.signal[static_cast<std::size_t>(2 * k)].getValue(),
                                   data->outputs.signal[static_cast<std::size_t>(2 * k + 1)].getValue());
  }
  out.noise.resize(d.noise_detectors, d.signal_values);
  for (Index k = 0; k < out.noise.size(); ++k) {
    out.noise.data()[k] = data->outputs.noise[static_cast<std::size_t>(k)].getValue();
  }
  return {std::move(out), std::unique_ptr<SimulationState>(new SimulationState(plan_, std::move(data)))};
}

VectorR Simulator::backward(const SimulationState& state, const OutputCotangents& ct) {
  std::lock_guard<std::mutex> lock(tape_mutex());
  if (state.plan_ptr() != plan_) {
    throw StateError("Simulator.backward: the state was recorded by a simulator with a different plan.");
  }
  if (!state.is_current()) {
    throw StateError(
        "Simulator.backward: the state is stale (a newer forward_with_state or another tape user ran "
        "since).");
  }
  const SimulationState::Data& d = *state.data_;
  check_cotangent("carrier", ct.carrier.rows(), ct.carrier.cols(), d.carrier_ports, d.carrier_values);
  check_cotangent("signal", ct.signal.rows(), ct.signal.cols(), d.signal_variables, d.signal_values);
  check_cotangent("noise", ct.noise.rows(), ct.noise.cols(), d.noise_detectors, d.signal_values);

  auto& tape = ActiveReal::getTape();
  tape.clearAdjoints();
  auto& outputs = const_cast<RecordedOutputs&>(d.outputs);  // gradients live in the tape, not the values
  for (Index k = 0; k < ct.carrier.size(); ++k) {
    outputs.carrier[static_cast<std::size_t>(2 * k)].setGradient(ct.carrier.data()[k].real());
    outputs.carrier[static_cast<std::size_t>(2 * k + 1)].setGradient(ct.carrier.data()[k].imag());
  }
  for (Index k = 0; k < ct.signal.size(); ++k) {
    outputs.signal[static_cast<std::size_t>(2 * k)].setGradient(ct.signal.data()[k].real());
    outputs.signal[static_cast<std::size_t>(2 * k + 1)].setGradient(ct.signal.data()[k].imag());
  }
  for (Index k = 0; k < ct.noise.size(); ++k) {
    outputs.noise[static_cast<std::size_t>(k)].setGradient(ct.noise.data()[k]);
  }
  tape.evaluate();

  VectorR grad(static_cast<Index>(d.inputs.size()));
  for (std::size_t i = 0; i < d.inputs.size(); ++i) grad(static_cast<Index>(i)) = d.inputs[i].getGradient();
  tape.clearAdjoints();
  return grad;
}

}  // namespace dfm
