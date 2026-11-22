#include "dfm/component_ops.hpp"

#include <string>
#include <utility>

#include "dfm/errors.hpp"

namespace dfm {
namespace {

using CF = ComponentFunction;

// Calls fn.template operator()<F>() for the runtime value f. This is the
// only runtime dispatch; everything inside is instantiated per kernel.
template <class Fn>
decltype(auto) dispatch(ComponentFunction f, Fn&& fn) {
  switch (f) {
    case CF::ForceCalculationRight: return fn.template operator()<CF::ForceCalculationRight>();
    case CF::Laser: return fn.template operator()<CF::Laser>();
    case CF::Surface: return fn.template operator()<CF::Surface>();
    case CF::Space: return fn.template operator()<CF::Space>();
    case CF::SpaceModulation: return fn.template operator()<CF::SpaceModulation>();
    case CF::SignalFunction: return fn.template operator()<CF::SignalFunction>();
    case CF::DummyFunction: return fn.template operator()<CF::DummyFunction>();
    case CF::VacuumQuantumNoise: return fn.template operator()<CF::VacuumQuantumNoise>();
    case CF::LossQuantumNoise: return fn.template operator()<CF::LossQuantumNoise>();
    case CF::SpaceLower: return fn.template operator()<CF::SpaceLower>();
    case CF::LaserAmplitudeModulation: return fn.template operator()<CF::LaserAmplitudeModulation>();
    case CF::LaserFrequencyModulation: return fn.template operator()<CF::LaserFrequencyModulation>();
    case CF::Susceptibility: return fn.template operator()<CF::Susceptibility>();
    case CF::ForceCalculationLeft: return fn.template operator()<CF::ForceCalculationLeft>();
    case CF::CorrectedOptomechanicalPhaseLeft: return fn.template operator()<CF::CorrectedOptomechanicalPhaseLeft>();
    case CF::CorrectedOptomechanicalPhaseRight: return fn.template operator()<CF::CorrectedOptomechanicalPhaseRight>();
    case CF::SpaceModulationLower: return fn.template operator()<CF::SpaceModulationLower>();
    case CF::Squeezer: return fn.template operator()<CF::Squeezer>();
    case CF::LaserFrequencyModulationLower: return fn.template operator()<CF::LaserFrequencyModulationLower>();
  }
  throw ValidationError("unknown component function");
}

template <ComponentFunction F>
inline void evaluate_one(const double* params, Complex* out) {
  constexpr int K = arity(F);
  Complex p[K > 0 ? K : 1];
  for (int k = 0; k < K; ++k) p[k] = Complex(params[k], 0.0);
  const auto y = kernels::evaluate<F>(p);
  for (int j = 0; j < kComponentOutputs; ++j) out[j] = y[j];
}

// Jet with at least one direction so that zero-arity kernels compile.
template <ComponentFunction F>
using JetFor = Jet<(arity(F) > 0 ? arity(F) : 1)>;

template <ComponentFunction F>
inline auto jet_outputs(const double* params) {
  constexpr int K = arity(F);
  using J = JetFor<F>;
  J p[K > 0 ? K : 1];
  for (int k = 0; k < K; ++k) p[k] = J::variable(params[k], k);
  return kernels::evaluate<F>(p);
}

template <ComponentFunction F>
inline void vjp_one(const double* params, const Complex* cot, double* grad) {
  constexpr int K = arity(F);
  if constexpr (K == 0) {
    return;
  } else {
    const auto y = jet_outputs<F>(params);
    for (int k = 0; k < K; ++k) {
      double g = 0.0;
      for (int j = 0; j < kComponentOutputs; ++j) g += std::real(std::conj(cot[j]) * y[j].d[k]);
      grad[k] += g;
    }
  }
}

}  // namespace

ComponentFunction component_function_from_index(int index) {
  if (index < 0 || index >= kNumComponentFunctions) {
    throw ValidationError("component function index " + std::to_string(index) + " is out of range [0, " +
                          std::to_string(kNumComponentFunctions) + ").");
  }
  return static_cast<ComponentFunction>(index);
}

void evaluate_component(ComponentFunction f, const double* params, Complex* out) {
  dispatch(f, [&]<ComponentFunction F>() { evaluate_one<F>(params, out); });
}

void component_jacobian(ComponentFunction f, const double* params, Complex* out, Complex* jacobian) {
  dispatch(f, [&]<ComponentFunction F>() {
    constexpr int K = arity(F);
    const auto y = jet_outputs<F>(params);
    for (int j = 0; j < kComponentOutputs; ++j) {
      out[j] = y[j].v;
      for (int k = 0; k < K; ++k) jacobian[j * K + k] = y[j].d[k];
    }
  });
}

void component_vjp(ComponentFunction f, const double* params, const Complex* cotangent, double* grad) {
  dispatch(f, [&]<ComponentFunction F>() {
    for (int k = 0; k < arity(F); ++k) grad[k] = 0.0;
    vjp_one<F>(params, cotangent, grad);
  });
}

void evaluate_components(ComponentFunction f, Index count, const double* params, Index stride, Complex* out) {
  dispatch(f, [&]<ComponentFunction F>() {
    for (Index e = 0; e < count; ++e) {
      evaluate_one<F>(params + e * stride, out + e * kComponentOutputs);
    }
  });
}

void component_vjp_batch(ComponentFunction f, Index count, const double* params, Index stride,
                         const Complex* cotangents, double* grad) {
  dispatch(f, [&]<ComponentFunction F>() {
    for (Index e = 0; e < count; ++e) {
      vjp_one<F>(params + e * stride, cotangents + e * kComponentOutputs, grad + e * stride);
    }
  });
}

}  // namespace dfm
