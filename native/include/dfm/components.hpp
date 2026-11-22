#pragma once

// Native port of the 19 registered component functions of upstream
// differometor/components.py (the FUNCTIONS list), plus the helpers they use.
//
// Conventions (identical to upstream)
// -----------------------------------
// - Angles in degrees, frequencies in Hz, lengths in m.
// - Every kernel maps a parameter vector to a fixed-length complex vector of
//   4 entries (upstream `standardize_output` zero-pads to 4).
// - Upstream evaluates on complex128 parameters. The kernels here are
//   written once, templated on the scalar type S, and use complex arithmetic
//   in upstream's operation order:
//     S = Complex  -> primal value
//     S = Jet<K>   -> value and the exact 4 x K Jacobian in one pass
//   Inputs are real (upstream's parameters have zero imaginary part); complex
//   branches such as asin(z) for |z| > 1 still follow upstream.
// - Kernel inputs: the first `arity(f)` parameters. Upstream pads every
//   function's input row to 7 values; extra values are ignored.
//
// Kernels are header-only and statically dispatched (one switch per batch,
// see component_ops.hpp), so they inline into the batch loops.

#include <array>
#include <cmath>
#include <complex>
#include <string_view>

#include "dfm/constants.hpp"
#include "dfm/jet.hpp"
#include "dfm/types.hpp"

namespace dfm {

/// Index into upstream FUNCTIONS. Order and values must not change.
enum class ComponentFunction : int {
  ForceCalculationRight = 0,
  Laser = 1,
  Surface = 2,
  Space = 3,
  SpaceModulation = 4,
  SignalFunction = 5,
  DummyFunction = 6,
  VacuumQuantumNoise = 7,
  LossQuantumNoise = 8,
  SpaceLower = 9,
  LaserAmplitudeModulation = 10,
  LaserFrequencyModulation = 11,
  Susceptibility = 12,
  ForceCalculationLeft = 13,
  CorrectedOptomechanicalPhaseLeft = 14,
  CorrectedOptomechanicalPhaseRight = 15,
  SpaceModulationLower = 16,
  Squeezer = 17,
  LaserFrequencyModulationLower = 18,
};

inline constexpr int kNumComponentFunctions = 19;
inline constexpr int kComponentOutputs = 4;   ///< upstream standardize_output length
inline constexpr int kComponentInputWidth = 7;  ///< upstream padded input row width

/// Number of parameters each kernel reads.
constexpr int arity(ComponentFunction f) noexcept {
  switch (f) {
    case ComponentFunction::ForceCalculationRight: return 3;
    case ComponentFunction::Laser: return 2;
    case ComponentFunction::Surface: return 7;
    case ComponentFunction::Space: return 3;
    case ComponentFunction::SpaceModulation: return 3;
    case ComponentFunction::SignalFunction: return 2;
    case ComponentFunction::DummyFunction: return 0;
    case ComponentFunction::VacuumQuantumNoise: return 0;
    case ComponentFunction::LossQuantumNoise: return 1;
    case ComponentFunction::SpaceLower: return 3;
    case ComponentFunction::LaserAmplitudeModulation: return 0;
    case ComponentFunction::LaserFrequencyModulation: return 1;
    case ComponentFunction::Susceptibility: return 2;
    case ComponentFunction::ForceCalculationLeft: return 1;
    case ComponentFunction::CorrectedOptomechanicalPhaseLeft: return 6;
    case ComponentFunction::CorrectedOptomechanicalPhaseRight: return 7;
    case ComponentFunction::SpaceModulationLower: return 3;
    case ComponentFunction::Squeezer: return 2;
    case ComponentFunction::LaserFrequencyModulationLower: return 1;
  }
  return 0;
}

/// Upstream Python function name.
constexpr std::string_view name(ComponentFunction f) noexcept {
  switch (f) {
    case ComponentFunction::ForceCalculationRight: return "force_calculation_right";
    case ComponentFunction::Laser: return "laser";
    case ComponentFunction::Surface: return "surface";
    case ComponentFunction::Space: return "space";
    case ComponentFunction::SpaceModulation: return "space_modulation";
    case ComponentFunction::SignalFunction: return "signal_function";
    case ComponentFunction::DummyFunction: return "dummy_function";
    case ComponentFunction::VacuumQuantumNoise: return "vacuum_quantum_noise";
    case ComponentFunction::LossQuantumNoise: return "loss_quantum_noise";
    case ComponentFunction::SpaceLower: return "space_lower";
    case ComponentFunction::LaserAmplitudeModulation: return "laser_amplitude_modulation";
    case ComponentFunction::LaserFrequencyModulation: return "laser_frequency_modulation";
    case ComponentFunction::Susceptibility: return "susceptibility";
    case ComponentFunction::ForceCalculationLeft: return "force_calculation_left";
    case ComponentFunction::CorrectedOptomechanicalPhaseLeft: return "corrected_optomechanical_phase_left";
    case ComponentFunction::CorrectedOptomechanicalPhaseRight: return "corrected_optomechanical_phase_right";
    case ComponentFunction::SpaceModulationLower: return "space_modulation_lower";
    case ComponentFunction::Squeezer: return "squeezer";
    case ComponentFunction::LaserFrequencyModulationLower: return "laser_frequency_modulation_lower";
  }
  return "unknown";
}

namespace kernels {

// jnp.pi / jnp.e as float64.
inline constexpr double PI = 3.141592653589793;
inline constexpr double E = 2.718281828459045;

template <class S>
using Outputs = std::array<S, kComponentOutputs>;

inline const Complex I(0.0, 1.0);

// jnp.radians(x) == x * (pi / 180)
template <class S>
inline S radians(const S& x) {
  return x * (PI / 180.0);
}

// ------------------------------------------------------------- helpers
template <class S>
inline S incidence_angle_left(const S& alpha) {
  using std::cos;
  return cos(radians(alpha));
}

template <class S>
inline S incidence_angle_right(const S& alpha, const S& refractive_index_left, const S& refractive_index_right) {
  using std::asin;
  using std::cos;
  using std::sin;
  return cos(asin(refractive_index_left / refractive_index_right * sin(radians(alpha))));
}

template <class S>
inline S phase_shift(const S& tuning, const S& frequency, const S& refractive_index) {
  return 2.0 * radians(tuning) * refractive_index * (1.0 + frequency / constants::F0);
}

template <class S>
inline S reflectivity_left(const S& reflectivity, const S& tuning, const S& frequency,
                           const S& refractive_index_left, const S& alpha) {
  using std::exp;
  using std::sqrt;
  return -sqrt(reflectivity) *
         exp(I * phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha));
}

template <class S>
inline S reflectivity_right(const S& reflectivity, const S& tuning, const S& frequency,
                            const S& refractive_index_left, const S& refractive_index_right, const S& alpha) {
  using std::exp;
  using std::sqrt;
  return -sqrt(reflectivity) *
         exp(-I * phase_shift(tuning, frequency, refractive_index_right) *
             incidence_angle_right(alpha, refractive_index_right, refractive_index_left));
}

template <class S>
inline S transmissivity(const S& transmissivity_value, const S& tuning, const S& frequency,
                        const S& refractive_index_left, const S& refractive_index_right, const S& alpha) {
  using std::exp;
  using std::sqrt;
  const S left_term = phase_shift(tuning, frequency, refractive_index_left) * incidence_angle_left(alpha);
  const S right_term = phase_shift(tuning, frequency, refractive_index_right) *
                       incidence_angle_right(alpha, refractive_index_left, refractive_index_right);
  return -sqrt(transmissivity_value) * exp(I * (PI / 2.0 + 0.5 * (left_term - right_term)));
}

template <class S>
inline S optomechanical_phase_left(const S& alpha) {
  return I * constants::X_SCALE * 2.0 * PI / constants::LAMBDA * incidence_angle_left(alpha);
}

template <class S>
inline S optomechanical_phase_right(const S& alpha, const S& refractive_index_left, const S& refractive_index_right) {
  return I * constants::X_SCALE * 2.0 * PI / constants::LAMBDA *
         incidence_angle_right(alpha, refractive_index_left, refractive_index_right);
}

template <class S>
inline S tuning_correction(const S* p) {
  using std::exp;
  return exp(I * radians(p[1]) * p[0] / constants::F0);
}

// ------------------------------------------------- registered functions
template <class S>
inline Outputs<S> force_calculation_right(const S* p) {
  return {incidence_angle_right(p[0], p[1], p[2]) / (constants::C_LIGHT * constants::X_SCALE), S(0.0), S(0.0),
          S(0.0)};
}

template <class S>
inline Outputs<S> laser(const S* p) {
  using std::exp;
  using std::sqrt;
  return {sqrt(2.0 * p[0] / constants::EPSILON_0C) * exp(I * radians(p[1])), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> surface(const S* p) {
  const S absolute_reflectivity = (1.0 - p[0]) * p[1];
  const S absolute_transmissivity = (1.0 - p[0]) * (1.0 - p[1]);
  return {reflectivity_left(absolute_reflectivity, p[2], p[3], p[4], p[6]),
          transmissivity(absolute_transmissivity, p[2], p[3], p[4], p[5], p[6]),
          reflectivity_right(absolute_reflectivity, p[2], p[3], p[4], p[5], p[6]), S(0.0)};
}

template <class S>
inline Outputs<S> space(const S* p) {
  using std::exp;
  return {-exp(-I * 2.0 * PI * p[0] * p[1] * p[2] / constants::C_LIGHT), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> space_modulation(const S* p) {
  using std::exp;
  using std::sin;
  const S w_g = 2.0 * PI * p[0];
  const double w_0 = 2.0 * PI * constants::F0;
  const S m_g = -0.5 * w_0 / w_g * sin(w_g * p[1] * p[2] / 2.0 / constants::C_LIGHT);
  const S phi_sb = -w_g * p[1] * p[2] / 2.0 / constants::C_LIGHT;
  return {I * m_g * exp(I * phi_sb), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> signal_function(const S* p) {
  using std::exp;
  return {p[0] * exp(I * radians(p[1])), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> dummy_function(const S* /*p*/) {
  return {S(1.0), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> vacuum_quantum_noise(const S* /*p*/) {
  return {S(constants::UNIT_VACUUM / 2.0), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> loss_quantum_noise(const S* p) {
  return {p[0] / 2.0, S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> space_lower(const S* p) {
  const S q[3] = {-p[0], p[1], p[2]};
  return space(q);
}

template <class S>
inline Outputs<S> laser_amplitude_modulation(const S* /*p*/) {
  return {S(constants::EPSILON_0C * 0.5), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> laser_frequency_modulation(const S* p) {
  return {constants::EPSILON_0C * 0.5 / p[0], S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> susceptibility(const S* p) {
  const S w = 2.0 * PI * p[0];
  const S factor = -1.0 / (p[1] * (w * w));
  return {-factor, S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> force_calculation_left(const S* p) {
  return {-(incidence_angle_left(p[0]) / (constants::C_LIGHT * constants::X_SCALE)), S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> corrected_optomechanical_phase_left(const S* p) {
  const S absolute_reflectivity = (1.0 - p[3]) * p[2];
  const S factor = tuning_correction(p) * optomechanical_phase_left(p[5]) *
                   reflectivity_left(absolute_reflectivity, p[1], p[0], p[4], p[5]);
  return {-factor, S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> corrected_optomechanical_phase_right(const S* p) {
  using std::conj;
  const S absolute_reflectivity = (1.0 - p[3]) * p[2];
  const S factor = conj(tuning_correction(p)) * -1.0 * optomechanical_phase_right(p[6], p[4], p[5]) *
                   reflectivity_right(absolute_reflectivity, p[1], p[0], p[4], p[5], p[6]);
  return {-factor, S(0.0), S(0.0), S(0.0)};
}

template <class S>
inline Outputs<S> space_modulation_lower(const S* p) {
  const S q[3] = {-p[0], p[1], p[2]};
  return space_modulation(q);
}

template <class S>
inline Outputs<S> squeezer(const S* p) {
  using std::conj;
  using std::cosh;
  using std::exp;
  using std::sinh;
  const double vacuum_unit = constants::UNIT_VACUUM / 2.0;
  const S squeezing_parameter = p[0] / (20.0 * std::log10(E));
  const S phase = exp(I * 2.0 * radians(p[1]));
  const S upper_diagonal = vacuum_unit * cosh(2.0 * squeezing_parameter);
  const S upper_off_diagonal = vacuum_unit * sinh(2.0 * squeezing_parameter) * phase;
  return {upper_diagonal, upper_off_diagonal, conj(upper_diagonal), conj(upper_off_diagonal)};
}

template <class S>
inline Outputs<S> laser_frequency_modulation_lower(const S* p) {
  const S q[1] = {-p[0]};
  return laser_frequency_modulation(q);
}

/// Compile-time dispatch: kernel F evaluated on scalar type S.
template <ComponentFunction F, class S>
inline Outputs<S> evaluate(const S* p) {
  using CF = ComponentFunction;
  if constexpr (F == CF::ForceCalculationRight) return force_calculation_right(p);
  else if constexpr (F == CF::Laser) return laser(p);
  else if constexpr (F == CF::Surface) return surface(p);
  else if constexpr (F == CF::Space) return space(p);
  else if constexpr (F == CF::SpaceModulation) return space_modulation(p);
  else if constexpr (F == CF::SignalFunction) return signal_function(p);
  else if constexpr (F == CF::DummyFunction) return dummy_function(p);
  else if constexpr (F == CF::VacuumQuantumNoise) return vacuum_quantum_noise(p);
  else if constexpr (F == CF::LossQuantumNoise) return loss_quantum_noise(p);
  else if constexpr (F == CF::SpaceLower) return space_lower(p);
  else if constexpr (F == CF::LaserAmplitudeModulation) return laser_amplitude_modulation(p);
  else if constexpr (F == CF::LaserFrequencyModulation) return laser_frequency_modulation(p);
  else if constexpr (F == CF::Susceptibility) return susceptibility(p);
  else if constexpr (F == CF::ForceCalculationLeft) return force_calculation_left(p);
  else if constexpr (F == CF::CorrectedOptomechanicalPhaseLeft) return corrected_optomechanical_phase_left(p);
  else if constexpr (F == CF::CorrectedOptomechanicalPhaseRight) return corrected_optomechanical_phase_right(p);
  else if constexpr (F == CF::SpaceModulationLower) return space_modulation_lower(p);
  else if constexpr (F == CF::Squeezer) return squeezer(p);
  else return laser_frequency_modulation_lower(p);
}

}  // namespace kernels
}  // namespace dfm
