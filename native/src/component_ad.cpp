#include "dfm/component_ad.hpp"

#include <codi.hpp>

#include "dfm/errors.hpp"

namespace dfm {
namespace {

using CF = ComponentFunction;
using PushHelper = codi::StatementPushHelper<ActiveReal>;

// `input(e, k)` returns the k-th active input of entry e by reference, so
// inputs are read in place (no copies, hence no copy statements).
template <ComponentFunction F, class Input>
void record_batch(Index count, Input&& input, ActiveReal* out) {
  constexpr int K = arity(F);
  constexpr int KJ = K > 0 ? K : 1;
  const bool recording = ActiveReal::getTape().isActive();
  PushHelper push;

  for (Index e = 0; e < count; ++e) {
    auto in = [&](int k) -> const ActiveReal& { return input(e, k); };
    ActiveReal* y = out + e * 2 * kComponentOutputs;

    bool any_active = false;
    if constexpr (K > 0) {
      for (int k = 0; k < K; ++k) any_active = any_active || (in(k).getIdentifier() != 0);
    }

    // Reset outputs to passive so a statement with no active arguments
    // cannot leave a stale identifier behind.
    for (int r = 0; r < 2 * kComponentOutputs; ++r) y[r] = 0.0;

    if (!recording || !any_active) {
      Complex p[KJ];
      for (int k = 0; k < K; ++k) p[k] = Complex(in(k).getValue(), 0.0);
      const auto v = kernels::evaluate<F>(p);
      for (int j = 0; j < kComponentOutputs; ++j) {
        y[2 * j] = v[j].real();
        y[2 * j + 1] = v[j].imag();
      }
      continue;
    }

    Jet<KJ> p[KJ];
    for (int k = 0; k < K; ++k) p[k] = Jet<KJ>::variable(in(k).getValue(), k);
    const auto v = kernels::evaluate<F>(p);
    for (int j = 0; j < kComponentOutputs; ++j) {
      push.startPushStatement();
      for (int k = 0; k < K; ++k) push.pushArgument(in(k), v[j].d[k].real());
      push.endPushStatement(y[2 * j], v[j].v.real());

      push.startPushStatement();
      for (int k = 0; k < K; ++k) push.pushArgument(in(k), v[j].d[k].imag());
      push.endPushStatement(y[2 * j + 1], v[j].v.imag());
    }
  }
}

template <class Input>
void record_dispatch(ComponentFunction f, Index count, Input&& input, ActiveReal* out) {
  switch (f) {
#define DFM_CASE(NAME)   case CF::NAME:           return record_batch<CF::NAME>(count, input, out);
    DFM_CASE(ForceCalculationRight)
    DFM_CASE(Laser)
    DFM_CASE(Surface)
    DFM_CASE(Space)
    DFM_CASE(SpaceModulation)
    DFM_CASE(SignalFunction)
    DFM_CASE(DummyFunction)
    DFM_CASE(VacuumQuantumNoise)
    DFM_CASE(LossQuantumNoise)
    DFM_CASE(SpaceLower)
    DFM_CASE(LaserAmplitudeModulation)
    DFM_CASE(LaserFrequencyModulation)
    DFM_CASE(Susceptibility)
    DFM_CASE(ForceCalculationLeft)
    DFM_CASE(CorrectedOptomechanicalPhaseLeft)
    DFM_CASE(CorrectedOptomechanicalPhaseRight)
    DFM_CASE(SpaceModulationLower)
    DFM_CASE(Squeezer)
    DFM_CASE(LaserFrequencyModulationLower)
#undef DFM_CASE
  }
  throw ValidationError("record_components: unknown component function");
}

}  // namespace

void record_components(ComponentFunction f, Index count, const ActiveReal* params, Index stride, ActiveReal* out) {
  record_dispatch(
      f, count, [params, stride](Index e, int k) -> const ActiveReal& { return params[e * stride + k]; }, out);
}

void record_components_indirect(ComponentFunction f, Index count, const ActiveReal* const* inputs, ActiveReal* out) {
  record_dispatch(
      f, count, [inputs](Index e, int k) -> const ActiveReal& { return *inputs[e * kComponentInputWidth + k]; }, out);
}

}  // namespace dfm
