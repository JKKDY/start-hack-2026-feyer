#pragma once

// Full simulation pipeline (native counterpart of upstream `simulate`):
//
//   1. carrier: assemble, solve A_c x_c = b_c for every carrier sweep value;
//   2. signal: assemble with the carrier couplings, solve M_b y_b = r_b for
//      every frequency / sweep value (sidebands + optomechanics);
//   3. quantum noise (MISO): with carrier selection vectors s_q,
//        w = M^{-H} s,  u = V w,  c = M^{-1} u,  S_q = Re(s^H c),
//        noise = sqrt(2 S_q * UNIT_VACUUM * H_PLANCK * F0 / 4).
//      The three solves of one frequency share one SuperLU factorisation.
//
// Outputs follow upstream: carrier (carrier_ports, carrier_values), signal
// (signal_variables, signal_values), noise (noise_detectors, signal_values)
// before upstream's squeeze.
//
// simulate_passive runs on plain doubles. simulate_recorded runs the same
// steps on CoDiPack types: kernels as preaccumulated statements, every
// solve and the noise product as one external function (implicit adjoint).

#include <vector>

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/plan.hpp"
#include "dfm/simulator.hpp"
#include "dfm/types.hpp"

namespace dfm {

ForwardOutputs simulate_passive(const SimulationPlan& plan, const Eigen::Ref<const VectorR>& optimized_values);

/// Active outputs, interleaved complex (re, im) where complex:
///   carrier[2 * (i * CV + c) (+1)], signal[2 * (r * B + b) (+1)], noise[q * B + b].
struct RecordedOutputs {
  std::vector<ActiveReal> carrier;
  std::vector<ActiveReal> signal;
  std::vector<ActiveReal> noise;
};

RecordedOutputs simulate_recorded(const SimulationPlan& plan, const ActiveReal* optimized_values);

}  // namespace dfm
