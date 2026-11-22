#pragma once

// Detector readouts (upstream components.amplitude_detector,
// power_detector, signal_detector). Elementwise kernels over contiguous
// memory, written with Eigen array expressions so they vectorise.

#include "dfm/types.hpp"

namespace dfm {

/// out = sqrt(0.5 * EPSILON_0C) * |in|, elementwise over n values.
void amplitude_detector(const Complex* in, Real* out, Index n);

/// out = 0.5 * EPSILON_0C * |in|^2, elementwise over n values.
void power_detector(const Complex* in, Real* out, Index n);

/// Demodulated signal for N carrier ports:
///   out = conj(carrier) * signal[0:N] + carrier * signal[N:2N]
/// carrier is (N, Kc), signal is (M, Ks) with M >= 2N, and the result is
/// (N, K) with K = max(Kc, Ks). Kc and Ks must each be 1 or K (column
/// broadcasting, as NumPy does for these shapes). Throws ValidationError.
MatrixC signal_detector(const Eigen::Ref<const MatrixC>& carrier,
                        const Eigen::Ref<const MatrixC>& signal);

}  // namespace dfm
