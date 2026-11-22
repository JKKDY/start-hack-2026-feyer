#pragma once

// Physical constants and thresholds, identical to upstream components.py.

namespace dfm::constants {

inline constexpr double C_LIGHT = 299'792'458.0;        ///< m/s
inline constexpr double LAMBDA = 1064e-9;               ///< m
inline constexpr double UNIT_VACUUM = 1.0;
inline constexpr double F0 = C_LIGHT / LAMBDA;          ///< Hz
inline constexpr double F = 0.0;                        ///< laser offset, Hz
inline constexpr double H_PLANCK = 6.626'070'15e-34;    ///< J s
inline constexpr double X_SCALE = 1e-9;                 ///< m
inline constexpr double EPSILON_0C = 1.0;
inline constexpr double DEFAULT_REFRACTIVE_INDEX = 1.0;
inline constexpr double SOFT_SIDE_POWER_THRESHOLD = 2e3;   ///< W
inline constexpr double HARD_SIDE_POWER_THRESHOLD = 3.5e6; ///< W
inline constexpr double DETECTOR_POWER_THRESHOLD = 1e-2;   ///< W

}  // namespace dfm::constants
