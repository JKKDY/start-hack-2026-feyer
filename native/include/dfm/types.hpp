#pragma once

// Common numeric types of the native engine.
//
// All public numeric data is float64 / complex128. Matrices are row-major so
// that they map onto C-contiguous NumPy arrays without copying.

#include <Eigen/Core>

#include <complex>
#include <cstdint>

namespace dfm {

using Index = Eigen::Index;
using Real = double;
using Complex = std::complex<double>;

using VectorR = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using VectorC = Eigen::Matrix<Complex, Eigen::Dynamic, 1>;
using VectorI = Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>;

using MatrixR = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace dfm
