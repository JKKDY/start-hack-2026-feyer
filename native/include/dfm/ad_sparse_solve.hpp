#pragma once

// Differentiable sparse solve for CoDiPack reverse mode.
//
// The solve y = A^{-1} b is recorded as ONE external function on the tape.
// SuperLU's internal arithmetic is never recorded: the reverse sweep calls
// the adjoint solve A^H on the same factors instead. Passive matrices are
// the expected use, since the optical matrix is assembled from parameters
// while its entries are constants for the solve.
//
// Complex values are split into interleaved real components on the tape:
// b = (b_0.re, b_0.im, b_1.re, ...). The reverse rule for real-valued
// adjoints ybar_re + i*ybar_im is
//     bbar = A^{-H} ybar,   d L/d b_re = Re(bbar),   d L/d b_im = Im(bbar).

#include <Eigen/Dense>
#include <codi.hpp>

#include <complex>

#include "dfm/sparse_lu.hpp"

namespace dfm {

using ActiveReal = codi::RealReverse;
using ActiveComplex = std::complex<ActiveReal>;
using ActiveVectorC = Eigen::Matrix<ActiveComplex, Eigen::Dynamic, 1>;

/// Records y = A^{-1} b on the active tape and returns y.
///
/// `lu` is borrowed: it must outlive every tape evaluation that reaches this
/// entry. Works when the tape is passive too (then it is a plain solve).
ActiveVectorC solve_recorded(const ComplexSparseLU& lu, const ActiveVectorC& b);

}  // namespace dfm
