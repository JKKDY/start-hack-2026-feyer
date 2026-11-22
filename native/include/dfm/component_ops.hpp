#pragma once

// Passive (double / complex) operations on the component kernels:
// evaluation, exact Jacobians and vector-Jacobian products, single entries
// and batches.
//
// Dispatch happens once per call (a switch on ComponentFunction that
// instantiates a fully inlined loop for that kernel), never per entry.
//
// VJP convention: for a real loss L and complex output y, the cotangent is
//     ybar = dL/dRe(y) + i dL/dIm(y),
// and the result is dL/dx for the real inputs x:
//     xbar_k = sum_j Re( conj(ybar_j) * dy_j/dx_k ).

#include "dfm/components.hpp"
#include "dfm/types.hpp"

namespace dfm {

/// out[0..3] = f(params[0..arity-1]).
void evaluate_component(ComponentFunction f, const double* params, Complex* out);

/// out[0..3] = f(params) and jacobian[j * arity + k] = d out_j / d params_k.
void component_jacobian(ComponentFunction f, const double* params, Complex* out, Complex* jacobian);

/// grad[k] = sum_j Re(conj(cotangent_j) * d out_j / d params_k), k < arity.
void component_vjp(ComponentFunction f, const double* params, const Complex* cotangent, double* grad);

/// Batch evaluation: entry e reads params[e * stride + k] and writes
/// out[e * 4 + j].
void evaluate_components(ComponentFunction f, Index count, const double* params, Index stride, Complex* out);

/// Batch VJP. Accumulates into grad[e * stride + k] (k < arity), so results
/// can be scattered onto shared parameters by the caller.
void component_vjp_batch(ComponentFunction f, Index count, const double* params, Index stride,
                         const Complex* cotangents, double* grad);

/// Runtime index -> enum, with range check (throws ValidationError).
ComponentFunction component_function_from_index(int index);

}  // namespace dfm
