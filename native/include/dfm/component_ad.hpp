#pragma once

// Recording component kernels on the CoDiPack tape.
//
// Each entry's exact 4 x arity Jacobian is computed with a complex jet in
// one pass and pushed as one statement per real output (8 statements of at
// most `arity` arguments). The kernel's intermediate operations (exp, sqrt,
// asin, ...) are never recorded, so the tape grows by a fixed small amount
// per entry regardless of how complicated the kernel is.
//
// Entries whose inputs are all passive (or when the tape is not recording)
// skip the Jacobian and evaluate the primal only.

#include "dfm/ad_sparse_solve.hpp"
#include "dfm/components.hpp"

namespace dfm {

/// Entry e reads active inputs params[e * stride + k] (k < arity) and
/// writes 8 active reals out[e * 8 + 2 j] = Re y_j, out[e * 8 + 2 j + 1] =
/// Im y_j. `out` may contain previous values; they are overwritten.
void record_components(ComponentFunction f, Index count, const ActiveReal* params, Index stride, ActiveReal* out);

/// Same, with inputs given as pointers: entry e reads *inputs[e * 7 + k].
/// Lets callers mix shared active parameters and per-batch constants
/// without copying active values (copies would add tape statements).
void record_components_indirect(ComponentFunction f, Index count, const ActiveReal* const* inputs, ActiveReal* out);

/// Same, with output pointers: entry e writes its 8 reals to outputs[e]
/// (lets batched kernels write straight into their final storage).
void record_components_indirect(ComponentFunction f, Index count, const ActiveReal* const* inputs,
                                ActiveReal* const* outputs);

}  // namespace dfm
