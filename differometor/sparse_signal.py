"""Experimental CPU signal solver with a shared sparse factorization."""

import os
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# The default JAX path also uses SciPy during lowering. Load it before tracing
# there, while the AOT path can wait until the first sparse solve.
if os.environ.get("DIFFEROMETOR_NATIVE") != "1":
    import scipy.sparse.linalg


class PackedSignal(NamedTuple):
    template: jax.Array
    update_slots: jax.Array
    coupling_slots: jax.Array
    conjugate: jax.Array


class PackedNoise(NamedTuple):
    template: jax.Array
    update_slots: jax.Array
    rows: jax.Array
    columns: jax.Array


def pack_noise_template(template, update_indices):
    """Give repeated covariance writes the same destination slot."""
    matrix = np.asarray(template)
    initial = np.asarray(np.nonzero(matrix))
    updates = np.asarray(update_indices)
    coordinates, inverse = np.unique(np.concatenate((initial, updates), axis=1).T, axis=0, return_inverse=True)
    rows, columns = coordinates.T
    return PackedNoise(*(np.asarray(value) for value in (
        matrix[rows, columns], inverse[initial.shape[1]:], rows, columns,
    )))


def build_signal_pattern(template, update_indices, coupling_indices):
    """Find entries that can be nonzero before frequency or parameter sweeps."""
    size = template.shape[0]
    possible = np.asarray(template)[:, :-1] != 0
    for indices in (update_indices, coupling_indices):
        rows, columns = np.asarray(indices)
        columns = columns % template.shape[1]
        matrix_entries = columns < size
        possible[rows[matrix_entries], columns[matrix_entries]] = True
    columns, rows = np.nonzero(possible.T)
    pointers = np.concatenate(([0], np.cumsum(np.bincount(columns, minlength=size))))
    return tuple(np.asarray(array, dtype=np.int32) for array in (rows, columns, pointers))


def pack_signal_template(template, pattern, update_indices, coupling_indices, carrier_size):
    """Map matrix writes to coefficient slots, with the right-hand side at the end."""
    dense = np.asarray(template)
    size = len(dense)
    rows, columns = map(np.asarray, pattern[:2])
    slots = np.full(dense.shape, -1, dtype=np.int32)
    slots[rows, columns] = np.arange(len(rows))
    slots[:, -1] = np.arange(len(rows), len(rows) + size)
    update_slots = slots[tuple(np.asarray(update_indices))]
    coupling_slots = slots[tuple(np.asarray(coupling_indices))]
    packed = np.concatenate((dense[rows, columns], dense[:, -1]))

    slot_rows = np.concatenate((rows, np.arange(size)))
    slot_columns = np.concatenate((columns, np.full(size, size)))
    lower_rows = (slot_rows >= carrier_size) & (slot_rows < 2 * carrier_size)
    conjugate = (lower_rows & (slot_columns >= carrier_size)) | (
        (slot_rows >= 2 * carrier_size) & (slot_columns < carrier_size)
    )
    return PackedSignal(*(np.asarray(value) for value in (packed, update_slots, coupling_slots, conjugate)))


def pack_optical_signal(carrier, size, updates, coupling):
    """Construct CSC entries directly from both carrier sidebands and potential writes."""
    nc = len(carrier)
    r, c = np.nonzero(carrier[:, :nc])
    sideband = c * size + r
    diagonal = np.arange(2 * nc, size)
    initial = np.concatenate((sideband, sideband + nc * (size + 1), diagonal * (size + 1)))
    def keys(indices):
        row, col = indices
        return (col % (size + 1)) * size + (row % size)
    update, coupled = keys(updates), keys(coupling)
    boundary = size * size
    occupied = np.unique(np.concatenate((initial, update[update < boundary], coupled[coupled < boundary])))
    cols, rows = np.divmod(occupied, size)
    pointers = np.r_[0, np.cumsum(np.bincount(cols, minlength=size))]
    pattern = tuple(x.astype(np.int32) for x in (rows, cols, pointers))
    def slots(k):
        return np.where(k >= boundary, len(rows) + k - boundary, np.searchsorted(occupied, k)).astype(np.int32)
    values = np.zeros(len(rows) + size, dtype=carrier.dtype)
    values[slots(initial)] = np.r_[carrier[r, c], carrier[r, c], np.ones(len(diagonal))]
    rr = np.r_[rows, np.arange(size)]
    cc = np.r_[cols, np.full(size, size)]
    conjugate = ((rr >= nc) & (rr < 2 * nc) & (cc >= nc)) | ((rr >= 2 * nc) & (cc < nc))
    return pattern, PackedSignal(values, slots(update), slots(coupled), conjugate)


def pack_optical_noise(size, indices):
    """The initial covariance is zero. Every possible nonzero has a write instruction."""
    r, c = indices
    keys, slots = np.unique((r % size) * size + (c % size), return_inverse=True)
    rows, cols = np.divmod(keys, size)
    return PackedNoise(np.zeros(len(keys), dtype=complex), slots, rows, cols)


# Fixed-precision iterative refinement steps applied after every sparse solve.
#
# Why: the solution vectors span ~36 orders of magnitude (optical fields vs. mechanical
# unknowns). SuperLU is normwise accurate (~1e-14) but can get the *tiny* components
# wrong by orders of magnitude, and the noise covariance weights some of those components
# heavily. Measured on the Voyager setup: noise off by 4.8e-3 in ln (limit 1e-5) without
# refinement, 3e-13 with it. Each step costs one sparse mat-vec plus one triangular solve
# pair with the existing factors (no refactorisation).
REFINEMENT_STEPS = 2


def _refine(solve, apply, rhs, solution):
    """x <- x + solve(b - A x), repeated REFINEMENT_STEPS times."""
    for _ in range(REFINEMENT_STEPS):
        solution = solution + solve(rhs - apply(solution))
    return solution


def _factor_and_solve(coefficients, rows, pointers, signal_rhs, detector_rhs, *, transpose):
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    coefficients, rows, pointers, signal_rhs, detector_rhs = map(
        np.asarray, (coefficients, rows, pointers, signal_rhs, detector_rhs)
    )
    signal = np.empty_like(signal_rhs)
    detectors = np.empty_like(detector_rhs)
    size = signal_rhs.shape[1]
    if not len(coefficients):
        return signal, detectors

    # A frequency sweep has one CSC structure. Keep the public inputs immutable
    # and refresh only the numeric buffers for each new factorization.
    matrix = csc_matrix((coefficients[0].copy(), rows, pointers), shape=(size, size))
    canonical = matrix.has_canonical_format
    if canonical:
        signal_operator = matrix.T if transpose else matrix
        if detector_rhs.shape[-1]:
            conjugate = matrix.copy()
            detector_operator = conjugate if transpose else conjugate.T
    for frequency, entries in enumerate(coefficients):
        if canonical:
            matrix.data[:] = entries
            if detector_rhs.shape[-1]:
                np.conjugate(entries, out=conjugate.data)
        else:
            matrix = csc_matrix((entries, rows, pointers), shape=(size, size))
        factors = splu(matrix)
        if transpose:      # A^T x = b
            solve_signal = lambda b: factors.solve(b, trans="T")  # noqa: E731
            apply_signal = (signal_operator if canonical else matrix.T).__matmul__
        else:              # A x = b
            solve_signal = factors.solve
            apply_signal = matrix.__matmul__
        rhs = signal_rhs[frequency]
        signal[frequency] = _refine(solve_signal, apply_signal, rhs, solve_signal(rhs))
        if detector_rhs.shape[-1]:
            rhs = detector_rhs[frequency]
            # The linear transpose of the detector's A^H system is conj(A).
            if transpose:  # conj(A) u = d
                solve_detector = lambda b: factors.solve(b.conj()).conj()  # noqa: E731
                apply_detector = (detector_operator if canonical else matrix.conj()).__matmul__
            else:          # A^H u = d
                solve_detector = lambda b: factors.solve(b, trans="H")  # noqa: E731
                apply_detector = (detector_operator if canonical else matrix.conj().T).__matmul__
            detectors[frequency] = _refine(solve_detector, apply_detector, rhs, solve_detector(rhs))
    return signal, detectors


def solve_signal_and_adjoint(coefficients, pattern, signal_rhs, detector_rhs):
    """Solve A x = b and A^H u = d with implicit derivatives through A."""
    rows, columns, pointers = pattern
    result_spec = tuple(
        jax.ShapeDtypeStruct(rhs.shape, rhs.dtype) for rhs in (signal_rhs, detector_rhs)
    )

    def matvec(vectors):
        signal, detectors = vectors
        signal_product = jnp.zeros_like(signal).at[:, rows].add(coefficients * signal[:, columns])
        detector_product = jnp.zeros_like(detectors).at[:, columns].add(
            coefficients.conj()[..., None] * detectors[:, rows, :]
        )
        return signal_product, detector_product

    def solve(_, rhs, *, transpose=False):
        return jax.pure_callback(
            partial(_factor_and_solve, transpose=transpose),
            result_spec, coefficients, rows, pointers, *rhs, vmap_method="sequential",
        )

    return jax.lax.custom_linear_solve(
        matvec, (signal_rhs, detector_rhs), solve, partial(solve, transpose=True),
    )
