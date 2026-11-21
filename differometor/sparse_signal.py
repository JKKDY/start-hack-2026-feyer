"""Experimental CPU signal solver with a shared sparse factorization."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


class PackedSignal(NamedTuple):
    template: jax.Array
    update_slots: jax.Array
    coupling_slots: jax.Array
    conjugate: jax.Array


class PackedNoise(NamedTuple):
    template: jax.Array
    update_slots: jax.Array | None
    rows: jax.Array
    columns: jax.Array


def pack_noise_template(template, update_indices, *, zero_template=False):
    """Give repeated covariance writes the same destination slot."""
    matrix = np.asarray(template)
    initial = np.empty((2, 0), dtype=int) if zero_template else np.asarray(np.nonzero(matrix))
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


def built_signal_pattern(carrier, size, updates, couplings):
    """CSC pattern of the two carrier blocks, auxiliary identity, and possible writes."""
    width = carrier.shape[0]
    rows, cols = np.nonzero(carrier[:, :width])
    keys = [cols * size + rows, (cols + width) * size + rows + width,
            np.arange(2 * width, size) * (size + 1)]
    for rows_to_write, columns_to_write in (updates, couplings):
        write_rows = np.asarray(rows_to_write, dtype=np.intp) % size
        write_columns = np.asarray(columns_to_write, dtype=np.intp) % (size + 1)
        in_matrix = write_columns < size
        keys.append(write_columns[in_matrix] * size + write_rows[in_matrix])
    cols, rows = np.divmod(np.unique(np.concatenate(keys)), size)
    pointers = np.concatenate(([0], np.cumsum(np.bincount(cols, minlength=size))))
    return tuple(value.astype(np.int32) for value in (rows, cols, pointers))


def pack_signal_template(template, pattern, update_indices, coupling_indices, carrier_size):
    """Map matrix writes to coefficient slots, with the right-hand side at the end."""
    dense = np.asarray(template)
    size = len(dense)
    rows, columns = map(np.asarray, pattern[:2])
    keys = columns.astype(np.int64) * size + rows

    def slots(indices):
        write_rows, write_columns = np.asarray(indices, dtype=np.intp)
        write_rows = write_rows % size
        write_columns = write_columns % (size + 1)
        matrix_slots = np.searchsorted(keys, write_columns * size + write_rows)
        return np.where(write_columns == size, len(rows) + write_rows, matrix_slots).astype(np.int32)

    update_slots = slots(update_indices)
    coupling_slots = slots(coupling_indices)
    packed = np.concatenate((dense[rows, columns], dense[:, -1]))

    slot_rows = np.concatenate((rows, np.arange(size)))
    slot_columns = np.concatenate((columns, np.full(size, size)))
    lower_rows = (slot_rows >= carrier_size) & (slot_rows < 2 * carrier_size)
    conjugate = (lower_rows & (slot_columns >= carrier_size)) | (
        (slot_rows >= 2 * carrier_size) & (slot_columns < carrier_size)
    )
    return PackedSignal(*(np.asarray(value) for value in (packed, update_slots, coupling_slots, conjugate)))


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
    coefficients, rows, pointers, signal_rhs, detector_rhs = map(
        np.asarray, (coefficients, rows, pointers, signal_rhs, detector_rhs)
    )
    signal = np.empty_like(signal_rhs)
    detectors = np.empty_like(detector_rhs)
    size = signal_rhs.shape[1]
    for frequency, entries in enumerate(coefficients):
        matrix = csc_matrix((entries, rows, pointers), shape=(size, size))
        factors = splu(matrix)
        if transpose:      # A^T x = b
            solve_signal = lambda b: factors.solve(b, trans="T")  # noqa: E731
            apply_signal = matrix.T.__matmul__
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
                apply_detector = matrix.conj().__matmul__
            else:          # A^H u = d
                solve_detector = lambda b: factors.solve(b, trans="H")  # noqa: E731
                apply_detector = matrix.conj().T.__matmul__
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
