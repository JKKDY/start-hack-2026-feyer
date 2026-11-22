"""Python smoke tests for the dfm_native extension.

Reference values come from NumPy dense linear algebra, so these tests do not
depend on SciPy or on JAX.
"""

import numpy as np
import pytest

import dfm_native


def _test_matrix(n: int):
    """Same deterministic complex matrix as tests/cpp/test_core.cpp, dense."""
    a = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        a[i, i] = complex(4.0 + 0.1 * i, 0.5 - 0.05 * i)
        if i + 1 < n:
            a[i, i + 1] = complex(-1.0, 0.3 * (i % 3))
            a[i + 1, i] = complex(0.2, -0.7)
        if i + 3 < n:
            a[i, i + 3] = complex(0.0, 0.4)
    return a


def _to_csc(a: np.ndarray):
    """Dense -> CSC arrays (indptr int32, indices int32, data complex128)."""
    n = a.shape[0]
    indptr = [0]
    indices = []
    data = []
    for col in range(n):
        rows = np.nonzero(a[:, col])[0]
        indices.extend(rows.tolist())
        data.extend(a[rows, col].tolist())
        indptr.append(len(indices))
    return (
        np.asarray(indptr, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=np.complex128),
    )


def _rhs(n: int) -> np.ndarray:
    return np.array([complex(1.0 + i, -0.5 * i) for i in range(n)], dtype=np.complex128)


def test_version_is_string():
    assert isinstance(dfm_native.version(), str)
    assert dfm_native.version()


@pytest.mark.parametrize("n", [1, 7, 12])
def test_sparse_solve_matches_numpy(n):
    a = _test_matrix(n)
    b = _rhs(n)
    lu = dfm_native.ComplexSparseLU(*_to_csc(a))

    assert lu.size == n
    x = np.asarray(lu.solve(b))
    x_ref = np.linalg.solve(a, b)

    assert x.dtype == np.complex128
    np.testing.assert_allclose(x, x_ref, rtol=1e-11, atol=1e-13)


def test_adjoint_solve_uses_conjugate_transpose():
    n = 12
    a = _test_matrix(n)
    b = _rhs(n)
    lu = dfm_native.ComplexSparseLU(*_to_csc(a))

    y = np.asarray(lu.solve_adjoint(b))
    y_ref = np.linalg.solve(a.conj().T, b)

    np.testing.assert_allclose(y, y_ref, rtol=1e-11, atol=1e-13)


def test_factorisation_is_reused_for_many_right_hand_sides():
    n = 9
    a = _test_matrix(n)
    lu = dfm_native.ComplexSparseLU(*_to_csc(a))
    for scale in (1.0, -2.0, 0.5j):
        b = _rhs(n) * scale
        np.testing.assert_allclose(np.asarray(lu.solve(b)), np.linalg.solve(a, b), rtol=1e-11)


def test_wrong_rhs_length_raises():
    a = _test_matrix(4)
    lu = dfm_native.ComplexSparseLU(*_to_csc(a))
    with pytest.raises(ValueError):
        lu.solve(_rhs(3))


def test_singular_matrix_raises():
    a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(RuntimeError):
        dfm_native.ComplexSparseLU(*_to_csc(a))


def test_invalid_index_raises():
    indptr = np.array([0, 1, 2], dtype=np.int32)
    indices = np.array([0, 5], dtype=np.int32)  # row 5 is out of range for n=2
    data = np.array([1.0, 1.0], dtype=np.complex128)
    with pytest.raises(ValueError):
        dfm_native.ComplexSparseLU(indptr, indices, data)


def test_complex_gradient_smoke_matches_analytic():
    value, gradient, max_error = dfm_native.complex_gradient_smoke([0.3, -0.2, 0.7])
    assert np.isfinite(value)
    assert np.asarray(gradient).shape == (3,)
    assert max_error < 1e-10
