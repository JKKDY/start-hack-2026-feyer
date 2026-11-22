"""Native kernels and sparse adjoints against the original JAX implementation.

The reference is a verbatim copy of upstream components.py
(tests/reference/differometor_upstream/components.py), evaluated with JAX in float64 /
complex128 exactly as upstream `simulate` does (complex parameters with zero
imaginary part).

Gradients use arbitrary complex cotangents c: the real loss is
    L(x) = sum_j Re(conj(c_j) * y_j(x)),
which in the native convention is the VJP with ybar = c. JAX differentiates
the same L with jax.grad with respect to real inputs, so no convention
mismatch can hide in the comparison.

Skipped when JAX is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from differometor_native import components as native_components  # noqa: E402
from differometor_native._native import kernels, linalg  # noqa: E402

from differometor_upstream import components as upstream  # noqa: E402

FREQ, TUNING, REFL, LOSS = (100.0, 3000.0), (0.0, 90.0), (0.1, 0.9), (0.0, 0.1)
N, ALPHA, ANGLE, LENGTH = (1.0, 1.4), (-40.0, 40.0), (-180.0, 180.0), (1.0, 4000.0)
SIGNED_FREQ = (-3000.0, 3000.0)

# Physically valid ranges for the first `arity` parameters, in FUNCTIONS order.
RANGES = {
    "force_calculation_right": [ALPHA, N, N],
    "laser": [(0.1, 10.0), ANGLE],
    "surface": [LOSS, REFL, TUNING, SIGNED_FREQ, N, N, ALPHA],
    "space": [SIGNED_FREQ, LENGTH, N],
    "space_modulation": [FREQ, LENGTH, N],
    "signal_function": [(0.1, 5.0), ANGLE],
    "dummy_function": [],
    "vacuum_quantum_noise": [],
    "loss_quantum_noise": [(0.0, 1.0)],
    "space_lower": [SIGNED_FREQ, LENGTH, N],
    "laser_amplitude_modulation": [],
    "laser_frequency_modulation": [FREQ],
    "susceptibility": [FREQ, (0.1, 100.0)],
    "force_calculation_left": [ALPHA],
    "corrected_optomechanical_phase_left": [FREQ, TUNING, REFL, LOSS, N, ALPHA],
    "corrected_optomechanical_phase_right": [FREQ, TUNING, REFL, LOSS, N, N, ALPHA],
    "space_modulation_lower": [FREQ, LENGTH, N],
    "squeezer": [(0.0, 15.0), ANGLE],
    "laser_frequency_modulation_lower": [FREQ],
}

INDICES = list(range(len(upstream.FUNCTIONS)))


def sample(index, rng, rows=None):
    name = upstream.FUNCTIONS[index].__name__
    shape = (7,) if rows is None else (rows, 7)
    x = np.zeros(shape)
    for k, (lo, hi) in enumerate(RANGES[name]):
        x[..., k] = rng.uniform(lo, hi, size=shape[:-1])
    return x


def reference(index, x):
    return np.asarray(upstream.FUNCTIONS[index](jnp.asarray(x, dtype=jnp.complex128)))


def reference_grad(index, x, cotangent):
    c = jnp.asarray(cotangent)

    def loss(xr):
        y = upstream.FUNCTIONS[index](xr.astype(jnp.complex128))
        return jnp.sum(jnp.real(jnp.conj(c) * y))

    return np.asarray(jax.grad(loss)(jnp.asarray(x, dtype=jnp.float64)))


def assert_close(actual, expected, rtol):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=rtol * scale)


def test_registry_order_and_arity_match_upstream():
    assert kernels.names() == [f.__name__ for f in upstream.FUNCTIONS]
    for i, name in enumerate(kernels.names()):
        assert kernels.arity(i) == len(RANGES[name])


@pytest.mark.parametrize("index", INDICES, ids=[f.__name__ for f in upstream.FUNCTIONS])
def test_values_match_jax(index):
    rng = np.random.default_rng(100 + index)
    for _ in range(5):
        x = sample(index, rng)
        expected = reference(index, x)
        assert_close(kernels.evaluate(index, x), expected, 1e-13)
        # Public Python API (same function name as upstream).
        name = upstream.FUNCTIONS[index].__name__
        assert_close(getattr(native_components, name)(x), expected, 1e-13)


@pytest.mark.parametrize("index", INDICES, ids=[f.__name__ for f in upstream.FUNCTIONS])
def test_gradients_match_jax_for_complex_cotangents(index):
    rng = np.random.default_rng(200 + index)
    rows = 4
    x = sample(index, rng, rows)
    cot = rng.normal(size=(rows, 4)) + 1j * rng.normal(size=(rows, 4))
    expected = np.stack([reference_grad(index, x[r], cot[r]) for r in range(rows)])
    native_jet = np.stack([kernels.vjp(index, x[r], cot[r]) for r in range(rows)])
    native_tape = kernels.vjp_recorded(index, np.ascontiguousarray(x), np.ascontiguousarray(cot))
    assert_close(native_jet, expected, 1e-10)
    assert_close(native_tape, expected, 1e-10)


@pytest.mark.parametrize("index", INDICES, ids=[f.__name__ for f in upstream.FUNCTIONS])
def test_batch_evaluation_matches_single(index):
    rng = np.random.default_rng(300 + index)
    x = sample(index, rng, 6)
    batch = kernels.evaluate_batch(index, np.ascontiguousarray(x))
    single = np.stack([kernels.evaluate(index, row) for row in x])
    np.testing.assert_array_equal(batch, single)


def test_complex_branch_total_internal_reflection_matches_jax():
    # n_left / n_right * sin(alpha) > 1: asin leaves the real axis. Upstream's
    # complex128 evaluation gives a complex cosine; the port must agree.
    index = [f.__name__ for f in upstream.FUNCTIONS].index("force_calculation_right")
    x = np.array([70.0, 1.5, 1.0, 0, 0, 0, 0])
    expected = reference(index, x)
    assert abs(expected[0].imag) > 0
    assert_close(kernels.evaluate(index, x), expected, 1e-13)
    assert_close(kernels.vjp(index, x, np.array([1 + 2j, 0, 0, 0])),
                 reference_grad(index, x, np.array([1 + 2j, 0, 0, 0])), 1e-10)


def test_helper_formulas_match_jax():
    args = (0.6, 12.0, 250.0, 1.2, 1.0, 20.0)
    for name, call in [
        ("reflectivity_left", lambda m: m.reflectivity_left(*args[:4], args[5])),
        ("reflectivity_right", lambda m: m.reflectivity_right(*args)),
        ("transmissivity", lambda m: m.transmissivity(*args)),
        ("phase_shift", lambda m: m.phase_shift(12.0, 250.0, 1.2)),
        ("incidence_angle_right", lambda m: m.incidence_angle_right(20.0, 1.2, 1.4)),
        ("mirror_matrix", lambda m: m.mirror_matrix(1e-5, 0.7, 10.0, 0.0, 1.0, 1.0)),
        ("beamsplitter_matrix", lambda m: m.beamsplitter_matrix(1e-5, 0.5, 10.0, 0.0, 1.0, 1.0, 45.0)),
    ]:
        assert_close(call(native_components), np.asarray(call(upstream)), 1e-13)


# ------------------------------------------------------- sparse adjoints
def _system(n, m, seed):
    rng = np.random.default_rng(seed)
    dense = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        dense[i, i] = 4 + rng.normal() + 1j * rng.normal()
        if i + 1 < n:
            dense[i, i + 1] = rng.normal() + 1j * rng.normal()
            dense[i + 1, i] = rng.normal() + 1j * rng.normal()
        if i + 3 < n:
            dense[i + 3, i] = rng.normal() + 1j * rng.normal()
    indptr, indices, data = [0], [], []
    for c in range(n):
        rows = np.nonzero(dense[:, c])[0]
        indices += rows.tolist()
        data += dense[rows, c].tolist()
        indptr.append(len(indices))
    b = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    cot = rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))
    csc = (np.array(indptr, np.int32), np.array(indices, np.int32), np.array(data, np.complex128))
    return dense, csc, b, cot


def _jax_reference(operation, dense, b, cot):
    def loss(ar, ai, br, bi):
        a = ar + 1j * ai
        out = operation(a, (br + 1j * bi).T).T
        return jnp.sum(jnp.real(jnp.conj(cot) * out))

    args = (dense.real, dense.imag, b.real, b.imag)
    out = operation(jnp.asarray(dense), jnp.asarray(b).T).T
    g = jax.grad(loss, argnums=(0, 1, 2, 3))(*map(jnp.asarray, args))
    return np.asarray(out), np.asarray(g[0] + 1j * g[1]), np.asarray(g[2] + 1j * g[3])


@pytest.mark.parametrize("conjugate_transpose", [False, True])
@pytest.mark.parametrize("kind", ["solve", "matmul"])
def test_sparse_operation_adjoints_match_jax(kind, conjugate_transpose):
    n, m = 8, 3
    dense, (indptr, indices, data), b, cot = _system(n, m, 7 + conjugate_transpose)

    def op_matrix(a):
        return jnp.conj(a).T if conjugate_transpose else a

    if kind == "solve":
        operation = lambda a, rhs: jnp.linalg.solve(op_matrix(a), rhs)  # noqa: E731
        native = linalg.solve_vjp
    else:
        operation = lambda a, rhs: op_matrix(a) @ rhs  # noqa: E731
        native = linalg.matmul_vjp

    x, a_bar, b_bar = native(indptr, indices, data, np.ascontiguousarray(b), np.ascontiguousarray(cot),
                             conjugate_transpose)
    x_ref, a_bar_dense, b_bar_ref = _jax_reference(operation, dense, b, cot)

    cols = np.repeat(np.arange(n), np.diff(indptr))
    assert_close(x, x_ref, 1e-12)
    assert_close(a_bar, a_bar_dense[indices, cols], 1e-10)
    assert_close(b_bar, b_bar_ref, 1e-10)
