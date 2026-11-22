import unittest

import jax
import jax.numpy as jnp
import numpy as np

from differometor.sparse_signal import build_signal_pattern, solve_signal_and_adjoint


class SparseSignalTest(unittest.TestCase):
    def test_direct_optical_packing_matches_dense_reference(self):
        from differometor.sparse_signal import (
            build_signal_pattern, pack_noise_template, pack_optical_noise,
            pack_optical_signal, pack_signal_template,
        )

        carrier = np.array([[0, 2j, 1], [3, -4, 0]], dtype=complex)
        size = 6
        template = np.eye(size, size + 1, dtype=complex)
        template[:2, :2] = template[2:4, 2:4] = carrier[:, :2]
        updates = np.array([[1, 1, 1, 3, 4], [4, 4, -1, -1, 1]])
        coupling = np.array([[0, 4], [2, 1]])
        reference = build_signal_pattern(template, updates, coupling)
        pattern, packed = pack_optical_signal(carrier, size, updates, coupling)
        expected = pack_signal_template(template, reference, updates, coupling, 2)
        for left, right in zip((*pattern, *packed), (*reference, *expected), strict=True):
            np.testing.assert_array_equal(left, right)
        noise_writes = updates.copy()
        noise_writes[1] %= size
        noise = pack_optical_noise(size, noise_writes)
        expected_noise = pack_noise_template(np.zeros((size, size), complex), noise_writes)
        for left, right in zip(noise, expected_noise, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_complex_signal_and_two_detectors_match_dense_gradients(self):
        matrix = jnp.asarray([[4 + 1j, 0, 1j], [-1j, 3, 1], [0, -2, 5 - 1j]])
        rhs = jnp.asarray([[1j, 2, 3], [4, 1, -2j]])
        detectors = jnp.stack((rhs.conj(), rhs * (1 + 2j)), axis=-1)
        augmented = jnp.column_stack((matrix, jnp.zeros(3)))
        no_writes = jnp.empty((2, 0), dtype=int)
        pattern = build_signal_pattern(augmented, no_writes, no_writes)
        rows, columns, _ = pattern

        def systems(tuning):
            return jnp.stack((matrix, matrix.at[0, 0].add(tuning * (1 - 1j))))

        def dense(tuning):
            blocks = systems(tuning)
            signal = jax.vmap(jnp.linalg.solve)(blocks, rhs)
            weights = jnp.linalg.solve(jnp.swapaxes(blocks.conj(), -1, -2), detectors)
            return signal, weights

        def sparse(tuning):
            return solve_signal_and_adjoint(systems(tuning)[:, rows, columns], pattern, rhs, detectors)

        def loss(solve, tuning):
            signal, weights = solve(tuning)
            return jnp.sum(jnp.abs(signal) ** 2) + jnp.sum(jnp.real(weights))

        for expected, actual in zip(dense(0.2), jax.jit(sparse)(0.2), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
        expected = jax.grad(lambda tuning: loss(dense, tuning))(0.2)
        actual = jax.jit(jax.grad(lambda tuning: loss(sparse, tuning)))(0.2)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
