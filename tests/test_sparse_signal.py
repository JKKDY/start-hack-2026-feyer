import unittest

import jax
import jax.numpy as jnp
import numpy as np

from differometor.sparse_signal import build_signal_pattern, solve_signal_and_adjoint


class SparseSignalTest(unittest.TestCase):
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
