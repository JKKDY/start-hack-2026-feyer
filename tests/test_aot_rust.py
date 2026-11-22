"""Compare the direct Rust sweep and its summed pullback with the AOT engine."""
import unittest

import numpy as np

from differometor.rust_kernel import installed_kernel


@unittest.skipUnless(installed_kernel(), "requires a host-built Rust kernel")
class RustSweepTest(unittest.TestCase):
    def test_component_values_and_sweep_derivatives(self):
        from differometor._native_simulate import coefficients
        from differometor._rust_sweep import Sweep
        from differometor.components import FUNCTIONS

        rng = np.random.default_rng(17)
        inputs = np.arange(7)[None, :]
        groups = [(inputs, np.zeros(4, dtype=int), np.arange(4)) for _ in FUNCTIONS]
        instruction = (groups, rng.permutation(4 * len(groups)))
        x = rng.uniform(0.1, 0.4, (3, 7)).astype(np.complex128)
        x += 1j * rng.uniform(-0.01, 0.01, x.shape)
        for varying in ([], [2], list(range(7))):
            with self.subTest(varying=varying):
                parameters = np.broadcast_to(x[0], x.shape).copy()
                parameters[:, varying] = x[:, varying]
                sweep = Sweep(instruction, varying, 7, installed_kernel())
                expected = coefficients(parameters, instruction)
                np.testing.assert_allclose(sweep(parameters, None), expected, rtol=2e-11, atol=1e-9)
                seed = rng.normal(size=expected.shape) + 1j * rng.normal(size=expected.shape)
                backward = coefficients(parameters, instruction, seed, np.ones(7, dtype=bool))
                actual = sweep(parameters, seed)
                shared = np.setdiff1d(np.arange(7), varying)
                np.testing.assert_allclose(actual[:, varying], backward[:, varying], rtol=2e-11, atol=1e-9)
                np.testing.assert_allclose(actual[:, shared].sum(0), backward[:, shared].sum(0),
                                           rtol=2e-11, atol=1e-9)
