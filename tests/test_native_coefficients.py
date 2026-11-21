"""Check the optional native path against JAX for all registered optical functions."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from differometor.components import FUNCTIONS
from differometor.rust_kernel import installed_kernel
from differometor.simulate import _sweep_coefficients


@unittest.skipUnless(
    installed_kernel() is not None, "optional CPU kernel has not been built"
)
class NativeCoefficientTest(unittest.TestCase):
    def test_values_and_derivatives(self):
        from differometor.native_coefficients import evaluate

        groups = tuple(
            (np.arange(7)[None, :], np.zeros(4, dtype=int), np.arange(4))
            for _ in FUNCTIONS
        )
        varying = np.arange(7)
        rng = np.random.default_rng(4)
        x = jnp.asarray(
            [
                [0.12, 0.31, 0.42, 0.62, 1.1, 1.2, 0.3],
                [0.23, 0.34, 0.51, 0.72, 1.02, 1.1, 0.5],
            ],
            dtype=jnp.complex128,
        )
        x += 1j * rng.uniform(-0.005, 0.005, x.shape)

        def reference(x):
            return _sweep_coefficients(x, None, None, groups, None, varying)

        def native(x):
            return evaluate(x, groups, None, varying, reference)

        shape = reference(x).shape
        cotangent = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        direction = jnp.asarray(
            rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
        )

        def pullback(f):
            return lambda x: jax.vjp(f, x)[1](cotangent)[0]

        for name, transform in (
            ("value", lambda f: f(x)),
            ("reverse", lambda f: pullback(f)(x)),
            ("forward", lambda f: jax.jvp(f, (x,), (direction,))[1]),
            ("second", lambda f: jax.jvp(pullback(f), (x,), (direction,))[1]),
            ("batched", lambda f: jax.vmap(f)(jnp.stack((x, x)))),
        ):
            with self.subTest(transform=name):
                np.testing.assert_allclose(
                    transform(native), transform(reference), rtol=2e-11, atol=1e-9
                )
