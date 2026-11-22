"""Compare the fused complex covariance pullback with its NumPy fallback."""
import unittest
from unittest.mock import patch

import numpy as np

try:
    from differometor import _native_simulate as engine
except ModuleNotFoundError as exc:
    if exc.name != "differometor._native_components":
        raise
    engine = None


@unittest.skipUnless(engine is not None and engine._covariance_kernel() is not None,
                     "optional native covariance kernel unavailable")
class OrderedCovarianceTest(unittest.TestCase):
    def test_repeated_and_overlapping_writes(self):
        rng = np.random.default_rng(23)

        def complex_array(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        cases = [(2, 5, 4, 13), (0, 3, 2, 4), (2, 0, 3, 3),
                 (1, 4, 3, 0), (1, 1, 1, 12)]
        for detectors, frequencies, fields, entries in cases:
            with self.subTest(shape=(detectors, frequencies, fields, entries)):
                weights = complex_array((detectors, frequencies, fields * 2))[..., ::2]
                covariance = complex_array((frequencies, entries * 2))[:, ::2]
                seeds = rng.normal(size=(detectors, frequencies, 1))
                rows = rng.integers(fields, size=entries)
                columns = rows[::-1]
                args = (weights, covariance, seeds, rows, columns)
                actual = engine.covariance_pullback(*args)
                with patch.object(engine, "_covariance_kernel", return_value=None):
                    expected = engine.covariance_pullback(*args)
                for result, reference in zip(actual, expected):
                    np.testing.assert_allclose(result, reference, rtol=3e-15, atol=3e-15)
