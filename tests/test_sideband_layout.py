import unittest

import numpy as np

from differometor.build import _sideband_indices


class SidebandLayoutTest(unittest.TestCase):
    def test_mixed_stencil_preserves_duplicate_write_order_and_rhs(self):
        # Include both optical indices, each mixed case, the appended signal
        # block, duplicate writes, and the negative RHS-column convention.
        rows = np.array([1, 4, 0, 4, 1, 4])
        cols = np.array([2, 2, 4, 5, 2, -1])
        expanded_rows, expanded_cols, sources = _sideband_indices(rows, cols, 3)
        np.testing.assert_array_equal(rows, [1, 7, 0, 7, 1, 7])
        np.testing.assert_array_equal(cols, [2, 2, 7, 8, 2, -1])
        np.testing.assert_array_equal(sources, [0, 1, 2, 3, 4, 5, 2, 1, 5, 0, 4])
        np.testing.assert_array_equal(expanded_rows, [1, 7, 0, 7, 1, 7, 3, 7, 7, 4, 4])
        np.testing.assert_array_equal(expanded_cols, [2, 2, 7, 8, 2, -1, 7, 5, 2, 5, 5])
