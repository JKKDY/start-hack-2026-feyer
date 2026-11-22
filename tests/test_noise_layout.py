import unittest

import numpy as np

from differometor.sparse_signal import pack_noise_template


class NoiseLayoutTest(unittest.TestCase):
    def test_original_write_order_with_aliases_and_existing_values(self):
        template = np.diag(np.array([1j, 2, 3]))
        updates = np.array([[0, 0, 0, 2], [-1, 2, -1, 1]])
        layout = pack_noise_template(template, updates)
        np.testing.assert_array_equal(layout.rows, [0, 0, 0, 1, 2, 2])
        np.testing.assert_array_equal(layout.columns, [-1, 0, 2, 1, 1, 2])
        np.testing.assert_array_equal(layout.update_slots, [0, 2, 0, 4])
        np.testing.assert_array_equal(layout.template, [0, 1j, 0, 2, 0, 3])

    def test_empty_covariance(self):
        layout = pack_noise_template(np.zeros((2, 2)), np.empty((2, 0), dtype=int))
        self.assertTrue(all(a.shape == (0,) for a in layout))
