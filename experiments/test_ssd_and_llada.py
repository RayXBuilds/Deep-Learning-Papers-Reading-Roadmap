import unittest

from llada_weight import expected_loss
from ssd_duality import dense, scan


class SSDTests(unittest.TestCase):
    def test_signed_kernel_with_reset(self):
        args = ([0.2, 0.5, 0], [[1, 2], [0, -1], [1, 0]],
                [[0, 1], [1, -1], [-2, 3]], [2, 3, 1])
        # States after each write: (2,4), (1,-1), (1,0).
        self.assertEqual(scan(*args), [4, 2, -2])
        self.assertEqual(dense(*args), [4, 2, -2])

    def test_future_input_does_not_change_prefix(self):
        args = ([1, 0.5], [[1], [2]], [[-1], [1]])
        self.assertEqual(dense(*args, [3, 4]), [-3, 9.5])
        self.assertEqual(dense(*args, [3, -4]), [-3, -6.5])


class MaskTests(unittest.TestCase):
    def test_one_site(self):
        self.assertEqual(expected_loss([8], 0.25), 8)
        self.assertEqual(expected_loss([8], 0.25, False), 2)

    def test_full_mask(self):
        self.assertEqual(expected_loss([1, 3], 1), 4)
        # At t=1/2 the four equally likely masked losses are 0, 1, 3, 6.
        self.assertEqual(expected_loss([1, 3], 0.5, neighbor_penalty=2), 5)

    def test_zero_probability_excluded(self):
        with self.assertRaises(ValueError):
            expected_loss([2], 0)
