import unittest

from early_mechanisms import dropout_means, impulse_response


class EarlyMechanismsTest(unittest.TestCase):
    def test_unequal_mask_probabilities(self):
        # Two independent contributions. Only the two positive outcomes survive.
        self.assertEqual(dropout_means((4, -8), 2, 0.25), (1, 2.25))

    def test_keep_endpoints(self):
        self.assertEqual(dropout_means((-4,), 2, 0), (2, 2))
        self.assertEqual(dropout_means((-4,), 2, 1), (-2, 0))

    def test_separable_reference(self):
        # Coefficients of (1+x+x²)^3, independently along each axis.
        coeff = (1, 3, 6, 7, 6, 3, 1)
        expected = {(x-3, y-3): a*b for x, a in enumerate(coeff)
                    for y, b in enumerate(coeff)}
        self.assertEqual(impulse_response([1, 1, 1]), expected)

    def test_single_kernel(self):
        expected = {(x, y): 1 for x in range(-3, 4) for y in range(-3, 4)}
        self.assertEqual(impulse_response([3]), expected)
