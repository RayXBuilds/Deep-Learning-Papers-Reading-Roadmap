import unittest
from bn_jacobian import forward, jacobian, finite_difference, matvec, statistics


class BatchNormJacobianTests(unittest.TestCase):
    def test_two_element_hand_reference(self):
        self.assertEqual(forward([-1, 1], 3), [-0.5, 0.5])
        self.assertEqual(jacobian([-1, 1], 3), [[0.1875, -0.1875], [-0.1875, 0.1875]])

    def test_constant_batch_has_nonzero_transverse_derivative(self):
        j = jacobian([4, 4, 4], 0.25)
        self.assertEqual(forward([4, 4, 4], 0.25, beta=3), [3, 3, 3])
        self.assertAlmostEqual(j[0][0], 4/3)
        self.assertAlmostEqual(j[0][1], -2/3)
        for v in matvec(j, [1, 1, 1]):
            self.assertAlmostEqual(v, 0)

    def test_singleton_is_algebraic_degeneracy(self):
        self.assertEqual(forward([3], 0.1, beta=2), [2])
        self.assertEqual(jacobian([3], 0.1), [[0]])

    def test_signed_affine_parameters_and_multiple_batches(self):
        for x in ([-2.0, 0.0, 3.0], [1.0, 1.0, 1.0], [2.0, 7.0]):
            for gamma in (-2, 0, 1.5):
                a, b = jacobian(x, 0.5, gamma), finite_difference(x, 0.5, gamma)
                for row, ref in zip(a, b):
                    for v, w in zip(row, ref):
                        self.assertAlmostEqual(v, w, places=8)

    def test_frozen_negative_control_for_shared_bias(self):
        # Sum of outputs: true training derivative in shared-bias direction is zero.
        x = [-1.0, 1.0]
        h = 1e-5
        true = (sum(forward([v+h for v in x], 3))-sum(forward([v-h for v in x], 3)))/(2*h)
        frozen = (sum(forward([v+h for v in x], 3, frozen=(0, 1)))-
                  sum(forward([v-h for v in x], 3, frozen=(0, 1))))/(2*h)
        self.assertAlmostEqual(true, 0)
        self.assertAlmostEqual(frozen, 1)

    def test_invalid_and_zero_denominator(self):
        for x, eps in (([], 1), ([1, 1], 0), ([1, 2], -1), ([float('nan')], 1)):
            with self.assertRaises(ValueError):
                statistics(x, eps)


if __name__ == '__main__':
    unittest.main()
