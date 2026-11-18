import unittest
from potential_gaussian import apply, circulation, inverse, jacobian


class PotentialGaussianTests(unittest.TestCase):
    def test_inverse_hand_reference(self):
        self.assertEqual(inverse([[2, 1], [1, 1]]), [[1, -1], [-1, 2]])
        with self.assertRaises(ValueError):
            inverse([[1, 1], [1, 1]])

    def test_midpoint_noncommuting_hand_reference(self):
        j = jacobian([[1, 0], [0, 4]], [[2, 1], [1, 2]], 0.5)
        expected = [[10/17, 4/17], [16/17, -14/17]]
        for i in range(2):
            for k in range(2):
                self.assertAlmostEqual(j[i][k], expected[i][k])
        self.assertAlmostEqual(circulation(j), 12/17)

    def test_pure_rotation_and_gradient_loop(self):
        self.assertEqual(circulation([[0, -1], [1, 0]]), 2)
        self.assertEqual(circulation([[2, 3], [3, -1]]), 0)

    def test_equal_covariances_midpoint_zero(self):
        a = [[2, 1], [1, 3]]
        self.assertEqual(jacobian(a, a, 0.5), [[0, 0], [0, 0]])

    def test_velocity_derivative_matches_jacobian(self):
        j = jacobian([[1, 0], [0, 4]], [[2, 1], [1, 2]], 0.25)
        x, h = [0.4, -0.7], 1e-6
        for k in range(2):
            plus, minus = x[:], x[:]
            plus[k] += h
            minus[k] -= h
            for i, (p, m) in enumerate(zip(apply(j, plus), apply(j, minus))):
                self.assertAlmostEqual((p-m)/(2*h), j[i][k], places=8)


if __name__ == '__main__':
    unittest.main()
