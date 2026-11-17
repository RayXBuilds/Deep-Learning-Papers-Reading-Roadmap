import math
import unittest
from euler_error import coefficient, euler


class EulerErrorTests(unittest.TestCase):
    def test_two_steps_hand_calculation(self):
        self.assertEqual(euler(1, 2, 2), 4.5)
        self.assertEqual(euler(-1, 2, 2), 0.5)

    def test_constant_and_zero_solutions(self):
        self.assertEqual(euler(0, 7, 20), 7)
        self.assertEqual(euler(2, 0, 20), 0)
        self.assertEqual(coefficient(0, 7), 0)

    def test_scaling_initial_condition_changes_error_sign(self):
        for initial in (-2, 0.5):
            self.assertAlmostEqual(euler(1, initial, 100), initial*euler(1, 1, 100))
            self.assertAlmostEqual(coefficient(1, initial), -initial*math.e/2)

    def test_propagated_g_solves_variation_equation(self):
        # g(t)=r^2*t*exp(r*t); a(t)=r^2*exp(r*t), J=r.
        for r in (-1, 1):
            t, h = 0.3, 1e-6
            g = lambda s: r*r*s*math.exp(r*s)
            derivative = (g(t+h)-g(t-h))/(2*h)
            self.assertAlmostEqual(derivative, r*g(t)+r*r*math.exp(r*t), places=8)

    def test_invalid_step_count(self):
        for n in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                euler(1, 1, n)


if __name__ == '__main__':
    unittest.main()
