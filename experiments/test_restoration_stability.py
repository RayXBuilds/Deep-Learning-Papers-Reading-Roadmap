import math
import unittest
from restoration_stability import exact, euler, forcing_integral, gain


class RestorationStabilityTests(unittest.TestCase):
    def test_constant_forcing_including_zero_time(self):
        self.assertEqual(exact(0, 3, 2, 4, 0.5), 8)
        self.assertEqual(exact(2, 3, 7, 4, 0), 7)
        self.assertAlmostEqual(forcing_integral(1e-12), 1, places=10)

    def test_initial_and_condition_sensitivity_are_distinct(self):
        self.assertAlmostEqual(gain(0, 0, True), 1)
        self.assertEqual(gain(2, 0, False), 0)
        self.assertAlmostEqual(gain(0, -3, False), 3)

    def test_closed_solution_satisfies_ode_by_finite_difference(self):
        for a in (-2, 0, 1):
            t, h = 0.4, 1e-6
            derivative = (exact(a, 3, -1, 2, t+h) - exact(a, 3, -1, 2, t-h)) / (2*h)
            self.assertAlmostEqual(derivative, a*exact(a, 3, -1, 2, t)+6, places=7)

    def test_euler_two_steps_hand_reference(self):
        # x0=1, f=x+2, h=.5 -> x1=2.5 -> x2=4.75.
        self.assertEqual(euler(1, 2, 1, 1, 2), 4.75)
        self.assertEqual(euler(-4, 0, 1, 0, 1), -3)

    def test_condition_forcing_survives_state_contraction(self):
        self.assertAlmostEqual(gain(-1, 10), 10*(1-1/math.e))
        self.assertGreater(gain(-1, 10), 1)
        self.assertLess(gain(-1, 0, True), 1)

    def test_nonpositive_and_noninteger_steps_rejected(self):
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                euler(1, 2, 0, 1, steps)


if __name__ == '__main__':
    unittest.main()
