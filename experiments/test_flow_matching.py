import math
import unittest
from flow_matching import moments, velocity, exact_flow, losses, euler


class FlowMatchingTests(unittest.TestCase):
    def test_endpoint_distributions(self):
        self.assertEqual(moments(0)[:2], (0, 1))
        self.assertEqual(moments(1)[:2], (2, 0.25))
        self.assertEqual(exact_flow(1, -2), 1)

    def test_equal_variance_midpoint_constant_velocity(self):
        for x in (-100, 0, 100):
            self.assertEqual(velocity(0.5, x, mu=3, sigma=1), 3)

    def test_hand_velocity(self):
        # At t=1/2, S=5/16, Cov(U,X)=-3/8, mean=1.
        self.assertAlmostEqual(velocity(0.5, 2), 0.8)
        self.assertAlmostEqual(velocity(0.5, 0), 3.2)

    def test_loss_with_zero_predictor(self):
        cfm, fm, gap = losses(0.5, 0, 0)
        self.assertAlmostEqual(cfm, 5.25)
        self.assertAlmostEqual(fm, 4.45)
        self.assertAlmostEqual(gap, 0.8)

    def test_conditional_straight_does_not_make_marginal_linear(self):
        # Identical endpoint variances still contract at intermediate time.
        self.assertAlmostEqual(exact_flow(0.5, 1, mu=0, sigma=1), math.sqrt(0.5))
        self.assertNotAlmostEqual(exact_flow(0.5, 1, mu=0, sigma=1), 1)

    def test_euler_first_step_and_clock(self):
        self.assertEqual(euler(1, 1), 2)
        self.assertEqual(euler(1, 1, reverse_clock=True), 2)
        self.assertEqual(euler(1, 1, reverse_clock=True, wrong_sign=True), 0)

    def test_invalid_parameters(self):
        for t, sigma in ((-0.1, 1), (1.1, 1), (0.5, 0), (math.nan, 1)):
            with self.assertRaises(ValueError):
                moments(t, sigma=sigma)
        with self.assertRaises(ValueError):
            euler(0, 0)


if __name__ == '__main__':
    unittest.main()
