import unittest
from residual_gate import forward


class ResidualGateTests(unittest.TestCase):
    def test_two_steps_hand_reference(self):
        self.assertEqual(forward(2, 0.5, 1, 2), (7, 2.25))

    def test_gate_blocks_positive_input(self):
        self.assertEqual(forward(1, -2, 0, 1), (0, 0))
        self.assertEqual(forward(1, -2, 0, 1, False), (-1, -1))

    def test_constant_branch_cancellation_away_from_kink(self):
        for x in (-3, 1, 7):
            self.assertEqual(forward(x, -1, 0.5, 1), (0.5, 0))

    def test_kink_convention_is_not_classical_derivative(self):
        self.assertEqual(forward(0, 0, 0, 1)[1], 0)
        h = 1e-6
        central = (forward(h, 0, 0, 1)[0]-forward(-h, 0, 0, 1)[0])/(2*h)
        self.assertEqual(central, 0.5)

    def test_zero_depth_and_invalid_depth(self):
        self.assertEqual(forward(-2, 4, 3, 0), (-2, 1))
        for depth in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                forward(1, 0, 0, depth)


if __name__ == '__main__':
    unittest.main()
