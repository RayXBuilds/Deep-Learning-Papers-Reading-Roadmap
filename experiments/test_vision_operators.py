import unittest

from vision_operators import dense, roi_level, separated


class OperatorTests(unittest.TestCase):
    def test_hand_mixed_channels(self):
        # Channel filters return 3 and -2. Mix with two different signed rows.
        patch = [[1, 2], [3, 4]]
        self.assertEqual(separated(patch, [[-1, 2], [2, -2]], [[1, 2], [-2, 1]]),
                         [-1, -8])

    def test_dense_can_preserve_two_independent_taps(self):
        identity = [[[1, 0]], [[0, 1]]]
        responses = [dense([probe], identity) for probe in ([1, 0], [0, 1])]
        (a, b), (c, d) = responses
        self.assertEqual(a * d - b * c, 1)
        # Any two-step single-input filter instead yields collinear impulse responses.
        (a, b), (c, d) = [separated([probe], [[3, 5]], [[2], [-1]])
                           for probe in ([1, 0], [0, 1])]
        self.assertEqual(a * d - b * c, 0)

    def test_power_of_two_thresholds(self):
        for below, at, expected in [(111, 112, 3), (223, 224, 4), (447, 448, 5)]:
            self.assertEqual(roi_level(below, below), expected - 1)
            self.assertEqual(roi_level(at, at), expected)

    def test_fixed_area_and_clipping(self):
        self.assertEqual(roi_level(14, 896), 3)
        self.assertEqual(roi_level(896, 14), 3)
        self.assertEqual(roi_level(1, 1), 2)
        self.assertEqual(roi_level(4096, 4096), 5)
        with self.assertRaises(ValueError):
            roi_level(0, 56)
