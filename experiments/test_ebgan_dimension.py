import unittest
from ebgan_dimension import run_experiment
from ebgan_energy import pulling_away


class DimensionTests(unittest.TestCase):
    def test_three_vectors_hand_value(self):
        report = run_experiment()
        self.assertAlmostEqual(report['cases']['three_equiangular']['pt'], 0.25)
        self.assertTrue(report['passed'], report['checks'])

    def test_repeated_axes_in_three_dimensions(self):
        features = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
        # Three identical pairs out of 15 unordered pairs, matching (6-3)/(3*5).
        self.assertAlmostEqual(pulling_away(features), 0.2)


if __name__ == '__main__':
    unittest.main()
