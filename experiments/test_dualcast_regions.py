import unittest
from dualcast_regions import region_weights, weighted_loss, run_experiment


class RegionTests(unittest.TestCase):
    def test_overlap_and_dropout(self):
        self.assertEqual(region_weights([1, 1], [1, 0]), [3, 2])
        self.assertEqual(region_weights([0, 0], [1, 0]), [2, 0.5])

    def test_channel_denominator(self):
        self.assertAlmostEqual(weighted_loss([[1], [3]], [2], epsilon=1), 4)

    def test_zero_weights(self):
        self.assertEqual(weighted_loss([[8, 9]], [0, 0]), 0)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            region_weights([0.2], [1])
        with self.assertRaises(ValueError):
            weighted_loss([[1, 2]], [1])
        with self.assertRaises(ValueError):
            weighted_loss([[1]], [-1])

    def test_mechanisms(self):
        self.assertTrue(run_experiment()['passed'])


if __name__ == '__main__':
    unittest.main()
