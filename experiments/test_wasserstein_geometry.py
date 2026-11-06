import itertools
import math
import unittest
from wasserstein_geometry import transport, empirical_js, run_experiment


class TransportTests(unittest.TestCase):
    def test_against_all_assignments(self):
        real, generated = [3, -1, 1], [0, 4, -2]
        exhaustive = min(sum(abs(x-y) for x,y in zip(real,p))/3 for p in itertools.permutations(generated))
        self.assertAlmostEqual(transport(real, generated), exhaustive)

    def test_hand_value_and_permutation(self):
        self.assertEqual(transport([0, 4], [1, 2]), 1.5)
        self.assertEqual(transport([4, 0], [2, 1]), 1.5)

    def test_duplicate_mass_js(self):
        self.assertAlmostEqual(empirical_js([0, 0], [0, 1]), 0.75*math.log(4/3))

    def test_symmetry_translation(self):
        self.assertEqual(transport([-1, 2], [0, 4]), transport([0, 4], [-1, 2]))
        self.assertEqual(transport([-1, 2], [0, 4]), transport([9, 12], [10, 14]))

    def test_invalid_inputs(self):
        for real, generated in [([], []), ([0], [0, 1]), ([math.inf], [0])]:
            with self.assertRaises(ValueError):
                transport(real, generated)

    def test_mechanisms(self):
        report = run_experiment()
        self.assertTrue(report['passed'], report['checks'])


if __name__ == '__main__':
    unittest.main()
