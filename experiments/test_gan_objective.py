import math
import unittest
from gan_objective import compare, losses, run_experiment


class GANObjectiveTests(unittest.TestCase):
    def test_hand_discriminator(self):
        result = compare([0.75, 0.25], [0.25, 0.75])
        self.assertEqual(result['discriminator'], [0.75, 0.25])
        self.assertAlmostEqual(result['value'], 1.5 * math.log(0.75) + 0.5 * math.log(0.25))

    def test_zero_mass_and_disjoint(self):
        result = compare([1, 0, 0], [0, 1, 0])
        self.assertEqual(result['value'], 0)
        self.assertAlmostEqual(result['js'], math.log(2))

    def test_symmetry(self):
        self.assertAlmostEqual(compare([0.2, 0.8], [0.6, 0.4])['js'], compare([0.6, 0.4], [0.2, 0.8])['js'])

    def test_scalar_losses(self):
        self.assertEqual(losses(0), (-math.log(2), math.log(2)))
        self.assertTrue(all(math.isfinite(v) for v in losses(-1000) + losses(1000)))

    def test_invalid_distributions(self):
        for probabilities in [[-1, 2], [0.2, 0.2], [], [float('nan'), 1]]:
            with self.assertRaises(ValueError):
                compare(probabilities, [0.5, 0.5])

    def test_mechanisms(self):
        result = run_experiment()
        self.assertTrue(result['passed'], result['checks'])


if __name__ == '__main__':
    unittest.main()
