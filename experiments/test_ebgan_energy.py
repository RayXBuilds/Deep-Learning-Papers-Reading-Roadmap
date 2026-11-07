import math
import unittest
from ebgan_energy import discriminator_loss, pulling_away, run_experiment


class EnergyTests(unittest.TestCase):
    def test_hand_loss_including_above_margin(self):
        self.assertEqual(discriminator_loss([0.75, 0.25], [0.25, 0.75], [0, 3], 2), 1.25)

    def test_equal_distribution_best_response_is_not_joint_equilibrium(self):
        energies = [0.5, 1.5]
        self.assertEqual(discriminator_loss([0.75, 0.25], [0.75, 0.25], energies, 2), 2)
        current_generator_loss = 0.75 * energies[0] + 0.25 * energies[1]
        self.assertGreater(current_generator_loss, min(energies))

    def test_zero_real_mass_allows_energy_above_margin(self):
        self.assertEqual(discriminator_loss([1, 0], [0, 1], [0, 7], 2), 0)

    def test_pt_ordered_pair_hand_value(self):
        # Two ordered pairs have cosine^2=1, four have zero. Self-pairs are excluded.
        self.assertAlmostEqual(pulling_away([[1, 0], [1, 0], [0, 1]]), 1 / 3)

    def test_pt_permutation_and_sign(self):
        expected = pulling_away([[1, 0], [1, 1], [0, 1]])
        self.assertAlmostEqual(pulling_away([[0, -1], [-1, -1], [1, 0]]), expected)

    def test_invalid_probability_energy_margin(self):
        for real, generated, energies, margin in [([], [], [], 2), ([1], [0.5], [0], 2),
                ([1], [1], [-1], 2), ([1], [1], [math.inf], 2),
                ([1], [1], [0], 0), ([1], [1], [0], math.nan),
                ([1, 0], [1], [0], 2), ([-1, 2], [0, 1], [0, 0], 2)]:
            with self.subTest(inputs=(real, generated, energies, margin)):
                with self.assertRaises(ValueError):
                    discriminator_loss(real, generated, energies, margin)

    def test_invalid_features(self):
        for features in ([], [[1]], [[], []], [[0, 0], [1, 0]],
                         [[1], [1, 0]], [[1], [math.nan]]):
            with self.subTest(features=features):
                with self.assertRaises(ValueError):
                    pulling_away(features)

    def test_all_mechanisms(self):
        report = run_experiment()
        self.assertTrue(report['passed'], report['checks'])


if __name__ == '__main__':
    unittest.main()
