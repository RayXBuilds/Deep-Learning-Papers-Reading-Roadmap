import math
import unittest

from attention_tiles import tiled
from preference_odds import optimal_policy, preference


class TestTileReduction(unittest.TestCase):
    def test_growing_maximum_and_partial_final_tile(self):
        # The last entry has four times the weight of either earlier entry.
        self.assertAlmostEqual(tiled([0, 0, math.log(4)], [1, 3, 8], 2), 6)

    def test_tied_scores_across_unequal_tiles(self):
        self.assertAlmostEqual(tiled([10000] * 3, [0, 0, 9], 2), 3)


class TestPreferenceOdds(unittest.TestCase):
    def test_equal_rewards_recover_reference(self):
        reference = [0.125, 0.875]
        for value, target in zip(optimal_policy(reference, [12, 12], 2), reference):
            self.assertAlmostEqual(value, target)
        self.assertAlmostEqual(preference(reference, reference, 0, 1, 2), 0.5)

    def test_policy_odds_double_reference_odds(self):
        policy, reference = [0.5, 0.5], [1/3, 2/3]
        self.assertAlmostEqual(preference(policy, reference, 0, 1, 2), 4/5)
        self.assertAlmostEqual(preference(policy, reference, 1, 0, 2), 1/5)
