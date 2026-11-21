import itertools
import unittest

from masked_patches import masked_mse, restore


class MaskedPatchesTest(unittest.TestCase):
    def test_every_four_patch_layout(self):
        tokens = ["sky", "sea", "sand", "boat"]
        for order in itertools.permutations(range(4)):
            for count in range(5):
                keep = order[:count]
                features = [tokens[i] for i in keep]
                expected = [token if i in keep else None for i, token in enumerate(tokens)]
                self.assertEqual(restore(features, order), expected)

    def test_only_masked_residuals_matter(self):
        target = [11, 12, 13]
        mask = [True, False, True]
        for middle in (-100, 12, 500):
            self.assertEqual(masked_mse([9, middle, 16], target, mask), 6.5)

    def test_empty_mask_and_invalid_order(self):
        with self.assertRaises(ValueError):
            masked_mse([1], [1], [False])
        with self.assertRaises(ValueError):
            restore([1], [0, 0])
