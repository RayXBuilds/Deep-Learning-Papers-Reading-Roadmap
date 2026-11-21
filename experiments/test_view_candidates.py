import math
import unittest

from view_candidates import anchor_loss


class ViewCandidatesTest(unittest.TestCase):
    def test_identical_views_count_only_other_keys(self):
        for n in (2, 4, 8):
            vectors = [(1, 0)] * n
            for i in range(n):
                self.assertAlmostEqual(anchor_loss(vectors, i, i ^ 1, 0.1), math.log(n - 1))

    def test_joint_permutation_preserves_nonidentical_pair(self):
        vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        actual = anchor_loss(vectors, 0, 1, 0.5)
        self.assertAlmostEqual(actual, math.log(2 + math.exp(-2)))
        self.assertEqual(actual, anchor_loss(vectors[::-1], 3, 2, 0.5))

    def test_lower_temperature_penalizes_wrong_partner(self):
        vectors = [(1, 0), (-1, 0), (1, 0), (0, 1)]
        self.assertGreater(anchor_loss(vectors, 0, 1, 0.1), anchor_loss(vectors, 0, 1, 1))
