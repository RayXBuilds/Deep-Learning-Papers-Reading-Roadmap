import math
import unittest

from additive_attention import attention, run_experiment


class AdditiveAttentionTests(unittest.TestCase):
    def test_hand_computed_nonlinear_scores(self):
        contexts, weights = attention([[0.]], [[0.], [1.]], [[1.]], [[1.]], [1.], 2)
        p = math.exp(math.tanh(1.)) / (1. + math.exp(math.tanh(1.)))
        self.assertAlmostEqual(weights[0][1], p)
        self.assertAlmostEqual(contexts[0][0], p)

    def test_zero_scorer_is_uniform_over_valid_positions(self):
        contexts, weights = attention([[7.]], [[2.], [8.], [1e6]], [[1.]], [[1.]], [0.], 2)
        self.assertEqual(weights, [[0.5, 0.5, 0.]])
        self.assertEqual(contexts, [[5.]])

    def test_single_valid_source_is_exact_copy(self):
        contexts, weights = attention([[2.], [-5.]], [[2., 3.], [99., -99.]],
                                     [[1.]], [[1., 1.]], [2.], 1)
        self.assertEqual(contexts, [[2., 3.], [2., 3.]])
        self.assertEqual(weights, [[1., 0.], [1., 0.]])

    def test_masked_annotation_intervention(self):
        base = attention([[0.]], [[1.], [2.]], [[1.]], [[1.]], [1.], 1)
        changed = attention([[0.]], [[1.], [10000.]], [[1.]], [[1.]], [1.], 1)
        self.assertEqual(base, changed)

    def test_linear_scorer_query_offset_cancels(self):
        contexts, weights = attention([[2.], [-3.]], [[-1.], [2.]],
                                     [[0.7]], [[0.4]], [1.3], 2, nonlinear=False)
        for a, b in zip(weights[0], weights[1]):
            self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(contexts[0][0], contexts[1][0])

    def test_incompatible_projection_rejected(self):
        with self.assertRaises(ValueError):
            attention([[1., 2.]], [[1.]], [[1.]], [[1.]], [1.], 1)

    def test_empty_all_masked_and_bad_lengths_rejected(self):
        for length in (0, -1, 2, True, 0.5):
            with self.subTest(length=length), self.assertRaises(ValueError):
                attention([[1.]], [[1.]], [[1.]], [[1.]], [1.], length)
        with self.assertRaises(ValueError):
            attention([[1.]], [], [[1.]], [[1.]], [1.], 0)

    def test_nonfinite_and_ragged_input_rejected(self):
        for invalid in ([[math.nan]], [[math.inf]], [[1.], [1., 2.]]):
            with self.assertRaises(ValueError):
                attention([[1.]], invalid, [[1.]], [[1.]], [1.], 1)

    def test_query_changes_context_and_controls_pass(self):
        report = run_experiment()
        self.assertTrue(report['passed'], report['checks'])
        self.assertGreater(report['metrics']['weights'][0][1], report['metrics']['weights'][1][1])


if __name__ == '__main__':
    unittest.main()
