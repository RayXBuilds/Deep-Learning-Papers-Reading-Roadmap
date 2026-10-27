"""Semantic checks against independent, hand-computable expectations."""
import math
import unittest

import run


class MechanismTests(unittest.TestCase):
    def test_normalization_known_axes(self):
        x = [[1., 2.], [3., 6.]]
        bn = run.batch_norm(x)
        ln = run.layer_norm(x)
        self.assertAlmostEqual(bn[0][0], -1 / math.sqrt(1 + 1e-5))
        self.assertAlmostEqual(bn[0][1], -2 / math.sqrt(4 + 1e-5))
        self.assertAlmostEqual(ln[0][0], -0.5 / math.sqrt(0.25 + 1e-5))
        self.assertAlmostEqual(ln[1][1], 1.5 / math.sqrt(2.25 + 1e-5))

    def test_normalization_batch_intervention(self):
        x = [[1., 2., 4.], [3., 5., 8.], [9., 7., 6.]]
        changed = [x[0], [100., -200., 0.], x[2]]
        self.assertEqual(run.layer_norm(x)[0], run.layer_norm(changed)[0])
        self.assertNotEqual(run.batch_norm(x)[0], run.batch_norm(changed)[0])

    def test_constant_normalization_is_finite_zero(self):
        self.assertEqual(run.layer_norm([[5., 5.]]), [[0., 0.]])
        self.assertEqual(run.batch_norm([[5., 2.]]), [[0., 0.]])

    def test_matrix_shape_validation(self):
        for invalid in ([], [[]], [[1., 2.], [3.]], [[math.nan]]):
            with self.assertRaises(ValueError):
                run.layer_norm(invalid)

    def test_zero_residual_branch_is_identity(self):
        self.assertEqual(run.residual_forward(2., 30, 0., True), (2., 1.))
        self.assertEqual(run.residual_forward(2., 30, 0., False), (0., 0.))

    def test_residual_gradient_against_finite_difference(self):
        for skip in (True, False):
            x, h = -0.3, 1e-5
            actual = run.residual_forward(x, 4, 0.2, skip)[1]
            difference = (run.residual_forward(x + h, 4, 0.2, skip)[0]
                          - run.residual_forward(x - h, 4, 0.2, skip)[0]) / (2 * h)
            self.assertAlmostEqual(actual, difference, places=8)

    def test_adam_first_step_bias_correction(self):
        theta, _, _ = run.adam_step([1., -2.], [2., -4.], [0., 0.], [0., 0.], 1)
        self.assertAlmostEqual(theta[0], 1. - 0.1 * 2 / (2 + 1e-8))
        self.assertAlmostEqual(theta[1], -2. + 0.1 * 4 / (4 + 1e-8))
        uncorrected, _, _ = run.adam_step([1.], [2.], [0.], [0.], 1, bias_correction=False)
        self.assertGreater(abs(uncorrected[0] - theta[0]), 0.2)

    def test_adam_zero_gradient_does_not_move(self):
        theta, _, _ = run.adam_step([3.], [0.], [0.], [0.], 1)
        self.assertEqual(theta, [3.])

    def test_attention_uniform_hand_computed(self):
        q = [[0., 0.], [0., 0.], [0., 0.]]
        k = [[1., 2.], [3., 4.], [5., 6.]]
        v = [[2.], [4.], [9.]]
        full, _ = run.attention(q, k, v)
        causal, _ = run.attention(q, k, v, causal=True)
        self.assertAlmostEqual(full[0][0], 5.)
        for actual, expected in zip(causal, [2., 3., 5.]):
            self.assertAlmostEqual(actual[0], expected)

    def test_attention_future_values_do_not_leak(self):
        q = k = [[1., 0.], [0., 1.], [1., 1.]]
        v = [[1.], [2.], [3.]]
        before, _ = run.attention(q, k, v, causal=True)
        after, _ = run.attention(q, k, [[1.], [2.], [10000.]], causal=True)
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[2], after[2])

    def test_attention_scaling_and_stability(self):
        # q.k/sqrt(4)=1; softmax([1,0]) has logistic(1) as first weight.
        _, weights = run.attention([[1., 1., 1., 1.]],
                                   [[0.5, 0.5, 0.5, 0.5], [0., 0., 0., 0.]], [[1.], [0.]])
        self.assertAlmostEqual(weights[0][0], math.exp(1) / (math.exp(1) + 1))
        self.assertEqual(run.softmax([10000., 10000.]), [0.5, 0.5])

    def test_attention_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            run.attention([[1., 2.]], [[1.]], [[1.]])

    def test_forward_noise_hand_computed(self):
        self.assertAlmostEqual(run.forward_noise(2., -1., 0.25), 1 - math.sqrt(0.75))
        with self.assertRaises(ValueError):
            run.forward_noise(2., 1., 1.5)

    def test_line_fit_known_solution(self):
        slope, intercept = run.fit_line([(-1., -1.), (0., 1.), (1., 3.), (2., 5.)])
        self.assertAlmostEqual(slope, 2.)
        self.assertAlmostEqual(intercept, 1.)

    def test_diffusion_multiple_seeds(self):
        # Stochastic tolerances are pedagogical acceptance limits, not significance tests.
        for seed in (7, 42, 2026):
            with self.subTest(seed=seed):
                result = run.diffusion_experiment(seed)
                self.assertTrue(all(result['checks'].values()), result['checks'])

    def test_reproducible_metrics(self):
        self.assertEqual(run.diffusion_experiment(7), run.diffusion_experiment(7))


if __name__ == '__main__':
    unittest.main()
