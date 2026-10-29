import math
import unittest

from vae_objectives import exact_terms, kl_normal, log_normal, run_experiment, sampled_terms, skip_log_density


class VAEObjectivesTests(unittest.TestCase):
    def test_standard_normal_kl_zero(self):
        self.assertEqual(kl_normal(0, 1), 0)
        self.assertAlmostEqual(kl_normal(2, 1), 2)

    def test_independent_decoder_prior_is_exact_posterior(self):
        # slope=0 makes x independent of z; q=prior must give the exact evidence.
        recon, kl, bound = exact_terms(2, 0, 0, 0, 3)
        self.assertEqual(kl, 0)
        self.assertAlmostEqual(bound, log_normal(2, 0, 3))
        self.assertEqual(recon, bound)

    def test_pathwise_derivatives_by_fixed_noise_finite_difference(self):
        x, mu, lv, slope, variance, eps = 1.2, 0.4, -0.7, 1.5, 0.7, -0.9
        sample = sampled_terms(x, mu, lv, slope, variance, eps)
        h = 1e-6
        fd_mu = (sampled_terms(x, mu+h, lv, slope, variance, eps)[1]
                 - sampled_terms(x, mu-h, lv, slope, variance, eps)[1]) / (2*h)
        fd_lv = (sampled_terms(x, mu, lv+h, slope, variance, eps)[1]
                 - sampled_terms(x, mu, lv-h, slope, variance, eps)[1]) / (2*h)
        self.assertAlmostEqual(sample[2], fd_mu, places=8)
        self.assertAlmostEqual(sample[3], fd_lv, places=8)

    def test_zero_skip_is_identity(self):
        self.assertAlmostEqual(skip_log_density([2], [1], 3, 0), log_normal(2, 1, 3))

    def test_density_integrates_to_one_and_missing_jacobian_does_not(self):
        # Composite Simpson quadrature, independent of the algebraic logpdf check.
        n, low, high = 4000, -10, 10
        h = (high-low)/n
        total = 0
        for i in range(n+1):
            x = low+i*h
            weight = 1 if i in (0, n) else (4 if i % 2 else 2)
            total += weight * math.exp(skip_log_density([x], [0.2], 0.04, 0.8))
        mass = total*h/3
        self.assertAlmostEqual(mass, 1, places=10)
        self.assertAlmostEqual(mass/0.2, 5, places=9)

    def test_two_dimensional_jacobian(self):
        expected = log_normal(1, 0, 1) + log_normal(-2, 1, 1)
        self.assertAlmostEqual(skip_log_density([1, -2], [0, 0.2], 0.04, 0.8), expected)

    def test_invalid_inputs_rejected(self):
        for alpha in (-0.1, 1, float('nan')):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                skip_log_density([0], [0], 1, alpha)
        for xs, means in (([], []), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                skip_log_density(xs, means, 1, 0.8)
        for variance in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                log_normal(0, 0, variance)

    def test_all_three_seeds_against_analytic_reference(self):
        report = run_experiment()
        self.assertEqual([r['seed'] for r in report['mc_runs']], [7, 42, 2026])
        self.assertTrue(report['passed'], report['checks'])


if __name__ == '__main__':
    unittest.main()
