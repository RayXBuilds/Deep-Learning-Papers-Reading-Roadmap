#!/usr/bin/env python3
"""Scalar Gaussian ELBO and constant input-skip mechanisms; no model training."""
import argparse
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path


def finite(*values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('parameters must be finite')


def log_normal(x, mean, variance):
    finite(x, mean, variance)
    if variance <= 0:
        raise ValueError('variance must be positive')
    return -0.5 * (math.log(2 * math.pi * variance) + (x - mean) ** 2 / variance)


def kl_normal(mean, variance, target_mean=0.0, target_variance=1.0):
    finite(mean, variance, target_mean, target_variance)
    if min(variance, target_variance) <= 0:
        raise ValueError('variances must be positive')
    return 0.5 * (math.log(target_variance / variance)
                  + (variance + (mean - target_mean) ** 2) / target_variance - 1)


def exact_terms(x, mu, logvar, slope, noise_variance):
    """E_q log p(x|z), KL(q||prior), ELBO for p(x|z)=N(slope*z, noise)."""
    finite(x, mu, logvar, slope, noise_variance)
    if noise_variance <= 0:
        raise ValueError('noise variance must be positive')
    variance = math.exp(logvar)
    reconstruction = -0.5 * (math.log(2 * math.pi * noise_variance)
        + ((x - slope * mu) ** 2 + slope ** 2 * variance) / noise_variance)
    kl = kl_normal(mu, variance)
    return reconstruction, kl, reconstruction - kl


def sampled_terms(x, mu, logvar, slope, noise_variance, epsilon):
    """One pathwise ELBO estimate and analytic derivatives wrt mu and logvar."""
    finite(epsilon)
    _, kl, _ = exact_terms(x, mu, logvar, slope, noise_variance)
    sigma = math.exp(0.5 * logvar)
    z = mu + sigma * epsilon
    reconstruction = log_normal(x, slope * z, noise_variance)
    dz = slope * (x - slope * z) / noise_variance
    return reconstruction, reconstruction - kl, dz - mu, (
        dz * 0.5 * sigma * epsilon - 0.5 * (math.exp(logvar) - 1))


def skip_log_density(xs, decoder_means, residual_variance, alpha):
    """Normalized isotropic density of X=alpha*X+f(z)+noise at fixed z."""
    finite(alpha)
    if not 0 <= alpha < 1:
        raise ValueError('constant alpha must be in [0, 1)')
    if not xs or len(xs) != len(decoder_means):
        raise ValueError('equal nonempty coordinate vectors required')
    b = 1 - alpha
    return len(xs) * math.log(b) + sum(
        log_normal(b * x, f, residual_variance) for x, f in zip(xs, decoder_means))


def run_experiment():
    x, mu, logvar, slope, noise = 1.2, 0.4, -0.7, 1.5, 0.7
    reconstruction, kl, elbo = exact_terms(x, mu, logvar, slope, noise)
    marginal = log_normal(x, 0, slope * slope + noise)
    post_variance = noise / (slope * slope + noise)
    post_mean = slope * x / (slope * slope + noise)
    posterior_gap = kl_normal(mu, math.exp(logvar), post_mean, post_variance)
    tight_elbo = exact_terms(x, post_mean, math.log(post_variance), slope, noise)[2]
    grad_mu = slope * (x - slope * mu) / noise - mu
    grad_logvar = -0.5 * slope * slope * math.exp(logvar) / noise - 0.5 * (math.exp(logvar) - 1)
    runs = []
    for seed in (7, 42, 2026):
        rng = random.Random(seed)
        totals = [0.0] * 4
        for _ in range(100000):
            values = sampled_terms(x, mu, logvar, slope, noise, rng.gauss(0, 1))
            totals = [a + b for a, b in zip(totals, values)]
        values = [v / 100000 for v in totals]
        runs.append(dict(seed=seed, reconstruction=values[0], kl=kl, elbo=values[1],
                         grad_mu=values[2], grad_logvar=values[3]))

    # Same fixed model in residual and observation coordinates, not separate fits.
    alpha = 0.8
    b = 1 - alpha
    xs, means = [1.2, -0.8], [0.6, 0.3]
    original_logp = sum(log_normal(v, m, noise) for v, m in zip(xs, means))
    transformed_logp = skip_log_density(xs, [b * m for m in means], b * b * noise, alpha)
    original_mse = statistics.mean((v - m) ** 2 for v, m in zip(xs, means))
    skip_mse = statistics.mean((v - (b * m + alpha * v)) ** 2 for v, m in zip(xs, means))
    # Removing the Jacobian creates an unnormalized density with mass b**(-d).
    missing_jacobian_logp = transformed_logp - len(xs) * math.log(b)
    # Matched latent and observation noise yield identical generated samples.
    original_sample = 0.6 + math.sqrt(noise) * 0.3
    skip_sample = (b * 0.6 + math.sqrt(b * b * noise) * 0.3) / b

    # Independent analytic population control: data N(0,1), zero decoder.
    # Fix residual variance at 1 instead of rescaling it: generator becomes N(0,25).
    bad_variance = 1 / (b * b)
    base_nll = 0.5 * (math.log(2 * math.pi) + 1)
    bad_nll = 0.5 * (math.log(2 * math.pi * bad_variance) + 1 / bad_variance)
    checks = {
        'elbo_gap_equals_posterior_kl': abs(marginal - elbo - posterior_gap) < 1e-12,
        'exact_posterior_makes_bound_tight': abs(marginal - tight_elbo) < 1e-12,
        'mc_reconstruction_all_seeds': all(abs(r['reconstruction'] - reconstruction) < 0.03 for r in runs),
        'mc_gradients_all_seeds': all(abs(r['grad_mu'] - grad_mu) < 0.04 and abs(r['grad_logvar'] - grad_logvar) < 0.04 for r in runs),
        'skip_normalized_density_preserved': abs(original_logp - transformed_logp) < 1e-12,
        'skip_coupled_sample_preserved': abs(original_sample - skip_sample) < 1e-12,
        'skip_mse_ratio_is_b_squared': abs(skip_mse / original_mse - b * b) < 1e-12,
        'missing_jacobian_negative_control': abs(missing_jacobian_logp - original_logp) > 1,
        'fixed_residual_variance_worsens_nll': bad_nll > base_nll + 1 and b * b < 1,
    }
    return {
        'config': {'x': x, 'mu': mu, 'logvar': logvar, 'slope': slope, 'noise_variance': noise,
                   'seeds': [7, 42, 2026], 'samples_per_seed': 100000, 'alpha': alpha,
                   'training': 'none; fixed parameters, analytic reference, independent MC streams'},
        'analytic': {'reconstruction': reconstruction, 'kl': kl, 'elbo': elbo,
                     'log_marginal': marginal, 'posterior_kl': posterior_gap,
                     'grad_mu': grad_mu, 'grad_logvar': grad_logvar},
        'mc_runs': runs,
        'skip': {'original_log_density': original_logp, 'transformed_log_density': transformed_logp,
                 'original_mse': original_mse, 'skip_mse': skip_mse,
                 'mse_ratio': skip_mse / original_mse,
                 'sample_difference': abs(original_sample - skip_sample),
                 'missing_jacobian_log_density_error': missing_jacobian_logp - original_logp,
                 'fixed_residual_variance': bad_variance,
                 'base_population_nll': base_nll, 'fixed_residual_population_nll': bad_nll},
        'checks': checks, 'passed': all(checks.values()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'platform': platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_experiment()
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
