#!/usr/bin/env python3
"""One-dimensional Gaussian flow matching controls; standard library only."""
import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path


def moments(t, mu=2.0, sigma=0.5):
    if not all(math.isfinite(x) for x in (t, mu, sigma)) or not 0 <= t <= 1 or sigma <= 0:
        raise ValueError('finite t in [0,1], finite mean and positive sigma required')
    return t * mu, (1 - t) ** 2 + (t * sigma) ** 2, t * sigma ** 2 - (1 - t)


def velocity(t, x, mu=2.0, sigma=0.5):
    mean, variance, covariance = moments(t, mu, sigma)
    return mu + covariance / variance * (x - mean)


def exact_flow(t, z, mu=2.0, sigma=0.5):
    mean, variance, _ = moments(t, mu, sigma)
    return mean + math.sqrt(variance) * z


def losses(t, a, b, mu=2.0, sigma=0.5):
    """Exact population losses for v(x)=a*x+b and independent Gaussian endpoints."""
    mean, variance, covariance = moments(t, mu, sigma)
    mean_error = a * mean + b - mu
    cfm = mean_error ** 2 + a * a * variance + 1 + sigma ** 2 - 2 * a * covariance
    fm = mean_error ** 2 + (a - covariance / variance) ** 2 * variance
    return cfm, fm, 1 + sigma ** 2 - covariance ** 2 / variance


def euler(z, steps, reverse_clock=False, wrong_sign=False):
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError('positive integer steps required')
    x = z
    for k in range(steps):
        if reverse_clock:
            tau, dtau = 1 - k / steps, -1 / steps
            field = velocity(1 - tau, x) * (1 if wrong_sign else -1)
            x += dtau * field
        else:
            x += velocity(k / steps, x) / steps
    return x


def regression(seed, count=30000):
    rng = random.Random(seed)
    pairs = [(rng.gauss(0, 1), rng.gauss(2, 0.5)) for _ in range(count)]
    rows = []
    for t in (0.25, 0.5, 0.75):
        xs = [(1 - t) * z + t * y for z, y in pairs]
        us = [y - z for z, y in pairs]
        mx, my = sum(xs) / count, sum(us) / count
        slope = sum((x - mx) * (u - my) for x, u in zip(xs, us)) / sum((x - mx) ** 2 for x in xs)
        intercept = my - slope * mx
        mean, variance, covariance = moments(t)
        expected_slope = covariance / variance
        expected_intercept = 2 - expected_slope * mean
        rows.append({'t': t, 'slope': slope, 'intercept': intercept,
                     'expected_slope': expected_slope, 'expected_intercept': expected_intercept,
                     'max_coefficient_error': max(abs(slope - expected_slope), abs(intercept - expected_intercept))})
    return {'seed': seed, 'count': count, 'rows': rows}


def run_experiment():
    h = 1e-5
    # Positive endpoint noise checks the paper's Eq. (22)-(23), not only its limit.
    z, y, sigma_min = -1.5, 3.0, 0.1
    path = lambda t: (1 - (1 - sigma_min) * t) * z + t * y
    conditional_error = max(abs((path(t + h) - path(t - h)) / (2 * h)
                                - (y - (1 - sigma_min) * z)) for t in (0.2, 0.5, 0.8))
    latents = [-2, -1, 0, 1, 2]
    derivative_error = max(abs((exact_flow(t + h, z) - exact_flow(t - h, z)) / (2 * h)
                               - velocity(t, exact_flow(t, z)))
                           for t in (0.2, 0.5, 0.8) for z in latents)
    loss_rows = []
    for t in (0.25, 0.5, 0.75):
        for a, b in ((0, 0), (1, 0), (-1, 2), (3, -2)):
            cfm, fm, gap = losses(t, a, b)
            loss_rows.append({'t': t, 'a': a, 'b': b, 'cfm': cfm, 'fm': fm, 'expected_gap': gap})
    runs = [regression(seed) for seed in (7, 42, 2026)]
    ode = []
    for steps in (400, 800):
        outputs = [euler(z, steps) for z in latents]
        ode.append({'steps': steps, 'outputs': outputs,
                    'max_endpoint_error': max(abs(x - (2 + 0.5 * z)) for x, z in zip(outputs, latents))})
    reverse_error = max(abs(euler(z, 800) - euler(z, 800, reverse_clock=True)) for z in latents)
    wrong_endpoint = euler(0, 800, reverse_clock=True, wrong_sign=True)
    checks = {
        'conditional_path_derivative': conditional_error < 1e-9,
        'marginal_flow_derivative': derivative_error < 1e-8,
        'population_loss_gap_constant': max(abs(r['cfm'] - r['fm'] - r['expected_gap']) for r in loss_rows) < 1e-12,
        'regression_all_seeds': all(row['max_coefficient_error'] < 0.06 for run in runs for row in run['rows']),
        'euler_converges_to_exact_endpoint': ode[1]['max_endpoint_error'] < 0.01 and ode[1]['max_endpoint_error'] < 0.6 * ode[0]['max_endpoint_error'],
        'reverse_clock_same_solution': reverse_error < 1e-10,
        'wrong_sign_negative_control': abs(wrong_endpoint - 2) > 1,
    }
    return {'config': {'mu': 2, 'sigma': 0.5, 'sigma_min_for_derivative': sigma_min,
                       'sigma_min_for_gaussian_controls': 0, 'latents': latents,
                       'finite_difference_h': h, 'regression_coefficient_tolerance': 0.06},
            'conditional_derivative_error': conditional_error, 'marginal_derivative_error': derivative_error,
            'population_losses': loss_rows, 'regressions': runs, 'ode': ode,
            'reverse_clock_error': reverse_error, 'wrong_sign_endpoint': wrong_endpoint,
            'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_experiment()
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
    raise SystemExit(0 if report['passed'] else 1)
