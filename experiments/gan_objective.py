#!/usr/bin/env python3
"""Finite probability tables and scalar GAN gradients, using only the CPU."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def validate(distribution):
    if not distribution or any(not math.isfinite(x) or x < 0 for x in distribution):
        raise ValueError('finite nonnegative probabilities required')
    if abs(sum(distribution) - 1) > 1e-12:
        raise ValueError('probabilities must sum to one')


def compare(real, generated):
    validate(real)
    validate(generated)
    if len(real) != len(generated):
        raise ValueError('matching support required')
    optimum = [p / (p + q) if p + q else 0.5 for p, q in zip(real, generated)]
    value = sum(p * math.log(d) if p else 0 for p, d in zip(real, optimum))
    value += sum(q * math.log1p(-d) if q else 0 for q, d in zip(generated, optimum))
    mixture = [(p + q) / 2 for p, q in zip(real, generated)]
    js = sum(p * math.log(p / m) if p else 0 for p, m in zip(real, mixture)) / 2
    js += sum(q * math.log(q / m) if q else 0 for q, m in zip(generated, mixture)) / 2
    return {'discriminator': optimum, 'value': value, 'js': js}


def softplus(value):
    return max(value, 0) + math.log1p(math.exp(-abs(value)))


def losses(logit):
    return -softplus(logit), softplus(-logit)


def run_experiment():
    cases = {
        'matched': compare([0.75, 0.25], [0.75, 0.25]),
        'mismatched': compare([0.75, 0.25], [0.25, 0.75]),
        'disjoint': compare([1, 0], [0, 1]),
    }
    gradients = []
    step = 1e-5
    for logit in [-8, 0, 8]:
        probability = 1 / (1 + math.exp(-logit))
        analytic = [-probability, probability - 1]
        plus, minus = losses(logit + step), losses(logit - step)
        numeric = [(p - m) / (2 * step) for p, m in zip(plus, minus)]
        gradients.append({'logit': logit, 'analytic': analytic, 'finite_difference': numeric})
    checks = {
        'js_identity': all(abs(r['value'] + math.log(4) - 2 * r['js']) < 1e-12 for r in cases.values()),
        'matched_equilibrium': abs(cases['matched']['value'] + math.log(4)) < 1e-12,
        'disjoint_extreme': abs(cases['disjoint']['js'] - math.log(2)) < 1e-12,
        'optimal_beats_constant': cases['mismatched']['value'] > -math.log(4) + 0.1,
        'gradient_finite_difference': all(abs(a - b) < 1e-8 for row in gradients for a, b in zip(row['analytic'], row['finite_difference'])),
        'non_saturating_signal': abs(gradients[0]['analytic'][0]) < 0.001 and abs(gradients[0]['analytic'][1]) > 0.999,
    }
    return {'config': {'seed': None, 'randomness': 'none', 'difference_step': step},
            'cases': cases, 'gradients': gradients, 'checks': checks, 'passed': all(checks.values()),
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
