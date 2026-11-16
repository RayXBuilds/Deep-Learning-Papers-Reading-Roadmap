#!/usr/bin/env python3
"""Scalar ODE observation-sensitivity controls; no image model or training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def forcing_integral(a, time=1.0):
    """Integral of exp(a*s), retaining the removable singularity at a=0."""
    return time if a == 0 else math.expm1(a * time) / a


def exact(a, b, initial, observation, time=1.0):
    return math.exp(a * time) * initial + b * observation * forcing_integral(a, time)


def euler(a, b, initial, observation, steps, time=1.0):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError('steps must be a positive integer')
    value = initial
    dt = time / steps
    for _ in range(steps):
        value += dt * (a * value + b * observation)
    return value


def gain(a, b, initial_from_y=False, steps=None, noise_shift=0.0):
    y, delta, noise = 0.37, 0.001, -0.2
    outputs = []
    for offset in (0, 1):
        observation = y + offset * delta
        initial = noise + offset * noise_shift + (observation if initial_from_y else 0)
        outputs.append(exact(a, b, initial, observation) if steps is None else
                       euler(a, b, initial, observation, steps))
    return abs(outputs[1] - outputs[0]) / delta


def run_experiment():
    rows = []
    for a, b, from_y in [(1, 0, True), (1, 2, False), (0, 2, False), (-1, 10, False)]:
        measured = gain(a, b, from_y)
        bound = math.exp(abs(a)) if from_y else abs(b) * forcing_integral(abs(a))
        rows.append({'a': a, 'b': b, 'initial_from_y': from_y,
                     'exact_gain': measured, 'lipschitz_bound': bound})
    target = 2 * math.expm1(1)
    convergence = [{'steps': n, 'gain': gain(1, 2, steps=n)} for n in (10, 100, 1000)]
    for row in convergence:
        row['absolute_error'] = abs(row['gain'] - target)
    wrong_noise = gain(1, 0, True, noise_shift=0.1)
    coarse = gain(-4, 0, True, steps=1)
    checks = {
        'sharp_initial_bound': abs(rows[0]['exact_gain'] - math.e) < 1e-10,
        'sharp_condition_bound': abs(rows[1]['exact_gain'] - target) < 1e-10,
        'zero_state_limit': abs(rows[2]['exact_gain'] - 2) < 1e-10,
        'contractive_state_not_condition_control': 6 < rows[3]['exact_gain'] < rows[3]['lipschitz_bound'],
        'euler_converges': convergence[0]['absolute_error'] > convergence[1]['absolute_error'] > convergence[2]['absolute_error'] and convergence[2]['absolute_error'] < 0.003,
        'fixed_noise_required': wrong_noise > 100 * rows[0]['exact_gain'],
        'coarse_solver_breaks_contraction': abs(coarse - 3) < 1e-10 and math.exp(-4) < 0.02,
    }
    return {'config': {'seed': None, 'observation': 0.37, 'delta': 0.001, 'noise': -0.2, 'time': 1},
            'cases': rows, 'euler': convergence, 'mismatched_noise_gain': wrong_noise,
            'coarse_contracting_field_gain': coarse, 'continuous_contracting_field_gain': math.exp(-4),
            'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_experiment()
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(serialized)
    print(serialized)
    raise SystemExit(0 if result['passed'] else 1)
