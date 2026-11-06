#!/usr/bin/env python3
"""Equal-weight one-dimensional transport and constrained linear critics."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from gan_objective import compare


def transport(real, generated):
    if not real or len(real) != len(generated):
        raise ValueError('equal nonempty empirical sample sizes required')
    if any(not math.isfinite(x) for x in list(real) + list(generated)):
        raise ValueError('finite coordinates required')
    return sum(abs(x - y) for x, y in zip(sorted(real), sorted(generated))) / len(real)


def empirical_js(real, generated):
    transport(real, generated)  # Validate the supported equal-weight inputs.
    support = sorted(set(real) | set(generated))
    return compare([real.count(x) / len(real) for x in support],
                   [generated.count(x) / len(generated) for x in support])['js']


def run_experiment():
    shifts = [-1, -0.1, -0.001, 0, 0.001, 0.1, 1]
    rows = []
    for shift in shifts:
        slope = -1 if shift > 0 else 1 if shift < 0 else 0
        rows.append({'shift': shift, 'w1': transport([0], [shift]),
                     'js': empirical_js([0], [shift]), 'critic_gap': -slope * shift})
    real, generated = [0, 2], [0.3, 2.3]
    value = transport(real, generated)
    # A linear critic only observes the mean, so equal means supply a negative control.
    equal_real, equal_generated = [-1, 1], [0, 0]
    equal_mean_w1 = transport(equal_real, equal_generated)
    linear_gap = abs(sum(equal_real) / 2 - sum(equal_generated) / 2)
    scaled_gap = 10 * abs(sum(real) / 2 - sum(generated) / 2)
    step = 1e-5
    derivative = (transport([0], [0.3 + step]) - transport([0], [0.3 - step])) / (2 * step)
    checks = {
        'point_mass_primal_dual_agree': all(abs(r['w1'] - r['critic_gap']) < 1e-12 for r in rows),
        'disjoint_js_constant': all(abs(r['js'] - math.log(2)) < 1e-12 for r in rows if r['shift']),
        'coincident_js_zero': rows[3]['js'] == 0,
        'translated_pair_cost': abs(value - 0.3) < 1e-12,
        'positive_shift_gradient': abs(derivative - 1) < 1e-8,
        'limited_critic_misses_equal_means': equal_mean_w1 == 1 and linear_gap == 0,
        'unconstrained_scale_exceeds_w1': scaled_gap > value + 1,
    }
    return {'config': {'seed': None, 'randomness': 'none', 'difference_step': step},
            'shifts': rows, 'translated_pair_w1': value, 'gradient': derivative,
            'equal_mean_w1': equal_mean_w1, 'equal_mean_linear_gap': linear_gap,
            'slope_ten_gap': scaled_gap, 'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependency_sha256': hashlib.sha256(Path(__file__).with_name('gan_objective.py').read_bytes()).hexdigest(),
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
