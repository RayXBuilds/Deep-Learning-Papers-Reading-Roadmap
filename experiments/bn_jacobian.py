#!/usr/bin/env python3
"""Single-feature BN derivatives and frozen-statistic controls; no training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def statistics(x, epsilon):
    if not x or epsilon < 0 or not all(math.isfinite(v) for v in (*x, epsilon)):
        raise ValueError('finite nonempty batch and nonnegative epsilon required')
    mean = sum(x)/len(x)
    c = [v-mean for v in x]
    variance = sum(v*v for v in c)/len(x)
    scale = math.sqrt(variance+epsilon)
    if scale == 0:
        raise ValueError('zero denominator')
    return mean, c, variance, scale


def forward(x, epsilon, gamma=1.0, beta=0.0, frozen=None):
    if frozen is None:
        mean, _, _, scale = statistics(x, epsilon)
    else:
        mean, variance = frozen
        scale = math.sqrt(variance+epsilon)
    return [gamma*(v-mean)/scale+beta for v in x]


def jacobian(x, epsilon, gamma=1.0):
    _, c, _, scale = statistics(x, epsilon)
    n = len(x)
    return [[gamma/scale*((i == j)-1/n-c[i]*c[j]/(n*scale*scale))
             for j in range(n)] for i in range(n)]


def matvec(a, v):
    return [sum(x*y for x, y in zip(row, v)) for row in a]


def error(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def finite_difference(x, epsilon, gamma, step=1e-5, frozen=None):
    n = len(x)
    result = [[0.0]*n for _ in x]
    for j in range(n):
        plus, minus = x[:], x[:]
        plus[j] += step
        minus[j] -= step
        yp = forward(plus, epsilon, gamma, frozen=frozen)
        ym = forward(minus, epsilon, gamma, frozen=frozen)
        for i in range(n):
            result[i][j] = (yp[i]-ym[i])/(2*step)
    return result


def run_experiment():
    x, gamma, beta = [-3.0, -1.0, 2.0, 6.0], 1.5, 0.3
    rows = []
    for epsilon in (0.0, 0.25):
        mean, c, variance, scale = statistics(x, epsilon)
        j = jacobian(x, epsilon, gamma)
        fd = finite_difference(x, epsilon, gamma)
        frozen_fd = finite_difference(x, epsilon, gamma, frozen=(mean, variance))
        rows.append({
            'epsilon': epsilon, 'variance': variance, 'scale': scale,
            'jacobian': j, 'finite_difference': fd,
            'fd_error': max(error(a, b) for a, b in zip(j, fd)),
            'shift_direction': matvec(j, [1]*len(x)),
            'shift_output_error': error(forward(x, epsilon, gamma, beta),
                                        forward([v+5 for v in x], epsilon, gamma, beta)),
            'radial_direction': matvec(j, c),
            'radial_reference': [gamma*epsilon*v/scale**3 for v in c],
            'off_diagonal_max': max(abs(j[i][k]) for i in range(4) for k in range(4) if i != k),
            'frozen_fd_error': max(abs(frozen_fd[i][k]-(gamma/scale if i == k else 0))
                                   for i in range(4) for k in range(4)),
            'frozen_shift_gain': gamma/scale,
            'frozen_shift_output': [b-a for a, b in zip(
                forward(x, epsilon, gamma, beta, (mean, variance)),
                forward([v+5 for v in x], epsilon, gamma, beta, (mean, variance)))],
        })
    checks = {
        'jacobian_matches_finite_difference': all(r['fd_error'] < 1e-8 for r in rows),
        'training_shift_invariant': all(max(map(abs, r['shift_direction'])) < 1e-12 and
                                       r['shift_output_error'] < 1e-12 for r in rows),
        'batch_entries_coupled': all(r['off_diagonal_max'] > 0.01 for r in rows),
        'zero_epsilon_radial_null': max(map(abs, rows[0]['radial_direction'])) < 1e-12,
        'positive_epsilon_radial_leak': error(rows[1]['radial_direction'], rows[1]['radial_reference']) < 1e-12
                                       and max(map(abs, rows[1]['radial_direction'])) > 0.01,
        'frozen_statistics_differ': all(r['frozen_fd_error'] < 1e-8 and
                                       error(r['frozen_shift_output'], [5*r['frozen_shift_gain']]*4) < 1e-12
                                       and r['frozen_shift_gain'] > 0.1 for r in rows),
    }
    return {'config': {'input': x, 'gamma': gamma, 'beta': beta, 'shift': 5,
                       'difference_step': 1e-5, 'seed': None},
            'rows': rows, 'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    r = run_experiment()
    s = json.dumps(r, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(s)
    print(s)
    raise SystemExit(0 if r['passed'] else 1)
