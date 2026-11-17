#!/usr/bin/env python3
"""Principal Euler error in smooth scalar fields; not a matched-coupling FM study."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def euler(rate, initial, steps):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError('positive integer steps required')
    y = initial
    for _ in range(steps):
        y += rate * y / steps
    return y


def coefficient(rate, initial):
    return -0.5 * rate**2 * initial * math.exp(rate)


def run_experiment():
    runs = []
    for rate in (-1, 1):
        target = math.exp(rate)
        c = coefficient(rate, 1)
        rows = []
        for n in (20, 40, 80, 160):
            error = euler(rate, 1, n) - target
            rows.append({'steps': n, 'signed_error': error, 'scaled_error': n*error,
                         'remainder': error-c/n})
        wrong = -0.5*rate*math.expm1(rate)  # Drops J*g from the propagation equation.
        runs.append({'rate': rate, 'coefficient': c, 'unpropagated_coefficient': wrong, 'rows': rows})
    checks = {
        'negative_signed_error': all(r['signed_error'] < 0 for run in runs for r in run['rows']),
        'coefficient_limit': all(abs(run['rows'][-1]['scaled_error']-run['coefficient']) < 0.008 for run in runs),
        'first_order_error': all(1.9 < abs(run['rows'][-2]['signed_error']/run['rows'][-1]['signed_error']) < 2.1 for run in runs),
        'second_order_remainder': all(3.8 < abs(run['rows'][-2]['remainder']/run['rows'][-1]['remainder']) < 4.2 for run in runs),
        'propagation_negative_control': all(abs(run['rows'][-1]['scaled_error']-run['unpropagated_coefficient']) > 0.1 for run in runs),
    }
    return {'config': {'seed': None, 'initial': 1, 'time': 1}, 'runs': runs,
            'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    r = run_experiment()
    s = json.dumps(r, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(s)
    print(s)
    raise SystemExit(0 if r['passed'] else 1)
