#!/usr/bin/env python3
"""Independent centered 2D Gaussian FM: curl controls, not network training."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def combine(a, b, sa=1, sb=1):
    return [[sa*a[i][j]+sb*b[i][j] for j in range(2)] for i in range(2)]


def mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def inverse(a):
    d = a[0][0]*a[1][1]-a[0][1]*a[1][0]
    if abs(d) < 1e-14:
        raise ValueError('singular matrix')
    return [[a[1][1]/d, -a[0][1]/d], [-a[1][0]/d, a[0][0]/d]]


def jacobian(source, target, t):
    c = combine(source, target, (1-t)**2, t*t)
    b = combine(source, target, -(1-t), t)
    return mul(b, inverse(c))


def apply(a, x):
    return [sum(a[i][k]*x[k] for k in range(2)) for i in range(2)]


def circulation(j):
    # Counterclockwise unit square; midpoint is exact on each linear edge.
    corners = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    value = 0
    for p, q in zip(corners, corners[1:]):
        v = apply(j, [(p[k]+q[k])/2 for k in range(2)])
        value += sum(v[k]*(q[k]-p[k]) for k in range(2))
    return value


def distance(a, b):
    return max(abs(a[i][j]-b[i][j]) for i in range(2) for j in range(2))


def run_experiment():
    identity = [[1, 0], [0, 1]]
    anisotropic = [[1, 0], [0, 4]]
    rotated = [[2, 1], [1, 2]]
    cases = [('isotropic', identity, rotated), ('commuting', anisotropic, [[2, 0], [0, 3]]),
             ('noncommuting', anisotropic, rotated)]
    rows = []
    for name, source, target in cases:
        for t in (0.25, 0.5, 0.75):
            j = jacobian(source, target, t)
            c = combine(source, target, (1-t)**2, t*t)
            score_j = combine(identity, mul(source, inverse(c)), 1/t, -(1-t)/t)
            rows.append({'case': name, 't': t, 'source': source, 'target': target, 'jacobian': j,
                         'curl': j[1][0]-j[0][1], 'circulation': circulation(j),
                         'score_identity_error': distance(j, score_j)})
    j = rows[7]['jacobian']
    symmetric = [[(j[i][k]+j[k][i])/2 for k in range(2)] for i in range(2)]
    change = max(abs(a-b) for a, b in zip(apply(j, [1, 0]), apply(symmetric, [1, 0])))
    checks = {
        'score_identity': max(r['score_identity_error'] for r in rows) < 1e-12,
        'isotropic_and_commuting_curl_zero': all(abs(r['curl']) < 1e-12 for r in rows[:6]),
        'noncommuting_curl_nonzero': all(abs(r['curl']) > 0.1 for r in rows[6:]),
        'closed_loop_matches_curl': all(abs(r['curl']-r['circulation']) < 1e-12 for r in rows),
        'symmetrization_changes_field': change > 0.3 and abs(circulation(symmetric)) < 1e-12,
    }
    return {'config': {'seed': None, 'dimension': 2, 'coupling': 'independent centered Gaussians'},
            'rows': rows, 'symmetrization_velocity_change': change, 'checks': checks,
            'passed': all(checks.values()), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
