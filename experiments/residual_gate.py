#!/usr/bin/env python3
"""Affine branch and post-addition ReLU controls; no ResNet training."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def forward(x, slope, bias, depth, gated=True):
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError('nonnegative integer depth required')
    derivative = 1.0
    for _ in range(depth):
        z = x + slope*x + bias
        derivative *= (1+slope) * (float(z > 0) if gated else 1)
        x = max(0.0, z) if gated else z
    return x, derivative


def run_experiment():
    cases = [('identity', 0, 0, 20), ('blocked', -2, 0, 1),
             ('cancelled', -1, 0.5, 1), ('exploding', 1, 0, 20),
             ('contracting', -0.5, 0, 20)]
    rows = []
    for name, a, b, depth in cases:
        value, gradient = forward(1, a, b, depth)
        h = 1e-6
        fd = (forward(1+h, a, b, depth)[0]-forward(1-h, a, b, depth)[0])/(2*h)
        rows.append({'name': name, 'slope': a, 'bias': b, 'depth': depth,
                     'output': value, 'gradient': gradient, 'finite_difference': fd,
                     'normalized_error': abs(fd-gradient)/max(1, abs(gradient))})
    checks = {'identity_preserved': rows[0]['gradient'] == 1,
              'relu_blocks_shortcut': rows[1]['gradient'] == 0 and forward(1, -2, 0, 1, False)[1] == -1,
              'branch_cancels_identity': rows[2]['output'] == 0.5 and rows[2]['gradient'] == 0,
              'shortcut_can_explode': rows[3]['gradient'] == 2**20,
              'shortcut_can_contract': rows[4]['gradient'] == 2**-20,
              'finite_differences_match': all(r['normalized_error'] < 1e-8 for r in rows)}
    return {'config': {'seed': None, 'input': 1, 'difference_step': 1e-6}, 'rows': rows,
            'checks': checks, 'passed': all(checks.values()),
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
