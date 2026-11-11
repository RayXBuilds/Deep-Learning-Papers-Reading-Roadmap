#!/usr/bin/env python3
"""CPU controls for binary region weights and normalized squared residuals."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def region_weights(product, face):
    if not product or len(product) != len(face):
        raise ValueError('matching nonempty masks required')
    if any(x not in (0, 1) for x in list(product) + list(face)):
        raise ValueError('binary masks required')
    weights = {(0, 0): 0.5, (1, 0): 2.0, (0, 1): 2.0, (1, 1): 3.0}
    return [weights[p, f] for p, f in zip(product, face)]


def weighted_loss(residuals, weights, epsilon=1e-8):
    if not residuals or not weights or any(len(row) != len(weights) for row in residuals):
        raise ValueError('nonempty channels with matching spatial positions required')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    if any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError('finite nonnegative weights required')
    if any(not math.isfinite(x) for row in residuals for x in row):
        raise ValueError('finite residuals required')
    numerator = sum(w * r * r for row in residuals for w, r in zip(weights, row))
    return numerator / (len(residuals) * sum(weights) + epsilon)


def run_experiment():
    product, face = [0, 1, 0, 1], [0, 0, 1, 1]
    weights = region_weights(product, face)
    dropped = region_weights(product, [0, 0, 0, 0])
    residuals = [[1, 2, 3, 4], [4, 3, 2, 1]]
    loss = weighted_loss(residuals, weights)
    constant = weighted_loss([[2] * 4, [2] * 4], weights)
    doubled = weighted_loss(residuals, [2 * w for w in weights])
    checks = {
        'overlap_has_own_weight': weights == [0.5, 2, 2, 3],
        'dropout_recomputes_regions': dropped == [0.5, 2, 0.5, 2],
        'two_channels_match_hand_sum': abs(loss - 111.5 / (15 + 1e-8)) < 1e-12,
        'constant_error_scale': abs(constant - 4) < 1e-8,
        'common_scale_nearly_invariant': abs(loss - doubled) < 1e-8,
    }
    return {'config': {'seed': None, 'epsilon': 1e-8, 'time_weight': 1},
            'product': product, 'face': face, 'weights': weights, 'face_dropped_weights': dropped,
            'residuals': residuals, 'loss': loss, 'constant_error_loss': constant,
            'double_weight_loss': doubled, 'checks': checks, 'passed': all(checks.values()),
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
