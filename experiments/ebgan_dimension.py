#!/usr/bin/env python3
"""Dimension limits of the EBGAN pulling-away objective, using fixed vectors."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from ebgan_energy import pulling_away


def run_experiment():
    cases = {
        'two_orthogonal': [[1, 0], [0, 1]],
        'three_equiangular': [[1, 0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]],
        'four_repeated_axes': [[1, 0], [1, 0], [0, 1], [0, 1]],
        'four_collinear': [[1, 0], [2, 0], [-1, 0], [-2, 0]],
    }
    rows = {}
    for name, features in cases.items():
        count, dimension = len(features), len(features[0])
        normalized = [[x / math.hypot(*row) for x in row] for row in features]
        # Independently sum entries of U^T U; production PT uses sample pairs.
        covariance = [[sum(row[i] * row[j] for row in normalized)
                       for j in range(dimension)] for i in range(dimension)]
        frame_potential = sum(x * x for row in covariance for x in row)
        rows[name] = {'inputs': features, 'count': count, 'dimension': dimension,
                      'lower_bound': max(0, (count - dimension) / (dimension * (count - 1))),
                      'pt': pulling_away(features),
                      'matrix_reference': (frame_potential - count) / (count * (count - 1))}
    checks = {
        'pair_and_matrix_formulas_agree': all(abs(r['pt'] - r['matrix_reference']) < 1e-12 for r in rows.values()),
        'all_respect_dimension_bound': all(r['pt'] >= r['lower_bound'] - 1e-12 for r in rows.values()),
        'three_vectors_attain_quarter': abs(rows['three_equiangular']['pt'] - 0.25) < 1e-12,
        'duplicates_attain_bound': abs(rows['four_repeated_axes']['pt'] - 1 / 3) < 1e-12,
        'collinear_negative_control': abs(rows['four_collinear']['pt'] - 1) < 1e-12,
    }
    return {'config': {'seed': None, 'randomness': 'none', 'tolerance': 1e-12},
            'cases': rows, 'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependency_sha256': hashlib.sha256(Path(__file__).with_name('ebgan_energy.py').read_bytes()).hexdigest(),
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
