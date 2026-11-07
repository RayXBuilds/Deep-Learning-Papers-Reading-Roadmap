#!/usr/bin/env python3
"""Finite-support EBGAN energy objectives and pulling-away controls."""
import argparse
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path


def discriminator_loss(real, generated, energies, margin):
    if not real or len(real) != len(generated) or len(real) != len(energies):
        raise ValueError('matching nonempty probability and energy vectors required')
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('positive finite margin required')
    for probabilities in (real, generated):
        if any(not math.isfinite(x) or x < 0 for x in probabilities):
            raise ValueError('finite nonnegative probabilities required')
        if not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError('probabilities must sum to one')
    if any(not math.isfinite(x) or x < 0 for x in energies):
        raise ValueError('finite nonnegative energies required')
    return sum(p * energy + q * max(0, margin - energy)
               for p, q, energy in zip(real, generated, energies))


def pulling_away(features):
    if len(features) < 2 or not features[0]:
        raise ValueError('at least two nonempty feature vectors required')
    dimension = len(features[0])
    normalized = []
    for row in features:
        if len(row) != dimension or any(not math.isfinite(x) for x in row):
            raise ValueError('matching finite feature dimensions required')
        norm = math.hypot(*row)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError('finite nonzero feature norms required')
        normalized.append([x / norm for x in row])
    pairs = list(itertools.combinations(normalized, 2))
    # Each unordered pair occurs twice in the paper's ordered-pair mean.
    return sum(sum(x * y for x, y in zip(a, b)) ** 2 for a, b in pairs) / len(pairs)


def run_experiment():
    margin = 2.0
    cases = [([0.75, 0.25], [0.25, 0.75]),
             ([0.75, 0.25], [0.75, 0.25]),
             ([1.0, 0.0], [0.0, 1.0])]
    rows = []
    for real, generated in cases:
        energies = [margin if p < q else 0.0 for p, q in zip(real, generated)]
        optimum = discriminator_loss(real, generated, energies, margin)
        grid_minimum = min(discriminator_loss(real, generated, candidate, margin)
                           for candidate in itertools.product([0, 0.5, 1, 2, 3], repeat=2))
        total_variation = sum(abs(p - q) for p, q in zip(real, generated)) / 2
        rows.append({'real': real, 'generated': generated, 'energies': energies,
                     'loss': optimum, 'grid_minimum': grid_minimum,
                     'total_variation': total_variation})
    equal_losses = [discriminator_loss([0.75, 0.25], [0.75, 0.25], energies, margin)
                    for energies in ([0, 0], [0.5, 1.5], [2, 2])]
    step = 1e-5
    energy = 3.0
    generator_derivative = ((energy + step) - (energy - step)) / (2 * step)
    hinge_derivative = (max(0, margin - energy - step)
                        - max(0, margin - energy + step)) / (2 * step)
    orthogonal = pulling_away([[1, 0], [0, 1]])
    duplicate = pulling_away([[1, 0], [1, 0]])
    opposite = pulling_away([[1, 0], [-1, 0]])
    features = [[1, 0], [1, 1], [0, 1]]
    pt = pulling_away(features)
    scaled_pt = pulling_away([[3, 0], [0.5, 0.5], [0, 7]])
    checks = {
        'analytic_optimum_matches_grid': all(abs(r['loss'] - r['grid_minimum']) < 1e-12 for r in rows),
        'optimal_loss_matches_tv': all(abs(r['loss'] - margin * (1 - r['total_variation'])) < 1e-12 for r in rows),
        'equal_distribution_energy_nonunique': all(abs(x - margin) < 1e-12 for x in equal_losses),
        'generator_energy_gradient_above_margin': abs(generator_derivative - 1) < 1e-8 and hinge_derivative == 0,
        'pt_geometry_controls': orthogonal == 0 and duplicate == 1 and opposite == 1,
        'pt_scale_invariance': abs(pt - scaled_pt) < 1e-12 and abs(pt - 1 / 3) < 1e-12,
    }
    return {'config': {'seed': None, 'randomness': 'none', 'margin': margin, 'difference_step': step},
            'distributions': rows, 'equal_distribution_losses': equal_losses,
            'generator_energy_derivative': generator_derivative, 'hinge_derivative': hinge_derivative,
            'pt': {'orthogonal': orthogonal, 'duplicate': duplicate, 'opposite': opposite,
                   'three_vectors': pt, 'rescaled': scaled_pt},
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
