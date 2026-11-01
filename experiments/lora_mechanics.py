#!/usr/bin/env python3
"""LoRA algebra and two SGD steps on one synthetic example, not model fine-tuning."""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
from pathlib import Path


def shape(matrix):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError('nonempty rectangular matrix required')
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError('finite matrix required')
    return len(matrix), len(matrix[0])


def matvec(matrix, vector):
    _, width = shape(matrix)
    if len(vector) != width or not all(math.isfinite(v) for v in vector):
        raise ValueError('incompatible finite vector required')
    return [sum(a*b for a, b in zip(row, vector)) for row in matrix]


def transpose(matrix):
    shape(matrix)
    return [list(col) for col in zip(*matrix)]


def dimensions(w, a, b, alpha):
    d, k = shape(w)
    r, a_width = shape(a)
    b_height, b_width = shape(b)
    if (a_width, b_height, b_width) != (k, d, r) or not math.isfinite(alpha):
        raise ValueError('incompatible LoRA dimensions or nonfinite alpha')
    return d, k, r


def forward(w, a, b, x, alpha):
    _, _, r = dimensions(w, a, b, alpha)
    base = matvec(w, x)
    branch = matvec(b, matvec(a, x))
    return [u + alpha/r*v for u, v in zip(base, branch)]


def merge(w, a, b, alpha):
    d, k, r = dimensions(w, a, b, alpha)
    return [[w[i][j] + alpha/r*sum(b[i][t]*a[t][j] for t in range(r))
             for j in range(k)] for i in range(d)]


def loss_and_gradients(w, a, b, x, target, alpha):
    """Squared loss / 2, no bias, dropout, regularization or optimizer state.

    Frozen W is not updated; its transpose still contributes to the input gradient.
    """
    d, k, r = dimensions(w, a, b, alpha)
    if len(target) != d or not all(math.isfinite(v) for v in target):
        raise ValueError('incompatible finite target required')
    y = forward(w, a, b, x, alpha)
    g = [v-t for v, t in zip(y, target)]
    ax = matvec(a, x)
    btg = matvec(transpose(b), g)
    ga = [[alpha/r*btg[t]*x[j] for j in range(k)] for t in range(r)]
    gb = [[alpha/r*g[i]*ax[t] for t in range(r)] for i in range(d)]
    base_dx = matvec(transpose(w), g)
    branch_dx = matvec(transpose(a), btg)
    gx = [u+alpha/r*v for u, v in zip(base_dx, branch_dx)]
    return 0.5*sum(v*v for v in g), ga, gb, gx


def sgd(matrix, gradient, lr):
    if shape(matrix) != shape(gradient) or not math.isfinite(lr) or lr < 0:
        raise ValueError('compatible gradient and nonnegative finite lr required')
    return [[v-lr*g for v, g in zip(row, grad)] for row, grad in zip(matrix, gradient)]


def maximum(matrix):
    return max(abs(v) for row in matrix for v in row)


def distance(xs, ys):
    if len(xs) != len(ys):
        raise ValueError('incompatible vectors')
    return max(abs(x-y) for x, y in zip(xs, ys))


def finite_difference_errors(w, a, b, x, target, alpha, h=1e-6):
    _, ga, gb, gx = loss_and_gradients(w, a, b, x, target, alpha)
    errors = {}
    for name, matrix, gradient in [('a', a, ga), ('b', b, gb)]:
        diffs = []
        for i, row in enumerate(matrix):
            for j in range(len(row)):
                lo, hi = copy.deepcopy(matrix), copy.deepcopy(matrix)
                lo[i][j] -= h
                hi[i][j] += h
                left = loss_and_gradients(w, lo if name == 'a' else a,
                                          lo if name == 'b' else b, x, target, alpha)[0]
                right = loss_and_gradients(w, hi if name == 'a' else a,
                                           hi if name == 'b' else b, x, target, alpha)[0]
                diffs.append(abs((right-left)/(2*h)-gradient[i][j]))
        errors[name] = max(diffs)
    diffs = []
    for j in range(len(x)):
        lo, hi = list(x), list(x)
        lo[j] -= h
        hi[j] += h
        left = loss_and_gradients(w, a, b, lo, target, alpha)[0]
        right = loss_and_gradients(w, a, b, hi, target, alpha)[0]
        diffs.append(abs((right-left)/(2*h)-gx[j]))
    errors['x'] = max(diffs)
    return errors


def run_experiment():
    w = [[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]]
    a = [[0.2, -0.3, 0.4], [-0.1, 0.5, 0.2]]
    b = [[0.7, -0.2], [0.3, 0.6]]
    x, target, alpha, lr = [1.0, -2.0, 0.5], [0.3, -0.7], 4.0, 0.01
    saved_w = copy.deepcopy(w)
    separated = forward(w, a, b, x, alpha)
    merged = matvec(merge(w, a, b, alpha), x)
    fd_errors = finite_difference_errors(w, a, b, x, target, alpha)
    initializations = []
    for seed in (7, 42, 2026):
        rng = random.Random(seed)
        ai = [[rng.gauss(0, 0.1) for _ in range(3)] for _ in range(2)]
        bi = [[0.0, 0.0], [0.0, 0.0]]
        loss0, ga0, gb0, _ = loss_and_gradients(w, ai, bi, x, target, alpha)
        a1, b1 = sgd(ai, ga0, lr), sgd(bi, gb0, lr)
        loss1, ga1, gb1, _ = loss_and_gradients(w, a1, b1, x, target, alpha)
        a2, b2 = sgd(a1, ga1, lr), sgd(b1, gb1, lr)
        loss2 = loss_and_gradients(w, a2, b2, x, target, alpha)[0]
        initializations.append({
            'seed': seed, 'a0': ai, 'b0': bi, 'a1': a1, 'b1': b1, 'a2': a2, 'b2': b2,
            'losses': [loss0, loss1, loss2],
            'initial_output_error': distance(forward(w, ai, bi, x, alpha), matvec(w, x)),
            'first_a_gradient_max': maximum(ga0), 'first_b_gradient_max': maximum(gb0),
            'second_a_gradient_max': maximum(ga1),
        })
    zero_a, zero_b = [[0.0]*3 for _ in range(2)], [[0.0]*2 for _ in range(2)]
    zero_loss, zga, zgb, _ = loss_and_gradients(w, zero_a, zero_b, x, target, alpha)
    # Fixed inverted-dropout multiplier, p=0.5: a counterexample, NOT Monte Carlo.
    multiplier = [0.0, 2.0, 0.0]
    dropped_x = [v*m for v, m in zip(x, multiplier)]
    base, dropped_branch = matvec(w, x), matvec(b, matvec(a, dropped_x))
    dropout_output = [v + alpha/2*u for v, u in zip(base, dropped_branch)]
    # Merging and retaining the branch counts the update twice.
    double_counted = forward(merge(w, a, b, alpha), a, b, x, alpha)
    # A rank-one update does not force the final weight to have rank one.
    rank_update = [[3.0, 4.0], [6.0, 8.0]]
    full_weight = merge([[1.0, 0.0], [0.0, 1.0]], [[3.0, 4.0]], [[1.0], [2.0]], 1.0)
    det = lambda m: m[0][0]*m[1][1]-m[0][1]*m[1][0]
    checks = {
        'merge_matches_separate_branch': distance(separated, merged) < 1e-12,
        'factor_and_input_gradients_match_finite_difference': max(fd_errors.values()) < 1e-7,
        'zero_b_preserves_base_all_seeds': all(t['initial_output_error'] < 1e-12 for t in initializations),
        'first_a_gradient_zero_b_gradient_nonzero_all_seeds': all(
            t['first_a_gradient_max'] == 0 and t['first_b_gradient_max'] > 1e-8 for t in initializations),
        'a_receives_gradient_after_b_update_all_seeds': all(t['second_a_gradient_max'] > 1e-8 for t in initializations),
        'both_zero_is_stationary_despite_positive_loss': maximum(zga) == 0 and maximum(zgb) == 0 and zero_loss > 0,
        'frozen_base_unchanged': w == saved_w,
        'active_dropout_breaks_naive_merge': distance(dropout_output, merged) > 0.1,
        'double_counted_branch_negative_control': distance(double_counted, merged) > 0.1,
        'low_rank_update_can_leave_full_rank_weight': det(rank_update) == 0 and det(full_weight) != 0,
    }
    return {
        'config': {'w': w, 'a': a, 'b': b, 'x': x, 'target': target, 'alpha': alpha, 'rank': 2,
                   'loss': '0.5*sum((prediction-target)^2)', 'lr': lr, 'steps': 2,
                   'seeds': [7, 42, 2026], 'initial_a_std': 0.1, 'initial_b': 'zeros',
                   'finite_difference_h': 1e-6, 'dropout_multiplier': multiplier,
                   'data': 'one hand-defined example; no train/test split or generalization claim'},
        'metrics': {'separate_output': separated, 'merged_output': merged,
                    'merge_max_error': distance(separated, merged), 'finite_difference_errors': fd_errors,
                    'both_zero_loss': zero_loss, 'both_zero_gradient_max': max(maximum(zga), maximum(zgb)),
                    'dropout_merge_error': distance(dropout_output, merged),
                    'double_counted_error': distance(double_counted, merged),
                    'rank_one_update_determinant': det(rank_update), 'merged_weight_determinant': det(full_weight),
                    'parameter_example': {'d': 4096, 'k': 4096, 'r': 8,
                                          'full': 4096*4096, 'factors': 8*(4096+4096)}},
        'initializations': initializations, 'checks': checks, 'passed': all(checks.values()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'platform': platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_experiment()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': report['checks'], 'metrics': report['metrics']}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
