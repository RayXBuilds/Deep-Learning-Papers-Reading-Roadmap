#!/usr/bin/env python3
"""Deterministic additive-attention and padding-mask mechanism checks."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def matrix(x, name):
    if not x or not x[0] or any(len(row) != len(x[0]) for row in x):
        raise ValueError(f'{name} must be a nonempty rectangular matrix')
    if not all(math.isfinite(value) for row in x for value in row):
        raise ValueError(f'{name} must contain finite numbers')


def attention(queries, annotations, wq, wh, vector, valid_length, nonlinear=True):
    """e_ij = v.T tanh(Wq q_i + Wh h_j); context uses h_j as values.

    One source sequence per call. Padded positions have zero attention weight.
    Matrices Wq and Wh are [hidden, query_dim] and [hidden, annotation_dim].
    """
    for name, value in [('queries', queries), ('annotations', annotations), ('wq', wq), ('wh', wh)]:
        matrix(value, name)
    if (not vector or not all(math.isfinite(v) for v in vector)
            or len(wq) != len(wh) or len(vector) != len(wq)
            or len(wq[0]) != len(queries[0]) or len(wh[0]) != len(annotations[0])):
        raise ValueError('Incompatible query, annotation or projection shapes')
    if type(valid_length) is not int or not 1 <= valid_length <= len(annotations):
        raise ValueError('valid_length must select at least one source position')
    projected_h = [[sum(a * b for a, b in zip(row, h)) for row in wh] for h in annotations]
    contexts, weights = [], []
    for query in queries:
        projected_q = [sum(a * b for a, b in zip(row, query)) for row in wq]
        scores = [sum(v * (math.tanh(q + h) if nonlinear else q + h)
                      for v, q, h in zip(vector, projected_q, key))
                  for key in projected_h[:valid_length]]
        maximum = max(scores)
        exps = [math.exp(score - maximum) for score in scores]
        total = sum(exps)
        w = [value / total for value in exps] + [0.] * (len(annotations) - valid_length)
        # Accumulate only valid values: padding should not participate in context.
        context = [sum(w[j] * annotations[j][d] for j in range(valid_length))
                   for d in range(len(annotations[0]))]
        weights.append(w)
        contexts.append(context)
    return contexts, weights


def max_delta(a, b):
    return max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def run_experiment():
    queries = [[1.], [-1.]]
    h = [[-2., 4.], [0., 0.], [2., 4.]]
    wq, wh, vector = [[1.], [-1.]], [[1., 0.], [1., 0.]], [1., -1.]
    contexts, weights = attention(queries, h, wq, wh, vector, 3)
    padded = h + [[1000., -1000.], [-2000., 2000.]]
    padded_contexts, padded_weights = attention(queries, padded, wq, wh, vector, 3)
    unmasked_contexts, _ = attention(queries, padded, wq, wh, vector, 5)
    # Compare a shorter sequence with the same sequence embedded in padded storage.
    short_contexts, _ = attention(queries, h[:2], wq, wh, vector, 2)
    short_padded_contexts, _ = attention(queries, padded, wq, wh, vector, 2)
    uniform, _ = attention(queries, h, wq, wh, [0., 0.], 3)
    fixed_mean = [[0., 8. / 3.], [0., 8. / 3.]]
    linear_contexts, _ = attention(queries, h, wq, wh, vector, 3, nonlinear=False)
    metrics = {
        'contexts': contexts, 'weights': weights, 'fixed_mean_contexts': fixed_mean,
        'query_context_change': max_delta([contexts[0]], [contexts[1]]),
        'fixed_mean_query_change': max_delta([fixed_mean[0]], [fixed_mean[1]]),
        'linear_scorer_query_change': max_delta([linear_contexts[0]], [linear_contexts[1]]),
        'padding_context_change': max_delta(contexts, padded_contexts),
        'short_sequence_padding_change': max_delta(short_contexts, short_padded_contexts),
        'unmasked_padding_change': max_delta(contexts, unmasked_contexts),
        'zero_scorer_mean_error': max_delta(uniform, fixed_mean),
        'padded_weights': padded_weights,
    }
    checks = {
        'weights_sum_to_one': all(abs(sum(row) - 1.) < 1e-12 for row in padded_weights),
        'padded_weights_zero': all(value == 0. for row in padded_weights for value in row[3:]),
        'query_changes_context_gt_1': metrics['query_context_change'] > 1.,
        'fixed_mean_unchanged': metrics['fixed_mean_query_change'] == 0.,
        'linear_scorer_query_cancels': metrics['linear_scorer_query_change'] < 1e-12,
        'padding_invariant_lt_1e-12': metrics['padding_context_change'] < 1e-12,
        'short_sequence_invariant_lt_1e-12': metrics['short_sequence_padding_change'] < 1e-12,
        'unmasked_control_changes_gt_1': metrics['unmasked_padding_change'] > 1.,
        'zero_scorer_matches_mean_lt_1e-12': metrics['zero_scorer_mean_error'] < 1e-12,
    }
    return {'schema_version': 1, 'scope': 'hand-set additive attention; no training or BLEU reproduction',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'config': {'queries': queries, 'annotations': h, 'wq': wq, 'wh': wh, 'v': vector,
                       'padded_annotations': padded, 'valid_lengths': [3, 2],
                       'seed': None, 'randomness': 'none; hand-set inputs and projections'},
            'metrics': metrics, 'checks': checks, 'passed': all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_experiment()
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
