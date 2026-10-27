#!/usr/bin/env python3
"""Small, CPU-only mechanism experiments; no full-paper benchmark claims."""
import argparse
import hashlib
import json
import math
import platform
import random
import sys
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs)


def variance(xs):
    mu = mean(xs)
    return mean([(x - mu) ** 2 for x in xs])


def normalize(xs, eps=1e-5):
    mu, var = mean(xs), variance(xs)
    return [(x - mu) / math.sqrt(var + eps) for x in xs]


def validate_matrix(x):
    if not x or not x[0] or any(len(row) != len(x[0]) for row in x):
        raise ValueError('Expected a non-empty rectangular matrix')
    if not all(math.isfinite(v) for row in x for v in row):
        raise ValueError('Expected finite matrix entries')


def batch_norm(x):
    """Training-mode BN on B x D; gamma=1, beta=0, population variance."""
    validate_matrix(x)
    return [list(row) for row in zip(*(normalize(col) for col in zip(*x)))]


def layer_norm(x):
    """LN over D for each row of B x D; gamma=1, beta=0."""
    validate_matrix(x)
    return [normalize(row) for row in x]


def max_difference(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def residual_forward(x, depth, scale, skip):
    derivative = 1.0
    for _ in range(depth):
        t = math.tanh(x)
        derivative *= (1.0 if skip else 0.0) + scale * (1.0 - t * t)
        x = (x if skip else 0.0) + scale * t
    return x, derivative


def residual_experiment(seed):
    # Deterministic; seed is included in runner provenance but unused here.
    x, scale, delta = 0.2, 0.1, 1e-5
    rows = []
    checks = {}
    for depth in (1, 5, 20):
        row = {'depth': depth}
        for name, skip in (('plain', False), ('residual', True)):
            value, analytic = residual_forward(x, depth, scale, skip)
            plus = residual_forward(x + delta, depth, scale, skip)[0]
            minus = residual_forward(x - delta, depth, scale, skip)[0]
            numerical = (plus - minus) / (2 * delta)
            relative_error = abs(analytic - numerical) / max(abs(analytic), 1e-30)
            row[name] = {'output': value, 'input_derivative': analytic,
                         'finite_difference': numerical, 'relative_error': relative_error}
            checks[f'{name}_depth_{depth}_gradient_relative_error_lt_1e-6'] = relative_error < 1e-6
        rows.append(row)
    checks['depth_20_plain_derivative_lt_1e-18'] = abs(rows[-1]['plain']['input_derivative']) < 1e-18
    checks['depth_20_residual_derivative_gt_1'] = rows[-1]['residual']['input_derivative'] > 1
    return {'config': {'input': x, 'branch_scale': scale, 'difference_step': delta,
                       'branch': 'scale*tanh(x)', 'trainable_weights': False},
            'metrics': {'by_depth': rows}, 'checks': checks}


def normalization_experiment(seed):
    x = [[1., 2., 4.], [3., 6., 9.], [5., 8., 12.]]
    changed = [row[:] for row in x]
    changed[1] = [30., -60., 90.]
    bn, ln = batch_norm(x), layer_norm(x)
    bn_delta = max_difference(bn[0], batch_norm(changed)[0])
    ln_delta = max_difference(ln[0], layer_norm(changed)[0])
    bn_mean = max(abs(mean(col)) for col in zip(*bn))
    ln_mean = max(abs(mean(row)) for row in ln)
    return {'config': {'input': x, 'modified_batch': changed, 'epsilon': 1e-5,
                       'affine': 'gamma=1,beta=0', 'bn_mode': 'training'},
            'metrics': {'bn': bn, 'ln': ln, 'bn_first_row_change': bn_delta,
                        'ln_first_row_change': ln_delta,
                        'bn_max_abs_column_mean': bn_mean, 'ln_max_abs_row_mean': ln_mean},
            'checks': {'bn_changes_with_other_sample': bn_delta > 0.1,
                       'ln_independent_of_other_sample': ln_delta < 1e-12,
                       'bn_column_mean_zero': bn_mean < 1e-12,
                       'ln_row_mean_zero': ln_mean < 1e-12}}


def adam_step(theta, grad, m, v, step, lr=0.1, beta1=0.9, beta2=0.999,
              epsilon=1e-8, bias_correction=True):
    if step < 1:
        raise ValueError('Adam steps start at 1')
    m = [beta1 * a + (1 - beta1) * g for a, g in zip(m, grad)]
    v = [beta2 * a + (1 - beta2) * g * g for a, g in zip(v, grad)]
    mh = [a / (1 - beta1 ** step) if bias_correction else a for a in m]
    vh = [a / (1 - beta2 ** step) if bias_correction else a for a in v]
    theta = [p - lr * a / (math.sqrt(b) + epsilon) for p, a, b in zip(theta, mh, vh)]
    return theta, m, v


def quadratic(theta):
    return 0.5 * (theta[0] ** 2 + 25 * theta[1] ** 2)


def optimize(method, steps=400):
    theta, m, v = [4., -3.], [0., 0.], [0., 0.]
    initial = quadratic(theta)
    trajectory = [initial]
    lr = 0.02 if method == 'gradient_descent' else 0.1
    for step in range(1, steps + 1):
        grad = [theta[0], 25 * theta[1]]
        if method == 'gradient_descent':
            theta = [p - lr * g for p, g in zip(theta, grad)]
        else:
            theta, m, v = adam_step(theta, grad, m, v, step, lr=lr,
                                    bias_correction=method == 'adam')
        trajectory.append(quadratic(theta))
    return {'learning_rate': lr, 'initial_loss': initial, 'final_loss': trajectory[-1],
            'final_parameters': theta, 'first_ten_steps': trajectory[:11],
            'loss_every_50_steps': trajectory[::50]}


def adam_experiment(seed):
    results = {name: optimize(name) for name in ('gradient_descent', 'adam', 'adam_uncorrected')}
    return {'config': {'objective': '0.5*(x*x+25*y*y)', 'initial_parameters': [4., -3.],
                       'steps': 400, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
                       'comparison': 'illustrative fixed settings; not an optimizer ranking'},
            'metrics': results,
            'checks': {'all_final_losses_finite': all(math.isfinite(r['final_loss']) for r in results.values()),
                       'adam_final_loss_lt_1e-8': results['adam']['final_loss'] < 1e-8}}


def softmax(values):
    maximum = max(values)
    weights = [math.exp(v - maximum) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def attention(q, k, v, causal=False):
    for x in (q, k, v):
        validate_matrix(x)
    if len(k) != len(v) or len(q[0]) != len(k[0]):
        raise ValueError('Incompatible Q/K/V shapes')
    if causal and len(q) != len(k):
        raise ValueError('This causal example requires equal query/key lengths')
    weights, outputs = [], []
    for i, query in enumerate(q):
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(len(query))
                  if not causal or j <= i else -math.inf for j, key in enumerate(k)]
        w = softmax(scores)
        weights.append(w)
        outputs.append([sum(w[j] * v[j][d] for j in range(len(v))) for d in range(len(v[0]))])
    return outputs, weights


def attention_experiment(seed):
    q = [[1., 0.], [0., 1.], [1., 1.]]
    k = [[1., 0.], [0., 1.], [1., 1.]]
    v = [[1., 2.], [3., 4.], [5., 6.]]
    causal, weights = attention(q, k, v, causal=True)
    full, _ = attention(q, k, v)
    k2, v2 = [row[:] for row in k], [row[:] for row in v]
    k2[-1], v2[-1] = [50., 50.], [100., -100.]
    after_causal, _ = attention(q, k2, v2, causal=True)
    after_full, _ = attention(q, k2, v2)
    causal_delta = max(max_difference(a, b) for a, b in zip(causal[:2], after_causal[:2]))
    full_delta = max(max_difference(a, b) for a, b in zip(full[:2], after_full[:2]))
    return {'config': {'q': q, 'k': k, 'v': v, 'changed_future_k': k2[-1],
                       'changed_future_v': v2[-1], 'heads': 1},
            'metrics': {'causal_weights': weights, 'causal_output': causal,
                        'unmasked_output': full, 'causal_prefix_change': causal_delta,
                        'unmasked_prefix_change': full_delta},
            'checks': {'weight_rows_sum_to_one': all(abs(sum(w) - 1) < 1e-12 for w in weights),
                       'future_weights_zero': all(weights[i][j] == 0 for i in range(3) for j in range(i + 1, 3)),
                       'future_intervention_isolated': causal_delta < 1e-12,
                       'unmasked_control_changes': full_delta > 1}}


def forward_noise(x0, noise, alpha_bar):
    if not 0 < alpha_bar < 1:
        raise ValueError('alpha_bar must lie strictly between 0 and 1')
    return math.sqrt(alpha_bar) * x0 + math.sqrt(1 - alpha_bar) * noise


def gaussian_samples(rng, n, alpha_bar, mu=2., sigma=2.):
    pairs = []
    for _ in range(n):
        x0, epsilon = rng.gauss(mu, sigma), rng.gauss(0., 1.)
        pairs.append((forward_noise(x0, epsilon, alpha_bar), epsilon))
    return pairs


def fit_line(pairs):
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    slope = mean([(x - mx) * (y - my) for x, y in pairs]) / variance(xs)
    return slope, my - slope * mx


def diffusion_experiment(seed):
    alpha_bar, n = 0.6, 20000
    # A single RNG stream is consumed sequentially: the held-out samples are fresh.
    rng = random.Random(seed)
    train = gaussian_samples(rng, n, alpha_bar)
    test = gaussian_samples(rng, n, alpha_bar)
    slope, intercept = fit_line(train)
    theoretical_mean = math.sqrt(alpha_bar) * 2
    theoretical_variance = alpha_bar * 4 + (1 - alpha_bar)
    oracle_slope = math.sqrt(1 - alpha_bar) / theoretical_variance
    oracle_intercept = -oracle_slope * theoretical_mean
    xs = [x for x, _ in test]
    mse = mean([(slope * x + intercept - noise) ** 2 for x, noise in test])
    baseline = mean([noise ** 2 for _, noise in test])
    mean_error = abs(mean(xs) - theoretical_mean)
    variance_error = abs(variance(xs) - theoretical_variance)
    return {'config': {'alpha_bar': alpha_bar, 'data_mean': 2., 'data_variance': 4.,
                       'train_samples': n, 'test_samples': n, 'fit': 'ordinary least squares',
                       'split': 'independent sequential draws; no training samples reused'},
            'metrics': {'empirical_mean': mean(xs), 'theoretical_mean': theoretical_mean,
                        'empirical_variance': variance(xs), 'theoretical_variance': theoretical_variance,
                        'fitted_slope': slope, 'oracle_slope': oracle_slope,
                        'fitted_intercept': intercept, 'oracle_intercept': oracle_intercept,
                        'test_noise_mse': mse, 'zero_predictor_mse': baseline,
                        'optimal_expected_noise_mse': 1 - (1 - alpha_bar) / theoretical_variance},
            'checks': {'mean_error_lt_0_08': mean_error < 0.08,
                       'variance_error_lt_0_15': variance_error < 0.15,
                       'slope_error_lt_0_03': abs(slope - oracle_slope) < 0.03,
                       'held_out_mse_beats_zero_by_0_05': mse < baseline - 0.05}}


EXPERIMENTS = {'residual': residual_experiment, 'normalization': normalization_experiment,
               'adam': adam_experiment, 'attention': attention_experiment, 'diffusion': diffusion_experiment}


def run_all(selected, seed):
    names = list(EXPERIMENTS) if selected == 'all' else [selected]
    results = {name: EXPERIMENTS[name](seed) for name in names}
    for result in results.values():
        result['passed'] = all(result['checks'].values())
    return {'schema_version': 1, 'scope': 'educational mechanism checks, not paper benchmarks',
            'python': platform.python_version(), 'platform': platform.platform(), 'seed': seed,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'results': results, 'passed': all(r['passed'] for r in results.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all', *EXPERIMENTS], default='all')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_all(args.experiment, args.seed)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
