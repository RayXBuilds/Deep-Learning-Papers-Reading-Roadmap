#!/usr/bin/env python3
"""Deterministic Adam+L2 versus AdamW mechanics; not a generalization benchmark."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def initial_state(theta):
    if not theta or not all(math.isfinite(x) for x in theta):
        raise ValueError('nonempty finite parameters required')
    return {'theta': list(theta), 'm': [0.0] * len(theta),
            'v': [0.0] * len(theta), 'step': 0}


def update(state, gradient, lr, decay, mode, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Use old theta for decay; AdamW shrinkage is 1-lr*decay (API convention)."""
    theta, old_m, old_v = state['theta'], state['m'], state['v']
    if mode not in ('l2', 'adamw'):
        raise ValueError('mode must be l2 or adamw')
    if not theta or len({len(theta), len(gradient), len(old_m), len(old_v)}) != 1:
        raise ValueError('inconsistent nonempty vector dimensions')
    scalars = [lr, decay, beta1, beta2, epsilon] + list(theta) + list(gradient) + list(old_m) + list(old_v)
    if not all(math.isfinite(x) for x in scalars):
        raise ValueError('finite inputs required')
    if min(lr, decay) < 0 or epsilon <= 0 or not (0 <= beta1 < 1 and 0 <= beta2 < 1):
        raise ValueError('invalid hyperparameters')
    if any(x < 0 for x in old_v) or type(state['step']) is not int or state['step'] < 0:
        raise ValueError('invalid moment state')
    step = state['step'] + 1
    effective = [g + decay * p if mode == 'l2' else g for g, p in zip(gradient, theta)]
    m = [beta1 * old + (1-beta1) * g for old, g in zip(old_m, effective)]
    v = [beta2 * old + (1-beta2) * g*g for old, g in zip(old_v, effective)]
    scaled = [a / (1-beta1**step) / (math.sqrt(b / (1-beta2**step)) + epsilon)
              for a, b in zip(m, v)]
    new_theta = [p - lr * u - (lr * decay * p if mode == 'adamw' else 0)
                 for p, u in zip(theta, scaled)]
    return {'theta': new_theta, 'm': m, 'v': v, 'step': step}


def quadratic(theta):
    # Exact fixed objective, no data sampled and no held-out generalization claim.
    x, y = theta
    return 0.5 * ((x-2)**2 + 4*(y+1)**2), [x-2, 4*(y+1)]


def distance(xs, ys):
    return max(abs(x-y) for x, y in zip(xs, ys))


def run_experiment():
    state = initial_state([1.0, 10.0])
    l2 = update(state, [0, 0], 0.1, 0.1, 'l2')
    wd = update(state, [0, 0], 0.1, 0.1, 'adamw')
    # First corrected Adam step: m_hat=g, v_hat=g^2.
    l2_reference = [p - 0.1 * (0.1*p) / (abs(0.1*p)+1e-8) for p in state['theta']]
    wd_reference = [0.99, 9.9]
    # With identical EXOGENOUS gradient streams, AdamW moments exclude decay.
    stream = [[2, -3], [0, 0], [-1, 4]]
    plain, decayed = initial_state([1, 10]), initial_state([1, 10])
    moment_errors = []
    for grad in stream:
        plain = update(plain, grad, 0.1, 0, 'adamw')
        decayed = update(decayed, grad, 0.1, 0.1, 'adamw')
        moment_errors += [distance(plain['m'], decayed['m']), distance(plain['v'], decayed['v'])]
    # No gradient and no initial moments: product of shrinkage factors is exact.
    schedule = [0.1, 0.05, 0.01]
    scheduled = initial_state([2, -3])
    factor = 1.0
    for lr in schedule:
        scheduled = update(scheduled, [0, 0], lr, 0.1, 'adamw')
        factor *= 1-lr*0.1
    schedule_reference = [2*factor, -3*factor]
    # Plain SGD without momentum: L2 and matched decay have identical steps.
    theta, grad, lr, coefficient = [2, -3], [4, -5], 0.1, 0.2
    sgd_l2 = [p-lr*(g+coefficient*p) for p, g in zip(theta, grad)]
    sgd_wd = [(1-lr*coefficient)*p-lr*g for p, g in zip(theta, grad)]
    trajectories = {}
    for label, mode, decay in [('adam', 'adamw', 0), ('adam_l2', 'l2', 0.1), ('adamw', 'adamw', 0.1)]:
        current = initial_state([3, -2])
        curve = []
        for t in range(201):
            loss, grad = quadratic(current['theta'])
            curve.append({'step': t, 'loss': loss, 'theta': current['theta']})
            if t < 200:
                current = update(current, grad, 0.05, decay, mode)
        trajectories[label] = curve
    checks = {
        'l2_first_step_matches_hand_calculation': distance(l2['theta'], l2_reference) < 1e-12,
        'adamw_first_step_matches_uniform_shrinkage': distance(wd['theta'], wd_reference) < 1e-12,
        'l2_and_adamw_are_different': distance(l2['theta'], wd['theta']) > 0.08,
        'adamw_does_not_put_decay_into_moments': max(moment_errors) < 1e-12,
        'scheduled_decay_matches_product': distance(scheduled['theta'], schedule_reference) < 1e-12,
        'plain_sgd_equivalence': distance(sgd_l2, sgd_wd) < 1e-12,
        'all_fixed_budget_trajectories_finite': all(math.isfinite(p) and math.isfinite(row['loss']) for rows in trajectories.values() for row in rows for p in row['theta']),
    }
    return {
        'config': {'seed': None, 'randomness': 'none', 'lr_convention': 'shrinkage=1-lr*decay',
                   'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8,
                   'first_step': {'theta': [1, 10], 'gradient': [0, 0], 'lr': 0.1, 'decay': 0.1},
                   'exogenous_gradients': stream, 'zero_gradient_lr_schedule': schedule,
                   'trajectory': {'objective': '0.5*((x-2)^2+4*(y+1)^2)', 'theta0': [3,-2],
                                  'steps': 200, 'lr': 0.05, 'decay': 0.1, 'data': 'none; exact objective'}},
        'first_step': {'adam_l2': l2, 'adamw': wd},
        'metrics': {'moment_error': max(moment_errors), 'scheduled_parameters': scheduled['theta'],
                    'scheduled_reference': schedule_reference, 'sgd_equivalence_error': distance(sgd_l2, sgd_wd)},
        'trajectories': trajectories,
        'checks': checks, 'passed': all(checks.values()),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'platform': platform.platform(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_experiment()
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload, encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checks': report['checks'],
                      'final': {name: rows[-1] for name, rows in report['trajectories'].items()}}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
