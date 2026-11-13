#!/usr/bin/env python3
"""Gaussian-mixture FM: independent regression data and CDF sampling controls."""
import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from statistics import NormalDist

NORMAL = NormalDist()


def velocity(t, x, linear=False):
    if not 0 <= t <= 1 or not math.isfinite(x):
        raise ValueError('finite state and time in [0,1] required')
    variance = 4.25 if linear else 0.25
    s = (1-t)**2 + t*t*variance
    c = t*variance - (1-t)
    if linear:
        return c/s*x
    return c/s*x + (1-c*t/s)*2*math.tanh(2*t*x/s)


def target_cdf(x):
    return 0.5*(NORMAL.cdf((x-2)/0.5)+NORMAL.cdf((x+2)/0.5))


def pairs(seed, n):
    rng = random.Random(seed)
    return [(rng.gauss(0, 1), (2 if rng.random() < 0.5 else -2)+rng.gauss(0, 0.5)) for _ in range(n)]


def regression(seed):
    train, test = pairs(seed, 10000), pairs(seed+100000, 10000)
    rows = []
    for t in (0.25, 0.5, 0.75):
        xs = [(1-t)*z+t*y for z,y in train]
        us = [y-z for z,y in train]
        mx, mu = sum(xs)/len(xs), sum(us)/len(us)
        a = sum((x-mx)*(u-mu) for x,u in zip(xs,us))/sum((x-mx)**2 for x in xs)
        b = mu-a*mx
        linear_mse = oracle_mse = excess = 0
        for z,y in test:
            x, u = (1-t)*z+t*y, y-z
            v = velocity(t,x)
            linear_mse += (a*x+b-u)**2
            oracle_mse += (v-u)**2
            excess += (a*x+b-v)**2
        rows.append({'t':t, 'slope':a, 'intercept':b,
                     'linear_test_mse':linear_mse/len(test),
                     'oracle_test_mse':oracle_mse/len(test),
                     'field_mse':excess/len(test)})
    return {'seed':seed, 'test_seed':seed+100000, 'rows':rows}


def sample(steps, linear=False):
    if not isinstance(steps,int) or steps <= 0:
        raise ValueError('positive integer steps required')
    qs = [(i+0.5)/200 for i in range(200)]
    outputs = []
    for q in qs:
        x = NORMAL.inv_cdf(q)
        for k in range(steps):
            x += velocity(k/steps,x,linear)/steps
        outputs.append(x)
    return {'steps':steps, 'nfe_per_sample':steps,
            'max_cdf_error':max(abs(target_cdf(x)-q) for x,q in zip(outputs,qs)),
            'left_peak_mass':sum(-3 < x < -1 for x in outputs)/len(outputs),
            'right_peak_mass':sum(1 < x < 3 for x in outputs)/len(outputs),
            'outputs':outputs}


def run_experiment():
    runs = [regression(s) for s in (7,42,2026)]
    exact = [sample(n) for n in (200,400)]
    linear = sample(400,True)
    peak_mass = target_cdf(3)-target_cdf(1)
    checks = {
        'oracle_beats_fitted_linear_all_seeds': all(r['linear_test_mse'] > r['oracle_test_mse']+0.1 for run in runs for r in run['rows'] if r['t']==0.5),
        'field_error_at_midpoint_all_seeds': all(run['rows'][1]['field_mse'] > 0.1 for run in runs),
        'oracle_cdf_converges': exact[1]['max_cdf_error'] < 0.02 and exact[1]['max_cdf_error'] < exact[0]['max_cdf_error'],
        'linear_cdf_negative_control': linear['max_cdf_error'] > 0.1,
        'both_peak_masses_recovered': all(abs(exact[1][key]-peak_mass) < 0.02 for key in ('left_peak_mass','right_peak_mass')),
    }
    return {'config':{'component_means':[-2,2], 'component_std':0.5, 'weights':[0.5,0.5],
                      'train_n':10000,'test_n':10000,'quantile_n':200,'endpoint_noise':0},
            'regressions':runs,'oracle_sampling':exact,'population_linear_sampling':linear,
            'target_peak_mass':peak_mass,'checks':checks,'passed':all(checks.values()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform()}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    r=run_experiment()
    data=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(data)
    print(data)
    raise SystemExit(0 if r['passed'] else 1)
