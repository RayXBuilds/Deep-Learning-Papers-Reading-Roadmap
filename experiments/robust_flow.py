#!/usr/bin/env python3
"""Fixed-input controls for corruption residuals and comparative label scores."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def residual(t, x0, x1, a, b):
    return a*((1-t)*x0+t*x1)+b-(x1-x0)


def ccs(losses, observed):
    if not losses or len(losses[0]) < 2 or not 0 <= observed < len(losses[0]):
        raise ValueError('at least two labels and a valid observed index required')
    if any(len(row)!=len(losses[0]) or any(not math.isfinite(x) or x < 0 for x in row) for row in losses):
        raise ValueError('rectangular finite nonnegative losses required')
    margins=[row[observed]-min(x for j,x in enumerate(row) if j!=observed) for row in losses]
    return sum(margins)/len(margins)


def weight(score, temperature=1):
    if not math.isfinite(score) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite score and positive finite temperature required')
    z=score/temperature
    return math.exp(-z)/(1+math.exp(-z)) if z>=0 else 1/(1+math.exp(z))


def run_experiment():
    rows=[]
    for t in (0,0.25,0.5,0.75,1):
        clean=residual(t,0,1,2,0)
        noisy=residual(t,0,2,2,0)
        rows.append({'t':t,'state_shift':t,'clean':clean,'corrupted':noisy,
                     'difference':noisy-clean,'expected_difference':2*t-1})
    losses=[[2,1,9],[2,9,1]]
    correct=ccs(losses,0)
    wrong=2-min(5,5)
    n=10000
    late_mean=sum(math.sqrt((i+0.5)/n) for i in range(n))/n
    checks={
        'affine_residual_identity':all(abs(r['difference']-r['expected_difference'])<1e-12 for r in rows),
        'midpoint_cancellation':rows[2]['difference']==0 and rows[2]['state_shift']>0,
        'label_only_changes_field':residual(0.25,0,1,2,3)-residual(0.25,0,1,2,0)==3,
        'compare_before_time_average':correct==1 and wrong==-3,
        'weight_direction_and_tie':weight(-1)>weight(0)==0.5>weight(1),
        'late_time_mean':abs(late_mean-2/3)<1e-6,
    }
    return {'config':{'seed':None,'x0':0,'x1':1,'endpoint_delta':1,'a':2,'b':0,'temperature':1,'quadrature_n':n},
            'residuals':rows,'label_losses':losses,'ccs':correct,'wrong_order_score':wrong,
            'weights':{str(s):weight(s) for s in (-1,0,1)},'late_time_mean':late_mean,
            'checks':checks,'passed':all(checks.values()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    r=run_experiment();s=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(s)
    print(s)
    raise SystemExit(0 if r['passed'] else 1)
