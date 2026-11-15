#!/usr/bin/env python3
"""Deterministic sample-indexed TRS buffer teaching controls, not model training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def update(scores, observations, beta=0.5):
    if not 0 <= beta < 1:
        raise ValueError('beta must be in [0,1)')
    result=list(scores)
    for index, loss in observations.items():
        if not isinstance(index,int) or not 0 <= index < len(scores) or not math.isfinite(loss) or loss < 0:
            raise ValueError('valid index and nonnegative finite loss required')
        result[index]=loss if scores[index] is None else beta*scores[index]+(1-beta)*loss
    return result


def quantile(values, p):
    if not values or not 0 <= p <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError('finite nonempty values and quantile in [0,1] required')
    ordered=sorted(values);pos=(len(values)-1)*p;lo=int(pos);hi=min(lo+1,len(values)-1)
    return ordered[lo]+(pos-lo)*(ordered[hi]-ordered[lo])


def weights(scores, warmup=False, p=0.5, temperature=1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    seen=[s for s in scores if s is not None]
    if not seen:
        return [1.0]*len(scores),None
    center=quantile(seen,p)
    result=[]
    for s in scores:
        if warmup or s is None:
            result.append(1.0)
        else:
            z=(s-center)/temperature
            result.append(math.exp(-z)/(1+math.exp(-z)) if z>=0 else 1/(1+math.exp(z)))
    return result,center


def run_experiment():
    first=update([None]*4,{0:1,1:3})
    warm,_=weights(first,True)
    second=update(first,{0:5,2:9})
    active,center=weights(second)
    wrong_center=quantile([s if s is not None else 0 for s in first],0.5)
    checks={'first_visit_uses_observation':first==[1,3,None,None],
            'sample_indexed_ema':second==[3,3,9,None],
            'unseen_excluded_from_quantile':weights(first)[1]==2 and wrong_center==0.5,
            'warmup_keeps_unit_weights':warm==[1,1,1,1],
            'high_score_downweighted':center==3 and active[0]==active[1]==0.5 and active[2]<0.01,
            'unseen_has_no_reliability_claim':active[3]==1}
    return {'config':{'seed':None,'beta':0.5,'quantile':0.5,'temperature':1},
            'observations':[{0:1,1:3},{0:5,2:9}], 'first':first,'second':second,
            'warmup_weights':warm,'active_weights':active,'center':center,
            'wrong_zero_filled_center':wrong_center,'checks':checks,'passed':all(checks.values()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    r=run_experiment();s=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if args.output: args.output.write_text(s)
    print(s)
    raise SystemExit(0 if r['passed'] else 1)
