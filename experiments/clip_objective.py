#!/usr/bin/env python3
"""Fixed-embedding CLIP objective controls; no encoders or training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def normalize(rows):
    if not rows or not rows[0] or any(len(r)!=len(rows[0]) for r in rows):
        raise ValueError('nonempty rectangular embeddings required')
    result=[]
    for row in rows:
        norm=math.sqrt(sum(v*v for v in row))
        if not math.isfinite(norm) or norm==0:
            raise ValueError('finite nonzero embeddings required')
        result.append([v/norm for v in row])
    return result


def scores(images,texts,scale):
    a,b=normalize(images),normalize(texts)
    if len(a[0])!=len(b[0]) or not math.isfinite(scale) or scale<=0:
        raise ValueError('matching dimensions and positive finite scale required')
    return [[scale*sum(x*y for x,y in zip(u,v)) for v in b] for u in a]


def probabilities(row):
    terms=[math.exp(v-max(row)) for v in row]
    return [v/sum(terms) for v in terms]


def objective(matrix):
    n=len(matrix)
    if not n or any(len(row)!=n for row in matrix) or not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError('finite square paired score matrix required')
    def ce(rows):
        return sum(math.log(sum(math.exp(v-max(row)) for v in row))+max(row)-row[i]
                   for i,row in enumerate(rows))/n
    return (ce(matrix)+ce(list(zip(*matrix))))/2


def run_experiment():
    images=[[1,0],[0,1]]
    texts=[[1,0],[0,1]]
    matrix=scores(images,texts,2)
    base=objective(matrix)
    wrong=objective(scores(images,list(reversed(texts)),2))
    duplicate=objective(scores([[1,0],[1,0]],[[1,0],[1,0]],2))
    low=probabilities([1,0])[0]
    high=probabilities([2,0])[0]
    expanded=probabilities([2,0,2])[0]
    scaled=objective(scores([[3,0],[0,7]],texts,2))
    asymmetric=[[2,0],[1,3]]
    transposed=objective(list(zip(*asymmetric)))
    metrics={'loss':base,'reference':math.log1p(math.exp(-2)),
             'text_only_shuffle_loss':wrong,'joint_shuffle_loss':objective(scores(list(reversed(images)),list(reversed(texts)),2)),
             'duplicate_loss':duplicate,'rescaling_error':abs(scaled-base),
             'transpose_error':abs(objective(asymmetric)-transposed),
             'probability_scale1':low,'probability_scale2':high,'probability_added_duplicate_candidate':expanded}
    checks={'hand_reference':abs(base-metrics['reference'])<1e-12,
            'text_only_shuffle_breaks_pairs':wrong-base>1,
            'joint_shuffle_preserves_pairs':abs(metrics['joint_shuffle_loss']-base)<1e-12,
            'duplicates_have_log2_loss':abs(duplicate-math.log(2))<1e-12,
            'positive_rescaling_removed_by_normalization':metrics['rescaling_error']<1e-12,
            'symmetric_loss_transpose_invariant':metrics['transpose_error']<1e-12,
            'scale_and_candidate_set_change_probabilities':high>low and expanded<high}
    return {'config':{'images':images,'texts':texts,'scale':2,'seed':None,'randomness':'none'},
            'metrics':metrics,'checks':checks,'passed':all(checks.values()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=run_experiment()
    if args.output:
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
