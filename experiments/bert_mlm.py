#!/usr/bin/env python3
"""Fixed-logit MLM supervision checks; no encoder, sampling or training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def mlm_loss(logits, labels, weights, epsilon=1e-5):
    if not logits or len(logits) != len(labels) or len(labels) != len(weights):
        raise ValueError('matching nonempty rows, labels and weights required')
    width = len(logits[0])
    if not width or any(len(row) != width for row in logits):
        raise ValueError('rectangular logits required')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('positive finite epsilon required')
    losses = []
    for row, label, weight in zip(logits, labels, weights):
        if type(label) is not int or not 0 <= label < width:
            raise ValueError('labels must be valid vocabulary indices, including padded slots')
        if not all(math.isfinite(v) for v in row) or not math.isfinite(weight) or weight < 0:
            raise ValueError('finite logits and nonnegative finite weights required')
        maximum = max(row)
        losses.append(math.log(sum(math.exp(v-maximum) for v in row)) + (maximum-row[label]))
    return sum(w*l for w,l in zip(weights,losses))/(sum(weights)+epsilon)


def run_experiment():
    # Slots: masked, selected-but-unchanged, random replacement, padding/unselected.
    logits = [[math.log(3),0,0], [0,math.log(2),0], [0,0,math.log(4)], [5,-2,1]]
    labels, weights = [0,1,2,0], [1,1,1,0]
    base = mlm_loss(logits,labels,weights)
    unselected = mlm_loss(logits,[0,1,2,2],weights)
    unchanged = mlm_loss(logits,[0,0,2,0],weights)
    random_label = mlm_loss(logits,[0,1,0,0],weights)
    padding_logits = logits[:3]+[[-100,100,0]]
    masked_only = mlm_loss(logits,labels,[1,0,0,0])
    reference = (math.log(5/3)+math.log(2)+math.log(1.5))/(3+1e-5)
    metrics = {'loss':base,'hand_reference':reference,'unselected_label_delta':unselected-base,
               'unchanged_selected_label_delta':unchanged-base,'random_selected_label_delta':random_label-base,
               'padding_logits_delta':mlm_loss(padding_logits,labels,weights)-base,
               'padding_removed_delta':mlm_loss(logits[:3],labels[:3],weights[:3])-base,
               'masked_only_loss':masked_only,'all_zero_weight_loss':mlm_loss(logits,labels,[0]*4)}
    checks = {'matches_hand_reference':abs(base-reference)<1e-12,
              'unselected_label_is_ignored':abs(unselected-base)<1e-12,
              'unchanged_selected_label_matters':unchanged-base>0.1,
              'random_selected_label_matters':random_label-base>0.1,
              'padding_excluded_from_numerator_and_denominator':abs(metrics['padding_logits_delta'])<1e-12 and abs(metrics['padding_removed_delta'])<1e-12,
              'mask_id_only_is_wrong':abs(masked_only-base)>0.01,
              'zero_weights_return_zero':metrics['all_zero_weight_loss']==0}
    return {'config':{'logits':logits,'labels':labels,'weights':weights,'epsilon':1e-5,'seed':None,
                      'scope':'fixed prediction logits; slot labels supplied explicitly; no input corruption implementation'},
            'metrics':metrics,'checks':checks,'passed':all(checks.values()),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python':platform.python_version(),'platform':platform.platform()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    r=run_experiment()
    if args.output:
        args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r,indent=2))
    return 0 if r['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
