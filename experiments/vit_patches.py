#!/usr/bin/env python3
"""Fixed-input patch projection and attention permutation checks; not ViT training."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path


def image_shape(image, patch):
    if type(patch) is not int or patch <= 0:
        raise ValueError('positive integer patch size required')
    if not image or not image[0] or not image[0][0]:
        raise ValueError('nonempty HWC image required')
    h, w, c = len(image), len(image[0]), len(image[0][0])
    if any(len(row) != w or any(len(pixel) != c for pixel in row) for row in image):
        raise ValueError('rectangular HWC image required')
    if h % patch or w % patch:
        raise ValueError('image dimensions must be divisible by patch size; no implicit crop')
    if not all(math.isfinite(v) for row in image for pixel in row for v in pixel):
        raise ValueError('finite pixels required')
    return h, w, c


def patchify(image, patch):
    """Raster patch order, within each patch: row, column, channel (HWC)."""
    h, w, c = image_shape(image, patch)
    return [[image[y+i][x+j][ch] for i in range(patch) for j in range(patch) for ch in range(c)]
            for y in range(0, h, patch) for x in range(0, w, patch)]


def matrix_shape(rows):
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('nonempty rectangular matrix required')
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError('finite matrix required')
    return len(rows), len(rows[0])


def project(patches, weights):
    _, k = matrix_shape(patches)
    k2, d = matrix_shape(weights)
    if k != k2:
        raise ValueError('incompatible projection dimensions')
    return [[sum(p[t]*weights[t][j] for t in range(k)) for j in range(d)] for p in patches]


def conv_reference(image, kernel, patch):
    """Direct valid cross-correlation, kernel [p,p,c,d], stride p, no bias."""
    h, w, c = image_shape(image, patch)
    d = len(kernel[0][0][0])
    if len(kernel) != patch or any(len(row) != patch for row in kernel) or any(
            len(pixel) != c or any(len(out) != d for out in pixel) for row in kernel for pixel in row):
        raise ValueError('incompatible kernel dimensions')
    result = []
    for y in range(0, h, patch):
        for x in range(0, w, patch):
            out = [0.0]*d
            for ch in range(c):
                for j in range(patch):
                    for i in range(patch):
                        for k in range(d):
                            out[k] += image[y+i][x+j][ch]*kernel[i][j][ch][k]
            result.append(out)
    return result


def attention(tokens):
    """One head with identity Q/K/V projections, no mask, dropout, LN or MLP."""
    _, d = matrix_shape(tokens)
    outputs = []
    for query in tokens:
        scores = [sum(a*b for a, b in zip(query, key))/math.sqrt(d) for key in tokens]
        exponentials = [math.exp(s-max(scores)) for s in scores]
        weights = [e/sum(exponentials) for e in exponentials]
        outputs.append([sum(w*row[k] for w, row in zip(weights, tokens)) for k in range(d)])
    return outputs


def add(a, b):
    if matrix_shape(a) != matrix_shape(b):
        raise ValueError('incompatible addition dimensions')
    return [[x+y for x, y in zip(u, v)] for u, v in zip(a, b)]


def error(a, b):
    if matrix_shape(a) != matrix_shape(b):
        raise ValueError('incompatible comparison dimensions')
    return max(abs(x-y) for u, v in zip(a, b) for x, y in zip(u, v))


def run_experiment():
    image = [[[(y*8+x*2+c+1)/32 for c in range(2)] for x in range(4)] for y in range(4)]
    weights = [[(t+1)/10, (-1)**t*0.2] for t in range(8)]
    patches = patchify(image, 2)
    projected = project(patches, weights)
    kernel = [[[weights[(i*2+j)*2+c] for c in range(2)] for j in range(2)] for i in range(2)]
    convolved = conv_reference(image, kernel, 2)
    # Deliberately wrong: flatten channel first but retain the HWC weight mapping.
    wrong = [[image[y+i][x+j][c] for c in range(2) for i in range(2) for j in range(2)]
             for y in (0, 2) for x in (0, 2)]
    tokens = [[0.2, -0.1]] + projected
    positions = [[0, 0], [0.1, 0.2], [-0.3, 0.1], [0.4, -0.2], [-0.1, -0.4]]
    permutation = [0, 3, 1, 4, 2]  # Keep CLS at row zero.
    permute = lambda rows: [rows[i] for i in permutation]
    no_pos = attention(tokens)
    shuffled = attention(permute(tokens))
    positioned = attention(add(tokens, positions))
    content_only = attention(add(permute(tokens), positions))
    joint = attention(add(permute(tokens), permute(positions)))
    metrics = {
        'projection_conv_error': error(projected, convolved),
        'wrong_layout_error': error(project(wrong, weights), convolved),
        'no_position_equivariance_error': error(shuffled, permute(no_pos)),
        'no_position_cls_error': error(shuffled[:1], no_pos[:1]),
        'fixed_position_cls_change': error(content_only[:1], positioned[:1]),
        'joint_permutation_equivariance_error': error(joint, permute(positioned)),
        'joint_permutation_cls_error': error(joint[:1], positioned[:1]),
        'token_counts_224': {'patch16': 197, 'patch32': 50},
        'attention_score_entries_ratio': 197**2/50**2,
    }
    checks = {
        'patch_projection_matches_convolution': metrics['projection_conv_error'] < 1e-12,
        'wrong_channel_layout_detected': metrics['wrong_layout_error'] > 0.01,
        'no_position_attention_is_equivariant': metrics['no_position_equivariance_error'] < 1e-12,
        'no_position_cls_is_invariant': metrics['no_position_cls_error'] < 1e-12,
        'fixed_positions_distinguish_content_permutation': metrics['fixed_position_cls_change'] > 1e-4,
        'joint_content_position_permutation_is_equivariant': metrics['joint_permutation_equivariance_error'] < 1e-12,
        'joint_content_position_cls_is_invariant': metrics['joint_permutation_cls_error'] < 1e-12,
    }
    return {'config': {'seed': None, 'randomness': 'none', 'image_hwc': image, 'patch_size': 2,
                       'projection_weights': weights, 'positions': positions, 'permutation': permutation,
                       'cls': tokens[0], 'data': 'fixed synthetic image; no training or dataset'},
            'patches': patches, 'projected': projected, 'convolved': convolved,
            'attention_outputs': {'no_position': no_pos, 'shuffled_no_position': shuffled,
                                  'positioned': positioned, 'content_only': content_only, 'joint': joint},
            'metrics': metrics, 'checks': checks, 'passed': all(checks.values()),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_experiment()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'metrics': report['metrics'], 'checks': report['checks']}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
