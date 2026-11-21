"""Linear depthwise/pointwise filters and FPN RoI assignment on synthetic inputs."""
import hashlib
import json
import math
import random
from pathlib import Path


def separated(patch, depthwise, pointwise):
    """One valid output site, with flattened spatial coordinates per channel."""
    filtered = [sum(x * d for x, d in zip(channel, kernel))
                for channel, kernel in zip(patch, depthwise)]
    return [sum(p * value for p, value in zip(row, filtered)) for row in pointwise]


def dense(patch, kernels):
    return [sum(x * w for channel, kernel in zip(patch, bank)
                for x, w in zip(channel, kernel)) for bank in kernels]


def convolution_trial(seed):
    rng = random.Random(seed)
    channels, outputs, taps = 3, 4, 9
    depthwise = [[rng.uniform(-1, 1) for _ in range(taps)] for _ in range(channels)]
    pointwise = [[rng.uniform(-1, 1) for _ in range(channels)] for _ in range(outputs)]
    kernels = [[[scale * d for d in kernel] for scale, kernel in zip(row, depthwise)]
               for row in pointwise]
    errors = []
    for _ in range(16):
        patch = [[rng.uniform(-1, 1) for _ in range(taps)] for _ in range(channels)]
        errors.extend(abs(a - b) for a, b in zip(separated(patch, depthwise, pointwise),
                                                 dense(patch, kernels)))
    # At one input channel every pair of output kernels shares its spatial factor.
    first, second = kernels[0][0], kernels[1][0]
    minor = first[0] * second[1] - first[1] * second[0]
    return dict(seed=seed, max_error=max(errors), minor=minor,
                dense_macs=outputs * channels * taps,
                separated_macs=channels * (taps + outputs))


def roi_level(width, height):
    if min(width, height) <= 0:
        raise ValueError("positive RoI dimensions required")
    raw = math.floor(4 + math.log2(math.sqrt(width * height) / 224))
    return max(2, min(5, raw))


def routing_trial():
    boxes = [(28, 28), (56, 56), (112, 112), (223, 223), (224, 224),
             (448, 448), (896, 896), (7, 7168)]
    rows = []
    for width, height in boxes:
        level = roi_level(width, height)
        scale = math.sqrt(width * height)
        rows.append(dict(width=width, height=height, level=level,
                         fixed_p4_span=scale / 16, routed_span=scale / 2**level,
                         short_side_cells=min(width, height) / 2**level))
    return rows


def report():
    conv = [convolution_trial(seed) for seed in (7, 42, 2026)]
    routing = routing_trial()
    checks = {
        "linear_equivalence": all(row["max_error"] < 1e-12 for row in conv),
        "factor_minor_zero": all(abs(row["minor"]) < 1e-12 for row in conv),
        "routing_boundaries": [row["level"] for row in routing] == [2, 2, 3, 3, 4, 5, 5, 4],
        "octave_span": all(routing[i]["routed_span"] == 14 for i in (1, 2, 4, 5)),
        "thin_box": routing[-1]["short_side_cells"] < 1,
    }
    return dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                convolution=conv, routing=routing, checks=checks, passed=all(checks.values()))


if __name__ == "__main__":
    result = report()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
