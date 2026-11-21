"""Online softmax reduction of one attention row, using scalar values."""
import hashlib
import json
import math
import platform
import random
from pathlib import Path


def dense(scores, values):
    weights = [math.exp(s - max(scores)) for s in scores]
    return math.fsum(w * v for w, v in zip(weights, values)) / math.fsum(weights)


def tiled(scores, values, width):
    if not scores or len(scores) != len(values) or width < 1:
        raise ValueError("expected matching nonempty rows and a positive tile size")
    peak = -math.inf
    mass = numerator = 0.0
    for start in range(0, len(scores), width):
        block = scores[start:start + width]
        new_peak = max(peak, max(block))
        rescale = math.exp(peak - new_peak)
        weights = [math.exp(s - new_peak) for s in block]
        mass = mass * rescale + sum(weights)
        numerator = numerator * rescale + sum(
            w * values[start + i] for i, w in enumerate(weights)
        )
        peak = new_peak
    return numerator / mass


def run():
    errors = []
    for seed in (7, 42, 2026):
        rng = random.Random(seed)
        scores = [1000 + rng.uniform(-20, 20) for _ in range(17)]
        values = [rng.uniform(-1, 1) for _ in scores]
        for width in (1, 4, 8, 17):
            errors.append(abs(tiled(scores, values, width) - dense(scores, values)))
    scores, values = [0, 0, 10, 10], [0, 0, 1, 1]
    local_mean = (dense(scores[:2], values[:2]) + dense(scores[2:], values[2:])) / 2
    bias = abs(local_mean - dense(scores, values))
    return {"seeds": [7, 42, 2026], "widths": [1, 4, 8, 17], "row_length": 17,
            "max_abs_error": max(errors), "local_mean_error": bias,
            "passed": max(errors) < 1e-12 and bias > 0.49}


if __name__ == "__main__":
    result = run()
    result.update(python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
