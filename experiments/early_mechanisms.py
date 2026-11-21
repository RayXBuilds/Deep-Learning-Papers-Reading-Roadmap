"""Exact CPU controls for dropout scaling and stacked convolution support."""
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path


def dropout_means(contributions, bias, keep):
    """Average each subnetwork using its Bernoulli probability."""
    linear = rectified = 0.0
    for mask in product((0, 1), repeat=len(contributions)):
        retained = sum(mask)
        probability = keep**retained * (1 - keep)**(len(mask) - retained)
        value = bias + sum(w * m for w, m in zip(contributions, mask))
        linear += probability * value
        rectified += probability * max(0, value)
    return linear, rectified


def impulse_response(radii):
    """Propagate an impulse on an infinite integer grid with all-one kernels."""
    response = {(0, 0): 1}
    for radius in radii:
        expanded = defaultdict(int)
        offsets = range(-radius, radius + 1)
        for (x, y), value in response.items():
            for dx, dy in product(offsets, repeat=2):
                expanded[x + dx, y + dy] += value
        response = expanded
    return response


def measure():
    linear, relu = dropout_means((2, -4), 1, 0.5)
    scaled = 1 + 0.5 * (2 - 4)
    positive = dropout_means((2, -4), 5, 0.5)
    stacked = impulse_response([1, 1, 1])
    direct = impulse_response([3])
    checks = {
        "affine_mean": linear == scaled,
        "nonlinear_gap": relu == 1 and relu > max(0, scaled),
        "positive_control": positive == (4, 4),
        "support": stacked.keys() == direct.keys() and len(stacked) == 49,
        "path_counts": stacked[0, 0] == 49 and stacked[3, 3] == direct[0, 0] == 1,
        "total_mass": sum(stacked.values()) == 729 and sum(direct.values()) == 49,
    }
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": None,
        "dropout": {"linear_mean": linear, "relu_mean": relu, "scaled": scaled,
                    "positive_means": positive},
        "convolution": {"support": len(stacked), "center": stacked[0, 0],
                        "corner": stacked[3, 3], "stack_mass": sum(stacked.values()),
                        "direct_mass": sum(direct.values())},
        "checks": checks,
    }


if __name__ == "__main__":
    report = measure()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(report["checks"].values()) else 1)
