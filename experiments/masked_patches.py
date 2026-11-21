"""Audit MAE token restoration and reconstruction loss on scalar patches."""
import hashlib
import json
import math
from pathlib import Path
import random


def restore(visible, order, mask_token=None):
    if sorted(order) != list(range(len(order))) or len(visible) > len(order):
        raise ValueError("expected a permutation of all positions")
    result = [mask_token] * len(order)
    for index, value in zip(order, visible):
        result[index] = value
    return result


def masked_mse(prediction, target, masked):
    errors = [(p - t) ** 2 for p, t, m in zip(prediction, target, masked, strict=True) if m]
    if not errors:
        raise ValueError("an empty mask has no mean reconstruction error")
    return sum(errors) / len(errors)


def measure():
    target = list(range(16))
    runs = []
    for seed in (7, 42, 2026):
        order = target.copy()
        random.Random(seed).shuffle(order)
        visible = order[:4]
        layout = restore([target[i] for i in visible], order)
        masked = [value is None for value in layout]
        prediction = [y + (1 if m else 10) for y, m in zip(target, masked)]
        full = masked_mse(prediction, target, [True] * len(target))
        hidden = masked_mse(prediction, target, masked)
        restored = all(value == (target[i] if i in visible else None)
                       for i, value in enumerate(layout))
        runs.append({"seed": seed, "visible": visible, "restored": restored,
                     "masked_mse": hidden, "all_patch_mse": full})
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs, "passed": all(r["restored"] and r["masked_mse"] == 1
                                     and math.isclose(r["all_patch_mse"], 25.75)
                                     for r in runs),
    }


if __name__ == "__main__":
    report = measure()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
