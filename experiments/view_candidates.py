"""Audit SimCLR's per-anchor candidate set on unit-length synthetic views."""
import hashlib
import json
import math
from pathlib import Path


def anchor_loss(views, anchor, partner, temperature, *, include_self=False):
    if temperature <= 0 or anchor == partner:
        raise ValueError("expected positive temperature and distinct paired views")
    keys = [i for i in range(len(views)) if include_self or i != anchor]
    scores = {
        k: sum(x * y for x, y in zip(views[anchor], views[k], strict=True)) / temperature
        for k in keys
    }
    peak = max(scores.values())
    return peak - scores[partner] + math.log(sum(math.exp(s - peak) for s in scores.values()))


def measure():
    # Consecutive pairs are two identical views of each source.
    views = [(1, 0), (1, 0), (0, 1), (0, 1)]
    base = anchor_loss(views, 0, 1, 0.5)
    self_key = anchor_loss(views, 0, 1, 0.5, include_self=True)
    hard = anchor_loss(views + [(1, 0), (1, 0)], 0, 1, 0.5)
    expected = [math.log(k + 2 * math.exp(-2)) for k in (1, 2, 3)]
    values = [base, self_key, hard]
    return {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "temperature": 0.5, "seed": None, "views": views,
        "loss": dict(zip(("base", "self_included", "same_direction_negative_pair"), values)),
        "passed": all(abs(x - y) < 1e-12 for x, y in zip(values, expected)),
    }


if __name__ == "__main__":
    report = measure()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
