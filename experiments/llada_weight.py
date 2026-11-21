"""Exact expectation of masked NLL over all Bernoulli masks."""
import hashlib
import itertools
import json
import math
from pathlib import Path


def expected_loss(losses, t, weighted=True, neighbor_penalty=0.0):
    if not 0 < t <= 1:
        raise ValueError("mask probability must be in (0, 1]")
    total = 0.0
    for mask in itertools.product((0, 1), repeat=len(losses)):
        count = sum(mask)
        probability = t ** count * (1 - t) ** (len(mask) - count)
        selected_loss = sum(loss for bit, loss in zip(mask, losses) if bit)
        # Hiding site 1 halves the correct-token probability at site 0
        # when neighbor_penalty is log(2). Site 0 must itself be supervised.
        if len(mask) >= 2 and mask[0] and mask[1]:
            selected_loss += neighbor_penalty
        total += probability * selected_loss
    return total / t if weighted else total


if __name__ == "__main__":
    losses = [-math.log(p) for p in (0.25, 0.5, 0.8)]
    reference = sum(losses)
    rows = [{"t": t, "weighted": expected_loss(losses, t),
             "context_dependent": expected_loss(losses, t, neighbor_penalty=math.log(2)),
             "unweighted": expected_loss(losses, t, False)} for t in (0.125, 0.5, 1)]
    passed = all(abs(r["weighted"] - reference) < 1e-12 and
                 abs(r["unweighted"] - r["t"] * reference) < 1e-12 and
                 abs(r["context_dependent"] - reference - r["t"] * math.log(2)) < 1e-12
                 for r in rows)
    print(json.dumps({"reference": reference, "rows": rows, "passed": passed,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                     indent=2))
    raise SystemExit(0 if passed else 1)
