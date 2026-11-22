"""Test the readout in Eq. 4 with fixed scores, without fitting an encoder."""
from hashlib import sha256
from itertools import permutations
import json
from math import exp, log
from pathlib import Path


def probabilities(scores):
    peak = max(scores)
    weights = [exp(score - peak) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def measure():
    # Log integer weights make the reference probabilities exact fractions.
    weights = (1, 2, 4)
    scores = [log(w) for w in weights]
    errors = []
    for order in permutations(range(3)):
        actual = probabilities([scores[j] for j in order])
        errors.extend(abs(p - weights[j] / 7) for j, p in zip(order, actual))

    extended = probabilities(scores + [log(8)])
    # Both distributions have mean coordinate 1 on positions (0, 1, 2).
    rows = [probabilities([log(w) for w in row]) for row in [(1, 6, 1), (3, 2, 3)]]
    contexts = [sum(j * p for j, p in enumerate(row)) for row in rows]
    losses = [-log(row[1]) for row in rows]
    return {
        "permutation_max_error": max(errors),
        "fourth_position_probability": extended[3],
        "contexts": contexts,
        "middle_target_losses": losses,
    }


if __name__ == "__main__":
    result = measure()
    print(json.dumps({"source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "measurements": result}, indent=2))
