"""Compare direct and mean-centered action values at equal observed-action fit."""
import hashlib
import json
from pathlib import Path
from statistics import fmean


def q_values(value, advantages):
    mean_advantage = fmean(advantages)
    return [value + (advantage - mean_advantage) for advantage in advantages]


def fit_one_action(action_count, action, target):
    # At zero initialization, move in the gradient direction of Q[action].
    # Divide by its squared norm to fit the same target exactly in both heads.
    slope = [float(i == action) - 1 / action_count for i in range(action_count)]
    step = target / (1 + sum(x * x for x in slope))
    duel = q_values(step, [step * x for x in slope])
    direct = [0.0] * action_count
    direct[action] = target
    return direct, duel


def measure():
    direct, duel = fit_one_action(4, 0, 1.0)
    probes = {}
    for name, targets in [("shared", [1, 1, 1]), ("opposed", [-1, -1, -1])]:
        probes[name] = {
            head: fmean((q - y) ** 2 for q, y in zip(values[1:], targets, strict=True))
            for head, values in [("direct", direct), ("duel", duel)]
        }
    return {"direct": direct, "duel": duel, "held_out_mse": probes}


if __name__ == "__main__":
    print(json.dumps({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "results": measure()}, indent=2))
