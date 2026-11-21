"""Compare the reward and reference-corrected policy preference probabilities."""
import hashlib
import json
import math
import platform
from itertools import combinations
from pathlib import Path


def optimal_policy(reference, rewards, beta):
    logits = [math.log(p) + r / beta for p, r in zip(reference, rewards)]
    weights = [math.exp(x - max(logits)) for x in logits]
    return [w / sum(weights) for w in weights]


def preference(policy, reference, winner, loser, beta):
    advantage = math.log(policy[winner] / reference[winner])
    advantage -= math.log(policy[loser] / reference[loser])
    return 1 / (1 + math.exp(-beta * advantage))


def run():
    reference, rewards = [0.8, 0.15, 0.05], [-0.5, 0, 1]
    errors, uncorrected = [], []
    for beta in (0.25, 1, 4):
        policy = optimal_policy(reference, rewards, beta)
        for w, l in combinations(range(3), 2):
            target = 1 / (1 + math.exp(rewards[l] - rewards[w]))
            errors.append(abs(preference(policy, reference, w, l, beta) - target))
            uncorrected.append(abs(preference(policy, [1/3]*3, w, l, beta) - target))
    return {"seed": None, "reference": reference, "rewards": rewards,
            "betas": [0.25, 1, 4], "max_abs_error": max(errors),
            "omitted_reference_max_error": max(uncorrected),
            "passed": max(errors) < 1e-12 and max(uncorrected) > 0.5}


if __name__ == "__main__":
    result = run()
    result.update(python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
