"""Compare a scalar-transition SSM scan with its explicit causal kernel."""
import hashlib
import json
import math
import random
from pathlib import Path


def dot(left, right):
    return sum(x * y for x, y in zip(left, right))


def scan(decays, writes, reads, inputs):
    state = [0.0] * len(writes[0])
    outputs = []
    for decay, write, read, value in zip(decays, writes, reads, inputs):
        state = [decay * old + b * value for old, b in zip(state, write)]
        outputs.append(dot(read, state))
    return outputs


def dense(decays, writes, reads, inputs):
    outputs = []
    for t, read in enumerate(reads):
        outputs.append(sum(
            dot(read, writes[s]) * math.prod(decays[s + 1:t + 1]) * inputs[s]
            for s in range(t + 1)
        ))
    return outputs


def run(seed):
    rng = random.Random(seed)
    length, width = 13, 3
    decays = [rng.uniform(0.1, 0.95) for _ in range(length)]
    decays[6] = 0.0
    writes, reads = ([[rng.uniform(-1, 1) for _ in range(width)]
                       for _ in range(length)] for _ in range(2))
    inputs = [rng.uniform(-1, 1) for _ in range(length)]
    baseline = scan(decays, writes, reads, inputs)
    kernel = dense(decays, writes, reads, inputs)
    changed = scan(decays, writes, reads, [x + 1 for x in inputs[:6]] + inputs[6:])
    return {
        "seed": seed,
        "max_kernel_error": max(abs(a - b) for a, b in zip(baseline, kernel)),
        "post_reset_error": max(abs(a - b) for a, b in zip(baseline[6:], changed[6:])),
        "pre_reset_effect": max(abs(a - b) for a, b in zip(baseline[:6], changed[:6])),
    }


if __name__ == "__main__":
    rows = [run(seed) for seed in (7, 19, 41)]
    passed = all(r["max_kernel_error"] < 1e-12 and r["post_reset_error"] == 0
                 and r["pre_reset_effect"] > 1e-6 for r in rows)
    print(json.dumps({"rows": rows, "passed": passed,
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                     indent=2))
    raise SystemExit(0 if passed else 1)
