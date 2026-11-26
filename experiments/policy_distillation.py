"""Exact one-step decisions for a student that cannot distinguish two states."""
import hashlib
import json
from pathlib import Path
from statistics import fmean

from run import softmax


def evaluate(rows, action):
    return {
        "agreement": fmean(row[action] == max(row) for row in rows),
        "mean_return": fmean(row[action] for row in rows),
    }


def fit_shared_student(rows, temperature):
    targets = [softmax([q / temperature for q in row]) for row in rows]
    # The KL minimizer for one shared student distribution is the mean target.
    p_left = fmean(target[0] for target in targets)
    action = 0 if p_left >= 0.5 else 1
    return {"p_left": p_left, "action": action, **evaluate(rows, action)}


def measure():
    rows = [(0.01, 0)] * 3 + [(0, 0.1)]
    hard_left = fmean(row[0] > row[1] for row in rows)
    mean_q = [fmean(column) for column in zip(*rows)]
    mse_action = mean_q.index(max(mean_q))
    return {
        "teacher_rows": rows,
        "nll": {"p_left": hard_left, **evaluate(rows, 0 if hard_left >= 0.5 else 1)},
        "mse": {"mean_q": mean_q, **evaluate(rows, mse_action)},
        "kl": [{"temperature": t, **fit_shared_student(rows, t)} for t in (0.1, 0.01)],
    }


if __name__ == "__main__":
    root = Path(__file__).parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("policy_distillation.py", "run.py")}
    print(json.dumps({"source_sha256": hashes, "results": measure()}, indent=2))
