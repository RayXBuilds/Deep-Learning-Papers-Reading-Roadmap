"""Exact expected WER for a tiny character CTC distribution."""
import hashlib
import json
from pathlib import Path

from ctc_alignment import enumerate_transcripts


def word_error(reference, hypothesis):
    words = reference.split()
    if not words:
        raise ValueError("WER requires a nonempty reference")
    predicted = hypothesis.split()
    previous = list(range(len(predicted) + 1))
    for i, word in enumerate(words, 1):
        current = [i]
        for j, token in enumerate(predicted, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (word != token)))
        previous = current
    return previous[-1] / len(words)


def evaluate(space, final_b):
    frames = [{"A": 1.0}, {" ": space, "_": 1 - space}, {"B": final_b, "_": 1 - final_b}]
    distribution = enumerate_transcripts(frames)
    risk = sum(p * word_error("A B", text) for text, p in distribution.items())
    return distribution["A B"], risk


def measure():
    rows = []
    for space in (0.25, 0.5, 0.75):
        b = 0.5
        probability, risk = evaluate(space, b)
        _, with_b = evaluate(space, 1)
        _, without_b = evaluate(space, 0)
        rows.append({"space": space, "b": b, "reference_probability": probability,
                     "expected_wer": risk, "wer_logit_gradient": b * (1-b) * (with_b - without_b),
                     "nll_logit_gradient": b - 1})
    return rows


if __name__ == "__main__":
    root = Path(__file__).parent
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("ctc_word_risk.py", "ctc_alignment.py")}
    print(json.dumps({"source_sha256": hashes, "rows": measure()}, indent=2))
