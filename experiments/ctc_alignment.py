"""Small CTC forward sum for fixed frame probabilities; '_' denotes blank."""
from collections import defaultdict
from hashlib import sha256
from itertools import groupby, product
import json
from math import prod
from pathlib import Path


def collapse(path):
    return "".join(symbol for symbol, _ in groupby(path) if symbol != "_")


def enumerate_transcripts(frames):
    totals = defaultdict(float)
    for path in product(*frames):
        totals[collapse(path)] += prod(row[symbol] for row, symbol in zip(frames, path))
    return dict(totals)


def forward(frames, transcript):
    if "_" in transcript:
        raise ValueError("blank cannot occur in a transcript")
    states = ["_"]
    for symbol in transcript:
        states.extend((symbol, "_"))
    previous = [1.0] + [0.0] * (len(states) - 1)
    for row in frames:
        current = []
        for i, symbol in enumerate(states):
            mass = previous[i]
            if i > 0:
                mass += previous[i - 1]
            if i >= 2 and symbol != "_" and symbol != states[i - 2]:
                mass += previous[i - 2]
            current.append(mass * row.get(symbol, 0.0))
        previous = current
    return sum(previous[-2:])


def measure():
    frames = [{"_": 0.6, "A": 0.4}] * 3
    totals = enumerate_transcripts(frames)
    greedy = collapse([max(row, key=row.get) for row in frames])
    return {"greedy_transcript": greedy, "best_transcript": max(totals, key=totals.get),
            "enumerated": totals, "forward": {s: forward(frames, s) for s in totals},
            "double_A_two_frames": forward(frames[:2], "AA")}


if __name__ == "__main__":
    print(json.dumps({"source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                      "results": measure()}, indent=2))
