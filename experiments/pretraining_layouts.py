"""Exhaustive finite controls for XLNet visibility and T5 span serialization."""
import hashlib
import itertools
import json
from pathlib import Path


def visible(order, target, *, content=False):
    """Return original positions whose content may be read at target."""
    stop = order.index(target) + int(content)
    return frozenset(order[:stop])


def span_pair(tokens, missing):
    """Build encoder input and decoder labels (pre-EOS), with distinct sentinels."""
    if len(tokens) != len(missing):
        raise ValueError("mask length must match sequence")
    source, labels = [], []
    spans = 0
    for i, (token, hidden) in enumerate(zip(tokens, missing)):
        if not hidden:
            source.append(token)
            continue
        if i == 0 or not missing[i - 1]:
            sentinel = f"<extra_id_{spans}>"
            source.append(sentinel)
            labels.append(sentinel)
            spans += 1
        labels.append(token)
    labels.append(f"<extra_id_{spans}>")
    return source, labels


def recover(source, labels):
    """Replace each sentinel with the payload that follows it in decoder labels."""
    spans = {}
    for item in labels:
        if item.startswith("<extra_id_"):
            current = []
            spans[item] = current
        else:
            current.append(item)
    return [piece for item in source for piece in spans.get(item, [item])]


def measure():
    n = 4
    rows = leaks = baseline_leaks = right = 0
    for order in itertools.permutations(range(n)):
        for target in range(n):
            query = visible(order, target)
            rows += 1
            leaks += target in query
            baseline_leaks += target in visible(order, target, content=True)
            right += any(i > target for i in query)

    tokens = list("abcdef")
    masks = restored = length_matches = shorter = 0
    for mask in itertools.product([False, True], repeat=len(tokens)):
        if not any(mask) or all(mask):
            continue
        source, labels = span_pair(tokens, mask)
        hidden = sum(mask)
        # A new run begins on each transition from a visible to a hidden token.
        runs = int(mask[0]) + sum(b and not a for a, b in zip(mask, mask[1:]))
        masks += 1
        restored += recover(source, labels) == tokens
        length_matches += (len(source), len(labels)) == (
            len(tokens) - hidden + runs, hidden + runs + 1)
        shorter += len(labels) < len(tokens)
    return {
        "xlnet": {"queries": rows, "target_leaks": leaks, "content_self_reads": baseline_leaks,
                  "queries_with_right_context": right},
        "t5": {"mixed_masks": masks, "restored": restored, "length_matches": length_matches,
               "shorter_than_full_target": shorter},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": None,
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2, sort_keys=True))
