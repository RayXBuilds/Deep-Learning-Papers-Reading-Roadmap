from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import unittest

from ctc_alignment import collapse, enumerate_transcripts, forward, measure


class CTCAlignmentTests(unittest.TestCase):
    def test_blank_and_repeats(self):
        self.assertEqual(collapse("AA_BBA"), "ABA")
        self.assertEqual(collapse("AA_A"), "AA")
        self.assertEqual(forward([], ""), 1)
        self.assertEqual(forward([], "A"), 0)

    def test_forward_against_every_path(self):
        frames = [{"_": 0.2, "A": 0.3, "B": 0.5}, {"_": 0.1, "A": 0.7, "B": 0.2}] * 2
        totals = enumerate_transcripts(frames)
        self.assertAlmostEqual(sum(totals.values()), 1, delta=1e-12)
        for length in range(5):
            for chars in product("AB", repeat=length):
                s = "".join(chars)
                self.assertAlmostEqual(forward(frames, s), totals.get(s, 0), delta=1e-12)

    def test_hand_masses_and_record(self):
        root = Path(__file__).parent
        report = json.loads((root / "results/ctc-alignment.json").read_text())
        self.assertEqual(report["source_sha256"], sha256((root / "ctc_alignment.py").read_bytes()).hexdigest())
        for result in (measure(), report["results"]):
            self.assertEqual(result["greedy_transcript"], "")
            self.assertEqual(result["best_transcript"], "A")
            self.assertEqual(result["double_A_two_frames"], 0)
            for method in ("forward", "enumerated"):
                for label, expected in {"": 0.216, "A": 0.688, "AA": 0.096}.items():
                    self.assertAlmostEqual(result[method][label], expected, delta=1e-12)
