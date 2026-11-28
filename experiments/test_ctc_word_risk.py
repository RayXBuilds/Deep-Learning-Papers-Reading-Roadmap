import hashlib
import json
import math
from pathlib import Path
import unittest

from ctc_word_risk import evaluate, measure, word_error


class WordRiskTests(unittest.TestCase):
    def test_edits(self):
        for text, expected in [("A B", 0), ("A ", 0.5), ("", 1), ("AB", 1), ("A C B", 0.5)]:
            self.assertEqual(word_error("A B", text), expected)

    def test_logit_derivatives(self):
        h = 1e-5
        for row in measure():
            minus, plus = [evaluate(row["space"], 1 / (1 + math.exp(-z))) for z in (-h, h)]
            self.assertAlmostEqual((plus[1] - minus[1]) / (2*h), row["wer_logit_gradient"], delta=1e-9)
            self.assertAlmostEqual(math.log(minus[0] / plus[0]) / (2*h), row["nll_logit_gradient"], delta=1e-9)

    def test_results_and_sources(self):
        root = Path(__file__).parent
        saved = json.loads((root / "results/ctc-word-risk.json").read_text())
        for name in ("ctc_word_risk.py", "ctc_alignment.py"):
            self.assertEqual(saved["source_sha256"][name], hashlib.sha256((root / name).read_bytes()).hexdigest())
        self.assertEqual(saved["rows"], measure())
        for row, p in zip(measure(), (0.25, 0.5, 0.75), strict=True):
            self.assertEqual(row["space"], p)
            self.assertEqual(row["b"], 0.5)
            self.assertEqual(row["reference_probability"], p/2)
            self.assertEqual(row["expected_wer"], 0.75-p/2)
            self.assertEqual(row["wer_logit_gradient"], (0.5-p)/4)
