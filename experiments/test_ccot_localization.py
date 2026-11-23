import hashlib
import json
from math import sin, tau
from pathlib import Path
import unittest

from ccot_localization import coefficient, measure, score


class ContinuousResponseTests(unittest.TestCase):
    def test_phase_and_normalization(self):
        for size in (5, 9):
            sine = [sin(tau * j / size) for j in range(size)]
            self.assertLess(abs(coefficient(sine, 1) + 0.5j), 1e-12)
        self.assertAlmostEqual(score({1: -0.5j}, 0.0, 1), tau, delta=1e-12)

    def test_peak_and_alias_controls(self):
        result = measure()
        for row in result["peaks"]:
            self.assertGreater(row["grid_error"], 0.01)
            self.assertLess(row["refined_error"], 1e-12)
        self.assertLess(result["alias_sample_error"], 1e-12)
        self.assertAlmostEqual(result["alias_half_period_gap"], 2.0)

    def test_recorded_results(self):
        root = Path(__file__).parent
        report = json.loads((root / "results/ccot-localization.json").read_text())
        digest = hashlib.sha256((root / "ccot_localization.py").read_bytes()).hexdigest()
        self.assertEqual(report["source_sha256"], digest)
        expected = measure()
        saved = report["results"]
        for current, recorded in zip(expected["peaks"], saved["peaks"], strict=True):
            for key, value in current.items():
                self.assertAlmostEqual(value, recorded[key], delta=1e-12)
        for key in ("alias_sample_error", "alias_half_period_gap"):
            self.assertAlmostEqual(expected[key], saved[key], delta=1e-12)
