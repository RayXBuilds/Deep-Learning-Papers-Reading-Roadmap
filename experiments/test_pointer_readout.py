import json
import math
from hashlib import sha256
from pathlib import Path
import unittest

from pointer_readout import measure, probabilities


class PointerReadoutTests(unittest.TestCase):
    def test_dynamic_dictionary(self):
        self.assertEqual(probabilities([1000]), [1.0])
        for actual in probabilities([1000, 1000, 1000]):
            self.assertAlmostEqual(actual, 1 / 3, delta=1e-12)
        result = measure()
        self.assertLess(result["permutation_max_error"], 1e-12)
        self.assertAlmostEqual(result["fourth_position_probability"], 8 / 15, delta=1e-12)

    def test_context_collision(self):
        result = measure()
        for value in result["contexts"]:
            self.assertAlmostEqual(value, 1.0, delta=1e-12)
        for actual, expected in zip(result["middle_target_losses"], [math.log(4/3), math.log(4)], strict=True):
            self.assertAlmostEqual(actual, expected, delta=1e-12)

    def test_record(self):
        root = Path(__file__).parent
        record = json.loads((root / "results/pointer-readout.json").read_text())
        self.assertEqual(record["source_sha256"], sha256((root / "pointer_readout.py").read_bytes()).hexdigest())
        result = measure()
        for key, value in result.items():
            actual = record["measurements"][key]
            if isinstance(value, list):
                for x, y in zip(actual, value, strict=True):
                    self.assertAlmostEqual(x, y, delta=1e-12)
            else:
                self.assertAlmostEqual(actual, value, delta=1e-12)
