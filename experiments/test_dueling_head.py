import hashlib
import json
from pathlib import Path
import unittest

from dueling_head import fit_one_action, measure, q_values


class DuelingHeadTests(unittest.TestCase):
    def test_mean_and_shift(self):
        self.assertEqual(q_values(5, [2, 5, 8]), [2, 5, 8])
        self.assertEqual(q_values(5, [12, 15, 18]), [2, 5, 8])
        self.assertEqual(q_values(5, [-7]), [5])

    def test_selected_action_and_permutation(self):
        for action in range(4):
            baseline, duel = fit_one_action(4, action, -2)
            self.assertEqual(baseline[action], -2)
            for i, value in enumerate(duel):
                self.assertAlmostEqual(value, -2 if i == action else -6 / 7, delta=1e-12)

    def test_probes_and_provenance(self):
        root = Path(__file__).parent
        saved = json.loads((root / "results/dueling-head.json").read_text())
        digest = hashlib.sha256((root / "dueling_head.py").read_bytes()).hexdigest()
        self.assertEqual(saved["source_sha256"], digest)
        for result in (saved["results"], measure()):
            self.assertEqual(result["direct"], [1, 0, 0, 0])
            for actual, expected in zip(result["duel"], [1, 3/7, 3/7, 3/7], strict=True):
                self.assertAlmostEqual(actual, expected, delta=1e-12)
            for name, mse in [("shared", 16/49), ("opposed", 100/49)]:
                self.assertEqual(result["held_out_mse"][name]["direct"], 1)
                self.assertAlmostEqual(result["held_out_mse"][name]["duel"], mse, delta=1e-12)
