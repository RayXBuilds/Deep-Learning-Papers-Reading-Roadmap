import hashlib
import json
import math
from pathlib import Path
import unittest

from policy_distillation import fit_shared_student, measure


class PolicyDistillationTests(unittest.TestCase):
    def test_independent_cross_entropy_search(self):
        rows = [(0.01, 0)] * 3 + [(0, 0.1)]
        for t in (0.1, 0.01):
            targets = [1 / (1 + math.exp(-gap / t)) for gap in (0.01, 0.01, 0.01, -0.1)]

            def loss(index):
                p = index / 1000
                return -sum(y * math.log(p) + (1-y) * math.log1p(-p) for y in targets)

            grid_best = min(range(1, 1000), key=loss) / 1000
            result = fit_shared_student(rows, t)
            self.assertLess(abs(grid_best - result["p_left"]), 0.00051)

    def test_shift_and_scale(self):
        rows = [(0.01, 0)] * 3 + [(0, 0.1)]
        reference = fit_shared_student(rows, 0.1)["p_left"]
        shifted = [(a + i, b + i) for i, (a, b) in enumerate(rows)]
        scaled = [(a * 7, b * 7) for a, b in rows]
        self.assertAlmostEqual(fit_shared_student(shifted, 0.1)["p_left"], reference)
        self.assertAlmostEqual(fit_shared_student(scaled, 0.7)["p_left"], reference)

    def test_records_and_controls(self):
        root = Path(__file__).parent
        report = json.loads((root / "results/policy-distillation.json").read_text())
        for name in ("policy_distillation.py", "run.py"):
            self.assertEqual(report["source_sha256"][name], hashlib.sha256((root / name).read_bytes()).hexdigest())
        measured = measure()
        saved = report["results"]
        self.assertEqual(saved["teacher_rows"], [list(row) for row in measured["teacher_rows"]])
        for result in (measured, saved):
            self.assertEqual(result["nll"], {"p_left": 0.75, "agreement": 0.75, "mean_return": 0.0075})
            self.assertEqual(result["mse"], {"mean_q": [0.0075, 0.025], "agreement": 0.25, "mean_return": 0.025})
            for row, t, action, agree, reward in zip(result["kl"], (0.1, 0.01), (1, 0),
                                                     (0.25, 0.75), (0.025, 0.0075), strict=True):
                self.assertEqual((row["temperature"], row["action"], row["agreement"], row["mean_return"]),
                                 (t, action, agree, reward))
                target = (3 / (1 + math.exp(-0.01/t)) + 1 / (1 + math.exp(0.1/t))) / 4
                self.assertAlmostEqual(row["p_left"], target, delta=1e-12)
