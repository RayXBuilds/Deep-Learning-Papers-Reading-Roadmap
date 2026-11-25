import hashlib
import json
import math
from pathlib import Path
import unittest

from distillation_temperature import measure, scaled_gradient


class DistillationGradientTests(unittest.TestCase):
    def test_loss_finite_difference(self):
        # exp(teacher / T) is proportional to (4, 2, 1) at T=2.
        teacher = [2 * math.log(4), 2 * math.log(2), 0]
        student = [1.0, -0.7, 0.4]

        def loss(z):
            normalizer = math.log(sum(math.exp(x / 2) for x in z))
            return 4 * (normalizer - (4 * z[0] + 2 * z[1] + z[2]) / 14)

        for i, actual in enumerate(scaled_gradient(student, teacher, 2)):
            plus, minus = student.copy(), student.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            self.assertAlmostEqual(actual, (loss(plus) - loss(minus)) / 2e-5, delta=1e-9)

    def test_independent_shifts_and_swapped_classes(self):
        base = scaled_gradient([2, 0, 0], [4, 1, -2], 4)
        shifted = scaled_gradient([12, 10, 10], [0, -3, -6], 4)
        for a, b in zip(base, shifted, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-12)
        swap = scaled_gradient([2, 0, 0], [4, -2, 1], 4)
        self.assertLess(base[1], 0)
        self.assertGreater(swap[1], 0)
        self.assertAlmostEqual(base[2], swap[1], delta=1e-12)

    def test_saved_sweep(self):
        root = Path(__file__).parent
        report = json.loads((root / "results/distillation-temperature.json").read_text())
        for name in ("distillation_temperature.py", "run.py"):
            self.assertEqual(report["source_sha256"][name], hashlib.sha256((root / name).read_bytes()).hexdigest())
        measured = measure()
        saved = report["results"]
        for key in ("teacher", "student"):
            self.assertEqual(saved[key], measured[key])
        for a, b in zip(saved["swapped_teacher_T4"], measured["swapped_teacher_T4"], strict=True):
            self.assertAlmostEqual(a, b, delta=1e-12)
        for a, b in zip(saved["sweep"], measured["sweep"], strict=True):
            self.assertEqual(a["T"], b["T"])
            self.assertAlmostEqual(a["max_error_to_limit"], b["max_error_to_limit"], delta=1e-12)
            for x, y in zip(a["scaled_gradient"], b["scaled_gradient"], strict=True):
                self.assertAlmostEqual(x, y, delta=1e-12)
        self.assertLess(measured["sweep"][-1]["max_error_to_limit"], 0.01)
