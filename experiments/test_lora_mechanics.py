import copy
import unittest

from lora_mechanics import forward, loss_and_gradients, merge, sgd, finite_difference_errors, run_experiment


class LoRAMechanicsTests(unittest.TestCase):
    def test_hand_computed_forward_and_merge_with_nonunit_scaling(self):
        # Ax=3, B(Ax)=9, base=0, alpha/r=2.
        self.assertEqual(forward([[2, -1]], [[1, 1]], [[3]], [1, 2], 2), [18])
        self.assertEqual(merge([[2, -1]], [[1, 1]], [[3]], 2), [[8, 5]])

    def test_hand_computed_gradients_include_frozen_input_path(self):
        # y=18, target=14 -> g=4; dA=2*3*4*[1,2], dB=2*4*3.
        loss, ga, gb, gx = loss_and_gradients([[2, -1]], [[1, 1]], [[3]], [1, 2], [14], 2)
        self.assertEqual(loss, 8)
        self.assertEqual(ga, [[24, 48]])
        self.assertEqual(gb, [[24]])
        self.assertEqual(gx, [32, 20])  # Frozen base contributes [8,-4].

    def test_two_sgd_steps_use_simultaneous_gradients(self):
        w, a, b = [[1]], [[2]], [[0]]
        _, ga, gb, _ = loss_and_gradients(w, a, b, [1], [3], 1)
        self.assertEqual((ga, gb), ([[0]], [[-4]]))
        a1, b1 = sgd(a, ga, 0.1), sgd(b, gb, 0.1)
        _, ga1, gb1, _ = loss_and_gradients(w, a1, b1, [1], [3], 1)
        self.assertAlmostEqual(ga1[0][0], -0.48)
        self.assertAlmostEqual(gb1[0][0], -2.4)

    def test_both_zero_traps_gradient_with_nonzero_error(self):
        loss, ga, gb, gx = loss_and_gradients([[1]], [[0]], [[0]], [1], [3], 1)
        self.assertEqual((loss, ga, gb, gx), (2, [[0]], [[0]], [-2]))

    def test_finite_differences_at_nonzero_factors(self):
        errors = finite_difference_errors([[1, 2], [3, 4]], [[0.2, -0.1]],
                                          [[0.3], [-0.7]], [0.5, -0.4], [1, 2], 3)
        self.assertLess(max(errors.values()), 1e-7)

    def test_unmerged_rank_one_update_is_not_compression_of_base(self):
        w = merge([[1, 0], [0, 1]], [[3, 4]], [[1], [2]], 1)
        self.assertEqual(w, [[4, 4], [6, 9]])
        self.assertEqual(w[0][0]*w[1][1]-w[0][1]*w[1][0], 12)

    def test_functions_do_not_mutate_base_or_factors(self):
        args = [[[1, 2]], [[3, 4]], [[5]], [1, 2], [0]]
        old = copy.deepcopy(args)
        _, ga, gb, _ = loss_and_gradients(*args, 2)
        sgd(args[1], ga, 0.1)
        sgd(args[2], gb, 0.1)
        merge(*args[:3], 2)
        self.assertEqual(args, old)

    def test_invalid_dimensions_and_values_rejected(self):
        for w, a, b, x, alpha in [([[1]], [[1, 2]], [[1]], [1], 1),
                                  ([[1]], [[1]], [[1]], [1, 2], 1),
                                  ([[1]], [[1]], [[1]], [1], float('nan')),
                                  ([[1]], [[1]], [[float('inf')]], [1], 1),
                                  ([[1, 2], [3]], [[1]], [[1]], [1], 1)]:
            with self.assertRaises(ValueError):
                forward(w, a, b, x, alpha)
        with self.assertRaises(ValueError):
            loss_and_gradients([[1]], [[1]], [[1]], [1], [], 1)

    def test_report_includes_all_seeds_and_negative_controls(self):
        report = run_experiment()
        self.assertTrue(report['passed'], report['checks'])
        self.assertEqual([r['seed'] for r in report['initializations']], [7, 42, 2026])
        self.assertEqual(len(report['checks']), 10)
        self.assertGreater(report['metrics']['dropout_merge_error'], 0.1)
        self.assertGreater(report['metrics']['double_counted_error'], 0.1)


if __name__ == '__main__':
    unittest.main()
