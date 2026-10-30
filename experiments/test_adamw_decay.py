import copy
import math
import unittest
from adamw_decay import initial_state, update, run_experiment, quadratic


class AdamWDecayTests(unittest.TestCase):
    def test_first_step_independent_hand_values(self):
        state = initial_state([1, 10])
        wd = update(state, [0, 0], 0.1, 0.1, 'adamw')
        l2 = update(state, [0, 0], 0.1, 0.1, 'l2')
        self.assertAlmostEqual(wd['theta'][0], 0.99)
        self.assertAlmostEqual(wd['theta'][1], 9.9)
        self.assertAlmostEqual(l2['theta'][0], 0.900000009999999)
        self.assertAlmostEqual(l2['theta'][1], 9.900000001)
        self.assertEqual(wd['m'], [0, 0])
        self.assertAlmostEqual(l2['m'][1], 0.1)
        self.assertAlmostEqual(l2['v'][1], 0.001)

    def test_zero_decay_modes_match_for_multiple_steps(self):
        a, b = initial_state([2, -1]), initial_state([2, -1])
        for g in ([3, 1], [-2, 4], [0, 0]):
            a = update(a, g, 0.03, 0, 'l2')
            b = update(b, g, 0.03, 0, 'adamw')
        self.assertEqual(a, b)

    def test_decay_applies_to_old_parameter_not_gradient_update(self):
        # beta=0: normalized positive gradient is 1/(1+eps).
        s = update(initial_state([2]), [1], 0.1, 0.2, 'adamw', 0, 0)
        expected = 1.96 - 0.1/(1+1e-8)
        wrong_post_decay = (2-0.1/(1+1e-8))*0.98
        self.assertAlmostEqual(s['theta'][0], expected)
        self.assertGreater(abs(s['theta'][0]-wrong_post_decay), 0.001)

    def test_zero_current_gradient_can_move_due_to_existing_momentum(self):
        state = update(initial_state([2]), [1], 0.1, 0, 'adamw')
        after = update(state, [0], 0.1, 0, 'adamw')
        # m2/(1-beta1^2)=0.09/0.19, v2/(1-beta2^2)=0.000999/0.001999.
        expected = state['theta'][0] - 0.1*(0.09/0.19)/(math.sqrt(0.000999/0.001999)+1e-8)
        self.assertAlmostEqual(after['theta'][0], expected)
        self.assertLess(after['theta'][0], state['theta'][0])

    def test_does_not_mutate_inputs(self):
        state = initial_state([1, 2])
        old = copy.deepcopy(state)
        update(state, [2, 3], 0.1, 0.2, 'adamw')
        self.assertEqual(state, old)

    def test_quadratic_gradient_by_finite_difference(self):
        point, h = [3.1, -2.7], 1e-6
        _, grad = quadratic(point)
        for i in range(2):
            left, right = point.copy(), point.copy()
            left[i] -= h
            right[i] += h
            self.assertAlmostEqual(grad[i], (quadratic(right)[0]-quadratic(left)[0])/(2*h), places=7)

    def test_invalid_inputs(self):
        state = initial_state([1])
        for kwargs in ({'lr': -1}, {'decay': -1}, {'beta1': 1}, {'epsilon': 0}, {'mode': 'bad'}):
            args = dict(lr=0.1, decay=0.1, mode='adamw')
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                update(state, [0], **args)
        for gradient in ([], [0, 1], [float('nan')]):
            with self.assertRaises(ValueError):
                update(state, gradient, 0.1, 0.1, 'adamw')

    def test_experiment_controls(self):
        r = run_experiment()
        self.assertTrue(r['passed'], r['checks'])
        self.assertEqual(set(r['trajectories']), {'adam', 'adam_l2', 'adamw'})
        self.assertTrue(all(len(rows) == 201 for rows in r['trajectories'].values()))


if __name__ == '__main__':
    unittest.main()
