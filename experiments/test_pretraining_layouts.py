import unittest
from itertools import permutations
from pretraining_layouts import recover, span_pair, visible


class LayoutTests(unittest.TestCase):
    def test_original_position_survives_permutation(self):
        order = (2, 0, 3, 1)
        self.assertEqual(visible(order, 3), {0, 2})
        self.assertEqual(visible(order, 2), set())
        self.assertEqual(visible(order, 2, content=True), {2})

    def test_no_indirect_target_path(self):
        for order in permutations(range(4)):
            for target in order:
                ancestors = set(visible(order, target))
                for previous in list(ancestors):
                    ancestors.update(visible(order, previous, content=True))
                self.assertNotIn(target, ancestors)
                self.assertEqual(ancestors, {i for i in order if order.index(i) < order.index(target)})

    def test_span_targets_keep_order_and_boundaries(self):
        src, tgt = span_pair(list("abacd"), [True, True, False, False, True])
        self.assertEqual(src, ["<extra_id_0>", "a", "c", "<extra_id_1>"])
        self.assertEqual(tgt, ["<extra_id_0>", "a", "b", "<extra_id_1>", "d", "<extra_id_2>"])
        self.assertEqual(recover(src, tgt), list("abacd"))

    def test_adjacent_and_separate_cost(self):
        for mask, lengths in [([1, 1, 0, 0], (3, 4)), ([1, 0, 1, 0], (4, 5))]:
            src, tgt = span_pair(list("abcd"), mask)
            self.assertEqual((len(src), len(tgt)), lengths)
