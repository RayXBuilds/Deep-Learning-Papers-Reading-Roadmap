import unittest
from trs_buffer import update,quantile,weights


class BufferTests(unittest.TestCase):
    def test_unvisited_entry_and_input_preserved(self):
        original=[1,7,None]
        self.assertEqual(update(original,{0:5}),[3,7,None])
        self.assertEqual(original,[1,7,None])

    def test_first_visit_has_no_zero_bias(self):
        self.assertEqual(update([None],{0:10},0.99),[10])

    def test_quantile_interpolation(self):
        self.assertAlmostEqual(quantile([0,10,20,30],0.7),21)

    def test_warmup_still_allows_scoring(self):
        scores=update([None,None],{0:10})
        self.assertEqual(scores,[10,None])
        self.assertEqual(weights(scores,True)[0],[1,1])

    def test_equal_evidence_equal_weights(self):
        self.assertEqual(weights([2,2,2])[0],[0.5]*3)
        self.assertEqual(weights([None,None]),([1,1],None))

    def test_invalid_input(self):
        with self.assertRaises(ValueError): update([None],{1:2})
        with self.assertRaises(ValueError): quantile([],0.5)
        with self.assertRaises(ValueError): weights([1],temperature=0)


if __name__=='__main__': unittest.main()
