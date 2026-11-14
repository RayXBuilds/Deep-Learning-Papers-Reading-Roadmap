import unittest
from robust_flow import ccs, weight, residual


class RobustFlowTests(unittest.TestCase):
    def test_excludes_observed(self):
        self.assertEqual(ccs([[1,4,9]],0),-3)
        self.assertEqual(ccs([[1,4,9]],2),8)

    def test_timewise_competitor(self):
        self.assertEqual(ccs([[2,1,9],[2,9,1]],0),1)

    def test_weight_symmetry_extremes(self):
        self.assertAlmostEqual(weight(2)+weight(-2),1)
        self.assertEqual(weight(10000),0)
        self.assertEqual(weight(-10000),1)

    def test_hand_residual(self):
        self.assertEqual(residual(0.25,0,1,2,0),-0.5)
        self.assertEqual(residual(0.25,0,2,2,0),-1)

    def test_invalid(self):
        for losses,index in (([[1]],0),([[1,2],[3]],0),([[1,-1]],0),([[1,2]],2)):
            with self.assertRaises(ValueError): ccs(losses,index)
        with self.assertRaises(ValueError): weight(1,0)


if __name__=='__main__': unittest.main()
