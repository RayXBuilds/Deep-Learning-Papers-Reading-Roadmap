import unittest
from flow_mixture import velocity, target_cdf, pairs, sample


class MixtureTests(unittest.TestCase):
    def test_symmetry_and_endpoints(self):
        for x in (-3,-1,0,1,3):
            self.assertAlmostEqual(velocity(0,x),-x)
            self.assertAlmostEqual(velocity(1,x),x)
            self.assertAlmostEqual(velocity(0.5,x),-velocity(0.5,-x))
            self.assertAlmostEqual(target_cdf(x)+target_cdf(-x),1)

    def test_independent_bayes_reference(self):
        # Compute posterior weights from two Gaussian component densities.
        import math
        t,x=0.4,0.7
        s=(1-t)**2+t*t*0.25
        c=t*0.25-(1-t)
        weights=[math.exp(-(x-t*m)**2/(2*s)) for m in (-2,2)]
        expected=sum(w*(m+c/s*(x-t*m)) for w,m in zip(weights,(-2,2)))/sum(weights)
        self.assertAlmostEqual(velocity(t,x),expected)

    def test_nonlinear_field(self):
        self.assertGreater(abs(velocity(0.5,2)-2*velocity(0.5,1)),0.1)
        self.assertAlmostEqual(velocity(0.5,2,True),2*velocity(0.5,1,True))

    def test_disjoint_reproducible_streams(self):
        self.assertEqual(pairs(7,3),pairs(7,3))
        self.assertNotEqual(pairs(7,3),pairs(100007,3))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            velocity(2,0)
        with self.assertRaises(ValueError):
            sample(0)


if __name__=='__main__':
    unittest.main()
