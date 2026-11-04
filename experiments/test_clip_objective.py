import math
import unittest
from clip_objective import scores, objective, probabilities, run_experiment


class CLIPObjectiveTests(unittest.TestCase):
    def test_cosine_hand_value(self):
        self.assertAlmostEqual(scores([[3,4]],[[0,2]],5)[0][0],4)

    def test_two_direction_hand_value(self):
        m=[[2,0],[1,3]]
        expected=(2*math.log1p(math.exp(-2))+math.log1p(math.exp(-1))+math.log1p(math.exp(-3)))/4
        self.assertAlmostEqual(objective(m),expected)

    def test_uniform_and_single_pair(self):
        self.assertAlmostEqual(objective([[7,7],[7,7]]),math.log(2))
        self.assertEqual(objective([[7]]),0)

    def test_softmax_stability(self):
        self.assertEqual(probabilities([1000,1000]),[0.5,0.5])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): scores([[0,0]],[[1,0]],1)
        with self.assertRaises(ValueError): scores([[1]],[[1,0]],1)
        with self.assertRaises(ValueError): objective([[1,2]])

    def test_controls(self):
        r=run_experiment()
        self.assertTrue(r['passed'],r['checks'])
        self.assertAlmostEqual(r['metrics']['text_only_shuffle_loss']-r['metrics']['loss'],2)


if __name__=='__main__': unittest.main()
