import math
import unittest
from bert_mlm import mlm_loss, run_experiment


class MLMLossTests(unittest.TestCase):
    def test_uniform_logits_hand_reference(self):
        self.assertAlmostEqual(mlm_loss([[0,0,0]],[2],[1]),math.log(3)/(1+1e-5))

    def test_fractional_weights_and_padding(self):
        actual=mlm_loss([[math.log(3),0],[0,0],[99,-99]],[0,1,1],[0.5,2,0])
        self.assertAlmostEqual(actual,(0.5*math.log(4/3)+2*math.log(2))/(2.5+1e-5))

    def test_logit_offset_invariance_and_extreme_values(self):
        self.assertAlmostEqual(mlm_loss([[1001,1000]],[1],[1]),mlm_loss([[1,0]],[1],[1]))
        self.assertAlmostEqual(mlm_loss([[1000,-1000]],[1],[1]),2000/(1+1e-5))

    def test_all_zero_weights(self):
        self.assertEqual(mlm_loss([[1,2],[3,4]],[0,1],[0,0]),0)

    def test_invalid_inputs(self):
        for labels,weights in [([2],[1]),([0],[-1]),([],[1]),([0],[float('nan')])]:
            with self.assertRaises(ValueError):
                mlm_loss([[0,0]],labels,weights)

    def test_interventions(self):
        r=run_experiment()
        self.assertTrue(r['passed'],r['checks'])
        self.assertAlmostEqual(r['metrics']['unchanged_selected_label_delta'],math.log(2)/(3+1e-5))
        self.assertAlmostEqual(r['metrics']['random_selected_label_delta'],math.log(4)/(3+1e-5))


if __name__=='__main__':
    unittest.main()
