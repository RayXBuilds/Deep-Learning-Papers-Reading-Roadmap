import math
import unittest
from vit_patches import patchify, project, conv_reference, attention, error, run_experiment


class ViTPatchTests(unittest.TestCase):
    def test_raster_patch_order_preserves_two_channels(self):
        image = [[[1, 11], [2, 12], [3, 13], [4, 14]],
                 [[5, 15], [6, 16], [7, 17], [8, 18]]]
        self.assertEqual(patchify(image, 2), [[1,11,2,12,5,15,6,16], [3,13,4,14,7,17,8,18]])

    def test_projection_and_convolution_match_independent_hand_value(self):
        image = [[[1], [2]], [[3], [4]]]
        weights = [[1, 0], [10, 0], [100, 0], [1000, 1]]
        expected = [[4321, 4]]
        self.assertEqual(project(patchify(image, 2), weights), expected)
        kernel = [[[[1,0]], [[10,0]]], [[[100,0]], [[1000,1]]]]
        self.assertEqual(conv_reference(image, kernel, 2), expected)

    def test_patch_size_one_and_whole_image(self):
        image = [[[1], [2]], [[3], [4]]]
        self.assertEqual(patchify(image, 1), [[1], [2], [3], [4]])
        self.assertEqual(patchify(image, 2), [[1,2,3,4]])

    def test_attention_against_hand_computed_softmax(self):
        output = attention([[1], [0]])
        self.assertAlmostEqual(output[0][0], math.e/(math.e+1))
        self.assertEqual(output[1], [0.5])

    def test_equivariance_for_different_permutation_and_dimension(self):
        tokens = [[1,0,0], [0,2,0], [0,0,3]]
        p = [2,0,1]
        self.assertLess(error(attention([tokens[i] for i in p]), [attention(tokens)[i] for i in p]), 1e-12)

    def test_invalid_images_and_projection_dimensions(self):
        for image, patch in [([], 1), ([[[1],[2]]], 2), ([[[1]]], 0),
                             ([[[1]], [[2,3]]], 1), ([[[float('nan')]]], 1)]:
            with self.assertRaises(ValueError):
                patchify(image, patch)
        with self.assertRaises(ValueError):
            project([[1,2]], [[1]])

    def test_negative_controls_and_provenance(self):
        report = run_experiment()
        self.assertTrue(report['passed'], report['checks'])
        self.assertEqual(len(report['checks']), 7)
        self.assertIsNone(report['config']['seed'])
        self.assertGreater(report['metrics']['wrong_layout_error'], 0.01)
        self.assertGreater(report['metrics']['fixed_position_cls_change'], 1e-4)


if __name__ == '__main__':
    unittest.main()
