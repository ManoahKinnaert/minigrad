"""
This file contains the unittests for the minigrad.nn.Layer class
"""
import unittest
from minigrad.nn import Layer
from minigrad.core.function import Dot

class LayerTest(unittest.TestCase):
    def test_init_nin_is_none(self):
        self.assertRaises(ValueError, lambda: Layer(None, 1))

    def test_init_nin_is_smaller_than_1(self):
        self.assertRaises(ValueError, lambda: Layer(0, 1))

    def test_init_nout_is_none(self):
        self.assertRaises(ValueError, lambda: Layer(1, None))

    def test_init_nout_is_smaller_than_1(self):
        self.assertRaises(ValueError, lambda: Layer(1, 0))

    def test_init_activation_default_none(self):
        self.assertIsNone(Layer(1, 1).activation)

    def test_init_activation_set_is_not_none(self):
        self.assertIsNotNone(Layer(1, 1, Dot).activation)

    def test_init_activation_set_correct(self):
        self.assertEqual(Dot, Layer(1, 1, Dot).activation)


if __name__ == "__main__":
    unittest.main()