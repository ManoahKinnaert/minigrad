"""
This file contains the unittests for the minigrad.nn.Layer class
"""
import unittest
from minigrad.nn import Layer

class LayerTest(unittest.TestCase):
    def test_init_nin_is_none(self):
        self.assertRaises(ValueError, lambda: Layer(None, 1))

    def test_init_nin_is_smaller_than_1(self):
        self.assertRaises(ValueError, lambda: Layer(0, 1))

    def test_init_nout_is_none(self):
        self.assertRaises(ValueError, lambda: Layer(1, None))

    def test_init_nout_is_smaller_than_1(self):
        self.assertRaises(ValueError, lambda: Layer(1, 0))

if __name__ == "__main__":
    unittest.main()