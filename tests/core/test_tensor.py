"""
This file houses unittests for the minigrad.core.Tensor class.
"""
import unittest
import numpy as np
from minigrad import Tensor

class TensorTest(unittest.TestCase):
    def test_init_value_error(self):
        """Test data Nonetype""" 
        self.assertRaises(ValueError, lambda: Tensor(None, None))

    # test static methods
    def test_randn_1d_length5(self):
        self.assertEqual(5, Tensor.randn(5).data.size)

    def test_randn_1d_length100(self):
        self.assertEqual(100, Tensor.randn(100).data.size)

    def test_zeros_1d_length5(self):
        np.testing.assert_array_equal(np.zeros(5), Tensor.zeros(5).data)

    def test_ones(self):
        pass 

    def test_ones_like(self):
        pass 

if __name__ == "__main__":
    unittest.main()