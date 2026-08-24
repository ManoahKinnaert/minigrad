"""
This file houses unittests for the minigrad.core.Tensor class.
"""
import unittest
import numpy as np
from minigrad import Tensor
from minigrad.testing import TensorAssertions

class TensorTest(unittest.TestCase):
    def test_init_value_error(self):
        """Test data Nonetype""" 
        self.assertRaises(ValueError, lambda: Tensor(None, None))

    # test static methods
    def test_randn_1d_length5(self):
        TensorAssertions.assert_tensor_shape((5,), Tensor.randn(5))

    def test_randn_1d_length100(self):
        TensorAssertions.assert_tensor_shape((100,), Tensor.randn(100))

    def test_randn_2d_5_by_5(self):
        TensorAssertions.assert_tensor_shape((5, 5), Tensor.randn(5, 5))

    def test_randn_2d_5_by_100(self):
        TensorAssertions.assert_tensor_shape((5, 100), Tensor.randn(5, 100))

    def test_zeros_1d_length5(self):
        TensorAssertions.assert_compare_numpy(np.zeros(5), Tensor.zeros(5))

    def test_zeros_2d_5_by_5(self):
        TensorAssertions.assert_compare_numpy(np.zeros((5, 5)), Tensor.zeros((5, 5)))

    def test_zeros_2d_5_by_100(self):
        TensorAssertions.assert_compare_numpy(np.zeros((5, 100)), Tensor.zeros((5, 100)))

    def test_zeros_1d_length100(self):
        TensorAssertions.assert_compare_numpy(np.zeros(100), Tensor.zeros(100))

    def test_ones_1d_lenght5(self):
        TensorAssertions.assert_compare_numpy(np.ones(5), Tensor.ones(5))

    def test_ones_1d_length100(self):
        TensorAssertions.assert_compare_numpy(np.ones(100), Tensor.ones(100))

    def test_ones_like(self):
        pass 

if __name__ == "__main__":
    unittest.main()