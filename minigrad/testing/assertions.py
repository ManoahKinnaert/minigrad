"""
This file contains helper assertions that might prove to be useful in testing
"""
from minigrad.core import Tensor
import numpy as np

class TensorAssertions:
    @staticmethod
    def _nonetype_check(*args):
        for i, ar in enumerate(args):
            if ar is None: raise AssertionError(f"Argument {i} can't be Nonetype.")

    @staticmethod
    def _valid_shape_check(shape: tuple):
        for s in shape: 
            if not isinstance(s, (int, np.integer)): raise AssertionError("The elements of shape must be integers.")
            if s < 0: raise AssertionError("All elements of shape must be positive integers")
            
    @staticmethod
    def assert_equal(tensor1: Tensor, tensor2: Tensor):
        TensorAssertions._nonetype_check(tensor1, tensor2)
        if not np.array_equal(tensor1.data, tensor2.data):
            raise AssertionError("Tensors are not equal!")

    @staticmethod
    def assert_equiv(tensor1: Tensor, tensor2: Tensor):
        TensorAssertions._nonetype_check(tensor1, tensor2)
        if not np.array_equiv(tensor1.data, tensor2.data):
            raise AssertionError("Tensors are not equivalent!")

    @staticmethod
    def assert_compare_numpy(nparray: np.ndarray, tensor: Tensor):
        TensorAssertions._nonetype_check(nparray, tensor)
        if not np.array_equal(nparray, tensor.data):
            raise AssertionError("nparray is not equal to tensor data!")

    @staticmethod
    def assert_tensor_shape(shape: tuple, tensor: Tensor):
        TensorAssertions._nonetype_check(shape, tensor)
        TensorAssertions._valid_shape_check(shape)
        if tensor.data.shape != shape: raise AssertionError("tensor shape doesn't match shape!")