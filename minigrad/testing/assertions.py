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