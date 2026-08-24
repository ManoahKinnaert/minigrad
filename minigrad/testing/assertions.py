"""
This file contains helper assertions that might prove to be useful in testing
"""
from minigrad.core import Tensor
import numpy as np

class TensorAssertions:
    @staticmethod
    def _tensor_compare_nonetype_check(tensor1: Tensor, tensor2: Tensor):
        if tensor1 is None: raise AssertionError("tensor1 can't be of type Nonetype!")
        if tensor2 is None: raise AssertionError("tensor2 can't be of type Nonetype!")

    @staticmethod
    def assert_equal(tensor1: Tensor, tensor2: Tensor):
        TensorAssertions._tensor_compare_nonetype_check(tensor1, tensor2)
        if not np.array_equal(tensor1.data, tensor2.data):
            raise AssertionError("Tensors are not equal!")

    @staticmethod
    def assert_equiv(tensor1, tensor2):
        TensorAssertions._tensor_compare_nonetype_check(tensor1, tensor2)
        if not np.array_equiv(tensor1.data, tensor2.data):
            raise AssertionError("Tensors are not equivalent!")
