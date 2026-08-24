"""
This file houses the unittests for the minigrad.core.Context class.
"""
import unittest
from minigrad.core import Tensor, Context

class ContextTest(unittest.TestCase):
    testing_context = Context()

    def test_save_for_backward_one_tensor_not_None(self):
        one = Tensor([1, 2, 3])
        self.testing_context.save_for_backward(one)
        self.assertIsNotNone(self.testing_context._prev)

    def test_save_for_backward_one_tensor_data(self):
        one = Tensor([1, 2, 3])
        self.testing_context.save_for_backward(one)
        self.assertListEqual([list(one.data)], [list(t.data) for t in self.testing_context._prev])

    def test_save_for_backward_two_tensors_not_None(self):
        one = Tensor([1, 2, 3])
        two = Tensor([3, 4, 5, 2])
        self.testing_context.save_for_backward(one, two)
        self.assertIsNotNone(self.testing_context._prev)

    def test_save_for_backward_two_tensors_data(self):
        one = Tensor([1, 2, 3])
        two = Tensor([3, 4, 5, 2])
        self.testing_context.save_for_backward(one, two)
        self.assertListEqual([list(one.data), list(two.data)], [list(t.data) for t in self.testing_context._prev])

    def test_save_for_backward_illegal(self):
        self.assertRaises(ValueError, lambda: self.testing_context.save_for_backward(None))

if __name__ == "__main__":
    unittest.main()