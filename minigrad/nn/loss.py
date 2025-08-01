"""
Define loss functions.
"""
from minigrad.core import Tensor
from minigrad.nn import Module

class Loss(Module):
    def calculate(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("To implement loss you must define how to calculate the loss.") 

class MeanSquared(Loss):
    @staticmethod
    def calculate(pred, y) -> Tensor:
        return ((pred - y) ** 2).mean()