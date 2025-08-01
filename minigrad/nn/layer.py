"""
Define layers.
"""
from minigrad.core import Function 
from minigrad.core import Tensor 
from minigrad.nn import Module

class Layer(Module):
    def __init__(self, nin: int, nout: int, activation: Function=None, **kwargs):
        self._nneurons: int = nout
        self._ninputs: int = nin
        self.activation = activation
        # init weights
        self.w: Tensor = Tensor.randn(nin, nout)
        # init biases
        self.b: Tensor = Tensor.zeros(nout)

    def parameters(self):
        return [self.w, self.b]

    def __repr__(self):
        return f"Layer({self._ninputs} Inputs and {self._nneurons} Neurons)"

    def forward(self, tin: Tensor):
        l = tin.dot(self.w) + self.b
        if self.activation:
            return self.activation.forward(l)
        return l