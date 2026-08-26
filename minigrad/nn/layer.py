"""
Define layers.
"""
from minigrad.core import Function 
from minigrad.core import Tensor 
from minigrad.nn import Module

class Layer(Module):
    def __init__(self, nin: int, nout: int, activation: Function=None, **kwargs):
        if nin is None: raise ValueError("nin can't be None!")
        if nin < 1: raise ValueError("nin must be greater than or equal to 1!")
        if nout is None: raise ValueError("nout can't be None!")
        if nout < 1: raise ValueError("nout must be greater than or equal to 1!")

        self._nneurons: int = nout
        self._ninputs: int = nin
        self.activation = activation
        # init weights
        self.w: Tensor = Tensor.randn(nin, nout)
        # init biases
        self.b: Tensor = Tensor.zeros(nout)

    @property
    def parameters(self): return [self.w, self.b]

    def __repr__(self):
        return f"Layer({self._ninputs} Inputs and {self._nneurons} Neurons)"

    def forward(self, tin: Tensor):
        l = tin.dot(self.w) + self.b
        if self.activation:
            return self.activation.forward(l)
        return l