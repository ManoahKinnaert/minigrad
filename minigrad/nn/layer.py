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
        self._activation = activation
        # init weights
        self._w: Tensor = Tensor.randn(nin, nout)
        # init biases
        self._b: Tensor = Tensor.zeros(nout)

    @property
    def parameters(self): return [self.w, self.b]

    @property
    def activation(self) -> Function: return self._activation

    @activation.setter
    def activation(self, new: Function): self._activation = new 

    @property
    def nin(self) -> int: return self._ninputs

    @property
    def nout(self) -> int: return self._nneurons

    @property
    def weights(self) -> Tensor: return self._w

    @property
    def biases(self) -> Tensor: return self._b

    def __repr__(self):
        return f"Layer({self._ninputs} Inputs and {self._nneurons} Neurons)"

    def forward(self, tin: Tensor):
        l = tin.dot(self._w) + self._b
        if self._activation:
            return self._activation.forward(l)
        return l