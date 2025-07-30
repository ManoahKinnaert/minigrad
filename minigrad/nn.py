from minigrad.tensor import Tensor
from minigrad.function import *
import numpy as np

class Parameter(Tensor):
    def __new__(cls, data=None):
        if data is None:
            data = np.empty(0, dtype=float)
        super().__init__(data)

    def __repr__(self):
        return f"Parameter containing: {super().__repr__()}"
   
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self):
        pass 

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

class Model(Module):
    def __init__(self):
        self.layers = []

    def add_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
        self.layers.append(Layer(nin, nout, activation))

    def forward(self):
        for layer in self.layers:
            layer.forward()

    # TODO: Loss to still be implemented 
    def backward(self, loss):
        pass 
