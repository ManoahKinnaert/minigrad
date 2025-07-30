from minigrad.tensor import Tensor
from minigrad.function import *
import numpy as np

   
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

class Loss(Module):
    def calculate(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("To implement loss you must define how to calculate the loss.") 

class MeanSquared(Loss):
    def calculate(self, pred, y) -> Tensor:
        return ((pred - y) ** 2).mean()

class Model(Module):
    def __init__(self):
        self.layers = []
        self.X = None 
        self.y = None 
        self.loss_function: Loss = MeanSquared   # default to mean squared loss 
        self.optim = None   # no optimizer by default -> will change this in the future

    def add_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
        self.layers.append(Layer(nin, nout, activation))

    def forward(self):
        pred = self.X
        for layer in self.layers:
            pred = layer.forward()
        return pred 
    
    # TODO: Optimizer to still be implemented and multi batch too
    def train(self, lr, epochs, batch=1, debug=False):
        for epoch in range(epochs):
            # forward pass + loss
            pred = self.forward()
            loss = self.loss_function.calculate(pred, self.y)
            # run through backwards pass
            loss.backward()
            # update params
            for layer in self.layers:
                layer.w.data -= lr * layer.w.grad 
                layer.b.data -= lr * layer.b.grad
            # Reset
            for layer in self.layer:
                layer.w.zero_grad()
                layer.b.zero_grad()

            # print the loss and TODO: other useful things
            if debug:
                print(f"Epoch: {epoch}: loss = {loss.data:.6f}")
            