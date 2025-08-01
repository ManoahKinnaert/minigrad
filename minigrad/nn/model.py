"""
Define a model class.
"""
from minigrad.core import Tensor 
from minigrad.core import Function
from minigrad.nn import Layer
from minigrad.nn.loss import *

class Model(Module):
    def __init__(self):
        self.layers = []
        self.X: Tensor = None 
        self.y: Tensor = None 
        self.loss_function: Loss = MeanSquared   # default to mean squared loss 
        self.optim = None   # no optimizer by default -> will change this in the future

    def add_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
        self.layers.append(Layer(nin, nout, activation))

    def set_training_data(self, X: Tensor, y: Tensor):
        self.X = X 
        self.y = y 

    def forward(self):
        pred = self.X
        for layer in self.layers:
            pred = layer.forward(pred)
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
            for layer in self.layers:
                layer.w.zero_grad()
                layer.b.zero_grad()

            # print the loss and TODO: other useful things
            if debug and epoch % 100:
                print(f"Epoch: {epoch}: loss = {loss.data:.6f}")
    