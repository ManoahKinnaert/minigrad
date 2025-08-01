"""
Define a model class.
"""
from minigrad.core import Tensor 
from minigrad.core import Function
from minigrad.nn import Layer
from minigrad.nn.loss import *
import minigrad.nn.optim as optim
class Model(Module):
    def __init__(self):
        self.layers = []
        self.X: Tensor = None 
        self.y: Tensor = None 
        self.loss_function: Loss = MeanSquared   # default to mean squared loss 
        self.optim = optim.SGD   # SGD optimizer by default

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
            self.optim(lr=lr, model=self).step()

            # print the loss and TODO: other useful things
            if debug and epoch % 100:
                print(f"Epoch: {epoch}: loss = {loss.data:.6f}")
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]