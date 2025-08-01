"""
Define a model class.
"""
from minigrad.core import Tensor 
from minigrad.core import Function
from minigrad.nn import Layer
from minigrad.nn.loss import *
import minigrad.nn.optim as optim
class Model(Module):
    def __init__(self, X: Tensor=None, y: Tensor=None, loss: Loss=MeanSquared, optimizer: optim.Optimizer=optim.SGD):
        self.layers = []
        self.X: Tensor = X 
        self.y: Tensor = y 
        self.loss_function: Loss = loss  # default to mean squared loss 
        self.optim_class = optimizer  # SGD optimizer by default
        self.optim = None

    def create_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
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
        self.optim = self.optim_class(model=self, lr=lr)
        for epoch in range(epochs):
            # forward pass + loss
            pred = self.forward()
            loss = self.loss_function.calculate(pred, self.y)
            # run through backwards pass
            loss.backward()
            # update params
            self.optim.step()

            # print the loss and TODO: other useful things
            if debug and epoch % 100:
                print(f"Epoch: {epoch}: loss = {loss.data:.6f}")
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]