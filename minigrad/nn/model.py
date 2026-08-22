"""
Define a model class.
"""
from minigrad.core import Tensor 
from minigrad.core import Function
from minigrad.nn import Layer
from minigrad.nn.loss import *
import minigrad.nn.optim as optim
from tqdm import trange

class Model(Module):
    def __init__(self, X: Tensor=None, y: Tensor=None, loss: Loss=MeanSquared, optimizer: optim.Optimizer=optim.SGD):
        self.layers = []
        self.X: Tensor = X 
        self.y: Tensor = y 
        self.loss_function: Loss = loss  # default to mean squared loss 
        self.optim_class = optimizer  # SGD optimizer by default
        self.optim = None

    @property
    def n_samples(self): return 0 if self.X is None else self.X.data.shape[0]

    @property 
    def parameters(self): return [p for layer in self.layers for p in layer.parameters]

    def create_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
        self.layers.append(Layer(nin, nout, activation))

    def set_training_data(self, X: Tensor, y: Tensor):
        self.X = X 
        self.y = y 

    def clear_training_data(self):
        self.X = None 
        self.y = None

    def forward(self, X: Tensor):
        pred = X
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred 
    
    # TODO: properly implement batching
    def train(self, lr, epochs, batch_size=1, debug=False):
        if self.X is None: raise ValueError("Training data has not been set: (at least) X is missing!")
        if self.y is None: raise ValueError("Training data has not been set: y is missing!")

        n_samples = self.X.data.shape[0]

        self.optim = self.optim_class(model=self, lr=lr)
        for epoch in (t := trange(epochs)):
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                X = self.X[start:end]
                y = self.y[start:end]

                pred = self.forward(X)
                loss = self.loss_function.calculate(pred, y)

                loss.backward()
                self.optim.step()

                epoch_loss += loss.data 
                n_batches += 1

            # print the loss and other things (if debug is enabled)
            if debug:
                t.set_description(f"Epoch: {epoch + 1}, loss: {epoch_loss / n_batches:.6f}")

        self.clear_training_data()

    def train_on_data(self, X: Tensor, y: Tensor, lr, epochs, batch=1, debug=False):
        self.set_training_data(X, y)
        self.train(lr, epochs, batch, debug)
        self.clear_training_data()

   