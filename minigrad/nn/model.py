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
    """
    A model class to allow the client to create a neural network model.
    The client may enter X and y tensors used for training, but this may also be accomplished by using the set_training_data method.
    Furthermore it allows the client to define a loss type and an optimizer of choice.
    """
    def __init__(self, X: Tensor=None, y: Tensor=None, loss: Loss=MeanSquared, optimizer: optim.Optimizer=optim.SGD):
        self._layers = []
        self.X: Tensor = X 
        self.y: Tensor = y 
        self.loss_function: Loss = loss  # default to mean squared loss 
        self._optim_class = optimizer  # SGD optimizer by default
        self.optim = None

    @property
    def n_samples(self): 
        """
        Return num of samples, in the case that self.X is None we return 0.
        """
        return 0 if self.X is None else self.X.data.shape[0]

    @property 
    def parameters(self): 
        """
        Returns the layer params.
        """
        return [p for layer in self._layers for p in layer.parameters]

    @property 
    def layers(self): 
        """
        Return a copy of the internal layers.
        """
        return self._layers.copy()

    @property
    def n_layers(self):
        """
        Return the amount of layers.
        """
        return len(self._layers)

    def create_layer(self, nin: int, nout: int, activation: Function=None, **kwargs):
        """
        Create a new layer and add it to the internal layers representation.
        """
        if nin is None: raise ValueError("nin can't be None")
        if nout is None: raise ValueError("nout can't be None")
        if nin <= 0: raise ValueError("nin can't be smaller than or equal to zero")
        if nout <= 0: raise ValueError("nout can't be smaller than or equal to zero")

        self.layers.append(Layer(nin, nout, activation))

    def set_training_data(self, X: Tensor, y: Tensor):
        """
        Set the training data.
        """
        if X is None: raise ValueError("Tensor X can't be None.")
        if y is None: raise ValueError("Tensor y can't be None.")

        self.X = X 
        self.y = y 

    def clear_training_data(self):
        """
        Clear out the training data (hoping to save some memory for larger models).
        """
        self.X = None 
        self.y = None

    def forward(self, X: Tensor):
        """
        Implementation of the forward pass.
        """
        if X is None: raise ValueError("Tensor X can't be None.")

        pred = X
        for layer in self._layers:
            pred = layer.forward(pred)
        return pred 
    
    def train(self, lr: float, epochs: int, batch_size: int=1, debug: bool=False):
        """
        The train method which is called to train the model.
        The client may set the learning rate named lr, the epochs named epochs,
        the batch size, not amount of batches, named batch_size and may choose to train in debug mode.
        Debug mode displays amount of epochs passed and loss, it (debug) is set to False by default
        """
        if self.X is None: raise ValueError("Training data has not been set: (at least) X is missing!")
        if self.y is None: raise ValueError("Training data has not been set: y is missing!")
        if lr is None: raise ValueError("The learning rate can't be None.")
        if epochs is None: raise ValueError("The amount of epochs can't be None.")
        if epochs <= 0: raise ValueError("Epochs can't be smaller than or equal to 0, otherwise ")

        n_samples = self.n_samples

        self.optim = self._optim_class(model=self, lr=lr)
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
        """
        Method for training, basically identical to train but just to allow a more 'flexible' signature. We allow
        the client to pass the training data directly at the same time as training.
        """
        self.set_training_data(X, y)
        self.train(lr, epochs, batch, debug)
        self.clear_training_data()

   