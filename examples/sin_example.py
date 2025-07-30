from minigrad import Tensor 
from minigrad.core import function as f 
from minigrad.nn import Model
import numpy as np

import matplotlib.pyplot as plt

# create training data
np.random.seed(0)

X = Tensor(np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1))
y = Tensor(np.sin(X.data))

# define a model
model = Model()
model.add_layer(nin=1, nout=16, activation=f.Tanh)
model.add_layer(nin=16, nout=16, activation=f.Tanh)
model.add_layer(nin=16, nout=16, activation=f.Tanh)
model.add_layer(nin=16, nout=1)


def plot_fitted_curve():
    y_pred = model.forward()

    # plot 
    fix, ax = plt.subplots()
    ax.plot(X.data, y.data, label="Target sin", lw=4)
    ax.plot(X.data, y_pred.data, label="Prediction", lw=2)
    ax.legend()
    plt.show()

# train without the use of optimizers
def non_optim_train(lr=0.075, epochs=60000):
    model.set_training_data(X, y)
    model.train(lr=lr, epochs=epochs, debug=True)

# example without use of optimizers
def non_optim_example():
    non_optim_train()
    plot_fitted_curve()

if __name__ == "__main__":
    non_optim_example()