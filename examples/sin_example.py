from minigrad import Tensor 
from minigrad.core import function as f 
from minigrad.nn import Model
from minigrad.nn import optim
import numpy as np

import matplotlib.pyplot as plt

# create training data
np.random.seed(0)

X = Tensor(np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1))
y = Tensor(np.sin(X.data))

# define a model
model = Model(X, y, optimizer=optim.Adam)
model.create_layer(nin=1, nout=64, activation=f.Relu)
model.create_layer(nin=64, nout=64, activation=f.Relu)
model.create_layer(nin=64, nout=1)

def plot_fitted_curve():
    y_pred = model.forward()

    # plot 
    fix, ax = plt.subplots()
    ax.plot(X.data, y.data, label="Target sin", lw=4)
    ax.plot(X.data, y_pred.data, label="Prediction", lw=2)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    model.train(lr=0.2, epochs=10000, debug=True)
    plot_fitted_curve()
