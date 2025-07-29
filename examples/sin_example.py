from minigrad import Tensor 
from minigrad import function as f 
import numpy as np

import matplotlib.pyplot as plt

# create training data
np.random.seed(0)

X = Tensor(np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1))
y = Tensor(np.sin(X.data))

# define model 
W1 = Tensor.randn(1, 16)
b1 = Tensor.zeros(16)

W2 = Tensor.randn(16, 16)
b2 = Tensor.zeros(16)

W3 = Tensor.randn(16, 16)
b3 = Tensor.zeros(16)

W4 = Tensor.randn(16, 1)
b4 = Tensor.zeros(1)


def forward():
    l1 = (X.dot(W1) + b1).tanh()
    l2 = (l1.dot(W2) + b2).tanh()
    l3 = (l2.dot(W3) + b3).tanh()
    return l3.dot(W4) + b4

def plot_fitted_curve():
    y_pred = forward()

    # plot 
    fix, ax = plt.subplots()
    ax.plot(X.data, y.data, label="Target sin", lw=4)
    ax.plot(X.data, y_pred.data, label="Prediction", lw=2)
    ax.legend()
    plt.show()

# train without the use of optimizers
def non_optim_train(lr=0.075, epochs=60000):
    for epoch in range(epochs):
        y_pred = forward() 

        # Loss: mean squared error 
        loss = ((y_pred - y) * (y_pred - y)).mean()
        # backward
        loss.backward()

        # update 
        for param in [W1, b1, W2, b2, W3, b3, W4, b4]:
            param.data -= lr * param.grad 
        
        # Reset 
        for param in [W1, b1, W2, b2, W3, b3, W4, b4]:
            param.grad = 0

        # print loss every 100 epochs
        if not epoch % 100:
            print(f"Epoch {epoch}: loss = {loss.data:.6f}")


# example without use of optimizers
def non_optim_example():
    non_optim_train()
    plot_fitted_curve()

if __name__ == "__main__":
    non_optim_example()