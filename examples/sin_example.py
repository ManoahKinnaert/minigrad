from minigrad import Tensor 
from minigrad import function as f 
import numpy as np


# create training data
np.random.seed(0)

X = Tensor(np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1))
y = Tensor(np.sin(X.data))

# define model 
W1 = Tensor.randn(1, 16)
b1 = Tensor.zeros(16)

W2 = Tensor.randn(16, 16)
b2 = Tensor.zeros(16)

W3 = Tensor.randn(16, 1)
b3 = Tensor.zeros(1)

def forward():
    l1 = (X.dot(W1) + b1).tanh()
    l2 = (l1.dot(W2) + b2).tanh()
    return l2.dot(W3) + b3 

# train without the use of optimizers
def non_optim_train(lr=0.01, epochs=1000):
    for epoch in range(epochs):
        y_pred = forward() 


# example without use of optimizers
def non_optim_example():
    pass