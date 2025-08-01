"""
Define Tensor class.
"""
import numpy as np
import minigrad.core.function as f

class Tensor:
    def __init__(self, data, ctx=None):
        self.data = np.array(data, dtype=float) 
        self.grad = np.zeros_like(self.data) 
        # context used for autograd graph construction
        self.ctx = ctx

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    ##### ops ##### 
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Add.forward(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Mul.forward(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (other * -1)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self * Tensor(1.0 / other.data)
    
    def dot(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Dot.forward(self, other)
    
    def relu(self):
        return f.Relu.forward(self)
    
    def sigmoid(self):
        return f.Sigmoid.forward(self)
    
    def tanh(self):
        return f.Tanh.forward(self)
    
    def sum(self):
        return f.Sum.forward(self)
    
    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Pow.forward(self, other)
    
    def sqrt(self):
        return self ** Tensor(0.5)
    
    def logsoftmax(self):   
        return f.LogSoftmax.forward(self)

    def backward(self, grad_out=None):
        if grad_out is None:
            # default gradient is 1
            grad_out = np.ones_like(self.data)
        
        # Handle broadcasting: sum over broadcasted dimensions
        # and reshape to match self.data shape
        grad_out = np.array(grad_out)  # Ensure it's a numpy array
        
        # Sum over dimensions that were broadcasted
        while grad_out.ndim > self.data.ndim:
            grad_out = grad_out.sum(axis=0)
        
        for i in range(self.data.ndim):
            if self.data.shape[i] == 1 and grad_out.shape[i] > 1:
                grad_out = grad_out.sum(axis=i, keepdims=True)
        
        # Ensure shapes match exactly
        grad_out = grad_out.reshape(self.data.shape)
        
        self.grad += grad_out
        
        # check for leaf node
        if self.ctx is None:
            return
        
        grads = self.ctx.function.backward(self.ctx, grad_out)
        if not isinstance(grads, (tuple, list)):
            grads = (grads,)
        
        for tensor, grad in zip(self.ctx._prev, grads):
            tensor.backward(grad)
   
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def randn(*args, **kwargs):
        return Tensor(data=np.random.randn(*args, **kwargs))
    
    @staticmethod
    def zeros(*args, **kwargs):
        return Tensor(data=np.zeros(*args, **kwargs))
    
    @staticmethod
    def zeros_like(*args, **kwargs):
        return Tensor(data=np.zeros_like(*args, **kwargs))
    
    @staticmethod
    def ones(*args, **kwargs):
        return Tensor(data=np.ones(*args, **kwargs))
    
    @staticmethod
    def ones_like(*args, **kwargs):
        return Tensor(data=np.ones_like(*args, **kwargs))
