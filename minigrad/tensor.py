import numpy as np
import function as f

class Tensor:
    def __init__(self, data, ctx=None):
        self.data = np.array(data, dtype=float) 
        self.grad = np.zeros_like(self.data) 
        # context used for autograd graph construction
        self.ctx = ctx

    ##### ops ##### 
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Add.forward(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return f.Mul.forward(self, other)
    
    def backward(self, grad_out=None):
        if grad_out is None:
            # default gradient is 1
            grad_out = np.ones_like(self.data)

        self.grad += grad_out
        # check for leaf node
        if self.ctx is None:
            return 
        
        grads = self.ctx.function.backward(self.ctx, grad_out)
        for tensor, grad in zip(self.ctx._prev, grads):
            tensor.backward(grad)

    def randn(*args, **kwargs):
        return Tensor(data=np.random.randn(*args, **kwargs))
    
    def zeros(*args, **kwargs):
        return Tensor(data=np.zeros(*args, **kwargs))