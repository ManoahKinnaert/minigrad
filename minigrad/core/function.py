"""
Define activation functions.
"""
import numpy as np
from minigrad.core.tensor import Tensor 
from minigrad.core.context import Context 

class Function:
    @staticmethod
    def forward(*args, **kwargs):
        raise NotImplementedError("For a custom function you need to implement a forward function!")

    @staticmethod
    def backward(*args, **kwargs):
        raise NotImplementedError("For a custom function you need to implement a backward function!")
    
## some basic functions
class Add(Function):
    @staticmethod
    def forward(x: Tensor, y: Tensor):
        ctx = Context()
        ctx.save_for_backward(x, y)
        ctx.function = Add
        return Tensor(x.data + y.data, ctx=ctx) 
    
    @staticmethod
    def backward(ctx: Context, grad_out):
        x, y = ctx._prev 
        return grad_out, grad_out 
    
class Mul(Function):
    @staticmethod
    def forward(x: Tensor, y: Tensor):
        ctx = Context()
        ctx.save_for_backward(x, y)
        ctx.function = Mul 
        return Tensor(x.data * y.data, ctx=ctx)
    
    @staticmethod
    def backward(ctx: Context, grad_out):
        x, y = ctx._prev 
        return y.data * grad_out, x.data * grad_out 

class Dot(Function):
    @staticmethod
    def forward(x: Tensor, y: Tensor):
        ctx = Context()
        ctx.save_for_backward(x, y)
        ctx.function = Dot 
        return Tensor(np.dot(x.data, y.data), ctx=ctx)

    @staticmethod 
    def backward(ctx: Context, grad_out):
        x, y = ctx._prev 
        return np.dot(grad_out, y.data.T), np.dot(x.data.T, grad_out)

class Pow(Function):
    @staticmethod
    def forward(base, power):
        ctx = Context()
        ctx.save_for_backward(base, power)
        ctx.function = Pow
        return Tensor(np.power(base.data, power.data), ctx=ctx)

    @staticmethod
    def backward(ctx: Context, grad_out):
        base, power = ctx._prev
        grad_base = power.data * np.power(base.data, power.data - 1) * grad_out
        grad_power = np.power(base.data, power.data) * np.log(base.data) * grad_out
        return grad_base, grad_power

class Relu(Function):
    @staticmethod
    def forward(input):
        ctx = Context()
        ctx.save_for_backward(input)
        ctx.function = Relu
        return Tensor(np.maximum(0, input.data), ctx=ctx)

    @staticmethod
    def backward(ctx: Context, grad_out):
        input, = ctx._prev 
        return (input.data > 0) * grad_out 
    
class Sigmoid(Function):
    @staticmethod
    def forward(input):
        sig = 1 / (1 + np.exp(-input.data))
        ctx = Context()
        ctx.save_for_backward(Tensor(sig))
        ctx.function = Sigmoid
        return Tensor(sig, ctx=ctx)

    @staticmethod 
    def backward(ctx: Context, grad_out):
        sig, = ctx._prev 
        return grad_out * sig.data * (1 - sig.data)

class Tanh(Function):
    @staticmethod 
    def forward(input):
        tanh_val = np.tanh(input.data)
        ctx = Context()
        ctx.save_for_backward(Tensor(tanh_val))
        ctx.function = Tanh
        return Tensor(tanh_val, ctx=ctx)

    @staticmethod 
    def backward(ctx: Context, grad_out):
        tanh_val, = ctx._prev 
        return (1 - tanh_val.data ** 2) * grad_out 

class Sum(Function):
    @staticmethod
    def forward(input):
        ctx = Context()
        ctx.save_for_backward(input)
        ctx.function = Sum 
        return Tensor(input.data.sum(), ctx=ctx)
    
    @staticmethod
    def backward(ctx: Context, grad_out):
        input, = ctx._prev
        return grad_out * np.ones_like(input.data)
    
class LogSoftmax(Function):
    @staticmethod
    def forward(input: Tensor):
        ctx = Context()
        c_max = input.data.max(axis=1, keepdims=True)
        logsumexp = c_max + np.log(np.sum(np.exp(input.data - c_max), axis=1, keepdims=True))
        output = input.data - logsumexp
        ctx.save_for_backward(Tensor(output))
        ctx.function = LogSoftmax
        return Tensor(output, ctx=ctx)

    @staticmethod
    def backward(ctx: Context, grad_out):
        (log_softmax,) = ctx._prev
        softmax = np.exp(log_softmax.data)
        return grad_out - softmax * np.sum(grad_out, axis=1, keepdims=True)
