import numpy as np
import minigrad.tensor as t
import minigrad.context as c

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
    def forward(x: t.Tensor, y: t.Tensor):
        ctx = c.Context()
        ctx.save_for_backward(x, y)
        ctx.function = Add
        return t.Tensor(x.data + y.data, ctx=ctx) 
    
    @staticmethod
    def backward(ctx: c.Context, grad_out):
        x, y = ctx._prev 
        return grad_out, grad_out 
    
class Mul(Function):
    @staticmethod
    def forward(x: t.Tensor, y: t.Tensor):
        ctx = c.Context()
        ctx.save_for_backward(x, y)
        ctx.function = Mul 
        return t.Tensor(x.data * y.data, ctx=ctx)
    
    @staticmethod
    def backward(ctx: c.Context, grad_out):
        x, y = ctx._prev 
        return y.data * grad_out, x.data * grad_out 

class Dot(Function):
    @staticmethod
    def forward(x: t.Tensor, y: t.Tensor):
        ctx = c.Context()
        ctx.save_for_backward(x, y)
        ctx.function = Dot 
        return t.Tensor(np.dot(x.data, y.data), ctx=ctx)

    @staticmethod 
    def backward(ctx: c.Context, grad_out):
        x, y = ctx._prev 
        return np.dot(grad_out, y.data.T), np.dot(x.data.T, grad_out)

class Relu(Function):
    @staticmethod
    def forward(input):
        ctx = c.Context()
        ctx.save_for_backward(input)
        ctx.function = Relu
        return t.Tensor(np.maximum(0, input.data), ctx=ctx)

    @staticmethod
    def backward(ctx: c.Context, grad_out):
        input, = ctx._prev 
        return (input.data > 0) * grad_out 
    

class Sigmoid(Function):
    @staticmethod
    def forward(input):
        sig = 1 / (1 + np.exp(-input.data))
        ctx = c.Context()
        ctx.save_for_backward(t.Tensor(sig))
        ctx.function = Sigmoid
        return t.Tensor(sig, ctx=ctx)

    @staticmethod 
    def backward(ctx: c.Context, grad_out):
        sig, = ctx._prev 
        return grad_out * sig.data * (1 - sig.data)

class Tanh(Function):
    @staticmethod 
    def forward(input):
        tanh_val = np.tanh(input.data)
        ctx = c.Context()
        ctx.save_for_backward(t.Tensor(tanh_val))
        ctx.function = Tanh
        return t.Tensor(tanh_val, ctx=ctx)

    @staticmethod 
    def backward(ctx: c.Context, grad_out):
        tanh_val, = ctx._prev 
        return (1 - tanh_val.data ** 2) * grad_out 

class Sum(Function):
    @staticmethod
    def forward(input):
        ctx = c.Context()
        ctx.save_for_backward(input)
        ctx.function = Sum 
        return t.Tensor(input.data.sum(), ctx=ctx)
    
    @staticmethod
    def backward(ctx: c.Context, grad_out):
        input, = ctx._prev
        return grad_out * np.ones_like(input.data)
    
class LogSoftmax(Function):
    @staticmethod
    def forward(input: t.Tensor):
        ctx = c.Context()
        c_max = input.data.max(axis=1, keepdims=True)
        logsumexp = c_max + np.log(np.sum(np.exp(input.data - c_max), axis=1, keepdims=True))
        output = input.data - logsumexp
        ctx.save_for_backward(t.Tensor(output))
        ctx.function = LogSoftmax
        return t.Tensor(output, ctx=ctx)

    @staticmethod
    def backward(ctx: c.Context, grad_out):
        (log_softmax,) = ctx._prev
        softmax = np.exp(log_softmax.data)
        return grad_out - softmax * np.sum(grad_out, axis=1, keepdims=True)