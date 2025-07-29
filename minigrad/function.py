import numpy as np
import tensor as t
import context as c

class Function:
    @staticmethod
    def forward(*args):
        raise NotImplementedError("For a custom function you need to implement a forward function!")

    @staticmethod
    def backward(*args):
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
    
class Relu(Function):
    @staticmethod
    def forward(input):
        ctx = c.Context()
        ctx.save_for_backward(input)
        ctx.function = Relu
        return t.Tensor(np.maximum(0, input), ctx=ctx)

    @staticmethod
    def backward(ctx: c.Context, grad_out):
        out = ctx._prev 
        return (out.data > 0) * grad_out 
    

class Sigmoid(Function):
    @staticmethod
    def forward(input):
        ctx = c.Context()
        ctx.save_for_backward(input)
        ctx.function = Sigmoid
        return t.Tensor(1 / (1 * np.exp(-input.data)))

    @staticmethod 
    def backward(ctx: c.Context, grad_out):
        out = ctx._prev 
        return (out.data * (1 - out.data)) * grad_out 
