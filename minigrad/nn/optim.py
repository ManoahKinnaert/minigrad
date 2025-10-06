import numpy as np

class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr 
    
    def step(self):
        raise NotImplementedError("To implement an optimizer you must implement the step function.")

class SGD(Optimizer):
    def __init__(self, decay, momentum, dampening, maximize: bool=True, nesterov: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay = decay
        self.momentum = momentum
        self.dampening = dampening
        self.maximize = maximize
        self.nesterov = nesterov
    
    def step(self):
        # update params
        for param in self.model.parameters():
            if self.maximize:
                param.data -= self.lr * param.grad 
            else:
                param.data += self.lr * param.grad

            if self.decay != 0:
                param.data += self.decay * param.grad
            if self.momentum != 0:
                if self.nesterov:
                    pass 
                
            # reset grad
            param.zero_grad()
            

"""
For details look here: https://arxiv.org/abs/1412.6980
"""
class Adam(Optimizer):
    def __init__(self, model, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.b1 = b1 
        self.b2 = b2 
        self.eps = eps 
        self.m0 = [np.zeros_like(p.data) for p in model.parameters()]
        self.v0 = [np.zeros_like(p.data) for p in model.parameters()]
        self.t = 0

    def step(self):
        self.t += 1
        for i, t in enumerate(self.model.parameters()):
            if t.grad is None:
                continue
            # update biased moment estimates
            self.m0[i] = self.b1 * self.m0[i] + (1 - self.b1) * t.grad 
            self.v0[i] = self.b2 * self.v0[i] + (1 - self.b2) * (t.grad ** 2)
            # compute bias corrected moment
            m = self.m0[i] / (1 - self.b1 ** self.t)
            v = self.v0[i] / (1 - self.b2 ** self.t)
            # update parameter
            t.data = t.data - self.lr * m / (np.sqrt(v) + self.eps)
            # reset grad 
            t.zero_grad()

