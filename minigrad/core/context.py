"""
Define context class.
"""
class Context:
    def __init__(self, prev: list=None):
        self._prev = prev
        self._function = None 

    @property
    def function(self): return self._function

    @function.setter
    def function(self, function): self._function = function 

    @property 
    def prev(self): return self._prev 

    def save_for_backward(self, *tensors):
        if not tensors or any(t is None for t in tensors): raise ValueError("You can't save Nonetype for backward pass!")
        self._prev = tensors 

    def copy(self):
        ctx = Context(self._prev.copy())
        ctx.function = self._function
        return ctx