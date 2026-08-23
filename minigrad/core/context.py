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

    def save_for_backward(self, *tensors):
        self._prev = tensors 
