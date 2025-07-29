class Context:
    def __init__(self, prev: list=None):
        self._prev = prev
        self.function = None 
    
    def save_for_backward(self, *tensors):
        self._prev = tensors 
