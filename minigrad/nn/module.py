class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> list:
        raise NotImplementedError("Module.paramaters must return a list!")