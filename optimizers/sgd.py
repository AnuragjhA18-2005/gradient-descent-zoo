from . import Optimizer


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def step(self, params, grads):
        self.check_length(params, grads)
        for i in range(len(params)):
            if params[i].shape != grads[i].shape:
                raise ValueError(f"Shape of parameter {i} and gradient {i} must be the same")
            params[i] -= self.lr * grads[i]
