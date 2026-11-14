from . import Optimizer
import numpy as np


class Momentum(Optimizer):
    def __init__(self, lr, momentum=0.9):
        super().__init__(lr)
        self.momentum = float(momentum)
        if self.momentum < 0 or self.momentum >= 1:
            raise ValueError("Momentum should be in the range [0, 1)")
        self.velocity = None

    def step(self, params, grads):
        self.check_length(params,grads)
        if self.velocity is None:
            self.velocity = []
            for i in range(len(params)):
                self.velocity.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            if params[i].shape != grads[i].shape:
                raise ValueError(f"Shape of parameter {i} and gradient {i} must be the same")
            
            old_velocity = self.velocity[i]
            current_grad = grads[i]
            new_velocity = self.momentum * old_velocity + self.lr * current_grad
            self.velocity[i] = new_velocity
            params[i] -= new_velocity
