from . import Optimizer
import numpy as np

class AdaGrad(Optimizer):
    def __init__(self,lr=0.01,epsilon=1e-8):
        super().__init__(lr)
        self.epsilon=float(epsilon)
        if self.epsilon<=0:
            raise ValueError("Epsilon should be Positive")
        self.accumulated_grads=None

    def step(self,params,grads):
        self.check_length(params,grads)
        if self.accumulated_grads is None:
            self.accumulated_grads=[]
            for i in range(len(params)):
                self.accumulated_grads.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            if params[i].shape != grads[i].shape:
                raise ValueError(f"Shape of parameter {i} and gradient {i} must be the same")
            current_grads=grads[i]
            self.accumulated_grads[i]+=current_grads**2
            denominator=np.sqrt(self.accumulated_grads[i]+self.epsilon)
            params[i]-=(self.lr/denominator)*current_grads
            