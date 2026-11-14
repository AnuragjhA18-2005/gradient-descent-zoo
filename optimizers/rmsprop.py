from . import Optimizer 

import numpy as np 

class RMSprop(Optimizer):
    def __init__(self,lr=0.001,rho=0.9,epsilon=1e-8):
        super().__init__(lr)
        self.rho=float(rho)
        if self.rho<0 or self.rho>=1:
            raise ValueError("Rho should be in the range [0, 1)")
        self.epsilon=float(epsilon)
        if self.epsilon<=0:
            raise ValueError("Epsilon should be Positive")
        self.avg_squared_grads=None


    def step(self,params,grads):
        self.check_length(params,grads)
        if self.avg_squared_grads is None:
            self.avg_squared_grads=[]
            for i in range(len(params)):
                self.avg_squared_grads.append(np.zeros_like(params[i]))


        for i in range(len(params)):
            if params[i].shape != grads[i].shape:
                raise ValueError(f"Shape of parameter {i} and gradient {i} must be the same")

            current_grads=grads[i]
            self.avg_squared_grads[i]=self.rho*self.avg_squared_grads[i] + (1-self.rho)*(current_grads**2)
            rms_squared_grads=np.sqrt(self.avg_squared_grads[i]+self.epsilon)

            delta_params = -(self.lr/rms_squared_grads)*current_grads

            params[i]+=delta_params