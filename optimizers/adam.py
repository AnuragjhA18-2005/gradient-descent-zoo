from . import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        super().__init__(lr)
        self.beta1=float(beta1)
        self.beta2=float(beta2)
        self.epsilon=float(epsilon)
        if self.beta1<0 or self.beta1>=1:
            raise ValueError("Beta1 should be in the range [0, 1)")
        if self.beta2<0 or self.beta2>=1:
            raise ValueError("Beta2 should be in the range [0, 1)") 
        if self.epsilon<=0:
            raise ValueError("Epsilon should be Positive")
        self.mt=None
        self.vt=None
        self.t=0


    def step(self,params,grads):
        self.check_length(params,grads)
        self.t+=1

        if self.mt is None:
            self.mt=[]
            self.vt=[]
            for i in range(len(params)):
                self.mt.append(np.zeros_like(params[i]))
                self.vt.append(np.zeros_like(params[i]))

        for i in range(len(params)):
            if params[i].shape != grads[i].shape:
                raise ValueError(f"Shape of parameter {i} and gradient {i} must be the same")
            current_grads=grads[i]
            self.mt[i] = self.beta1*self.mt[i] + (1-self.beta1)*current_grads
            self.vt[i] = self.beta2*self.vt[i] + (1-self.beta2)*current_grads**2

            mt_hat = self.mt[i]/(1.0-self.beta1**self.t)
            vt_hat = self.vt[i]/(1.0-self.beta2**self.t)

            delta_params = -(self.lr/(np.sqrt(vt_hat)+self.epsilon))*mt_hat

            params[i]+=delta_params



        