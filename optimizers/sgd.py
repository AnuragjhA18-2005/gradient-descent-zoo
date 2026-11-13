from . import Optimizer 

class SGD(Optimizer):
    def __init__(self,lr):
        super().__init__(lr)

    def step(self,params,grads):
        for p,g in zip(params,grads):
            p -= self.lr*g