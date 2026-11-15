class Optimizer:
    def __init__(self,lr):
        self.lr=float(lr)
        if self.lr <=0:
            raise ValueError("Learning rate cannot be Negative or Zero")


    def step(self,params,grads):
        # params is a list of NumPy arrays, grads is a list of the same length and same shapes, and the method updates params in place and returns nothing
        raise NotImplementedError("This method should be overridden by subclasses")

    def check_length(self,params,grads):
        if len(params)!=len(grads):
            raise ValueError("Length of params and grads must be Equal")
        

from .sgd import SGD
from .momentum import Momentum
from .nag import NAG
from .adagrad import AdaGrad
from .adadelta import AdaDelta
from .rmsprop import RMSprop
from .adam import Adam

