

from optimizers import Optimizer


op = Optimizer(-0.1)

op.step([1, 2, 3], [0.1, 0.2, 0.3])
    