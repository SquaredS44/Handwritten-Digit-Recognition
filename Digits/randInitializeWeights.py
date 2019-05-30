import numpy as np


def randInit(Lin, Lout):
    epsilonInit = 0.12
    w = np.random.rand(Lout, 1 + Lin)*2*epsilonInit-epsilonInit
    return w


