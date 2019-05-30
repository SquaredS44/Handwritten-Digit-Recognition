import numpy as np

def sigmoidGradient(x):
    sigm = 1. / (1. + np.exp(-x))
    sigmoidgradient = sigm * (1. - sigm)
    return sigmoidgradient


