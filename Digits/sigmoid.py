import numpy as np

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm
