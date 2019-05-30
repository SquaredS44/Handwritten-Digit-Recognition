import numpy as np


def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape[0])
    perturb = np.zeros(theta.shape[0])
    e = 1e-4
    for p in range(theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2[0] - loss1[0]) / (2*e)
        perturb[p] = 0
    return numgrad
