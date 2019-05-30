import numpy as np
from sigmoid import sigmoid


def predict(theta1, theta2, x):
    m = len(x)
    x0 = np.ones((m,1))
    h1 = sigmoid(np.matmul(np.hstack((x0,x)), np.transpose(theta1)))
    h2 = sigmoid(np.matmul(np.hstack((x0,h1)), np.transpose(theta2)))
    p = np.argmax(h2,axis = 1)
    return p + 1