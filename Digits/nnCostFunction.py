import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lam):
    # reshape nn_params into Theta1 and Theta2
    theta1 = np.reshape(nn_params[:(hidden_layer_size*(input_layer_size+1))], (hidden_layer_size, input_layer_size + 1))
    theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):], (num_labels, hidden_layer_size + 1))
    # setup useful variables
    m = len(x)
    x0 = np.ones((m, 1))
    # recode y [1,3,4,5,...] to Y [[1000000000],[0010000000],...]
    I = np.identity(num_labels)
    Y = np.zeros((m, num_labels))
    for row in range(len(Y)):
        # y has to be an array of int
        Y[row, :] = I[(y[row]-1), :]
    # Feedforward Propagation (calculate hypothesis)
    a1 = np.hstack((x0, x))
    z2 = np.matmul(a1, np.transpose(theta1))
    a2bias = np.ones((len(z2), 1))
    a2 = np.hstack((a2bias, sigmoid(z2)))
    z3 = np.matmul(a2, np.transpose(theta2))
    h = sigmoid(z3)
    # calculate penalty (due to regularization)
    # remove bias units from Theta1 and Theta2
    theta1_unbias = theta1[:, 1:]
    theta2_unbias = theta2[:, 1:]
    theta1_pow = np.power(theta1_unbias, 2)
    theta2_pow = np.power(theta2_unbias, 2)
    colSum1 = theta1_pow.sum(axis=1)
    colSum2 = theta2_pow.sum(axis=1)
    p = np.sum(colSum1) + np.sum(colSum2)
    # calculate J (cost)
    cost1 = (-Y) * np.log(h)
    cost2 = (1 - Y) * np.log(1 - h)
    cost = cost1 - cost2
    colSumCost = cost.sum(axis=1)
    J = (1.0/m)*np.sum(colSumCost)+lam/(2*m)*p
    # calculate sigmas (error in each layer, for the last layer p = h - y)
    sigma3 = h - Y
    sigma2 = (np.matmul(sigma3, theta2))*sigmoidGradient(np.hstack((a2bias, z2)))
    sigma2 = sigma2[:, 1:]
    # accumulate gradients
    delta1 = np.matmul(np.transpose(sigma2), a1)
    delta2 = np.matmul(np.transpose(sigma3), a2)
    # calculate regularized gradient
    theta10 = np.zeros((len(theta1), 1))
    theta20 = np.zeros((len(theta2), 1))
    theta1_reg = np.hstack((theta10, theta1_unbias))
    theta2_reg = np.hstack((theta20, theta2_unbias))
    p1 = lam/m * theta1_reg
    p2 = lam/m * theta2_reg
    theta1_grad = delta1/m + p1
    theta2_grad = delta2/m + p2
    # unroll gradient
    grad = np.hstack((theta1_grad.flatten(), theta2_grad.flatten()))
    return [J, grad]
