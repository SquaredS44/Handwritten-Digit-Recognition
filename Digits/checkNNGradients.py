import numpy as np
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction
from computeNumericalGradient import computeNumericalGradient


def checkNNGradients(lambda_reg):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # generate some random test data
    theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    x = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.transpose(np.mod(range(m), num_labels))
    # unroll parameters theta1 and theta2
    nn_params = np.hstack((theta1.flatten(), theta2.flatten()))
    # short hand for cost function
    def costFunc(p):
        return nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg)
    J, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)
    print(np.c_[grad, numgrad])
    # compute the difference
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print(diff)
    return




