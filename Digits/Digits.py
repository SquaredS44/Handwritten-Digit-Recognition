import numpy as np
from nnCostFunction import nnCostFunction
import scipy.io
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInit
from checkNNGradients import checkNNGradients
from scipy.optimize import minimize
from predict import predict
from displayData import displayData
# parameters setup
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
# Load Training Data
mat = scipy.io.loadmat('ex4data1.mat')
x = mat["X"]
y = mat["y"]
m = len(x)
y = y.flatten()
# Visualize data
rand_indices = np.random.permutation(m)
sel = x[rand_indices[:100], :]
displayData(sel)
# Debugging parameters
mat = scipy.io.loadmat('ex4weights.mat')
theta1 = mat["Theta1"]
theta2 = mat["Theta2"]
nn_params = np.hstack((theta1.flatten(),theta2.flatten()))
# compute  the cost
lambda_reg = 0
J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg)
print(J) #yeyyyyy
# cost with regularization
lambda_reg = 1
J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg)
print(J) #yey
# evaluate sigmoid gradient
g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
#initialize theta1 and theta2
initial_theta1 = randInit(input_layer_size, hidden_layer_size)
initial_theta2 = randInit(hidden_layer_size, num_labels)
initial_nn_params = np.hstack((initial_theta1.flatten(),initial_theta2.flatten()))
# check gradient
checkNNGradients(lambda_reg) #yey
# train nn
maxiter = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, x, y, lambda_reg)
results = minimize(nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)
nn_params = results["x"]
theta1 = np.reshape(nn_params[:(hidden_layer_size*(input_layer_size+1))], (hidden_layer_size, input_layer_size + 1))
theta2 = np.reshape(nn_params[(hidden_layer_size*(input_layer_size+1)):], (num_labels, hidden_layer_size + 1))
# predict
pred = predict(theta1, theta2, x)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))










