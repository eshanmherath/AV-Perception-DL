import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


learning_rate = 0.5
x = np.array([1, 2])
y = np.array(0.5)

# Initial weights
w = np.array([0.5, -0.5])

# Calculate one gradient descent step for each weight
nn_output = sigmoid(np.dot(x, w))

# Calculate error of neural network
error = y - nn_output

# Calculate change in weights
del_w = learning_rate * error * nn_output * (1 - nn_output) * x

print('Neural Network output:' + str(nn_output))
print('Amount of Error:' + str(error))
print('Change in Weights:' + str(del_w))
