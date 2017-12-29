import numpy as np

# XOR training data
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

y = np.array([[0, 1, 1, 0]]).T


# Nonlinear Function is sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of the sigmoid function
def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Cost function and its derivative
def cost_function(prediction, answer):
    difference = prediction - answer
    error = np.square(difference) / 2
    return error


def cost_function_deriv(prediction, answer):
    return answer - prediction

# The network will have 3 layers in total

# Initializing weights
l1_weights = np.random.random((3, 6))
l2_weights = np.random.random((6, 1))

#
for e in range(10000):
        # input layer
        l1_activation = x

        # hidden layer
        l2_input = np.dot(l1_activation, l1_weights)
        l2_activation = sigmoid(l2_input)

        # output layer
        l3_input = np.dot(l2_activation, l2_weights)
        l3_activation = sigmoid(l3_input)

        # deltas
        delta_l3 = cost_function_deriv(l3_activation, y) * sigmoid_deriv(l3_activation)
        delta_l2 = np.dot(delta_l3, l2_weights.T) * sigmoid_deriv(l2_activation)

        # changes in weights
        change_w1 = np.dot(l1_activation.T, delta_l2)
        change_w2 = np.dot(l2_activation.T, delta_l3)

        l1_weights += change_w1
        l2_weights += change_w2

# testing data
test = np.array([[0,0,0], [1,1,1], [0,1,1]])

# feedforward of test data
l1_activation = test
l2_input = np.dot(l1_activation, l1_weights)
l2_activation = sigmoid(l2_input)
l3_input = np.dot(l2_activation, l2_weights)
l3_activation_Test = sigmoid(l3_input)

# expected: 0, 0, 1
print('Output of Test: \n{0} \n'.format(l3_activation_Test))