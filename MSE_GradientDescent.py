import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deri_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

x = np.array([0.1,0.3])
y = 0.2

weights = np.array([-0.8,0.5])

learnrate = 0.5

h = x[0]*weights[0] + x[1] * weights[1]

nn_output = sigmoid(h)

error = y - nn_output

outgradient = error * deri_sigmoid(h)

error_term = error*outgradient

del_w= [ learnrate* error_term*x[0], learnrate * error_term*x[1]]


print(del_w)
