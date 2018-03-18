import numpy as np

n_features = 10

weights =  np.random.normal(scale=1/n_features**.5,size=n_features)

print (weights)


# Initialize 2 dimensional weights for a layer
# This layer has 2 neurons and each neuron get 3 inputs
# w_xy: x :for matching up with input, y for matching up nth neuron
#
#   | w00 w01 |
#   | w10 w11 |
#   | w20 w21 |
#
    
weights2 = np.random.normal(scale = 1/n_features **.5,size=(3,2))

print (weights2)


# Experiment With Row and columns and Transpose

weights3 = weights2.T

print (weights3)

# Transpose second way vector as 2D
vec1 = np.array([1,2,3])

vec2 = vec1[:,None]

print (vec2)