import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])


weights_input_to_hidden  = test_w_i_h.copy()                    
weights_hidden_to_output = test_w_h_o.copy()


hidden_outputs = sigmoid(np.matmul(inputs,weights_input_to_hidden))
print(np.matmul(inputs,weights_input_to_hidden))
outputs = sigmoid(np.matmul(hidden_outputs,weights_hidden_to_output))

print (outputs)