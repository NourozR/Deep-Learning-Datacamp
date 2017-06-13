# A forward neural network with 2 inputs, 2 hidden layers and each hidden layer with 2 nodes

import numpy as np 

def relu(x): # using Relu as activation functions
	return max(0,x)

input_data =np.array([3,5]) # weight matrix for each node
weights = { 'node_0_0': np.array([2,4]),
            'node_0_1': np.array([4,-5]),
            'node_1_0': np.array([-1,2]),
            'node_1_1': np.array([1,2]),
            'output': np.array([2,7]) }

node_0_0_val = (input_data*weights['node_0_0']).sum()
node_0_0_val = relu(node_0_0_val)
node_0_1_val = (input_data*weights['node_0_1']).sum()
node_0_1_val = relu(node_0_1_val)
first_layer_output = np.array([node_0_0_val, node_0_1_val]) # output from first layer
node_1_0_val = (first_layer_output*weights['node_1_0']).sum()
node_1_0_val = relu(node_1_0_val)
node_1_1_val = (first_layer_output*weights['node_1_1']).sum()
node_1_1_val = relu(node_1_1_val)

output_val = np.array([node_1_0_val, node_1_1_val])
output = (output_val*weights['output']).sum() # final output
print output



