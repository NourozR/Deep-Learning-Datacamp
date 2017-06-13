import numpy as np 

def ReLU(x):
	return max(0,x)	

input_data = np.array([-2,2])

weights = {'node_0': np.array([1,1]),'node_1': np.array([-1,1]),'output': np.array([2,-1])}
node_0_input = (input_data*weights['node_0']).sum()
node_1_input = (input_data*weights['node_1']).sum()
node_0_output = ReLU(node_0_input)
node_1_output = ReLU(node_1_input)
hidden_layer_val = np.array([node_0_output, node_1_output])
output = (hidden_layer_val*weights['output']).sum()

print output  
         