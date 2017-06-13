# 3 different inputs with their corresponding weights to calculate mean square error of the system
import numpy as np
import math
# using Relu as activation functions
def relu(x): 
    return max(0,x)


def predict_with_network(input_data, weights):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)


input_data_1 = np.array([3,5]) 
input_data_2 = np.array([-2,5])
input_data_3 = np.array([1,-2])

# weight matrix for each node
weights_1 = { 'node_0_0': np.array([2,4]),
            'node_0_1': np.array([4,-5]),
            'node_1_0': np.array([-1,2]),
            'node_1_1': np.array([1,2]),
            'output': np.array([2,7]) }

weights_2 = { 'node_0_0': np.array([-1,4]),
            'node_0_1': np.array([4,-2]),
            'node_1_0': np.array([-1,1]),
            'node_1_1': np.array([3,2]),
            'output': np.array([2,1]) }

weights_3 = { 'node_0_0': np.array([-1,2]),
            'node_0_1': np.array([1,-2]),
            'node_1_0': np.array([-1,3]),
            'node_1_1': np.array([3,-2]),
            'output': np.array([3,-4]) }            

# Predicted outputs of inputs and corresponding weights
predicted_output_1 = predict_with_network(input_data_1, weights_1)
predicted_output_2 = predict_with_network(input_data_2, weights_2)
predicted_output_3 = predict_with_network(input_data_3, weights_3)
# all predicted outputs
predicted_output = np.array([predicted_output_1, predicted_output_2, predicted_output_3])
# actual outputs/target outputs
actual_output = 175
print predicted_output_1, '\n' , predicted_output_2, '\n', predicted_output_3
# calculation of mean squared error (MSE)
MSE = math.sqrt(((actual_output - predicted_output)**2).mean(axis = 0))
print MSE




