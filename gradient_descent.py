# Code for Gradient Descent Algorithm for any input size
import numpy as np 
import matplotlib.pyplot as plt 
n_updates = 20
mse_hist = []
def GD(input_data, target, weights, learning_rate):
	for i in range (n_updates):
		# calculation of slope of MSE
		prediction = (input_data*weights).sum() # not using transpose property of matrix yet
		error = prediction - target
		slope_mse = 2*error*input_data
		# updating weights iteratively
		weights = weights - learning_rate*slope_mse
		mse = (target - prediction)**2
		a = mse_hist.append(mse)
	# from gradient descent, we get weghts and MSEs at each iteration
	return weights, mse_hist	
	
	
# a test input, target, weights and learning rate
test_input = np.array([1,2,3])
test_target = 10
test_weights = np.array([0,-1,2])
test_learning_rate = 0.01
# implementing our defined gradient descent function 
updated_weights, test_mse = GD(test_input, test_target, test_weights, test_learning_rate)
# printing updated weights and MSEs for test set
print "updated weights:", updated_weights
print "Mean squared errors:", mse_hist
# plotting MSE vs number of iterations curve
plt.plot(mse_hist)
plt.xlabel('iterations')
plt.ylabel('mean squared error(MSE)')
plt.show()


	

