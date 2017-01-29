#!/usr/bin/env python
""" Tiny Toy Network: 2 Layer Neural Network
Summary: 	Creates and trains a 2 layer sigmoid neural network with 3 inputs and 1 output
Source:		http://iamtrask.github.io/2015/07/12/basic-python-network/
Date:		01/29/2017
"""
import numpy as np

# Sigmoid Function
def nonlin(x,deriv=False):
	"""This nonlinearity function is a sigmoid function
	A Sigmoid funtion maps any value to a value between 0 and 1

	When deriv=True	the derivative of the sigmoid funtion is given
	"""
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

# Input Dataset: Training Example
X = np.array([	[0,0,1],
		[0,1,1],
		[1,0,1],
		[1,1,1]	])
"""Input dataset matrix where each row is a training example"""
print("Input training Example")
print(X)

# Output Dataset
y = np.array([[0,0,1,1]]).T
"""Output dataset matrix where each row is a training example"""
print("Output training Example")
print(y)

# Seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# Initialize weights, between the 3 input and 1 output neuron, randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
"""First layer of weights, Synapse 0, connecting l0 to l1"""

for iter in range(10000):

	# Forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))

	# How much did we miss?
	l1_error = y-l1
	
	# Multiply how much we missed by the
	# slope of the sigmoid at the values in l1
	l1_delta = l1_error * nonlin(l1,True)

	# Update weights
	syn0 += np.dot(l0.T,l1_delta)
	if(iter==0):
#		print("Multiply how much we missed by the slope of the sigmoid at the values in l1")
#		print("l1_error = ",l1_error,"   How much we missed")
#		print("l1 = ",l1)
#		print("nonlin(l1,True) = ","1/(1+np.exp(-l1)) = ","1/(1+",np.exp(-l1),") = ",nonlin(l1,True),"   Slope of sigmoid at the values in l1")
#		print("l1_delta = ","l1_error * nonlin(l1,True) = ",l1_delta)
#		print("Update weights")
#		print("l0.T = ",l0.T)
#		print("syn0 += np.dot(l0.T,l1_delta) = ",syn0)
		print("Output after first iteration of training")
		print(l1)
print("Output After Last Iteration of Training:")
print(l1)

