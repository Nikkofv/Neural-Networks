#!/usr/bin/env python
""" A Slightly Harder Problem: 3 Layer Neural Network
Problem type:	Nonlinear pattern where there is a one-to-one relationship between a combination of inputs
Summary: 	Creates and trains a 3 layer sigmoid neural network with 3 inputs, 4 hidden nodes  and 1 output
Strategy:	Combine pixels into something that can have a one-to-one relationship with the output
		==> Add another layer
			First layer will comine the inputs
			Second layer will map first layer inputs to the output using the output of the first layer as input
		 
Source:		http://iamtrask.github.io/2015/07/12/basic-python-network/
Date:		01/29/2017
Possible applications:	shape recognition from image inputs
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
y = np.array([[0,1,1,0]]).T
"""Output dataset matrix where each row is a training example"""
print("Output training Example")
print(y)

# Seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1
"""First layer of weights, Synapse 0, connecting l0 to l1"""
syn1 = 2*np.random.random((4,1)) - 1

for iter in range(60000):

	# Feed forward through layers 0, 1, and 2
	l0 = X	# First layer of the network, specified by the input data
	l1 = nonlin(np.dot(l0,syn0))	# Second layer of the Network, other wise known as the hidden layer
	l2 = nonlin(np.dot(l1,syn1))	# Final layer of the netwok, which is our hypothesis, and should approximate the correct answer as we train.

	# How much did we miss the target value?
	l2_error = y-l2
	
	# Incrementally Report the model error
	if(iter% 10000) == 0:
		print("Error:",str(np.mean(np.abs(l2_error))))

	# In what direction is the target value?
	# Were we really sure? If so, don't change much
	l2_delta = l2_error * nonlin(l2,deriv=True)


	# How much did each l1 value contribute to the l2 error
	# (according to the wieghts)?
	l1_error = np.dot(l2_delta,syn1.T)	# Back propogation

	# In what direction is the target l1?
	# Were we really sure? If so, don't change much
	l1_delta = l1_error * nonlin(l1,deriv=True)

	# Update weights
	syn1 += np.dot(l1.T,l2_delta)
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

