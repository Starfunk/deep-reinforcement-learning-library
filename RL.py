#network_library reinforcment learning

# ~ # A library that implements the Network class

# BIG NOTE: THE INPUT ARRAY MUST BE THE SAME SIZE AS THE INPUT LAYER
# IF THESE LENGTHS ARE NOT THE SAME, THE NETWORK WILL NOT BE ABLE TO
# FORWARD PROPAGATE.


import numpy as np
import random
#### Defining the Neural Network class



class Network:
	
	"""Initialize network with variables for net length , netshape (i.e. 
	number of layers and number of neurons per each layer, the biases, 
	and the weights."""
	def __init__(self, netshape):
		self.netlength = len(netshape)
		self.netshape = netshape
		self.biases = self.init_bias(netshape)
		# ~ np.random.randn(1, i) for i in (netshape[1:])
		# Initialize an array of arrays. The nth subarray holds the 
		# weights for the nth layer. 
		self.weights = [np.random.randn(j, i) for i, j in 
			zip(netshape[0:], netshape[1:])]
		
		self.learning_rate = 0.1
		self.gamma = 0.9
			
	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		
		for i in range(self.netlength - 1):
			w = self.weights[i]
			b = self.biases[i]
			a = sigmoid(np.dot(w, a)+b)
		# ~ for b, w in zip(self.biases, self.weights):
			
			# ~ print(a)
			# ~ a = sigmoid(np.dot(w, a)+b)
		return a
		
	# returns an array of each layers activations - similar to feedforward
	# but with more information. Also returns the z values for each layer
	def get_activations(self, a):
		
		activations = [np.asarray(a)]
		zs = []
		for i in range(self.netlength - 1):
			w = self.weights[i]
			b = self.biases[i]
			z = np.dot(w, a) + b
			zs.append(z)
			a = sigmoid(z)
			activations.append(a)
	
		return activations, zs
		
		
	# Initialize the bias in this function. Run a loop, for the nth loop
	# create a 1 dimensional array that represents the biases for the nth
	# layer (we take the 0th index because the array created is within
	# another array! since biases are always 1 per neuron, the array
	# representing the biases for a given layer will always be 1-D, so
	# this is fine.	
	def init_bias(self, netshape):
		biases = []
		for i in netshape[1:]:
			biases.append(np.random.randn(1, i)[0])	
		return biases
		
	# We also need the first input activations for backprop!
	def backpropagation(self, reward, Q, activations, z, action): 
		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		print("The loss is: " + str(reward + self.gamma * max(Q) - activations[-1][action]))
		#may have to turn these arrays into np arrays to do hadamard product!
		# represents initially the last layer's error and then it represents the
		# current layer + 1's error in the for loop  below. The index -1 indexes 
		# the last element in the array. 
		error_L = (reward + self.gamma * max(Q) - activations[-1][action]) * sigmoid_prime(z[-1][action])
		
		nabla_b[-1] = error_L
		
		# The weight for the last layer is the error times the activation in the second last layer
		nabla_w[-1] = np.dot(np.asarray([error_L]).transpose(), np.asarray([activations[-2]]))
		
		# ~ print(nabla_w)
		# ~ print("FOR ERROR CAPTIAL L")
		# ~ print("error_L")
		# ~ print(error_L)
		# ~ print()
		# ~ print("Nabla W UPDATE")
		# ~ print(np.dot(np.asarray([error_L]).transpose(), np.asarray([activations[-2]])))
		# ~ print()
		
		# calculate the error_l for each layer BEFORE THE LAST LAYER, which
		# we have already calculated above thats why we have -2 rather than -1.
		for i in reversed(range(self.netlength - 2)):
			
			# ~ print()
			# ~ print("TOTAL RANGE: " + str(self.netlength - 2))
			# ~ print("This is the value of i: " + str(i))
			# ~ print()
			# NOTE RIGHT NOW ERROR IS TRANSPOSED NOT WEIGHTS, THIS IS DIFFERENT FROM NIELSEN!
	
			# ~ print(np.dot(self.weights[i],  error_L))
			# ~ print()
			# ~ print(sigmoid_prime(z[i]))
			# ~ print()
			# ~ print("weights")
			# ~ print(self.weights[i].transpose())
			# ~ print()
			# ~ print("error_L")
			# ~ print(error_L)
			# ~ print()
			# ~ print("sigmoid_prime(z)")
			# ~ print(sigmoid_prime(z[i-1]))
			# ~ print()
			# ~ print(np.dot(self.weights[i + 1].transpose(),  error_L))
			
			
			# THE INDEX IS ONLY O WHICH CORRESPONDS WITH THE WEIGHTS FROM LAYERS
			# 1 to 2. 
			error_L = np.dot(self.weights[i + 1].transpose(),  error_L) * sigmoid_prime(z[i][action])
			

			#PLAY AROUND WITH THESE INDICES THERE IS SOMETHING FISHY GOING ON HERE!
			nabla_b[i] = error_L
			
			# ~ print("NEW ERROR_L")
			# ~ print(error_L)
			# ~ print(np.asarray([error_L]).transpose())
			# ~ print()
			# ~ print("NEW ACTIV")
			# ~ print(activations[i])
			# ~ print()
			#PLAY AROUND WITH THESE INDICES THERE IS SOMETHING FISHY GOING ON HERE!
			nabla_w[i] = np.dot(np.asarray([error_L]).transpose(), np.asarray([activations[i]]))

		return nabla_b, nabla_w
	
	def update_net(self, reward, Q, activations, z, action): 
		nabla_b, nabla_w = self.backpropagation(reward, Q, activations, z, action)
		
		self.biases = self.biases - np.asarray(self.learning_rate) * nabla_b
	
		self.weights = self.weights - np.asarray(self.learning_rate) * nabla_w
	
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
	
def sigmoid_prime(z):
	return(np.exp(z) / ((np.exp(z) + 1) ** 2))
	
def cost_derivative(self, output_activations, y):
	"""Return the vector of partial derivatives \partial C_x / \partial 
	a for the output activations."""
	return (output_activations-y)
	

