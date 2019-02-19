"""This file implements a neural network library 'from scratch', i.e. 
only using numpy to implement the matrix data structures used to construct 
neural networks. Precisely, you can use this library to create 
feedforward neural networks; this library does not support the creation 
of CNNs or RNNs. 

Credit: Much of the code and functions in this library were inspired by 
Michael Nielsen's own neural network library: 
https://github.com/mnielsen/neural-networks-and-deep-learning, and his
online textbook: 'Neural networks and deep learning'. Without this
amazing textbook, this project would not have been possible.

Indeed, much of the code in this library looks similar to Nielsen's 
network.py file. However, there is a key aspect of this library that 
distinguishes it from Nielsen's: The biases of the neural network are
initialized in Nielsen's library to be vertical vectors, whereas they
are initialized to be horizontal vectors in this library. This minor
difference turns out to change the specifics of the matrix 
multiplication and arithmetic steps involved in the gradient descent and 
backpropagation functions. Another important distinction between
Nielsen's library and this library is that given the netshape = [2,3,2],
Nielsen's network.py program outputs a 2x3 matrix, that is, 
a matrix of 2 rows and 3 columns. On the other hand, a network initialized
using this library outputs a 1x2 vector. To me, a 1x2 vector makes much
more sense, as what we are interested in are the activations of
the final layer of neurons and it is easy to interpret the two elements
of a 1x2 vector, [a,b], as the outputs of the final layer of neurons.
It is unclear to me how one is to interpret a 2x3 matrix as the output.
Nevertheless, Nielsen's library served as an invaluable resource when
writing this library.  """


import numpy as np
import random
import json


class Network:
	
	"""The Network class is used to initialize neural
	networks.
	
	Networks are initialized with the shape given by netshape. For
	example, if netshape = [2,3,2], then the input layer of 
	neurons accepts numpy arrays (NPAs) of the form: [a,b], and outputs NPAs of
	of the form: [c,d]. This example network would have one hidden 
	layer of 3 neurons. Each layer of biases in the network
	are contained in a NPA of shape (n,) where n is the number of 
	neurons in that layer. For example, in our example network, the 
	biases could look something like this:
	
	[	
		array([ 1.24740072, -0.69648469,  2.04505759]), 
	
		array([ 0.39117851, -0.86469781])
	]
	
	This was a set of biases generated using this library for this 
	specific network architecture. Note that there are no biases for 
	the first layer of the network as is the standard convention for 
	neural networks. The first subarray represents the biases for the 
	3 neurons in the hidden layer. The final subarray represents the 
	biases for the two output neurons. 
	
	The weights are initialized in a similar way. Here, the first 
	subarray holds the weights connecting the first layer to the second 
	layer of the neural network. The first subarray has shape (3, 2) 
	which is determined by the 3 neurons in the second layer, and the
	2 neurons in the first layer. Each row in the 3x2 matrix represents 
	the weights between both neurons in the first layer and one of the
	neurons in the second layer:
	
	[	
		array([[-0.8272883 , -1.74170864],
			   [ 0.22531047,  0.76300333],
               [-0.14128084, -2.00334914]]), 
               
        array([[ 1.43465322, -0.69658175, -0.25336335],
			   [ 0.20888024,  0.00778669,  0.15188696]])
	] 
	
	The first element of this subarray, [-0.8272883 , -1.74170864], 
	is a NPA that represents the 
	weights connecting the two neurons in the first layer to the first
	(or 'top') neuron of the second layer. The remaining NPAs can be 
	similarly interpreted. The values for the weights and 
	biases are initialized with values taken from a normal distribution
	with a mean of 0 and a standard deviation of 1.
	
	Customizable parameters in this model include:
	
	- netshape: The shape of the neural network.
	
	- learning_rate: The rate at which the network learns. In other 
	words, this term controls how large of an impact the gradient 
	descent step has on the weights and biases of the network. If this
	term is too large, the network becomes unstable as it constantly 
	overshoots 'good values' for its weights and biases. However, if
	this term is too small, then it will take an extremely long time
	for the network to learn.
	
	- lmbda: Used in the gradient descent step. Lambda (written as 
	lmbda because 'lambda' is already a reserved word in Python) 
	determines the relative importance of minimizing the weights vs. 
	minimizing the cost function with respect to the weights. In other 
	words, this term controls how much of an impact L2 regularization
	has on the network's weights. 
	
	- mini_batch_size: Determines how large the mini batch is. For
	example, a mini batch size of 32 means each mini batch contains
	32 training images. 
	
	- output_activation: A boolean value that determines if the softmax 
	activation function is used on the final layer of output neurons. 
	
	
	- cost_function: This library contains two cost functions: the
	quadratic cost function and the cross entropy cost function. By
	default, the cross entropy cost function is used, however, a 
	network can be initialized to use the quadratic cost function by
	specifying "cost_function=quadratic_cost_derivative"."""
	
	def __init__(self, netshape, learning_rate, mini_batch_size, 
	softmax=False, cost_function=False):
		
		# Record the number of layers the network has.
		self.netlength = len(netshape)
		# Record the number of neurons in each layer.
		self.netshape = netshape
		# Initialize the biases by calling init_biases().
		self.biases = self.init_bias(netshape)
		# Initialize the weights of the network. Note that i and j
		# specify, the dimensions of each of the sublists in the network.
		# So np.random.randn(2, 3) creates an numpy matrix of dimensions
		# 2 x 3 with values taken from a normal distribution of mean 0
		# and standard deviation of 1. Note also that these weights are
		# on average much larger than weights typically are at 
		#initilization. This is because of Nielsen's observation
		# that large weight initialization does well when classifying 
		# MNIST images. 
		self.weights = [np.random.randn(j, i) for i, j in 
			zip(netshape[0:], netshape[1:])]
		self.learning_rate = learning_rate
		self.lmbda = 5
		self.mini_batch_size = mini_batch_size
		self.softmax = softmax
		self.cost_function = cost_function
		
	def init_bias(self, netshape):
		"""Initialize the biases of the network. Each layer of biases is
		represneted as a (1,n) array where n represented the number of
		neirons in that layer. Each of these numpy arrays (NPAs) 
		are then stored in a list."""
	
		biases = []
		for i in netshape[1:]:
			biases.append(np.random.randn(1, i)[0])	
		return biases
				
	
	def feedforward(self, a, output_activation=False):
		
		"""Return the output of the network if 'a' is input. If the network
		was initialized with an output activation function then that 
		function is used, otherwise the sigmoid activation function is used."""		
		
		for i in range(self.netlength - 1):
			w = self.weights[i]
			b = self.biases[i]
			z = np.dot(w, a) + b
			
			if i == self.netlength - 2:
				a = sigmoid(z)
			else:
				a = sigmoid(z)

		return a
		
		
	def get_activations(self, a):
		""" Calculates the activations and z values for each layer of the
		network. This function is similar to feedforward(), but
		get_activations() was specifically made as a helper function
		for backpropagation()."""
		
		activations = [a]
		zs = []
		if self.softmax:	
			for i in range(self.netlength - 1):
				w = self.weights[i]
				b = self.biases[i]
				z = np.dot(w, a) + b
				zs.append(z)
				if i == self.netlength - 2:
					a = softmax(z)
				else:
					a = sigmoid(z)
			activations.append(a)
		else:
			for b, w in zip(self.biases, self.weights):
				a = np.asarray(a)
				z = np.dot(w, a) + b
				a = sigmoid(z)
				zs.append(z)
				activations.append(a)
		return activations, zs
		
	
	def backpropagation(self, a, y): 
		"""Calculate the cost gradient with respect to the weights and 
	biases of the network using the backpropagation algorithm. 'a' is
	the input signal (i.e. in the case of MNIST this would be a list of 
	784 elements. 'y' is the label of the input signal, in the case
	of MNIST, this would be a one-hot encoding of a digit between 0 and 
	9 (e.g. the number 5 would be encoded as [0,0,0,0,0,1,0,0,0,0]."""
		
			
		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		activations, zs = self.get_activations(a, y)	
		
		#may have to turn these arrays into np arrays to do hadamard product!
		# represents initially the last layer's error and then it represents the
		# current layer + 1's error in the for loop  below. The index -1 indexes 
		# the last element in the array. 
		error = cross_entropy_cost_derivative(a, y)

		# Note that the error is a 1-D vector of the form 
		# [0, 1, 2, ... , n] where n is the number of output neurons. 
		nabla_b[-1] = error
		
		# The gradient of the cost function with respect to the 
		# final layer of weights is given by multiplying the transpose of
		# the error vector and the second last layer of activations. This
		# calculation returns an I x J matrix where I is the 
		# We can quickly verify that this calculation returns the correct
		# dimensions of the weight gradient.  
		nabla_w[-1] = np.dot(np.asarray([error]).transpose(), np.asarray([activations[-2]]))

		w = np.dot(np.asarray([activations[-2]]).transpose(), np.asarray([error]))
	
		# calculate the error for each layer BEFORE THE LAST LAYER, which
		# we have already calculated above thats why we have -2 rather than -1.
		for i in reversed(range(self.netlength - 2)):

		
			#PLAY AROUND WITH THESE INDICES THERE IS SOMETHING FISHY GOING ON HERE!
			nabla_b[i] = error
			
			nabla_w[i] = np.dot(np.asarray([error]).transpose(), np.asarray([activations[i]]))

		return nabla_b, nabla_w
	
	def update_net(self, activations, y, z):
		nabla_b, nabla_w = self.backpropagation(activations, y, z)
		
		self.biases = self.biases - np.asarray(self.learning_rate) * nabla_b

		self.weights = self.weights - np.asarray(self.learning_rate) * nabla_w
		
	def mini_batch_update_net(self, mini_batch, dataset_length, L2=False, 
		monitor=False):
		
		"""Computes one round of mini-batch gradient descent
		on the network. L2 is a boolean value that determines whether L2
		regularization is used in the gradient descent step. L2 greatly
		improves the network's ability to generalize the features it 
		learns and so it is recommended to have this feature turned on. """
		
		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		for a, y in mini_batch:

			# calculate change in nabla_b, and change in nabla_w
			d_nabla_b, d_nabla_w = self.backpropagation(a, y)
	
			
			nabla_b = np.asarray(nabla_b) + np.asarray(d_nabla_b)
			nabla_w = np.asarray(nabla_w) + np.asarray(d_nabla_w)
		
		# I follow Nielsen's lead by dividing the nabla update by size of the minibatch
		self.biases = self.biases - np.asarray(self.learning_rate / len(mini_batch)) * nabla_b
		
		if L2:
			
			self.weights = np.asarray(1 - self.learning_rate * self.lmbda
			 / dataset_length) * self.weights - np.asarray(self.learning_rate 
			 / len(mini_batch)) * nabla_w
			
		else:
			self.weights = self.weights - np.asarray(self.learning_rate / 
				len(mini_batch)) * nabla_w
		
			
	def stochastic_gradient_descent(dataset, labels, epochs, L2=False, 
		monitor=False):
		
		"""Perform mini batch stochastic gradient descent on the neural 
		network using the dataset and labels provided by the function 
		arguments. L2 is a boolean value that determines whether L2
		regularization is used in the gradient descent step. L2 greatly
		improves the network's ability to generalize the features it 
		learns and so it is recommended to have this feature turned on.
		The variable, monitor, determines if 
		
		the program returns explicit information about the performance of 
		the neural network. In order to activate 'monitor', assign to it
		the list: [test_data, test_labels]. Then after each training epoch
		---i.e. after each mini-batch gradient descent step---the function
		will print the network's current peformance on the test data. 
		After the final training epoch, the function will plot the performance
		of the model (network and model are used interchangably here) across
		all training epochs on the test data. Note that having monitor on
		significantly slows down the training process as the program has to
		run many more feedforward signals through the network after every
		training epoch.
		
		""" 
	
		dataset_length = len(dataset)
		mini_batch = []
		
		if monitor:
			epochs_list = [i for i in range(epochs)]
			evaluations = []

		samples = np.random.randint(dataset_length, size=self.mini_batch_size)
		
		for i in range(epochs):
			for i in samples:
				training_example = dataset[i]
				label = labels[i]
				mini_batch.append([training_example, label])
				
			self.mini_batch_update_net(mini_batch, dataset_length, L2, monitor)	
			
			
			if monitor:
				accuracy = self.evaluation(test_data, labels)
				evaluations.append(accuracy)
				print("On training epoch: " + str(i) + 
					"The NN correctly classified: " + str(accuracy) 
					+ "% of the test dataset")
		
		if monitor:
			plt.plot(epoch, evaluations)
			plt.xlabel('Epoch')
			plt.ylabel('Accuracy on test data')
			plt.title('Evaluating model performance')
			plt.show()
	

	def evaluation(self, dataset, labels):
		"""Compare the output of the neural network with the set of labels. 
		This function was specifically made to test network performance on 
		the MNIST dataset. Here, we interpret the position of the highest
		value in the network's output list to be the network's 'guess' of
		which number the input list represents. The output of this function
		is the percentage of correct 'guesses' the network made on the 
		dataset."""	
		
		error = 0
		l = len(dataset)	

		for data, label in zip(dataset, labels):
	
			if np.argmax(self.feedforward(data)) == label:
					continue
			else:
				error += 1
					
		accuracy = (l - error)/ l * 100	
		
		return accuracy	
	
	
	def save(self, filename=False):
		"""Save the network in a JSON format. Note that numpy arrays cannot 
		be saved in the JSON format, so we must first convert the biases
		and weights to be lists before saving the network as a JSON. The
		JSON is saved in a text file in the same directory as this file. By
		default, the saved file is called 'network.txt' but this can be 
		changed if another file name is given in the function call."""
	
		biases_list = [i.tolist() for i in self.biases]
		weights_list = [i.tolist() for i in self.weights]
		data = {
			"netshape": self.netshape,
			"biases": biases_list,
			"weights": weights_list,
			"learning_rate": self.learning_rate,
			"mini_batch_size": self.mini_batch_size,
			"softmax": self.softmax,
			"cost_function": self.cost_function
		}
		if filename:
			with open(filename, 'w') as outfile:  
				json.dump(data, outfile)
		else:
			with open('network.txt', 'w') as outfile:  
				json.dump(data, outfile)
				

def load(filename):	
	"""Load a saved neural network. This function initializes a network 
		instance with the parameters of the loaded network. Note that the 
	file with the network parameters should be in the same directory
	as this file."""
	
	with open(filename) as json_file:  
		network_data = json.load(json_file)
		
	net = Network([network_data['netshape']], 0.1, 32)
	
    # We need to reconvert the sublists in the biases and weights lists
    # back to being numpy arrays.
	biases_numpy = [np.asarray(i) for i in network_data['biases']]
	weights_numpy = [np.asarray(i) for i in network_data['weights']]
	
	net.biases = biases_numpy
	net.weights = weights_numpy
	net.learning_rate = network_data['learning_rate']
	net.mini_batch_size = network_data['mini_batch_size']
	net.softmax = network_data['softmax']
	net.cost_function = network_data['cost_function']
	
	return net

	
def sigmoid(z):
	"""The sigmoid activation function. The sigmoid function 
	transforms all values on the number line to values between 0 and 1."""
	
	return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
	"""The derivative of the sigmoid activation function."""
	
	return sigmoid(z) * (1 - sigmoid(z))

	
def softmax(z):
	"""The softmax activation function. Can be used as the final
	activation function for the network."""	
	
	total = np.exp(z).sum()
	z_exp = np.exp(z)
	z[True] = total #Convert every element to be equal to total
	return z_exp / total	
	
def tanh(z):
	"""The tanh activation function. The tanh function transforms all values
	on the number line to values between -1 and 1."""
	e = np.exp(2 * z)
	return ((e - 1) / (e + 1))


def tanh_prime(z):
	"""The derivative of the tanh function."""
	e = np.exp(2 * z)
	
	return (4 * e) / (e + 1) ** 2


def relu(z):
	"""The ReLU activation function. ReLu is defined as f(x) = max(0,x)."""
	
	flag = isinstance(z, np.ndarray) # Test if z is an array
	if flag:
		z[z < 0] = 0
		return z
	else:
		z = max(0,z)
		return z
	return z
	

def relu_prime(z):
	"""The derivative of the ReLU activation function. Similar to relu(),
	first test if z is a list or integer and then transform the values 
	accordingly."""
	
	flag = isinstance(z, np.ndarray)
	if flag:
		z[z < 0] = 0
		z[z >= 0] = 1
		return z
	else:
		if z < 0:
			return 0
		else: 
			return 1 	

			
def linear(z):
	"""The linear activation function"""
	
	return z
	

def linear_prime(z):
	"""The derivative of the linear activation function"""
	
	return 1


def quadratic_cost_derivative(a, y, z):
	"""The derivative of the quadratic cost function. This function is used 
	in backpropagation to calculate the error between the output of the 
	network and the label. Where 'a' is the network input, 'y' is the
	label, and """
	
	return (a - y) * sigmoid_prime(z)
	

def cross_entropy_cost_derivative(a, y):
	"""The derivative of the cross-entropy cost function. This function is 
	used in backpropagation to calculate the error between the output of the 
	network and the label. Where 'a' is the network input and 'y' is the
	label."""
	
	return (a - y)
	
