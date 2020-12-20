"""Neural network model."""

from typing import Sequence

import numpy as np

class NeuralNetwork:
	"""A multi-layer fully-connected neural network. The net has an input
	dimension of N, a hidden layer dimension of H, and performs classification
	over C classes. We train the network with a cross-entropy loss function and
	L2 regularization on the weight matrices.

	The network uses a nonlinearity after each fully connected layer except for
	the last. The outputs of the last fully-connected layer are passed through
	a softmax, and become the scores for each class."""

	def __init__(
		self,
		input_size: int,
		hidden_sizes: Sequence[int],
		output_size: int,
		num_layers: int,
		adam = False,
	):
		"""Initialize the model. Weights are initialized to small random values
		and biases are initialized to zero. Weights and biases are stored in
		the variable self.params, which is a dictionary with the following
		keys:

		W1: 1st layer weights; has shape (D, H_1)
		b1: 1st layer biases; has shape (H_1,)
		...
		Wk: kth layer weights; has shape (H_{k-1}, C)
		bk: kth layer biases; has shape (C,)

		Parameters:
			input_size: The dimension D of the input data
			hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
				hidden layer i
			output_size: The number of classes C
			num_layers: Number of fully connected layers in the neural network
		"""
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		self.num_layers = num_layers

		assert len(hidden_sizes) == (num_layers - 1)
		sizes = [input_size] + hidden_sizes + [output_size]

		self.params = {}
		for i in range(1, num_layers + 1):
			self.params["W" + str(i)] = optimizer(adam, np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1]))
			self.params["b" + str(i)] = optimizer(adam, np.zeros(sizes[i]))

	def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
		"""Fully connected (linear) layer.

		Parameters:
			W: the weight matrix
			X: the input data
			b: the bias

		Returns:
			the output
		"""
		# TODO: implement me
		return np.dot(X, W) + b

	def relu(self, X: np.ndarray) -> np.ndarray:
		"""Rectified Linear Unit (ReLU).

		Parameters:
			X: the input data

		Returns:
			the output
		"""
		# TODO: implement me
		return np.maximum(X, 0)

	def softmax(self, X: np.ndarray) -> np.ndarray:
		"""The softmax function.

		Parameters:
			X: the input data

		Returns:
			the output
		"""
		# TODO: implement me
		return np.exp(X - np.max(X)) / sum(np.exp(X - np.max(X)))

	def forward(self, X: np.ndarray) -> np.ndarray:
		"""Compute the scores for each class for all of the data samples.

		Hint: this function is also used for prediction.

		Parameters:
			X: Input data of shape (N, D). Each X[i] is a training or
				testing sample

		Returns:
			Matrix of shape (N, C) where scores[i, c] is the score for class
				c on input X[i] outputted from the last layer of your network
		"""
		self.outputs = {}
		self.outputs[0] = X
		# TODO: implement me. You'll want to store the output of each layer in
		# self.outputs as it will be used during back-propagation. You can use
		# the same keys as self.params. You can use functions like
		# self.linear, self.relu, and self.softmax in here.
		for i in range(1, self.num_layers + 1):
			w = self.params["W" + str(i)].x
			b = self.params["b" + str(i)].x
			X = self.linear(w, X, b)
			self.outputs[i] = X
			if i != self.num_layers:
				X = self.relu(X)

		Z = X
		for i in range(X.shape[0]):
			Z[i] = self.softmax(X[i])
		return Z

	def backward(
		self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
	) -> float:
		"""Perform back-propagation and update the parameters using the
		gradients.

		Parameters:
			X: Input data of shape (N, D). Each X[i] is a training sample
			y: Vector of training labels. y[i] is the label for X[i], and each
				y[i] is an integer in the range 0 <= y[i] < C
			lr: Learning rate
			reg: Regularization strength

		Returns:
			Total loss for this batch of training samples
		"""
		self.gradients = {}
		loss = 0.0
		# TODO: implement me. You'll want to store the gradient of each layer
		# in self.gradients if you want to be able to debug your gradients
		# later. You can use the same keys as self.params. You can add
		# functions like self.linear_grad, self.relu_grad, and
		# self.softmax_grad if it helps organize your code.
		out = self.forward(X)
		out_label = np.argmax(out,axis=1)
		num = out.shape[0]
		loss = np.sum(-np.log(out[np.arange(num), y]))
		loss = loss/num
		for i in range(1, self.num_layers + 1):
			loss += reg * np.sum(self.params["W" + str(i)].x * self.params["W" + str(i)].x)
		
		out[np.arange(num),y] -= 1
		out = out/num

		for i in range(self.num_layers,0,-1):
			grad_w = self.relu(self.outputs[i - 1].T).dot(out) + 2 * reg * self.params["W" + str(i)].x
			grad_b = out.sum(axis=0) + reg * 2 * self.params["b" + str(i)].x
			out = out.dot(self.params["W" + str(i)].x.T) * (self.outputs[i - 1] > 0)
			self.params["W" + str(i)].update(grad_w, lr)
			self.params["b" + str(i)].update(grad_b, lr)

		return out_label, loss

	def predict(self, X: np.ndarray, y: np.ndarray, batch_size=1000):
		preds = []
		labels = []
		for batch in range(X.shape[0] // batch_size):
			X_batch = X[batch * batch_size: (batch + 1) * batch_size, :]
			y_batch = y[batch * batch_size: (batch + 1) * batch_size]
			labels.extend(y_batch.tolist())
			out = self.forward(X_batch)
			out_label = np.argmax(out,axis=1)
			preds.extend(out_label.tolist())
		return preds, (np.sum(np.array(labels) == np.array(preds)) / len(preds))

class optimizer:
	def __init__(self, adam, x):
		self.x = x
		self.adam = adam
		if adam:
			self.beta_1 = 0.9
			self.beta_2 = 0.99
			self.epsilon = 1e-8
			self.t = 0
			self.m_t = np.zeros_like(x)
			self.v_t = np.zeros_like(x)

	def update(self, grad, lr = 1e-3):
		if self.adam:
			self.t += 1
			self.m_t = self.beta_1 * self.m_t + (1.0 - self.beta_1) * grad
			self.v_t = self.beta_2 * self.v_t + (1.0 - self.beta_2) * (grad * grad)
			M = self.m_t / (1.0 - (self.beta_1 ** self.t))
			V = self.v_t / (1.0 - (self.beta_2 ** self.t))
			new_grad = M/(np.sqrt(V)+self.epsilon)
			self.x -= lr * new_grad
		else:
			self.x -= lr * grad

