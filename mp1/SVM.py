"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
	def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
		"""Initialize a new classifier.

		Parameters:
			n_class: the number of classes
			lr: the learning rate
			epochs: the number of epochs to train for
			reg_const: the regularization constant
		"""
		self.w = None  # TODO: change this
		self.lr = lr
		self.epochs = epochs
		self.reg_const = reg_const
		self.n_class = n_class
		self.batch_size = 32

	def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
		"""Calculate gradient of the svm hinge loss.

		Inputs have dimension D, there are C classes, and we operate on
		mini-batches of N examples.

		Parameters:
			X_train: a numpy array of shape (N, D) containing a mini-batch
				of data
			y_train: a numpy array of shape (N,) containing training labels;
				y[i] = c means that X[i] has label c, where 0 <= c < C

		Returns:
			the gradient with respect to weights w; an array of the same shape
				as w
		"""
		# TODO: implement me
		N, D = X_train.shape
		grads = self.reg_const * self.w  # gradient for regularization
		ground_truth_batch = self.batch_score[np.arange(N), y_train]  # (n,)
		ground_truth_batch = ground_truth_batch[:, np.newaxis]
		score_for_grad = ((ground_truth_batch - self.batch_score) < 1).astype(int)
		score_for_grad[np.arange(N), y_train] = 0 
		ground_truth_grad = score_for_grad.sum(axis=1)[:, np.newaxis] * X_train
		for i in range(N):
			grads[:, y_train[i]] -= ground_truth_grad[i]
			grads += X_train[i][:, np.newaxis] * score_for_grad[i]
		
		return grads

	def train(self, X_train: np.ndarray, y_train: np.ndarray):
		"""Train the classifier.

		Hint: operate on mini-batches of data for SGD.

		Parameters:
			X_train: a numpy array of shape (N, D) containing training data;
				N examples with D dimensions
			y_train: a numpy array of shape (N,) containing training labels
		"""
		# TODO: implement me
		N, D = X_train.shape
		self.w = np.random.randn(D, 10)
		iters = N // self.batch_size
		for epoch in range(self.epochs):
			self.lr = self.lr / (1+epoch)
			print("epoch: " + str(epoch), " lr: " + str(self.lr))
			for i in range(iters):
				x = X_train[i*self.batch_size: (i+1) * self.batch_size]
				y = y_train[i*self.batch_size: (i+1) * self.batch_size]
				self.batch_score = x.dot(self.w)
				self.w = self.w - self.lr * self.calc_gradient(x, y)
		return

	def predict(self, X_test: np.ndarray) -> np.ndarray:
		"""Use the trained weights to predict labels for test data points.

		Parameters:
			X_test: a numpy array of shape (N, D) containing testing data;
				N examples with D dimensions

		Returns:
			predicted labels for the data in X_test; a 1-dimensional array of
				length N, where each element is an integer giving the predicted
				class.
		"""
		# TODO: implement me
		pred = X_test.dot(self.w)
		return pred.argmax(axis=1)
