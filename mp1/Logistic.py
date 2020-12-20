"""Logistic regression model."""

import numpy as np


class Logistic:
	def __init__(self, lr: float, epochs: int):
		"""Initialize a new classifier.

		Parameters:
			lr: the learning rate
			epochs: the number of epochs to train for
		"""
		self.w = None  # TODO: change this
		self.b = 0
		self.lr = lr
		self.epochs = epochs
		self.threshold = 0.5

	def sigmoid(self, z: np.ndarray) -> np.ndarray:
		"""Sigmoid function.

		Parameters:
			z: the input

		Returns:
			the sigmoid of the input
		"""
		# TODO: implement me
		return 1/(1+np.exp(-z))

	def train(self, X_train: np.ndarray, y_train: np.ndarray):
		"""Train the classifier.

		Use the logistic regression update rule as introduced in lecture.

		Parameters:
			X_train: a numpy array of shape (N, D) containing training data;
				N examples with D dimensions
			y_train: a numpy array of shape (N,) containing training labels
		student note:
			the function we wrote have x be(D,N) and y be (1,N)
		"""
		y_train = y_train[:,None]
		y_train = y_train.T
		X_train = X_train.T
		self.w = np.zeros((X_train.shape[0], 1))
		for i in range(self.epochs):
			# grads = self.costfunc(X_train, y_train)
			m = X_train.shape[1]
			P = self.sigmoid(np.dot(self.w.T, X_train)+self.b)
			dz = (P-y_train)
			dw = np.dot(X_train, dz.T)/m
			db = np.sum(dz)/m
			grads = {"dw": dw, "db": db}

			dw_ = grads["dw"]  # get dw
			db_ = grads["db"]  # get db
			#update
			self.w = self.w - self.lr*dw_
			self.b = self.b - self.lr*db_

		grads = {"dw": dw_, "db": db_}

	# def costfunc(self, x, y):
	#     m = x.shape[1]  # number of examples
	#     P = self.sigmoid(np.dot(self.w.T, x)+self.b)
	#     #calculate gradient
	#     dz = (P-y)
	#     dw = np.dot(x, dz.T)/m
	#     db = np.sum(dz)/m
	#     assert(dw.shape == self.w.shape)
	#     assert(db.dtype == float)

	#     grads = {"dw": dw, "db": db}
	#     return grads  # , cost

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
		X_test = X_test.T
		m = X_test.shape[1]  # number of examples
		labels = np.zeros((1, m))
		self.w = self.w.reshape(X_test.shape[0], 1)

		# sig function probability
		P = self.sigmoid(np.dot(self.w.T, X_test)+self.b)
		print(P)

		for i in range(P.shape[1]):
			labels[0, i] = np.where(P[0, i] > 0.5, P[0, i], 0)
			labels[0, i] = np.where(labels[0, i] < 0.5, labels[0, i], 1)
			pass
		labels = labels.T
		assert(labels.shape == (m, 1))
		return labels

