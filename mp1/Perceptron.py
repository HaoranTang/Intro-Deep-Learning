"""Perceptron model."""

import numpy as np
import scipy

class Perceptron:
	def __init__(self, n_class: int, lr: float, epochs: int):
		"""Initialize a new classifier.

		Parameters:
			n_class: the number of classes
			lr: the learning rate
			epochs: the number of epochs to train for
		"""
		self.w = None # TODO: change this
		self.lr = lr
		self.epochs = epochs
		self.n_class = n_class

	def train(self, X_train: np.ndarray, y_train: np.ndarray):
		"""Train the classifier.

		Use the perceptron update rule as introduced in the Lecture.

		Parameters:
			X_train: a number array of shape (N, D) containing training data;
				N examples with D dimensions
			y_train: a numpy array of shape (N,) containing training labels
		"""
		# TODO: implement me
		self.w = np.random.randn(X_train.shape[1], 10)
		for epoch in range(self.epochs):
			print("epoch: " + str(epoch))
			for i, x in enumerate(X_train):
				label = y_train[i]
				score = x.dot(self.w)  # (10,)
				update = (score > score[label])  # (10,) 
				sum_update = np.sum(update)
				update = x[:, np.newaxis] * update # (D, 10)
				
				self.w[:, label] = self.w[:, label] + self.lr * sum_update * x
				self.w = self.w - self.lr * update

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
		print("start predicting")
		pred = X_test.dot(self.w).argmax(axis=1)
		return pred
