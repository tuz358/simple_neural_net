#-*- coding: utf-8 -*-

import numpy as np


class Affine(object):
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		return np.dot(self.x, self.W) + self.b

	def backward(self, dout):
		self.dW = np.dot(self.x.T, dout)
		self.db = dout
		dx = np.dot(dout, self.W.T)
		return dx

	def fit(self, lr):
		self.W -= lr * self.dW
		self.b -= lr * self.db

class ReLu(object):
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0)
		y = x.copy()
		y[self.mask] = 0
		return y

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout
		return dx

	def fit(self, lr):
		pass

class Sigmoid(object):
	def __init__(self):
		self.y = None

	def forward(self, x):
		self.y = 1 / (1 + np.exp(-x))
		return self.y

	def backward(self, dout):
		dx = dout * (1.0 - self.y) * self.y
		return dx

	def fit(self, lr):
		pass

class Mse(object):
	''' Mean Squared Error'''
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, y, t):
		self.y = y
		self.t = t
		self.loss = 0.5 * np.sum((y - t)**2)
		return self.loss

	def backward(self, dout=1):
		dx = self.y - self.t
		return dx
