#!/usr/bin/python
#-*- coding: utf-8 -*-

import nnet
import numpy as np
from nnet import Linear
from nnet import ReLU
from nnet import Sigmoid
from nnet import Mse
from collections import OrderedDict
import pickle
import sys


class SimpleNNet(object):
	def __init__(self):
		self.params = {}
		self.params['W1'] = 0.01 * np.random.randn(2, 3) # Gaussian distribution
		self.params['b1'] = np.zeros((1, 3))
		self.params['W2'] = 0.01 * np.random.randn(3, 1) # Gaussian distribution
		self.params['b2'] = np.zeros((1, 1))

		self.layers = OrderedDict()
		self.layers['Linear1'] = Linear(W=self.params['W1'], b=self.params['b1'])
		self.layers['ReLU'] = ReLU()
		self.layers['Linear2'] = Linear(W=self.params['W2'], b=self.params['b2'])
		self.layers['Sigmoid'] = Sigmoid()
		self.lastlayer = Mse()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	def one_cycle(self, x, t):
		x = x.reshape(1, x.shape[0]) # shape (2,) -> (1, 2) : to use x.T

		# feed forward
		y = self.predict(x)
		self.loss = self.lastlayer.forward(y, t)

		sys.stdout.write('[*] loss : {0}\r'.format(self.loss))
		sys.stdout.flush()

		# backward and update parameters
		dout = self.lastlayer.backward()
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
			layer.fit(lr=0.01) # lr : learning rate

	def save_model(self):
		self.params['W1'] = self.layers['Linear1'].W
		self.params['b1'] = self.layers['Linear1'].b
		self.params['W2'] = self.layers['Linear2'].W
		self.params['b2'] = self.layers['Linear2'].b
		pickle.dump(self.params, open(sys.argv[1], 'wb'), protocol=2)

if __name__=='__main__':
	# XOR
	x = np.array([[0, 0], [0, 1.0], [1.0, 0], [1.0, 1.0]])
	t = np.array([0, 1.0, 1.0, 0])

	simple_nnet = SimpleNNet()
	try:
		while 1:
			for i in range(x.shape[0]):
				simple_nnet.one_cycle(x[i], t[i])
	except KeyboardInterrupt:
		print()
		simple_nnet.save_model() # pickle (protocol 2)
