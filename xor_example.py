#!/usr/bin/python
#-*- coding: utf-8 -*-

import nnet
import numpy as np
from nnet import Linear
from nnet import ReLU
from nnet import Sigmoid
from collections import OrderedDict
import pickle
import sys


class XORNNet(object):
	def __init__(self, params={}):
		self.params = params

		self.layers = OrderedDict()
		self.layers['Linear1'] = Linear(W=self.params['W1'], b=self.params['b1'])
		self.layers['ReLU'] = ReLU()
		self.layers['Linear2'] = Linear(W=self.params['W2'], b=self.params['b2'])
		self.layers['Sigmoid'] = Sigmoid()

	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

if __name__=='__main__':
	x = np.array([[0, 0], [0, 1.0], [1.0, 0], [1.0, 1.0]])

	model = pickle.load(open(sys.argv[1], 'rb'))
	xor_nnet = XORNNet(params=model)

	for i in range(x.shape[0]):
		out = xor_nnet.predict(x[i])
		y = 1 if out >= 0.5 else 0
		print('in: {0}  ->  out: {1} ({2})'.format(x[i], y, float(out)))
