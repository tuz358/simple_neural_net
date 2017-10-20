#!/usr/bin/python
#-*- coding: utf-8 -*-

import nnet
import numpy as np
from nnet import Affine
from nnet import ReLu
from nnet import Sigmoid
from collections import OrderedDict
import pickle
import sys


class XORNNet(object):
	def __init__(self, params={}):
		self.params = params

		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(W=self.params['W1'], b=self.params['b1'])
		self.layers['ReLu'] = ReLu()
		self.layers['Affine2'] = Affine(W=self.params['W2'], b=self.params['b2'])
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
