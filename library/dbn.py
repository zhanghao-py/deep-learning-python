#!/usr/bin/env python

import numpy as np
import scipy.signal
import scipy.ndimage
import datetime

from library import util
from library import autoencoder

def setup(stack, trainData, opts):

	inputSize, numOfSamples = trainData.shape

	for l in xrange(1, len(stack.layers)):

		stack.layers[l].alpha = opts.alpha
		stack.layers[l].momentun = opts.momentun

		stack.layers[l].W = np.zeros([stack.layers[l].size, stack.layers[l-1].size])
		stack.layers[l].deltaW = np.zeros([stack.layers[l].size, stack.layers[l-1].size])

		stack.layers[l].a = np.zeros([stack.layers[l-1].size, 1])
		stack.layers[l].deltaa = np.zeros([stack.layers[l-1].size, 1])

		stack.layers[l].b = np.zeros([stack.layers[l].size, 1])
		stack.layers[l].deltab = np.zeros([stack.layers[l].size, 1])

	return stack

def rbmtrain(layer, trainData, opts):

	inputSize, numOfSamples = trainData.shape
	numOfBatches = numOfSamples / opts.batchsize

	for i in xrange(opts.numepochs):

		error = 0

		for l in xrange(numOfBatches):

			batchStart = l * opts.batchsize
			batchEnd = (l+1) * opts.batchsize

			batchTrainData = trainData[:, batchStart:batchEnd]

			v1 = batchTrainData
			h1 = sigmoidrnd(layer.W.dot(v1) + layer.b)
			v2 = sigmoidrnd(layer.W.T.dot(h1) + layer.a)
			h2 = autoencoder.sigmoid(layer.W.dot(v2) + layer.b)

			c1 = h1.dot(v1.T)
			c2 = h1.dot(v2.T)

			deltaa = np.sum(v1 - v2, axis = 1)
			deltab = np.sum(h1 - h2, axis = 1)

			layer.deltaW = layer.momentun * layer.deltaW + layer.alpha * (c1 - c2) / opts.batchsize
			layer.detlaa = layer.momentun * layer.deltaa + layer.alpha * deltaa.reshape([deltaa.size, 1]) / opts.batchsize
			layer.deltab = layer.momentun * layer.deltab + layer.alpha * deltab.reshape([deltab.size, 1]) / opts.batchsize

			layer.W = layer.W + layer.deltaW 
			layer.a = layer.a + layer.detlaa
			layer.b = layer.b + layer.deltab

			error = error + np.power(v1 - v2, 2).sum() / opts.batchsize

		print 'epoch %d/%d. Average reconstruction error is: %3f' % (i+1, opts.numepochs, (error / numOfBatches))

	return layer

def sigmoidrnd(x):
	dim1, dim2 = x.shape
	return ( autoencoder.sigmoid(x) > np.random.rand(dim1, dim2) ).astype(int)

def train(stack, trainData, opts):

	depth = len(stack.layers)

	stack.layers[1] = rbmtrain(stack.layers[1], trainData, opts)

	for i in xrange(2, depth):

		trainData = autoencoder.sigmoid(stack.layers[i-1].W.dot(trainData) + stack.layers[i-1].b)
		stack.layers[i] = rbmtrain(stack.layers[i], trainData, opts)

	return stack

def feedforward(stack, trainData):
	depth = len(stack.layers)

	feature = trainData

	for i in xrange(1, depth):

		if stack.layers[i].type == 'rbm':
			feature = autoencoder.sigmoid(stack.layers[i].W.dot(feature) + stack.layers[i].b)

	return feature

