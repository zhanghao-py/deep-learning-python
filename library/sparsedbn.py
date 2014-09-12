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

		stack.layers[l].c = np.zeros([stack.layers[l-1].size, 1])
		stack.layers[l].deltac = np.zeros([stack.layers[l-1].size, 1])

		stack.layers[l].b = np.zeros([stack.layers[l].size, 1])
		stack.layers[l].deltab = np.zeros([stack.layers[l].size, 1])

	return stack

def rbmtrain(layer, trainData, opts):

	weightDecay = 0.0002
	sparseQ = 0
	sparseLambda = 0.001

	inputSize, numOfSamples = trainData.shape
	numOfBatches = numOfSamples / opts.batchsize

	for i in xrange(opts.numepochs):

		# [start] ----> sparse term [to learn, more detals]
		dsW = np.zeros([layer.size, inputSize])
		dsB = np.zeros([layer.size, 1])

		vis0 = trainData
		hid0 = sigmoidrnd(layer.W.dot(vis0) + layer.b)

		dH = np.multiply( hid0, (1 - hid0) )
		mH = np.mean(hid0, axis = 1)
		mH = mH.reshape([mH.size, 1])

		sdH = np.sum(dH, axis = 1)
		sdH = sdH.reshape([sdH.size, 1])

		svdH = dH.dot(vis0.T)


		da = sparseQ-mH
		dsW = dsW + sparseLambda * 2.0 * (da - svdH)
		dsB = dsB + sparseLambda * 2.0 * np.multiply(da, sdH)
		# [end] ---->


		error = 0

		for l in xrange(numOfBatches):

			batchStart = l * opts.batchsize
			batchEnd = (l+1) * opts.batchsize

			batchTrainData = trainData[:, batchStart:batchEnd]

			v1 = batchTrainData
			h1 = sigmoidrnd(layer.W.dot(v1) + layer.b)
			v2 = sigmoidrnd(layer.W.T.dot(h1) + layer.c)
			h2 = autoencoder.sigmoid(layer.W.dot(v2) + layer.b)

			c1 = h1.dot(v1.T)
			c2 = h1.dot(v2.T)

			deltac = np.sum(v1 - v2, axis = 1)
			deltab = np.sum(h1 - h2, axis = 1)

			layer.deltaW = layer.momentun * layer.deltaW + layer.alpha * (c1 - c2) / opts.batchsize
			layer.detlac = layer.momentun * layer.deltac + layer.alpha * deltac.reshape([deltac.size, 1]) / opts.batchsize
			layer.deltab = layer.momentun * layer.deltab + layer.alpha * deltab.reshape([deltab.size, 1]) / opts.batchsize 

			layer.deltaW = layer.deltaW + opts.batchsize / numOfSamples * dsW
			layer.deltab = layer.deltab + opts.batchsize / numOfSamples * dsB

			layer.W = layer.W + layer.deltaW - weightDecay * layer.W
			layer.c = layer.c + layer.detlac
			layer.b = layer.b + layer.deltab

			error = error + np.power(v1 - v2, 2).sum() / opts.batchsize

		print 'epoch %d/%d. Average reconstruction error is: %3f' % (i+1, opts.numepochs, (error / numOfBatches))

	return layer

def gaussianrnd(mu, sigma):
	# dim1, dim2 = mu.shape
	# return sigma * np.random.normal(0, 1, (dim1, dim2)) + mu

	dim1, dim2 = mu.shape
	return sigma * np.random.randn(dim1, dim2) + mu

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

def feedforward(stack, data):
	depth = len(stack.layers)

	feature = data

	for i in xrange(1, depth):

		if stack.layers[i].type == 'rbm':
			feature = autoencoder.sigmoid(stack.layers[i].W.dot(feature) + stack.layers[i].b)

	return feature

def dbn2nn(stack, numOfClasses):

	nnstack = util.Empty()
	nnstack.layers = []
	nnstack.alpha = 0.1

	depth = len(stack.layers)
	for i in xrange(1, depth):

		if stack.layers[i].type == 'rbm':

			layer = util.Empty()
			layer.type = 'hidden'
			layer.size = stack.layers[i].size
			layer.W = stack.layers[i].W
			layer.b = stack.layers[i].b

			nnstack.layers.append(layer)


	output = util.Empty()
	output.type = 'output'
	output.size = numOfClasses
	output.W = np.random.rand(numOfClasses, nnstack.layers[depth-1].size) - 0.5) * 2 * np.sqrt(6.0 / (numOfClasses + nnstack.layers[depth-1].size))
	output.b = np.zeros([numOfClasses, 1])
	nnstack.layers.append(output)

	return nnstack



