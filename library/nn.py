#!/usr/bin/env python

import numpy as np

from library import util
from library import autoencoder


def setup(stack):

	depth = len(stack.layers)
	for i in xrange(1, depth):

		stack.layers[i-1].W = (np.random.rand(stack.layers[i].size, stack.layers[i-1].size) - 0.5) * 2 * np.sqrt(6.0 / (stack.layers[i].size + stack.layers[i-1].size))
		stack.layers[i-1].b = np.zeros([stack.layers[i].size, 1])

	return stack

def train(stack, trainData, trainLabels, opts):

	stack.alpha = opts.alpha
	inputSize, numOfSamples = trainData.shape
	numOfBatches = numOfSamples / opts.batchsize

	depth = len(stack.layers)
	numOfClasses = stack.layers[depth-1].size

	for i in xrange(opts.numepochs):

		for l in xrange(numOfBatches):

			batchStart = l * opts.batchsize
			batchEnd = (l+1) * opts.batchsize

			batchTrainData = trainData[:, batchStart:batchEnd]
			batchTrainLabels = trainLabels[batchStart:batchEnd]

			stack = feedforward(stack, batchTrainData)
			stack = calculateLoss(stack, batchTrainLabels)
			stack = backpropagation(stack)
			stack = applygrads(stack)
			
		print 'epoch %d/%d. Mini-batch mean squared error on training set is: %3f' % (i+1, opts.numepochs, 0)

	return stack

def calculateLoss(stack, trainLabels):

	depth = len(stack.layers)
	numOfClasses = stack.layers[depth-1].size

	numOfSamples = trainLabels.size

	groundTruth = np.zeros([numOfClasses, numOfSamples])
	groundTruth[trainLabels.ravel('F'), np.arange(numOfSamples)] = 1

	stack.error = groundTruth - stack.layers[depth-1].a
	stack.loss = 0.5 * np.mean(stack.error)

	return stack

def feedforward(stack, data):

	inputSize, numOfSamples = data.shape

	stack.layers[0].a = data

	depth = len(stack.layers)


	# feedforward pass / hidden layer
	for i in xrange(0, depth):

		if stack.layers[i].type == 'hidden':
			stack.layers[i+1].a = autoencoder.sigmoid( stack.layers[i].W.dot(stack.layers[i].a) + stack.layers[i].b )

		# if stack.layers[i].type == 'output':
			# stack.result = autoencoder.sigmoid( stack.layers[i].W.dot(stack.layers[i].a) + stack.layers[i].b )

	# output layer

	return stack

def backpropagation(stack):

	depth = len(stack.layers)
	delta = [0 for _ in xrange(depth)]

	delta[depth-1] = - stack.error * stack.layers[depth-1].a * (1 - stack.layers[depth-1].a)

	for i in xrange(depth-2, 0, -1):
		d_act = stack.layers[i].a * (1 - stack.layers[i].a)
		delta[i] = stack.layers[i].W.T.dot(delta[i+1]) * d_act

	for i in xrange(depth-1):
		stack.layers[i].deltaW = (1.0/delta[i+1].shape[0]) * delta[i+1].dot(stack.layers[i].a.T)
		stack.layers[i].deltab = (1.0/delta[i+1].shape[0]) * np.sum(delta[i+1], 1).reshape([stack.layers[i+1].size, 1])

	return stack


def applygrads(stack):

	depth = len(stack.layers)
	for i in xrange(depth-1):

		stack.layers[i].W = stack.layers[i].W - stack.alpha * stack.layers[i].deltaW
		stack.layers[i].b = stack.layers[i].b - stack.alpha * stack.layers[i].deltab


	return stack


def predict(stack, testData):

	stack = feedforward(stack, testData)
	depth = len(stack.layers)
	pred = stack.layers[depth-1].a
	return np.argmax(pred, axis=0)
