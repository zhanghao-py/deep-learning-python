#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import mnist
from library import softmax
from library import autoencoder
from library import stackedae
from library import util

import scipy.optimize


if __name__ == '__main__':

	DISPLAY = False
	maxfun = 100

	inputSize = 28 * 28
	numOfClasses = 10
	hiddenSizeL1 = 200
	hiddenSizeL2 = 200
	sparsityParam = 0.1
	lamb = 3e-3
	beta = 3

	trainData = mnist.load_images('../data/train-images-idx3-ubyte')
	trainLabels = mnist.load_labels('../data/train-labels-idx1-ubyte')


	r = np.load("result.npz")
	W11 = r["W11"]
	b11 = r["b11"]
	W12 = r["W12"]
	b12 = r["b12"]

	saeSoftmaxOptTheta = r["saeSoftmaxOptTheta"]


	stack = [util.Empty(), util.Empty()]

	stack[0].w = W11
	stack[0].b = b11
	stack[1].w = W12
	stack[1].b = b12

	# np.savez("result.npz", W11 = W11, W12 = W12, b11 = b11, b12 = b12, saeSoftmaxOptTheta = saeSoftmaxOptTheta)

	stackParams, netconfig = stackedae.stack2params(stack)
	stackedaeTheta = np.append(saeSoftmaxOptTheta, stackParams).ravel('F')

	# Finetuning
	fn = lambda theta: stackedae.cost(theta, inputSize, hiddenSizeL2, numOfClasses, netconfig, lamb, trainData, trainLabels)
	stackedaeOptTheta, loss, d = scipy.optimize.fmin_l_bfgs_b(fn, stackedaeTheta, maxfun=maxfun, iprint=1)

	# Test
	testData = mnist.load_images('../data/t10k-images-idx3-ubyte')
	testLabels = mnist.load_labels('../data/t10k-labels-idx1-ubyte')

	pred = stackedae.predict(stackedaeTheta, inputSize, hiddenSizeL2, numOfClasses, netconfig, testData)
	acc = (testLabels == pred).mean()
	print 'Before Finetuning Test Accuracy: %0.3f%%\n' % (acc * 100)

	pred = stackedae.predict(stackedaeOptTheta, inputSize, hiddenSizeL2, numOfClasses, netconfig, testData)
	acc = (testLabels == pred).mean()
	print 'After Finetuning Test Accuracy: %0.3f%%\n' % (acc * 100)