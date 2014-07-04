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

	# Train the first sparse autoencoder.
	# Randomly initialize the parameters
	sae1Theta = autoencoder.initializeParameters(hiddenSizeL1, inputSize)

	fn = lambda theta: autoencoder.sparseAutoencoderCost(theta, inputSize, hiddenSizeL1, lamb, sparsityParam, beta, trainData)
	sae1OptTheta, loss, d = scipy.optimize.fmin_l_bfgs_b(fn, sae1Theta, maxfun=maxfun, iprint=1)

	if DISPLAY:
  		W1, W2, b1, b2 = autoencoder.unflatten(sae1OptTheta, inputSize, hiddenSizeL1)
  		util.display_network(W1.T)

	sae1Features = autoencoder.feedforwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData)

	# Train the second sparse autoencoder.
	# Randomly initialize the parameters
	sae2Theta = autoencoder.initializeParameters(hiddenSizeL2, hiddenSizeL1)

	fn = lambda theta: autoencoder.sparseAutoencoderCost(theta, hiddenSizeL1, hiddenSizeL2, lamb, sparsityParam, beta, sae1Features)
	sae2OptTheta, loss, d = scipy.optimize.fmin_l_bfgs_b(fn, sae2Theta, maxfun=maxfun, iprint=1)

	if DISPLAY:
  		W11, W21, b11, b21 = autoencoder.unflatten(sae1OptTheta, inputSize, hiddenSizeL1)
  		W12, W22, b12, b22 = autoencoder.unflatten(sae2OptTheta, hiddenSizeL1, hiddenSizeL2)
  		# figure out how to display a 2-level network
  		# util.display_network( np.log(W11.T / (1-W11.T)).dot(W12.T) )

	sae2Features = autoencoder.feedforwardAutoencoder(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features)


	# Train the softmax classifier.
	saeSoftmaxTheta = 0.005 * np.random.randn(hiddenSizeL2 * numOfClasses, 1)

	softmaxModel = softmax.train(hiddenSizeL2, numOfClasses, 1e-4, sae2Features, trainLabels, maxfun=maxfun)

	saeSoftmaxOptTheta = softmaxModel['optTheta'].ravel('F')

	stack = [util.Empty(), util.Empty()]
	W11, W21, b11, b21 = autoencoder.unflatten(sae1OptTheta, inputSize, hiddenSizeL1)
	W12, W22, b12, b22 = autoencoder.unflatten(sae2OptTheta, hiddenSizeL1, hiddenSizeL2)
	stack[0].w = W11
	stack[0].b = b11
	stack[1].w = W12
	stack[1].b = b12

	np.savez("result.npz", W11 = W11, W12 = W12, b11 = b11, b12 = b12, saeSoftmaxOptTheta = saeSoftmaxOptTheta)

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