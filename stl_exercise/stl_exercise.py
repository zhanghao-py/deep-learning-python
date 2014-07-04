#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import mnist
from library import softmax
from library import autoencoder
from library import util

import scipy.optimize


if __name__ == '__main__':

	inputSize = 28 * 28
	numOfLabels = 5
	hiddenSize = 200
	sparsityParam = 0.1
	lamb = 3e-3
	beta = 3
	maxfun = 400

	print 'Loading raw MNIST data...'
	mnist_data = mnist.load_images('../data/train-images-idx3-ubyte')
	mnist_labels = mnist.load_labels('../data/train-labels-idx1-ubyte')

	# Simulate a Labeled and Unlabeled set

	print 'Splitting MNIST data...'
	labeledSet = mnist_labels <= 4
	unlabeledSet = mnist_labels >= 5

	unlabeledData = mnist_data[:, unlabeledSet]
	labeledData = mnist_data[:, labeledSet]
	labels = mnist_labels[labeledSet]

	numOfTrain = labels.size / 2

	trainData = labeledData[:, :numOfTrain]
	trainLabels = labels[:numOfTrain]

	testData = labeledData[:, numOfTrain:]
	testLabels = labels[numOfTrain:]

	# Output some statistics
	print '# examples in unlabeled set: %d' % unlabeledData.shape[1]
	print '# examples in supervised training set: %d' % trainData.shape[1]
	print '# examples in supervised testing set: %d' % testData.shape[1]

	# Randomly initialize the parameters
	theta = autoencoder.initializeParameters(hiddenSize, inputSize)

	fn = lambda theta: autoencoder.sparseAutoencoderCost(theta, inputSize, hiddenSize, lamb, sparsityParam, beta, unlabeledData)

	optTheta, cost, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=1, m=20)

	W1, W2, b1, b2 = autoencoder.unflatten(optTheta, inputSize, hiddenSize)

	np.savez("result.npz", W1 = W1)

	# util.display_network(W1.T)

	# trainFeatures = autoencoder.feedforwardAutoencoder(optTheta, hiddenSize, inputSize, trainData)
	# testFeatures = autoencoder.feedforwardAutoencoder(optTheta, hiddenSize, inputSize, testData)

	# lamb = 1e-4
	# numOfClasses = len(set(trainLabels))
	# softmaxModel = softmax.train(hiddenSize, numOfClasses, lamb, trainFeatures, trainLabels, maxfun=100)

	# pred = softmax.predict(softmaxModel, testFeatures)
	# acc = (testLabels == pred).mean()
	# print 'Accuracy: %0.3f' % (acc * 100)


