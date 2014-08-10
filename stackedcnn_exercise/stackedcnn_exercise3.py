#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import datetime
import pickle

from library import mnist
from library import cnn
from library import util

import scipy.optimize


if __name__ == '__main__':

	imageDim = 28
	numOfClasses = 10
	kernelDim = 9
	numOfKernels = 20
	poolDim = 2

	maxfun = 10
	batchSize = 100

	trainData = mnist.load_images('../data/train-images-idx3-ubyte')
	trainLabels = mnist.load_labels('../data/train-labels-idx1-ubyte')
	testData = mnist.load_images('../data/t10k-images-idx3-ubyte')
	testLabels = mnist.load_labels('../data/t10k-labels-idx1-ubyte')

	# Build the cnn structure.
	theta = cnn.initParams(imageDim, kernelDim, numOfKernels, poolDim, numOfClasses)

	# The cnn setup & training
	optTheta = cnn.train(theta, trainData, trainLabels, numOfClasses, kernelDim, numOfKernels, poolDim, batchSize, maxfun)

	pred = cnn.predict(optTheta, testData, kernelDim, numOfKernels, poolDim, numOfClasses)

	print pred
	print testLabels

	acc = (testLabels == pred).mean()
	print 'Test Accuracy: %0.3f%%\n' % (acc * 100)
