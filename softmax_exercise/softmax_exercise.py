#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import mnist
from library import softmax


if __name__ == '__main__':

	inputSize = 28 * 28
	numOfClasses = 10
	lamb = 1e-4

	trainImages = mnist.load_images('../data/train-images-idx3-ubyte')
	trainLabels = mnist.load_labels('../data/train-labels-idx1-ubyte')

	softmaxModel = softmax.train(inputSize, numOfClasses, lamb, trainImages, trainLabels, maxfun=100)

	testImages = mnist.load_images('../data/t10k-images-idx3-ubyte')
	testLabels = mnist.load_labels('../data/t10k-labels-idx1-ubyte')

	pred = softmax.predict(softmaxModel, testImages)

	acc = (testLabels == pred).mean()

	print 'Accuracy: %0.3f' % (acc * 100)

