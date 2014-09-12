#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import datetime
import pickle

from library import mnist
from library import nn
from library import sparsedbn
from library import util

import scipy.optimize


if __name__ == '__main__':

	DISPLAY = False

	inputSize = 28 * 28
	numOfClasses = 10

	trainData = mnist.load_images('../data/train-images-idx3-ubyte')
	trainLabels = mnist.load_labels('../data/train-labels-idx1-ubyte')
	testData = mnist.load_images('../data/t10k-images-idx3-ubyte')
	testLabels = mnist.load_labels('../data/t10k-labels-idx1-ubyte')

	# Build the cnn structure.
	stack = util.Empty()
	stack.layers = [util.Empty(), util.Empty()]
	stack.layers[0].type = 'i'
	stack.layers[0].size = inputSize

	stack.layers[1].type = 'rbm'
	stack.layers[1].size = 400

	# stack.layers[2].type = 'rbm'
	# stack.layers[2].size = 400

	# stack.layers[3].type = 'rbm'
	# stack.layers[3].size = 900

	opts = util.Empty()
	opts.alpha = 0.1
	opts.batchsize = 100
	opts.momentun = 0
	opts.numepochs = 1


	# The deep brief nets setup & training
	stack = sparsedbn.setup(stack, trainData, opts)
	stack = sparsedbn.train(stack, trainData, opts)

	# if DISPLAY:
  		# util.display_network(stack.layers[1].W.T)


	# Softmax Classifier training
  	# trainFeature = sparsedbn.feedforward(stack, trainData)
  	# testFeature = sparsedbn.feedforward(stack, testData)

  	opts.numepochs = 1
	opts.batchsize = 100
  	nnstack = sparsedbn.dbn2nn(stack, numOfClasses)
  	nnstack = nn.train(nnstack, trainData, trainLabels, opts)

  	# Predict
  	pred = nn.predict(nnstack, testData)
  	print pred
  	print pred.shape
	# acc = (testLabels == pred).mean()
	# print 'After Finetuning Accuracy: %0.3f' % (acc * 100)

