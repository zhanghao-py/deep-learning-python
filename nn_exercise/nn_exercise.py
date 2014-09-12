#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import datetime
import pickle

from library import mnist
from library import nn
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

	# Build the nn structure.
	stack = util.Empty()
	stack.layers = [util.Empty(), util.Empty(), util.Empty()]
	stack.layers[0].type = 'hidden'
	stack.layers[0].size = inputSize

	stack.layers[1].type = 'hidden'
	stack.layers[1].size = 50

	stack.layers[2].type = 'output'
	stack.layers[2].size = numOfClasses

	opts = util.Empty()
	opts.alpha = 0.1
	opts.batchsize = 100
	opts.momentun = 0
	opts.numepochs = 10


	# The nn setup & training
	stack = nn.setup(stack)
	stack = nn.train(stack, trainData, trainLabels, opts)

	if DISPLAY:
  		util.display_network(stack.layers[1].W.T)

  	pred = nn.predict(stack, testData)

  	print pred
  	print testLabels

	acc = (testLabels == pred).mean()

	print 'Accuracy: %0.3f' % (acc * 100)
