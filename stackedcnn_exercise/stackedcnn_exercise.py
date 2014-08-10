#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import datetime
import pickle

from library import mnist
from library import softmax
from library import autoencoder
from library import stackedcnn
from library import util

import scipy.optimize


if __name__ == '__main__':

	DISPLAY = False
	maxfun = 100

	inputSize = 28 * 28
	numOfClasses = 10

	trainData = mnist.load_images('../data/train-images-idx3-ubyte')
	trainLabels = mnist.load_labels('../data/train-labels-idx1-ubyte')
	testData = mnist.load_images('../data/t10k-images-idx3-ubyte')
	testLabels = mnist.load_labels('../data/t10k-labels-idx1-ubyte')

	# Build the cnn structure.
	stack = util.Empty()
	stack.layers = [util.Empty(), util.Empty(), util.Empty(), util.Empty(), util.Empty()]
	stack.layers[0].type = 'i'

	stack.layers[1].type = 'c'
	stack.layers[1].outputmaps = 6
	stack.layers[1].kernelsize = 5

	stack.layers[2].type = 's'
	stack.layers[2].scale = 2

	stack.layers[3].type = 'c'
	stack.layers[3].outputmaps = 12
	stack.layers[3].kernelsize = 5

	stack.layers[4].type = 's'
	stack.layers[4].scale = 2

	opts = util.Empty()
	opts.alpha = 1
	opts.batchsize = 50
	opts.numepochs = 1

	# The cnn setup & training
	# starttime = datetime.datetime.now()

	# stack = stackedcnn.setup(stack, trainData, numOfClasses)
	# stack = stackedcnn.train(stack, trainData, trainLabels, opts)

	# endtime = datetime.datetime.now()
	# print 'training stackedcnn last time : %s s' % (endtime-starttime)

	# Save CNN Stack
	# with open('stack.pickle', 'wb') as f:
		# pickle.dump(stack, f)

	starttime = datetime.datetime.now()
	
	with open('stack.pickle', 'rb') as f:
		stack = pickle.load(f)

	endtime = datetime.datetime.now()
	print 'training stackedcnn last time : %s s' % (endtime-starttime)

	# Test
	starttime = datetime.datetime.now()

	pred = stackedcnn.predict(stack, testData)

	endtime = datetime.datetime.now()
	print 'test stackedcnn last time : %s s' % (endtime-starttime)

	print pred
	print testLabels

	acc = (testLabels == pred).mean()
	print 'Test Accuracy: %0.3f%%\n' % (acc * 100)
