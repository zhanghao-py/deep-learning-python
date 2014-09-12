#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import datetime
import pickle

from library import mnist
from library import autoencoder
from library import softmax
from library import dbn
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
	stack.layers = [util.Empty(), util.Empty(), util.Empty()]
	stack.layers[0].type = 'i'
	stack.layers[0].size = inputSize

	stack.layers[1].type = 'rbm'
	stack.layers[1].size = 100

	stack.layers[2].type = 'rbm'
	stack.layers[2].size = 100

	opts = util.Empty()
	opts.alpha = 1
	opts.batchsize = 100
	opts.momentun = 0
	opts.numepochs = 50


	# The cnn setup & training
	# starttime = datetime.datetime.now()

	stack = dbn.setup(stack, trainData, opts)
	stack = dbn.train(stack, trainData, opts)

	if DISPLAY:
  		util.display_network(stack.layers[1].W.T)
  		util.display_network(stack.layers[2].W.T)


  	trainFeature = dbn.feedforward(stack, trainData)
  	testFeature = dbn.feedforward(stack, testData)

  	lamb = 1e-4
  	maxfun = 400
  	softmaxModel = softmax.train(trainFeature.shape[0], numOfClasses, lamb, trainFeature, trainLabels, maxfun=maxfun)
  	pred = softmax.predict(softmaxModel, testFeature)

	acc = (testLabels == pred).mean()

	print 'Accuracy: %0.3f' % (acc * 100)

	# endtime = datetime.datetime.now()
	# print 'training stackedcnn last time : %s s' % (endtime-starttime)
