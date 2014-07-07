#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import stlsubset
from library import softmax
from library import cnn
from library import util

import scipy.optimize


if __name__ == '__main__':

	stepSize = 50;
	hiddenSize = 400;

	imageDim = 64;
	patchDim = 8;
	poolDim = 19;

	# Loading data...
	print 'Loading raw STLTrainSubset data...'
	r = np.load("trainImages.npz")
	trainImages = r["trainImages"]
	trainLabels = r["trainLabels"]

	print 'Loading raw STLTestSubset data...'
	r = np.load("testImages.npz")
	testImages = r["testImages"]
	testLabels = r["testLabels"]

	numTrainImages = trainImages.shape[1]
	numTestImages = testImages.shape[1]

	r = np.load("pooledFeatures.npz")
	pooledFeaturesTrain = r["pooledFeaturesTrain"]
	pooledFeaturesTest = r["pooledFeaturesTest"]

	# A = np.transpose(pooledFeaturesTrain, (2, 3, 0, 1))
	# A = A[:, :, :, 1]
	# A = A.reshape([9, 400])
	# util.display_network(A)


	# Train Softmax Classifier
	softmaxLambda = 1e-4;
	numOfClasses = 4;

	inputSize = np.size(pooledFeaturesTrain) / numTrainImages
	softmaxX = np.transpose(pooledFeaturesTrain, (0, 2, 3, 1))
	softmaxX = softmaxX.reshape([inputSize, numTrainImages])
	softmaxY = np.int_(trainLabels) - 1

	softmaxModel = softmax.train(inputSize, numOfClasses, softmaxLambda, softmaxX, softmaxY, maxfun=400)
	
	# Test
	inputSize = np.size(pooledFeaturesTest) / numTestImages
	softmaxX = np.transpose(pooledFeaturesTest, (0, 2, 3, 1))
	softmaxX = softmaxX.reshape([inputSize, numTestImages])
	softmaxY = np.int_(testLabels) - 1
	softmaxY = softmaxY.reshape([1, numTestImages])

	pred = softmax.predict(softmaxModel, softmaxX)

	acc = (softmaxY == pred).mean()
	print 'Accuracy: %0.3f' % (acc * 100)

