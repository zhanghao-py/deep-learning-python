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

	trainImages = trainImages[:,0:2]
	trainLabels = trainLabels[:,0:2]

	print 'Loading raw STLTestSubset data...'
	r = np.load("testImages.npz")
	testImages = r["testImages"]
	testLabels = r["testLabels"]

	testImages = testImages[:,0:2]
	testLabels = testLabels[:,0:2]

	numTrainImages = trainImages.shape[1]
	numTestImages = testImages.shape[1]

	print 'Loading linear decoder features...'
	r = np.load("../linear_decoder_exercise/result.npz")
	W = r["W1"]
	b = r["b1"]
	zcaWhite = r["zcaWhite"]
	meanPatch = r["meanPatch"]


	pooledFeaturesTrain = np.zeros( [hiddenSize, numTrainImages, np.floor((imageDim - patchDim + 1) / poolDim), np.floor((imageDim - patchDim + 1) / poolDim)] );
	pooledFeaturesTest = np.zeros( [hiddenSize, numTestImages, np.floor((imageDim - patchDim + 1) / poolDim), np.floor((imageDim - patchDim + 1) / poolDim)] );

	# Features Learning.
	for convPart in xrange(int(hiddenSize / stepSize)):

		featureStart = convPart * stepSize
		featureEnd = (convPart+1) * stepSize

		print 'Step %d: features %d to %d' % (convPart, featureStart, featureEnd)
		Wt = W[featureStart:featureEnd, :]
		bt = b[featureStart:featureEnd]

		print 'Convolving and pooling train images...'
		convolvedFeaturesThis = cnn.convolve(patchDim, stepSize, trainImages, Wt, bt, zcaWhite, meanPatch);
		pooledFeaturesThis = cnn.meanPooling(poolDim, convolvedFeaturesThis);
		pooledFeaturesTrain[featureStart:featureEnd, :, :, :] = pooledFeaturesThis;

		print 'Convolving and pooling test images'
		convolvedFeaturesThis = cnn.convolve(patchDim, stepSize, testImages, Wt, bt, zcaWhite, meanPatch);
		pooledFeaturesThis = cnn.meanPooling(poolDim, convolvedFeaturesThis);
		pooledFeaturesTest[featureStart:featureEnd, :, :, :] = pooledFeaturesThis;

	np.savez('pooledFeatures.npz', pooledFeaturesTrain = pooledFeaturesTrain, pooledFeaturesTest = pooledFeaturesTest)

	# print 'Convolving and pooling train images...'
	# convolvedFeaturesThis = cnn.convolve(patchDim, hiddenSize, trainImages, W, b, zcaWhite, meanPatch);
	# pooledFeaturesThis = cnn.meanPooling(poolDim, convolvedFeaturesThis);
	# pooledFeaturesTrain[:, :, :, :] = pooledFeaturesThis;

	# print 'Convolving and pooling test images'
	# convolvedFeaturesThis = cnn.convolve(patchDim, hiddenSize, testImages, W, b, zcaWhite, meanPatch);
	# pooledFeaturesThis = cnn.meanPooling(poolDim, convolvedFeaturesThis);
	# pooledFeaturesTest[:, :, :, :] = pooledFeaturesThis;

	# Train Softmax Classifier
	softmaxLambda = 1e-4;
	numOfClasses = 4;

	inputSize = np.size(pooledFeaturesTrain) / numTrainImages
	softmaxX = np.transpose(pooledFeaturesTrain, [0 2 3 1])
	softmaxX = softmaxX.reshape([inputSize, numTrainImages])
	softmaxY = trainLabels

	softmaxModel = softmax.train(inputSize, numOfClasses, softmaxLambda, softmaxX, softmaxY, maxfun=100)
	
	# Test
	inputSize = np.size(pooledFeaturesTest) / numTestImages
	softmaxX = np.transpose(pooledFeaturesTest, [0 2 3 1])
	softmaxX = softmaxX.reshape([inputSize, numTestImages])
	softmaxY = testLabels

	pred = softmax.predict(softmaxModel, softmaxX)
	acc = (softmaxY == pred).mean()
	print 'Accuracy: %0.3f' % (acc * 100)

