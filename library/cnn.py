#!/usr/bin/env python

import numpy as np
import scipy.signal

# from library import util
from library import autoencoder


def maxPooling(poolDim, convolvedFeatures):

	numOfFeatures = convolvedFeatures.shape[0]
	numOfImages = convolvedFeatures.shape[1]
	convolvedDim = convolvedFeatures.shape[2]

	resDim = np.floor(convolvedDim / poolDim)

	pooledFeatures = np.zeros([numOfFeatures, numOfImages, np.floor(convolvedDim / poolDim), np.floor(convolvedDim / poolDim)])

	for imageNum in xrange(numOfImages):
		for featureNum in xrange(numOfFeatures):
			for poolRow in xrange(int(resDim)):
				rowStart = poolRow * poolDim
				rowEnd = rowStart + poolDim

				for poolCol in xrange(int(resDim)):
					colStart = poolCol * poolDim
					colEnd = colStart + poolDim

					patch = convolvedFeatures[featureNum, imageNum, rowStart:rowEnd, colStart:colEnd]

					pooledFeatures[featureNum, imageNum, poolRow, poolCol] = np.max(patch);

	return pooledFeatures


def meanPooling(poolDim, convolvedFeatures):

	numOfFeatures = convolvedFeatures.shape[0]
	numOfImages = convolvedFeatures.shape[1]
	convolvedDim = convolvedFeatures.shape[2]

	resDim = np.floor(convolvedDim / poolDim)

	pooledFeatures = np.zeros([numOfFeatures, numOfImages, np.floor(convolvedDim / poolDim), np.floor(convolvedDim / poolDim)])

	for imageNum in xrange(numOfImages):
		for featureNum in xrange(numOfFeatures):
			for poolRow in xrange(int(resDim)):
				rowStart = poolRow * poolDim
				rowEnd = rowStart + poolDim

				for poolCol in xrange(int(resDim)):
					colStart = poolCol * poolDim
					colEnd = colStart + poolDim

					patch = convolvedFeatures[featureNum, imageNum, rowStart:rowEnd, colStart:colEnd]

					pooledFeatures[featureNum, imageNum, poolRow, poolCol] = np.mean(patch);

	return pooledFeatures


def convolve(patchDim, numOfFeatures, images, W, b, zcaWhite, meanPatch):

	inputSize, numOfImages = images.shape
	imageChannels = 3
	imageDimPower = inputSize / imageChannels
	imageDim = np.sqrt(imageDimPower)

	wt = W.dot(zcaWhite)
	bt = b - wt.dot(meanPatch)

	convolvedFeatures = np.zeros([numOfFeatures, numOfImages, imageDim - patchDim + 1, imageDim - patchDim + 1])

	for imageNum in xrange(numOfImages):
		for featureNum in xrange(numOfFeatures):

			convolvedImage = np.zeros([imageDim - patchDim + 1, imageDim - patchDim + 1])

			for channel in xrange(imageChannels):

				first = patchDim * patchDim * channel
				last = first + patchDim * patchDim

				feature = wt[featureNum, first:last].reshape([patchDim, patchDim])
				im = images[imageDimPower*channel:imageDimPower*(channel+1), imageNum]
				im = im.reshape([imageDim, imageDim])

				convolvedImage = convolvedImage + scipy.signal.fftconvolve(im, feature, mode = 'valid')
	

			convolvedImage = autoencoder.sigmoid(convolvedImage + bt[featureNum])
			convolvedFeatures[featureNum, imageNum, :, :] = convolvedImage

	return convolvedFeatures