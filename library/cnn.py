#!/usr/bin/env python

import numpy as np
import scipy.signal

# from library import util
from library import autoencoder

def flatten(Wc, Wd, bc, bd):
	return np.array(np.hstack([Wc.ravel('F'), Wd.ravel('F'), bc.ravel('F'), bd.ravel('F')]), order='F')

def meanPooling(poolDim, convolvedFeatures):

	numOfSamples = convolvedFeatures.shape[3]
	numOfKernels = convolvedFeatures.shape[2]
	convolvedDim = convolvedFeatures.shape[0]

	outputDim = np.floor(convolvedDim / poolDim)

	pooledFeatures = np.zeros([outputDim, outputDim, numOfKernels, numOfSamples])

	for sampleIdx in xrange(numOfSamples):
		for kernelIdx in xrange(numOfKernels):

			tmp = scipy.signal.convolve(convolvedFeatures[:, :, kernelIdx, sampleIdx], np.ones([poolDim, poolDim]), mode = 'valid')
			pooledFeatures[:, :, kernelIdx, sampleIdx] = 1.0 / np.power(poolDim, 2) * tmp[::poolDim, ::poolDim]

	return pooledFeatures

def initParams(imageDim, kernelDim, numOfKernels, poolDim, numOfClasses):

	sigma = 1e-1
	mu = 0

	convolvedDim = imageDim - kernelDim + 1
	outputDim = np.floor(convolvedDim / poolDim)

	hiddenSize = np.power(outputDim, 2) * numOfKernels

	Wc = sigma * np.random.randn(kernelDim, kernelDim, numOfKernels) + mu
	Wd = (np.random.rand(numOfClasses, hiddenSize) - 0.5) * 2 * np.sqrt(6.0 / (numOfClasses + hiddenSize))

	bc = np.zeros([numOfKernels, 1])
	bd = np.zeros([numOfClasses, 1])

	theta = flatten(Wc, Wd, bc, bd)

	return theta

def params2Stack(theta, imageDim, kernelDim, numOfKernels, poolDim, numOfClasses):
	convolvedDim = imageDim - kernelDim + 1
	outputDim = np.floor(convolvedDim / poolDim)

	hiddenSize = np.power(outputDim, 2) * numOfKernels

	indexStart = 0
	indexEnd = np.power(kernelDim, 2) * numOfKernels
	Wc = theta[indexStart:indexEnd].reshape([kernelDim, kernelDim, numOfKernels])

	indexStart = indexEnd;
	indexEnd = indexEnd + hiddenSize * numOfClasses
	Wd = theta[indexStart:indexEnd].reshape([numOfClasses, hiddenSize])

	indexStart = indexEnd;
	indexEnd = indexEnd + numOfKernels
	bc = theta[indexStart:indexEnd].reshape([numOfKernels, 1])

	indexStart = indexEnd
	indexEnd = indexEnd + numOfClasses
	bd = theta[indexStart:indexEnd].reshape([numOfClasses, 1])

	return Wc, Wd, bc, bd

def feedforward(theta, images, kernelDim, numOfKernels, poolDim, numOfClasses):

	inputSize, numOfSamples = images.shape
	imageDim = np.sqrt(inputSize)

	convolvedDim = imageDim - kernelDim + 1
	outputDim = np.floor(convolvedDim / poolDim)

	hiddenSize = np.power(outputDim, 2) * numOfKernels

	Wc, Wd, bc, bd = params2Stack(theta, imageDim, kernelDim, numOfKernels, poolDim, numOfClasses)

	convolvedFeatures = convolve(kernelDim, numOfKernels, images, Wc, bc)
	activationsPooled = meanPooling(poolDim, convolvedFeatures)

	# Reshape activations into 2-d matrix, hiddenSize x numImage for Softmax layer
	activationsPooled = activationsPooled.reshape([hiddenSize, numOfSamples])

	td = Wd.dot(activationsPooled) + bd
	td = td - td.max(0)
	td = np.exp(td)
	probs = td / td.sum(0)

	return convolvedFeatures, activationsPooled, probs

def cost(theta, images, labels, numOfClasses, kernelDim, numOfKernels, poolDim):

	inputSize, numOfSamples = images.shape
	imageDim = np.sqrt(inputSize)

	convolvedDim = imageDim - kernelDim + 1
	outputDim = np.floor(convolvedDim / poolDim)

	lamb = 3e-3

	Wc, Wd, bc, bd = params2Stack(theta, imageDim, kernelDim, numOfKernels, poolDim, numOfClasses)

	Wc_grad = np.zeros(Wc.shape)
	bc_grad = np.zeros(bc.shape)

	convolvedFeatures, activationsPooled, probs = feedforward(theta, images, kernelDim, numOfKernels, poolDim, numOfClasses)

	groundTruth = np.zeros([numOfClasses, numOfSamples])
	groundTruth[labels.ravel('F'), np.arange(numOfSamples)] = 1

	cost = (-1.0/numOfSamples) * np.multiply(groundTruth, np.log(probs)).sum() + (lamb/2) * ( np.power(Wd, 2).sum() + np.power(Wc, 2).sum() )

	delta_d = probs - groundTruth
	Wd_grad = (1.0/numOfSamples) * delta_d.dot(activationsPooled.T) + lamb * Wd
	bd_grad = (1.0/numOfSamples) * np.sum(delta_d, 1)

	delta_s = Wd.T.dot(delta_d)
	delta_s = delta_s.reshape([outputDim, outputDim, numOfKernels, numOfSamples])
	
	# calculate delta_c
	delta_c = np.zeros([convolvedDim, convolvedDim, numOfKernels, numOfSamples])

	for i in xrange(numOfSamples):
		for j in xrange(numOfKernels):
			delta_c[:, :, j, i] = (1.0/np.power(poolDim, 2)) * np.kron( delta_s[:, :, j, i], np.ones([poolDim, poolDim]) )

	delta_c = np.multiply( np.multiply(convolvedFeatures, (1-convolvedFeatures)), delta_c )

	# calculate Wc_grad
	for i in xrange(numOfKernels):
		Wc_i = np.zeros([kernelDim, kernelDim])

		for j in xrange(numOfSamples):
			k = delta_c[:, :, i, j]
        	Wc_i = Wc_i + scipy.signal.convolve(images[:,j].reshape([imageDim, imageDim]), rot180(k), mode = 'valid')

		Wc_grad[:, :, i] = (1.0/numOfSamples) * Wc_i + lamb * Wc[:, :, i]

		bc_i = delta_c[:, :, i, :]
		bc_i = bc_i[:]
		bc_grad[i] = np.sum(bc_i)/numOfSamples

	grad = flatten(Wc_grad, Wd_grad, bc_grad, bd_grad)

	return cost, grad

def train(theta, trainData, trainLabels, numOfClasses, kernelDim, numOfKernels, poolDim, batchSize = 100, maxfun = 3):

	inputSize, numOfSamples = trainData.shape
	numOfBatches = numOfSamples / batchSize

	fn = lambda theta: cost(theta, trainData, trainLabels, numOfClasses, kernelDim, numOfKernels, poolDim)
	theta, f, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=25, m=20)

	# for batchIdx in xrange(numOfBatches):

	# 	print 'Iteration numOfBatches: %d/%d' % (batchIdx+1, numOfBatches)

	# 	batchStart = batchIdx * batchSize
	# 	batchEnd = (batchIdx+1) * batchSize

	# 	batchTrainData = trainData[:, batchStart:batchEnd]
	# 	batchTrainLabels = trainLabels[batchStart:batchEnd]

	# 	fn = lambda theta: cost(theta, batchTrainData, batchTrainLabels, numOfClasses, kernelDim, numOfKernels, poolDim)
	# 	theta, f, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=25, m=20)

	return theta

def predict(optTheta, images, kernelDim, numOfKernels, poolDim, numOfClasses):

	convolvedFeatures, activationsPooled, probs = feedforward(optTheta, images, kernelDim, numOfKernels, poolDim, numOfClasses)
	pred = np.argmax(probs, axis = 0)

	return np.array(pred).flatten()

def convolve(kernelDim, numOfKernels, images, W, b):

	inputSize, numOfSamples = images.shape
	imageDim = np.sqrt(inputSize)

	convolvedDim = imageDim - kernelDim + 1

	convolvedFeatures = np.zeros([convolvedDim, convolvedDim, numOfKernels, numOfSamples])

	for sampleIdx in xrange(numOfSamples):
		for kernelIdx in xrange(numOfKernels):

			kernel = W[:, :, kernelIdx]
			kernel = rot180(kernel)

			bc = b[kernelIdx]

			image = images[:, sampleIdx]
			image = image.reshape([imageDim, imageDim])

			convolvedImage = scipy.signal.convolve(image, kernel, mode = 'valid')
			convolvedImage = autoencoder.sigmoid(convolvedImage + bc)

			convolvedFeatures[:, :, kernelIdx, sampleIdx] = convolvedImage

	return convolvedFeatures

def rot180(matrix):
	return np.rot90(matrix, k = 2)