#!/usr/bin/env python

import numpy as np
import scipy.signal
import scipy.ndimage
import datetime

from library import util
from library import autoencoder

def rot180(matrix):
	return np.rot90(matrix, k = 2)

def setup(stack, trainData, numOfClasses):

	inputmaps = 1

	inputSize, numOfSamples = trainData.shape
	patchSize = np.sqrt(inputSize)

	for l in xrange(len(stack.layers)):
		if stack.layers[l].type == 's':
			patchSize = patchSize / stack.layers[l].scale

			stack.layers[l].b = [0 for x in xrange(inputmaps)]

			for j in xrange(inputmaps):
				stack.layers[l].b[j] = 0

		if stack.layers[l].type == 'c':
			kernelSize = stack.layers[l].kernelsize
			outputmaps = stack.layers[l].outputmaps
			patchSize = patchSize - kernelSize + 1
			fanOut = outputmaps * kernelSize ^ 2

			stack.layers[l].k = [[0 for x in xrange(outputmaps)] for x in xrange(inputmaps)]
			stack.layers[l].b = [0 for x in xrange(outputmaps)]

			for j in xrange(outputmaps):
				fanIn = inputmaps * np.power(kernelSize, 2)

				for i in xrange(inputmaps):
					stack.layers[l].k[i][j] = (np.random.rand(kernelSize, kernelSize) - 0.5) * 2 * np.sqrt(6.0 / (fanIn + fanOut))

				stack.layers[l].b[j] = 0

			inputmaps = outputmaps

	numOfFeatures = patchSize * patchSize * inputmaps

	# stack.ffb = np.zeros((numOfClasses, 1))
	stack.ffTheta = (np.random.rand(numOfClasses, numOfFeatures) - 0.5) * 2 * np.sqrt(6.0 / (numOfClasses + numOfFeatures))
	stack.rLoss = []

	return stack

def train(stack, trainData, trainLabels, opts):

	inputSize, numOfSamples = trainData.shape
	numOfBatches = numOfSamples / opts.batchsize

	for i in xrange(opts.numepochs):

		for l in xrange(numOfBatches):

			if (l+1) % 10 == 0:
				print 'Opts.numepochs %d Iteration numOfBatches: %d/%d' % (i+1, l+1, numOfBatches)

			batchStart = l * opts.batchsize
			batchEnd = (l+1) * opts.batchsize

			batchTrainData = trainData[:, batchStart:batchEnd]
			batchTrainLabels = trainLabels[batchStart:batchEnd]

			stack = feedforward(stack, batchTrainData)
			stack = backpropagation(stack, batchTrainLabels)
			stack = applygrads(stack, opts)

			if stack.rLoss:
				# Not Empty
				stack.rLoss.append( 0.99 * stack.rLoss[-1] + 0.01 * stack.loss )
			else:
				# Empty
				stack.rLoss.append(stack.loss)

	return stack

def predict(stack, testData):

	stack = feedforward(stack, testData);
	pred = np.argmax(stack.o, axis = 0)
	
	return np.array(pred).flatten()

def feedforward(stack, trainData):

	inputmaps = 1

	stack.layers[0].a = [0 for x in xrange(inputmaps)]
	stack.layers[0].a[0] = trainData

	depth = len(stack.layers)

	for l in xrange(depth):
		if stack.layers[l].type == 'c':
			
			inputSize, numOfSamples = stack.layers[l-1].a[0].shape
			patchSize = np.sqrt(inputSize)
			kernelSize = stack.layers[l].kernelsize
			outputmaps = stack.layers[l].outputmaps

			stack.layers[l].a = [0 for x in xrange(outputmaps)]

			for j in xrange(outputmaps):
				
				z = np.zeros([(patchSize - kernelSize + 1) * (patchSize - kernelSize + 1), numOfSamples])
				
				for i in xrange(inputmaps):
					
					imgs = stack.layers[l-1].a[i].reshape([patchSize, patchSize, numOfSamples])
					kernel = stack.layers[l].k[i][j]

					rets = np.zeros([(patchSize - kernelSize + 1) * (patchSize - kernelSize + 1), numOfSamples])
					for imgIdx in xrange(numOfSamples):
						img = imgs[:, :, imgIdx]
						ret = scipy.signal.convolve(img, kernel, mode = 'valid')
						rets[:, imgIdx] = ret.flatten()

					z += rets

				stack.layers[l].a[j] = autoencoder.sigmoid(z + stack.layers[l].b[j])

			inputmaps = outputmaps

		elif stack.layers[l].type == 's':

			inputSize, numOfSamples = stack.layers[l-1].a[0].shape
			patchSize = np.sqrt(inputSize)
			scale = stack.layers[l].scale

			stack.layers[l].a = [0 for x in xrange(inputmaps)]

			for j in xrange(inputmaps):

				imgs = stack.layers[l-1].a[j].reshape([patchSize, patchSize, numOfSamples])
				kernel = np.ones([scale, scale]) / np.power(scale, 2)

				rets = np.zeros([(patchSize/scale) * (patchSize/scale), numOfSamples])
				for imgIdx in xrange(numOfSamples):
					img = imgs[:, :, imgIdx]
					ret = scipy.signal.convolve(img, kernel, mode = 'valid')
					ret = ret[::scale, ::scale]
					rets[:, imgIdx] = ret.flatten()
				
				stack.layers[l].a[j] = rets


	outputLayer = stack.layers[depth-1]
	numOfFeatures = outputLayer.a[0].shape[0]	
	fv = np.ones([0, numOfSamples])

	for j in xrange(len(outputLayer.a)):
		vector = outputLayer.a[j]
		fv = np.bmat('fv; vector')
	
	stack.fv = fv

	# Softmax Classifier
	td = stack.ffTheta.dot(stack.fv)
	td = td - td.max(0)
	td = np.exp(td)
	p = td / td.sum(0)

	stack.o = p

	return stack

def backpropagation(stack, trainLabels):

	lamb = 3e-3
	depth = len(stack.layers)

	numOfClasses, numOfSamples = stack.o.shape

	# constructe groundTruth
	groundTruth = np.zeros([numOfClasses, numOfSamples])
	groundTruth[trainLabels, np.arange(numOfSamples)] = 1

	# loss function
	stack.loss = (-1.0/numOfSamples) * (np.multiply(groundTruth, np.log(stack.o))).sum() + (lamb/2) * np.power(stack.ffTheta, 2).sum();
	# output delta
	stack.od = -(groundTruth - stack.o)
	# feature vector delta
	stack.fvd = stack.ffTheta.T.dot(stack.od)

	if stack.layers[depth-1].type == 'c':
		stack.fvd = np.multiply(stack.fvd, np.multiply(stack.fv, (1 - stack.fv)))

	# reshape feature vector deltas into output map style
	numOfFeatures, numOfSamples = stack.layers[depth-1].a[0].shape
	stack.layers[depth-1].d = [0 for x in xrange(len(stack.layers[depth-1].a))]
	for j in xrange(len(stack.layers[depth-1].a)):
		stack.layers[depth-1].d[j] = stack.fvd[j * numOfFeatures : (j+1) * numOfFeatures, :]

	for l in xrange(depth-1-1, 0, -1):
		if stack.layers[l].type == 'c':

			stack.layers[l].d = [0 for x in xrange(len(stack.layers[l].a))]
			
			for j in xrange(len(stack.layers[l].a)):

				delta = stack.layers[l+1].d[j]
				scale = stack.layers[l+1].scale

				inputSize, numOfSamples = delta.shape
				patchSize = np.sqrt(inputSize)

				rets = np.ones([inputSize * np.power(scale, 2), numOfSamples])

				for sampleIdx in xrange(numOfSamples):
					# ret = np.kron( delta[:, sampleIdx].reshape([patchSize, patchSize]), np.ones([scale, scale]) )
					ret = scipy.ndimage.zoom(delta[:, sampleIdx].reshape([patchSize, patchSize]), zoom = scale, order = 0)
					rets[:, sampleIdx] = ret.flatten()

				stack.layers[l].d[j] = (1.0 / np.power(scale, 2)) * np.multiply( np.multiply( stack.layers[l].a[j], (1 - stack.layers[l].a[j]) ), rets )

		elif stack.layers[l].type == 's':

			stack.layers[l].d = [0 for x in xrange(len(stack.layers[l].a))]

			for i in xrange(len(stack.layers[l].a)):
				
				z = np.zeros(stack.layers[l].a[0].shape)
				
				for j in xrange(len(stack.layers[l+1].a)):


					delta = stack.layers[l+1].d[j]
					kernel = stack.layers[l+1].k[i][j]
					kernelSize = stack.layers[l+1].kernelsize

					inputSize, numOfSamples = delta.shape
					patchSize = np.sqrt(inputSize)

					rets = np.zeros([(patchSize+kernelSize-1) * (patchSize+kernelSize-1), numOfSamples])
					for sampleIdx in xrange(numOfSamples):
						ret = scipy.signal.convolve(delta[:, sampleIdx].reshape([patchSize, patchSize]), rot180(kernel), mode = 'full')
						rets[:, sampleIdx] = ret.flatten()

					z = z + rets

				stack.layers[l].d[i] = z

	# calc gradients
	for l in xrange(depth):
		if stack.layers[l].type == 'c':

			stack.layers[l].dk = [[0 for x in xrange(len(stack.layers[l].a))] for x in xrange(len(stack.layers[l-1].a))]
			stack.layers[l].db = [0 for x in xrange(len(stack.layers[l].a))]

			for j in xrange(len(stack.layers[l].a)):

				inputSize, numOfSamples = stack.layers[l].d[j].shape

				for i in xrange(len(stack.layers[l-1].a)):

					patches = stack.layers[l-1].a[i]
					kernels =  stack.layers[l].d[j]

					patchSize, numOfSamples = patches.shape
					kernelSize, numOfSamples = kernels.shape

					patchSize = np.sqrt(patchSize)
					kernelSize = np.sqrt(kernelSize)

					scale = patchSize - kernelSize + 1

					ret = np.zeros([scale, scale])
					for idx in xrange(numOfSamples):
						ret += scipy.signal.convolve(patches[:, idx].reshape([patchSize, patchSize]), rot180(kernels[:, idx].reshape([kernelSize, kernelSize])), mode = 'valid')

					stack.layers[l].dk[i][j] = rot180(ret) / numOfSamples

				stack.layers[l].db[j] = np.sum(stack.layers[l].d[j][:]) / numOfSamples;


	# stack.dffTheta = stack.od.dot(stack.fv.T) / stack.od.shape[1]
	stack.dffTheta = 1.0/numOfSamples * stack.od.dot(stack.fv.T) + lamb * stack.ffTheta
	# stack.dffb = np.mean(stack.od, axis = 1)

	return stack

def applygrads(stack, opts):

	depth = len(stack.layers)
	for l in xrange(depth):

		if stack.layers[l].type == 'c':
			for j in xrange(len(stack.layers[l].a)):

				for i in xrange(len(stack.layers[l - 1].a)):
					stack.layers[l].k[i][j] = stack.layers[l].k[i][j] - opts.alpha * stack.layers[l].dk[i][j]

				stack.layers[l].b[j] = stack.layers[l].b[j] - opts.alpha * stack.layers[l].db[j]

	stack.ffTheta = stack.ffTheta - opts.alpha * stack.dffTheta
	# stack.ffb = stack.ffb - opts.alpha * stack.dffb

	return stack