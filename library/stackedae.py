#!/usr/bin/env python

import numpy as np
from library import util
from library import autoencoder

def stack2params(stack):
	params = np.empty([1,0], order='F')
	netconfig = util.Empty()
	netconfig.layersizes = []
	netconfig.inputsize = 0
	if stack:
		netconfig.inputsize = stack[0].w.shape[1]
		for entry in stack:
			params = np.hstack([params, entry.w.reshape([1, -1], order='F'), entry.b.reshape([1, -1], order='F')])
			netconfig.layersizes.append(entry.w.shape[0])

	return params.ravel('F'), netconfig

def params2stack(params, netconfig):
	depth = len(netconfig.layersizes)
	stack = []
	prev_layersize = netconfig.inputsize
	cur_pos = 0

	for d in xrange(depth):
		entry = util.Empty()

		# Extract weights
		layersize = netconfig.layersizes[d]
		wlen = layersize * prev_layersize
		entry.w = params[cur_pos:cur_pos + wlen].reshape([layersize, prev_layersize], order='F')
		cur_pos += wlen

		# Extract bias
		blen = layersize
		entry.b = params[cur_pos:cur_pos + blen].reshape([layersize, 1], order='F')
		cur_pos += blen

		prev_layersize = layersize
		stack.append(entry)

	return stack

def cost(theta, inputSize, hiddenSize, numOfClasses, netconfig, lamb, data, labels):

	softmaxTheta = theta[:hiddenSize * numOfClasses].reshape([numOfClasses, hiddenSize], order='F')

	# Extract out the "stack"
	stack = params2stack(theta[hiddenSize * numOfClasses:], netconfig)
	depth = len(stack)
	numOfSamples = data.shape[1]
	groundTruth = np.zeros([numOfClasses, numOfSamples])
	groundTruth[labels.ravel(), np.arange(numOfSamples)] = 1

	z = [0]
	a = [data]

	for layer in xrange(depth):
		z.append( stack[layer].w.dot(a[layer]) + stack[layer].b )
		a.append( autoencoder.sigmoid(z[layer+1]) )

	td = softmaxTheta.dot(a[depth])
	td = td - td.max(0)

	p = np.exp(td) / np.exp(td).sum(0)

	cost = (-1.0/numOfSamples) * (groundTruth * np.log(p)).sum() + (lamb/2) * np.power(softmaxTheta, 2).sum();
	softmaxThetaGrad = -1.0/numOfSamples * (groundTruth - p).dot(a[depth].T) + lamb * softmaxTheta

	delta = [0 for _ in xrange(depth+1)]

	delta[depth] = -(softmaxTheta.T.dot(groundTruth - p)) * a[depth] * (1-a[depth])

	for layer in range(depth-1, 0, -1):
		delta[layer] = stack[layer].w.T.dot(delta[layer+1]) * a[layer] * (1-a[layer])

	stackGrad = [util.Empty() for _ in xrange(depth)]
	for layer in range(depth-1, -1, -1):
		stackGrad[layer].w = (1.0/numOfSamples) * delta[layer+1].dot(a[layer].T)
		stackGrad[layer].b = (1.0/numOfSamples) * np.sum(delta[layer+1], 1)

	grad = np.append(softmaxThetaGrad.ravel('F'), stack2params(stackGrad)[0])

	assert (grad.shape==theta.shape)
	# assert grad.flags['F_CONTIGUOUS']

	return cost, grad

def predict(theta, input_size, hidden_size, num_classes, netconfig, data):
	# We first extract the part which compute the softmax gradient
	softmax_theta = theta[:hidden_size * num_classes].reshape([num_classes, hidden_size], order='F')

	# Extract out the "stack"
	stack = params2stack(theta[hidden_size * num_classes:], netconfig)

	depth = len(stack)
	z = [0]
	a = [data]

	for layer in xrange(depth):
 		z.append(stack[layer].w.dot(a[layer]) + stack[layer].b)
		a.append(autoencoder.sigmoid(z[layer+1]))

	return softmax_theta.dot(a[depth]).argmax(0)