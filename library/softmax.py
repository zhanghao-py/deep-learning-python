#!/usr/bin/env python

import numpy as np
import scipy.optimize

def cost(theta, numOfClasses, inputSize, lamb, data, labels):
	theta = theta.reshape([numOfClasses, inputSize], order='F')
	numOfSamples = data.shape[1]
	groundTruth = np.zeros([numOfClasses, numOfSamples])
	groundTruth[labels.ravel(), np.arange(numOfSamples)] = 1

	td = theta.dot(data)
	td = td - td.max(0)

	p = np.exp(td) / np.exp(td).sum(0)

	cost = (-1.0/numOfSamples) * (groundTruth * np.log(p)).sum() + (lamb/2) * np.power(theta, 2).sum();

	thetaGrad = -1.0/numOfSamples * (groundTruth - p).dot(data.T) + lamb * theta

	# Unroll into a vector
	grad = thetaGrad.ravel('F')

	return cost, grad


def train(inputSize, numOfClasses, lamb, data, labels, maxfun=400):
	theta = 0.005 * np.random.randn(numOfClasses * inputSize, 1)
	theta = np.asfortranarray(theta)

	fn = lambda theta: cost(theta, numOfClasses, inputSize, lamb, data, labels)

	optTheta, f, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=25, m=20)

	return dict(optTheta = optTheta.reshape([numOfClasses, inputSize], order='F'),
              inputSize = inputSize,
              numOfClasses = numOfClasses)


def predict(model, data):
	theta = model['optTheta']

	pred = theta.dot(data)
	return np.argmax(pred, axis=0)
	# return (theta.dot(data)).argmax(0)
