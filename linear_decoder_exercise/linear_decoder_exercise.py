#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import stl10
from library import softmax
from library import autoencoder
from library import util

import scipy.optimize


if __name__ == '__main__':

	maxfun = 400

	imageChannels = 3
	patchDim = 8
	numPatches = 100000
	inputSize = patchDim * patchDim * imageChannels
	outputSize = inputSize
	hiddenSize = 400
	sparsityParam = 0.035
	lamb = 3e-3
	beta = 5
	epsilon = 0.1

	print 'Loading raw STL10-Sampled-Patches data...'
	patches = stl10.load_images('../data/stlSampledPatches.csv')

	# np.savez("result.npz", train_data = train_data)

	# r = np.load("result.npz")
	# patches = r["train_data"]

	# Subtract mean patch (hence zeroing the mean of the patches)
	meanPatch = np.mean(patches, axis=1)

	patches = patches - meanPatch

	# Apply ZCA whitening
	covariance = patches.dot(patches.T) / numPatches
	u, sigma, vt = np.linalg.svd(covariance)
	zcaWhite = u.dot( np.diag(1 / np.sqrt(sigma + epsilon)) ).dot(u.T);

	patches = zcaWhite.dot(patches);

	# util.display_color_network(patches[:, 0:100])

	# use sparse autoencoder (with linear decoder) to learn feature
	theta = autoencoder.initializeParameters(hiddenSize, inputSize);

	fn = lambda theta: autoencoder.sparseAutoencoderLinearCost(theta, inputSize, hiddenSize, lamb, sparsityParam, beta, patches)

	optTheta, cost, d = scipy.optimize.fmin_l_bfgs_b(fn, theta, maxfun=maxfun, iprint=1, m=20)

	W1, W2, b1, b2 = autoencoder.unflatten(optTheta, inputSize, hiddenSize)

	# TODO: display W1
	# np.savez("result.npz", W1 = W1)


