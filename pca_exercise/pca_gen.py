#!/usr/bin/env python

import sys
sys.path.append('..')


import numpy as np
import scipy.io

from library import util
from library.imports import *

def sampleIMAGESRAW():

	mat = scipy.io.loadmat('IMAGES_RAW.mat')
	IMAGES = mat['IMAGESr']

	patchSize = 12
	numOfPatches = 10000

	width, height, size = IMAGES.shape

	patches = np.zeros([patchSize*patchSize, numOfPatches])

	p = 0
	for idx in xrange(size):

		numOfSamples = numOfPatches / size

		for s in xrange(numOfSamples):

			y = np.random.randint(width - patchSize + 1)
			x = np.random.randint(height - patchSize + 1)
			sample = IMAGES[y:y+patchSize, x:x+patchSize, idx]
			patches[:, p] = np.array(sample).flatten()
			p = p + 1

	return patches

if __name__ == '__main__':

	# Step 0a: Load data
	x = sampleIMAGESRAW()

	inputSize, numOfSamples = x.shape
	randsel = np.random.randint(0, numOfSamples, 200)
	util.display_network(x[:, randsel], 'Raw data')

	# Step 0b: Zero-mean the data (by column)
	avg = np.mean(x, axis = 0)
	x = x - avg

	# Step 1a: Implement PCA to obtain xRot
	sigma = x.dot(x.T) / numOfSamples
	U, S, V = np.linalg.svd(sigma)
	xRot = U.T.dot(x)

	# Step 1b: Check your implementation of PCA
	covar = xRot.dot(xRot.T) / numOfSamples

	plt.figure()
	plt.imshow(covar)
	plt.title('Visualization of covariance matrix')
	plt.show()

	# Step 2: Find k, the number of components to retain
	k = S[(np.cumsum(S) / np.sum(S)) < 0.99].size

	# Step 3: Implement PCA with dimension reduction
	xRot = U[:, 0:k].T.dot(x)
	xHat = U[:, 0:k].dot(xRot)
	util.display_network(xHat[:,randsel], 'PCA processed images ' + bytes(k) + ' dimensions')

	# Step 4a: Implement PCA with whitening and regularisation
	epsilon = 0.1
	xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot( U.T.dot(x) )

	# Step 4b: Check your implementation of PCA whitening

	# Step 5: Implement ZCA whitening
	xZCAWhite = U.dot(xPCAWhite)
	util.display_network(xZCAWhite[:,randsel], 'ZCA whitened images')




