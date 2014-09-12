#!/usr/bin/env python

import sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

def load_data(path):

	dataMat = []
	fr = open(path)
	for line in fr.readlines():
		lineArr = line.strip().split('  ')

		row = []
		for element in lineArr:
			row.append(float(element))

		dataMat.append(row)

	return np.mat(dataMat)

if __name__ == '__main__':

	# Step 0: Load data
	x = load_data('pcaData.txt')
	n, numOfSamples = x.shape

	# Step 1a: Implement PCA to obtain U
	sigma = x.dot(x.T) / numOfSamples
	U, S, V = np.linalg.svd(sigma)

	plt.figure(1)
	plt.title('Raw data')
	plt.plot([0, U[0,0]], [0, U[1,0]], 'go-')
	plt.plot([0, U[0,1]], [0, U[1,1]], 'go-')
	plt.plot(x[0, :], x[1, :], c='green', marker='+')

	# Step 1b: Compute xRot, the projection on to the eigenbasis
	xRot = U.T.dot(x)

	plt.figure(2)
	plt.title('xRot')
	plt.plot(xRot[0, :], xRot[1, :], c='red', marker='+')

	# Step 2: Reduce the number of dimensions from 2 to 1
	k = 1
	xRot = U[:, 0:k].T.dot(x)
	xHat = U[:, 0:k].dot(xRot)

	plt.figure(3)
	plt.title('xHat')
	plt.plot(xHat[0, :], xHat[1, :], c='blue', marker='+')

	# Step 3a: PCA Whitening
	epsilon = 1e-5
	xPCAWhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot( U.T.dot(x) )

	plt.figure(4)
	plt.title('xPCAWhite')
	plt.plot(xPCAWhite[0, :], xPCAWhite[1, :], c='red', marker='+')

	# Step 3b: ZCA Whitening
	xZCAWhite = U.dot(xPCAWhite)
	plt.figure(5)
	plt.title('xZCAWhite')
	plt.plot(xZCAWhite[0, :], xZCAWhite[1, :], c='green', marker='+')

	# display figure
	plt.show()


