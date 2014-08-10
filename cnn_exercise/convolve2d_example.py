#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import scipy.signal
import scipy.ndimage
import pickle

if __name__ == '__main__':


	A = [[1., 1., 1., 0., 0.], [0., 1., 1., 1., 0.], [0., 0., 1., 1., 1.], [0., 0., 1., 1., 0.], [0., 1., 1., 0., 0.],]
	W = [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]

	Z = [[0., 1., 1., 0., 0.]]


	A = np.mat(A)
	W = np.mat(W)
	Z = np.mat(Z).T

	with open('entry.pickle', 'wb') as f:
		pickle.dump(A, f)


	with open('entry.pickle', 'rb') as f:
		entry2 = pickle.load(f) 

		print entry2
	# print A
	# print Z
	# print np.bmat('A Z')
	# print np.bmat('Z;Z')

	# size = np.size(W)
	# print size

	print A
	B = scipy.ndimage.zoom(A, zoom = 2, order = 0)
	print B

	C = scipy.signal.convolve(A, W, mode = 'valid')
	print C

	# D = scipy.signal.fftconvolve(np.mat(A), np.mat(W), mode = 'valid')
	# print D

	# E = scipy.ndimage.filters.convolve(A, W, mode = 'reflect')
	# print E