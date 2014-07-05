#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
import scipy.signal
import scipy.ndimage.filters


if __name__ == '__main__':


	A = [[1., 1., 1., 0., 0.], [0., 1., 1., 1., 0.], [0., 0., 1., 1., 1.], [0., 0., 1., 1., 0.], [0., 1., 1., 0., 0.],]
	W = [[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]]

	# size = np.size(W)
	# print size

	C = scipy.signal.convolve2d(A, W, mode = 'valid')
	print C

	D = scipy.signal.fftconvolve(np.mat(A), np.mat(W), mode = 'valid')
	print D

	# E = scipy.ndimage.filters.convolve(A, W, mode = 'reflect')
	# print E