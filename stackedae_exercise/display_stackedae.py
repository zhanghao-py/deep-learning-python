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

	r = np.load("result.npz")
	W11 = r["W11"]

	print W11.shape
	util.display_network(W11.T)

	# print W12.shape
	# util.display_network(W12.T)