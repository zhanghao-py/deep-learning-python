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
	W1 = r["W1"]

	print W1.T.shape
	util.display_color_network(W1.T)