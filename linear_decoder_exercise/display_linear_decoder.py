#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import util

import scipy.optimize


if __name__ == '__main__':

	r = np.load("result.npz")
	W1 = r["W1"]
	b1 = r["b1"]
	zcaWhite = r["zcaWhite"]
	meanPatch = r["meanPatch"]

	# print W1.T.shape

	util.display_color_network( W1.T )

	util.display_color_network( W1.dot(zcaWhite).T )


