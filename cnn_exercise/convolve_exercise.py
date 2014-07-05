#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np

from library import stlsubset
from library import util
from library import cnn


if __name__ == '__main__':

	patchDim = 8;
	hiddenSize = 400;
	poolDim = 19;


	r = np.load("../linear_decoder_exercise/result.npz")
	W = r["W1"]
	b = r["b1"]
	zcaWhite = r["zcaWhite"]
	meanPatch = r["meanPatch"]

	# util.display_color_network( W.dot(zcaWhite).T )

	print 'Loading raw STLTrainSubset data...'
	# trainImages = stlsubset.load_images('../data/stlTrainSubset_images.csv')
	# np.savez('trainImages.npz', trainImages = trainImages)
	r = np.load("trainImages.npz")
	trainImages = r["trainImages"]
	print 'Loaded raw STLTrainSubset data.'

	convImages = trainImages[:, 0:8]

	# Convolve Features
	print 'Convolving Features...'
	convolvedFeatures = cnn.convolve(patchDim, hiddenSize, convImages, W, b, zcaWhite, meanPatch)
	print convolvedFeatures.shape

	print 'Pooling Features...'
	pooledFeatures = cnn.meanPooling(poolDim, convolvedFeatures);
	print pooledFeatures.shape

