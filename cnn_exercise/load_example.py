#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np

from library import stlsubset

if __name__ == '__main__':


	print 'Loading raw STLTrainSubset data...'
	trainImages = stlsubset.load_images('../data/stlTrainSubset_images.csv')
	trainLabels = stlsubset.load_labels('../data/stlTrainSubset_labels.csv')
	np.savez('trainImages.npz', trainImages = trainImages, trainLabels = trainLabels)


	print 'Loading raw STLTestSubset data...'
	testImages = stlsubset.load_images('../data/stlTestSubset_images.csv')
	testLabels = stlsubset.load_labels('../data/stlTestSubset_labels.csv')
	np.savez('testImages.npz', testImages = testImages, testLabels = testLabels)
	# r = np.load("trainImages.npz")
	# trainImages = r["trainImages"]