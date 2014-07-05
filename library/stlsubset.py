
import numpy as np

def load_images(path):
	dataMat = []
	fr = open(path)
	for line in fr.readlines():
		lineArr = line.strip().split(',')

		row = []
		for element in lineArr:
			row.append(float(element))

		dataMat.append(row)

	return np.mat(dataMat)

def load_labels(path):
	labelMat = [];
	fr = open(path);
	for line in fr.readlines():
		lineArr = line.strip().split(',')

		row = []
		for element in lineArr:
			row.append(float(element))

		labelMat.append(row)

	return np.mat(labelMat)