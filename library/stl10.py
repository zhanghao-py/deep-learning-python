
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
	dataMat = [];labelMat = [];
	fr = open(path);
	for line in fr.readlines():
		lineArr = line.strip().split();
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]);
		labelMat.append(int(lineArr[2]));
	return dataMat, labelMat;