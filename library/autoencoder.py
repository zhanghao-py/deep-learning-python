#!/usr/bin/env python

import numpy as np

# Flatten W1, W2, b1, b2 into a row vector
def flatten(W1, W2, b1, b2):
	return np.array(np.hstack([W1.ravel('F'), W2.ravel('F'), b1.ravel('F'), b2.ravel('F')]), order='F')

# Expand a row vector back into W1, W2, b1, b2
def unflatten(theta, visible_size, hidden_size):
	hv = hidden_size * visible_size
	W1 = theta[0:hv].reshape([hidden_size, visible_size], order='F')
	W2 = theta[hv:2*hv].reshape([visible_size, hidden_size], order='F')
	b1 = theta[2*hv:2*hv+hidden_size].reshape([hidden_size, 1], order='F')
	b2 = theta[2*hv+hidden_size:].reshape([visible_size, 1], order='F')
	return (W1, W2, b1, b2)


def initializeParameters(hiddenSize, visibleSize):
	r = np.sqrt(6) / np.sqrt(hiddenSize + visibleSize + 1)
	W1 = np.random.random([hiddenSize, visibleSize]) * 2 * r - r;
	W2 = np.random.random([visibleSize, hiddenSize]) * 2 * r - r;
	b1 = np.zeros([hiddenSize, 1])
	b2 = np.zeros([visibleSize, 1])
	return flatten(W1, W2, b1, b2)


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def kl(rho, rhohats):
	return np.sum( rho * np.log(rho / rhohats) + (1-rho) * np.log((1-rho) / (1-rhohats)) )

def kl_delta(rho, rhohats):
	return -(rho / rhohats) + (1-rho) / (1-rhohats);

def feedforwardAutoencoder(theta, hiddenSize, visibleSize, data):
	W1, W2, b1, b2 = unflatten(theta, visibleSize, hiddenSize)
	return sigmoid(W1.dot(data) + b1)

def sparseAutoencoderCost(theta, visibleSize, hiddenSize, lamb, sparsityParam, beta, data):
	m = data.shape[1]
	rho = sparsityParam

	W1, W2, b1, b2 = unflatten(theta, visibleSize, hiddenSize)

	z2 = W1.dot(data) + b1
	a2 = sigmoid(z2)
	z3 = W2.dot(a2) + b2
	a3 = sigmoid(z3)

	squares = np.power(a3 - data, 2)
	squaredError = 0.5 * (1.0/m) * np.sum(squares)

	weightDecay = (lamb/2.0) * ( np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)) )

	rhohats = np.mean(a2, 1)

	sparsityPenalty = beta * kl(rho, rhohats)

	cost = squaredError + weightDecay + sparsityPenalty

	delta3 = - np.multiply( np.multiply((data - a3), a3), (1-a3) )

	betaTerm = beta * kl_delta(rho, rhohats)
	betaTerm = betaTerm.reshape([hiddenSize, 1])

	delta2 = np.multiply( (W2.T.dot(delta3) + betaTerm), np.multiply(a2, (1-a2)) )

	W2grad = (1.0/m) * delta3.dot(a2.T) + lamb * W2
	b2grad = (1.0/m) * np.sum(delta3, 1)
	W1grad = (1.0/m) * delta2.dot(data.T) + lamb * W1
	b1grad = (1.0/m) * np.sum(delta2, 1)

	grad = flatten(W1grad, W2grad, b1grad, b2grad)

	return cost, grad

def sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lamb, sparsityParam, beta, data):
	m = data.shape[1]
	rho = sparsityParam

	W1, W2, b1, b2 = unflatten(theta, visibleSize, hiddenSize)

	z2 = W1.dot(data) + b1
	a2 = sigmoid(z2)
	z3 = W2.dot(a2) + b2
	a3 = z3

	squares = np.power(a3 - data, 2)
	squaredError = 0.5 * (1.0/m) * np.sum(squares)

	weightDecay = (lamb/2.0) * ( np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)) )

	rhohats = np.mean(a2, 1)[:, np.newaxis]

	sparsityPenalty = beta * kl(rho, rhohats)

	cost = squaredError + weightDecay + sparsityPenalty

	delta3 = -(data - a3)
	betaTerm = beta * kl_delta(rho, rhohats)
	delta2 = (W2.T.dot(delta3) + betaTerm) * a2 * (1-a2)

	W2grad = (1.0/m) * delta3.dot(a2.T) + lamb * W2
	b2grad = (1.0/m) * np.sum(delta3, 1)[:, np.newaxis]
	W1grad = (1.0/m) * delta2.dot(data.T) + lamb * W1
	b1grad = (1.0/m) * np.sum(delta2, 1)[:, np.newaxis]

	grad = flatten(W1grad, W2grad, b1grad, b2grad)

	return cost, grad



