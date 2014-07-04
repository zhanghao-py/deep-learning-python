#!/usr/bin/env python

import sys
sys.path.append('..')

import numpy as np
from library import mnist
from library import softmax
from library import autoencoder
from library import stackedae
from library import util

import scipy.optimize


if __name__ == '__main__':