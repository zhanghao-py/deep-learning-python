# -*- coding: utf-8 -*-
# <script type="text/javascript"
#   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
# </script>

# Various helper routines for the Deep Learning exercises.

from library.imports import *

# An empty class for sticking attributes onto
class Empty(object):
  pass

# Given a function and a function input, compute the gradient at that
# input, numerically.
#
# - `fn` - the function
# - `theta` - the point at which to compute the gradient
# - `epsilon` - the step-size
# - returns: \\( \\nabla_{\\mathrm{theta}} \\mathrm{fn}(\\mathrm{theta}) \\)
def compute_numerical_gradient(fn, theta, epsilon=1e-4):
  numgrad = np.zeros(theta.shape)

  for i in range(theta.size):
    theta_minus = theta.copy()
    theta_plus = theta.copy()
    theta_minus[i] = theta_minus[i] - epsilon
    theta_plus[i] = theta_plus[i] + epsilon
    numgrad[i] = (fn(theta_plus) - fn(theta_minus)) / (2 * epsilon)

  return numgrad

# Display a network or array.
#
# - `arr` - the data, one entry per column
# - `title` - optional title
# - `show` - if true, pause and display (call `plt.show()`)
def display_network(arr, title=None, show=True):
  arr = arr - np.mean(arr)
  L, M = arr.shape
  sz = np.sqrt(L)
  buf = 1

  # Figure out pleasant grid dimensions
  if M == np.floor(np.sqrt(M))**2:
    n = m = np.sqrt(M)
  else:
    n = np.ceil(np.sqrt(M))
    while (M%n) and n < 1.2*np.sqrt(M):
      n += 1
    m = np.ceil(M/n)

  array = np.zeros([buf+m*(sz+buf), buf+n*(sz+buf)])

  k = 0
  for i in range(0, int(m)):
    for j in range(0, int(n)):
      if k>=M:
        continue
      cmax = np.max(arr[:,k])
      cmin = np.min(arr[:,k])
      r = buf+i*(sz+buf)
      c = buf+j*(sz+buf)
      array[r:r+sz, c:c+sz] = (arr[:,k].reshape([sz,sz], order='F') - cmin) / (cmax-cmin)
      k = k + 1
  plt.figure()
  if title is not None:
    plt.title(title)
  plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
  if show:
    plt.show()

def display_network_with_wh(arr, width, height, title=None, show=True):
  arr = arr - np.mean(arr)
  L, M = arr.shape
  sz = np.sqrt(L)
  buf = 1

  # Figure out pleasant grid dimensions
  if M == np.floor(np.sqrt(M))**2:
    n = m = np.sqrt(M)
  else:
    n = np.ceil(np.sqrt(M))
    while (M%n) and n < 1.2*np.sqrt(M):
      n += 1
    m = np.ceil(M/n)

  array = np.zeros([buf+m*(height+buf), buf+n*(width+buf)])

  k = 0
  for i in range(0, int(m)):
    for j in range(0, int(n)):
      if k>=M:
        continue
      cmax = np.max(arr[:,k])
      cmin = np.min(arr[:,k])
      r = buf+i*(height+buf)
      c = buf+j*(width+buf)
      array[r:r+height, c:c+width] = (arr[:,k].reshape([height, width], order='F') - cmin) / (cmax-cmin)
      k = k + 1
  plt.figure()
  if title is not None:
    plt.title(title)
  plt.imshow(array, interpolation='nearest', cmap=plt.cm.gray)
  if show:
    plt.show()

# Display a network or array
def display_color_network(arr, title=None, show=True):
  arr = arr - np.mean(arr)
  L, M = arr.shape
  L = L/3
  sz = np.sqrt(L)
  buf = 1

  # Figure out pleasant grid dimensions
  if M == np.floor(np.sqrt(M))**2:
    n = m = np.sqrt(M)
  else:
    n = np.ceil(np.sqrt(M))
    while (M%n) and n < 1.2*np.sqrt(M):
      n += 1
    m = np.ceil(M/n)

  array = np.zeros([buf+m*(sz+buf), buf+n*(sz+buf), 3])


  k = 0
  for i in range(0, int(m)):
    for j in range(0, int(n)):
      if k>=M:
        continue

      r = buf+i*(sz+buf)
      c = buf+j*(sz+buf)

      image = arr[:,k]

      rChannel = image[0:L]
      gChannel = image[L:2*L]
      bChannel = image[2*L:3*L]

      rMax = np.max(rChannel)
      rMin = np.min(rChannel)
      gMax = np.max(gChannel)
      gMin = np.min(gChannel)
      bMax = np.max(bChannel)
      bMin = np.min(bChannel)

      array[r:r+sz, c:c+sz, 0] = (rChannel.reshape([sz,sz], order='F') - rMin) / (rMax - rMin)
      array[r:r+sz, c:c+sz, 1] = (gChannel.reshape([sz,sz], order='F') - gMin) / (gMax - gMin)
      array[r:r+sz, c:c+sz, 2] = (bChannel.reshape([sz,sz], order='F') - bMin) / (bMax - bMin)

      k = k + 1
  plt.figure()
  if title is not None:
    plt.title(title)
  plt.imshow(array, interpolation='nearest')
  if show:
    plt.show()



def display_plot(rLoss, xlabel, ylabel):
  
  y = rLoss
  x = np.arange(len(y))

  plt.figure()
  
  plt.plot(x, y, linestyle = 'dashed', marker = 'r--', color = 'red')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  plt.show()
