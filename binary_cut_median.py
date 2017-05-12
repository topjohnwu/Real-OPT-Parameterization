from numpy import sqrt, sin, cos, pi, median
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def qfunc(x):
	return 0.5 - 0.5 * special.erf(x / sqrt(2))
	
def avg(func, l, u):
	return integrate.quad(func, l, u)[0] / abs(l - u)
	
def var(func, l, u):
	return avg(lambda x: func(x) ** 2, l, u) - avg(func, l, u) ** 2
	
def err(func, l, u):
	return var(func, l, u) / ( N * (u - l) / (upper_bound - lower_bound) )

def func_wrapper(x):
	ret = obj_func(x) - g_median
	return 0 if ret < 0 else ret


# Objective functions
def xsinx(x):
	return x * sin(x)


# Parameters
obj_func = xsinx
lower_bound = 0
upper_bound = 50
N = 10
opt_loc = 0

# First round call obj_func directly
call_func = obj_func

while True:
	sample = np.empty((N, 2))

	# Sample randomly
	for i in range(N):
		sample[i][0] = random.uniform(lower_bound, upper_bound)
		sample[i][1] = call_func(sample[i][0])

	# Sort with value
	sample.view('f8,f8').sort(axis=0, order='f1')
	# print(sample)

	# Set the median parameter for next round
	g_median = sample[int(N / 2)][1]
	call_func = func_wrapper
	# print(g_median)

	# Discard half of the values
	sample = sample[int(N / 2):]
	# print(sample)

	# Sort with location
	sample.view('f8,f8').sort(axis=0, order='f0')
	print(sample)
	# Search index
	middle = (lower_bound + upper_bound) / 2
	idx = np.searchsorted(sample[:,0], middle, side='right')
	print(idx)

	left = sample[0:idx]
	right = sample[idx:]

	# TODO: Check left / right and update bounds

	
