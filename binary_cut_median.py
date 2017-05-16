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
	
def func_wrapper(x):
	org = obj_func(x)
	wrap = 0 if org < g_median else org - g_median
	return (org, wrap)

def first_wrapper(x):
	val = obj_func(x)
	return (val, val)

def check_uni():
	pre = sample[0][1]
	pass_opt = False
	for i in range(1, N):
		if (pass_opt):
			if (sample[i][1] >= pre):
				return False
		else:
			if (sample[i][1] <= pre):
				pass_opt = True
		pre = sample[i][1]
	return True


# Objective functions
def xsinx(x):
	return x * sin(x)


# Parameters
obj_func = xsinx
lower_bound = 0
upper_bound = 50
N = 10000

opt_loc = 45.575

# Result
param = 0
iterations = 0

# First round call special wrapper
call_func = first_wrapper
idx = N
# [0] = location; [1] = orignal value; [2] = wrapped value
sample = np.empty((N, 3))
g_median = 0;

while True:
	# Sample randomly
	r = range(-idx, N) if idx < 0 else range(idx)
	for i in r:
		sample[i][0] = random.uniform(lower_bound, upper_bound)
		v = call_func(sample[i][0])
		sample[i][1] = v[0]
		sample[i][2] = v[1]

	# Sort with value
	sample.view('f8,f8,f8').sort(axis=0, order='f2')

	# Set the median parameter for next round
	median = sample[int(N / 2)][2]
	g_median += median
	call_func = func_wrapper

	# Adjust the sample points
	for i in range(N):
		sample[i][2] = 0 if sample[i][2] < median else sample[i][2] - median

	# Sort with location
	sample.view('f8,f8,f8').sort(axis=0, order='f0')

	if (check_uni()):
		break

	iterations += 1

	# plt.scatter(sample[:, 0], sample[:, 1])
	# plt.show()

	# Search index
	middle = (lower_bound + upper_bound) / 2
	idx = np.searchsorted(sample[:, 0], middle, side='right')

	l_m = sample[:idx, 2].mean()
	l_v = sample[:idx, 2].var(ddof=1)
	r_m = sample[idx:, 2].mean()
	r_v = sample[idx:, 2].var(ddof=1)

	# random var X = N(l_m, l_v) - N(r_m, r_v) = N(l_m - r_m, l_v + r_v)

	# Optimal on left: P(X > 0) = Q( -(l_m - r_m) / (l_v + r_v) )
	# Optimal on right: P(X < 0) = 1 - P(X > 0)

	P = qfunc( (r_m - l_m) / sqrt(l_v + r_v) )
	if (opt_loc > middle):
		P = 1 - P
		lower_bound = middle
	else:
		upper_bound = middle
		idx = -idx

	param += np.log(P)


print((iterations, param))

	
