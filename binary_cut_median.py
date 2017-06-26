#!/usr/bin/env python3

from numpy import sqrt, sin, cos, pi, median
import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optproblems.cec2005 import *

def qfunc(x):
	return 0.5 - 0.5 * special.erf(x / sqrt(2))

def func_wrapper(x):
	org = obj_func(x)
	wrap = 0 if org < g_median else org - g_median
	return [org, wrap]

def first_wrapper(x):
	val = obj_func(x)
	return [val, val]

def terminate():
	# TODO!!!
	return

# Objective functions
def xsinx(x):
	return x[0] * sin(x[0] + x[1])

# The dimention of the input vector
dimen = 2
# choose a cec2005 problem
cec_probs = F14(dimen)
# Objective function: Any function that accepts a vector, returns a real value
# or: Assign a R^n -> R function
obj_func = lambda x: -cec_probs(x)  # cec2005 problems requires to be inverted
# The bounds of the function
bound = [ [-100, 100], [-100, 100] ]
# Sample number
N = 10000
# The position of the optimal value
opt_loc = cec_probs.get_optimal_solutions()[0].phenome  # use included opt position from cec2005

# Asserts, check dimensions
assert len(bound) == dimen
assert len(opt_loc) == dimen

# Results
param = 0
iterations = 0

# Intermediate values, don't edit
call_func = first_wrapper
g_median = 0;

# Print opt position
print('optimal: ' + str(opt_loc))

while True:
	# Sample

	# Data structure of sample:
	# 0: position, vector with size == dimen
	# 1: value of function in position, val[0] = raw value, used for terminate evaluation
	#                                   val[1] = wrapped value, used for actual calculation
	sample = []
	for _ in range(N):
		pos = []
		# Random generate position
		for i in range(dimen):
			pos.append(random.uniform(bound[i][0], bound[i][1]))
		sample.append([pos, call_func(pos)])

	# Update call function
	call_func = func_wrapper

	# Check termination
	if (terminate()):
		break

	# Sort by wrapped value
	sample.sort(key = lambda point : point[1][1])

	# Get median and remove half
	g_median = sample[int(N / 2)][1][1]
	sample = sample[int(N / 2):]
	for s in sample:
		s[1][1] -= g_median

	# Get variance of each axis
	axis_var = []
	for i in range(dimen):
		axis_var.append(np.var([pos[0][i] for pos in sample]))

	# Get the target axis
	target_axix, val = max(enumerate(axis_var), key=lambda p: p[1])
	axix_split = (bound[target_axix][0] + bound[target_axix][1]) / 2

	left = []
	right = []
	for point in sample:
		left.append(point[1][1]) if point[0][target_axix] < axix_split else right.append(point[1][1])

	l_m = np.mean(left)
	l_v = np.var(left, ddof=1)
	r_m = np.mean(right)
	r_v = np.var(right, ddof=1)

	# random var X = N(l_m, l_v) - N(r_m, r_v) = N(l_m - r_m, l_v + r_v)
	# Optimal on left: P(X > 0) = Q( -(l_m - r_m) / (l_v + r_v) )
	# Optimal on right: P(X < 0) = 1 - P(X > 0)

	P = qfunc( (r_m - l_m) / sqrt(l_v + r_v) )
	if (opt_loc[target_axix] > axix_split):
		P = 1 - P
		bound[target_axix][0] = axix_split
	else:
		bound[target_axix][1] = axix_split

	param += np.log(P)
	iterations += 1

	print('[param, iter] = ' + str([param, iterations]))

	command = input("Press p to plot, q to quit, enter to continue\n")

	if command == 'p':
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		X = [point[0][0] for point in sample]
		Y = [point[0][1] for point in sample]
		Z = [point[1][1] for point in sample]
		ax.scatter(X, Y, Z, s=1)
		plt.show()
	elif command == 'q':
		exit()


