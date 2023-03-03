#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from typing import List, Dict, Iterable, Tuple, Callable
import math
import os
import random
import sys
import numpy as np
import pandas as pd
import tqdm
from functools import partial, reduce
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


# In[2]:


# Functions

# def add(a, b): return a + b

Vector = List[float]

Tensor = list

def add(a, b): return a + b

def vector_sum(vectors):
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def scalar_multiply(c , v):
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """Computes the element-wise average"""
    n = len(vectors)
    m = np.sum(vectors,axis=0)
    vec_mean = np.multiply(1/n,m)
    return vec_mean

def de_mean(xs):
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = np.mean(xs)
    d_mean = [x - x_bar for x in xs]
    return d_mean

def dot(v, w):
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def variance(xs):
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    vari = sum_of_squares(deviations)/(n-1)
    return vari

# Standard deviation                        
def standard_deviation(xs):
    """The standard deviation is the square root of the variance"""
    std_dev = np.sqrt(variance(xs)) 
    return std_dev

def scale(data):
    """returns the mean and standard deviation for each position"""
    dim = len(data[0])
    
    # Vector Mean
#     n = len(data)
#     m = np.sum(data,axis=0)
#     means = np.multiply(1/n,m)
    means = vector_mean(data)
    
    # Standard Deviaiton
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means,stdevs

def rescale(data):
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)
    
    means = list(means)
    stdevs = list(stdevs)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]
    v0 = []
    for v in rescaled:
        v = list(v)
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
        v0.append(v)

    return v0

def gradient_step(v, gradient, step_size):
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    grad_step = np.add(v,step)
    return grad_step

# def predict(alpha, beta, x_i):
#     pred = beta * x_i + alpha
#     return pred

# def error(x, y, beta):
#     """
#     The error from predicting beta * x_i + alpha
#     when the actual value is y_i
#     """
#     err_fin = predict(alpha, beta, x_i) - y_i
#     return err_fin

def predict(x, beta):
    """assumes that the first element of x is 1"""
    return dot(x, beta)

def error(x, y, beta):
    return predict(x, beta) - y 

def sqerror_gradient(x, y, beta):
    err = error(x, y, beta)
    err_fin = [2 * err * x_i for x_i in x]
    return err_fin

def least_squares_fit(xs, ys, learning_rate = 0.001, num_steps = 1000, batch_size = 1):
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    # Start with a random guess
    guess = [np.random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def logistic_prime(x):
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x, y, beta):
    """The negative log likelihood for one data point""" 
    if y == 1:
        return -np.log(logistic(np.dot(x, beta)))
    else:
        return -np.log(1 - logistic(np.dot(x, beta)))
    
def negative_log_likelihood(xs, ys, beta):
    return sum(_negative_log_likelihood(x, y, beta)
               for x, y in zip(xs, ys))

def _negative_log_partial_j(x, y, beta, j):
    """
    The jth partial derivative for one data point.
    Here i is the index of the data point.
    """
    return -(y - logistic(np.dot(x, beta))) * x[j]

def _negative_log_gradient(x, y, beta):
    """
    The gradient for one data point.
    """
    return [_negative_log_partial_j(x, y, beta, j)
            for j in range(len(beta))]

def negative_log_gradient(xs, ys,beta):
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])

def split_data(data, prob):
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

def train_test_split(xs, ys, test_pct):
     # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test
                                                                
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                           # set theta to initial value
    target_fn = safe(target_fn)               # safe version of target_fn
    value = target_fn(theta)                  # value we're minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value
    
    
def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


def step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01,tolerance=0.000001):
    data = list(zip(x, y))
    
    # initial guess
    theta = theta_0
    
    # initial step size
    alpha = alpha_0
    
    # the minimum so far
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(negate(target_fn),
                          negate_all(gradient_fn),
                          x,
                          y,
                          theta_0,
                          alpha_0 = 0.01)
def	in_random_order(data):
				"""generator	that	returns	the	elements	of	data	in	random	order"""
				indexes	=	[i	for	i,	_	in	enumerate(data)]		#	create	a	list	of	indexes
				random.shuffle(indexes)																				#	shuffle	them
				for	i	in	indexes:																										#	return	the	data	in	that	order
								yield	data[i]
                        
def minimize_stochastic2(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01,tolerance=0.000001):
    data = list(zip(x, y))
    
    # initial guess
    theta = theta_0
    
    # initial step size
    alpha = alpha_0
    
    # the minimum so far
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
        # stop if we're "converging"
        next_value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if abs(value - next_value) < tolerance:
            break

    return min_theta,iterations_with_no_improvement

def maximize_stochastic2(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01,tolerance=0.000001):
    return minimize_stochastic2(negate(target_fn),
                          negate_all(gradient_fn),
                          x,
                          y,
                          theta_0,
                          alpha_0 = 0.01,tolerance=0.000001)                       
                        


# In[3]:


# Functions

def add(a, b): return a + b

def vector_sum(vectors):
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

def scalar_multiply(c , v):
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """Computes the element-wise average"""
    n = len(vectors)
    m = np.sum(vectors,axis=0)
    vec_mean = np.multiply(1/n,m)
    return vec_mean

def de_mean(xs):
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = np.mean(xs)
    d_mean = [x - x_bar for x in xs]
    return d_mean

def dot(v, w):
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def variance(xs):
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    vari = sum_of_squares(deviations)/(n-1)
    return vari

# Standard deviation                        
def standard_deviation(xs):
    """The standard deviation is the square root of the variance"""
    std_dev = np.sqrt(variance(xs)) 
    return std_dev

def scale(data):
    """returns the mean and standard deviation for each position"""
    dim = len(data[0])
    
    # Vector Mean
#     n = len(data)
#     m = np.sum(data,axis=0)
#     means = np.multiply(1/n,m)
    means = vector_mean(data)
    
    # Standard Deviaiton
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]
    return means,stdevs

def rescale(data):
    """
    Rescales the input data so that each position has
    mean 0 and standard deviation 1. (Leaves a position
    as is if its standard deviation is 0.)
    """
    dim = len(data[0])
    means, stdevs = scale(data)
    
    means = list(means)
    stdevs = list(stdevs)

    # Make a copy of each vector
    rescaled = [v[:] for v in data]
    v0 = []
    for v in rescaled:
        v = list(v)
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
        v0.append(v)

    return v0

def gradient_step(v, gradient, step_size):
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    grad_step = np.add(v,step)
    return grad_step

# def predict(alpha, beta, x_i):
#     pred = beta * x_i + alpha
#     return pred

# def error(x, y, beta):
#     """
#     The error from predicting beta * x_i + alpha
#     when the actual value is y_i
#     """
#     err_fin = predict(alpha, beta, x_i) - y_i
#     return err_fin

def predict(x, beta):
    """assumes that the first element of x is 1"""
    return dot(x, beta)

def error(x, y, beta):
    return predict(x, beta) - y 

def sqerror_gradient(x, y, beta):
    err = error(x, y, beta)
    err_fin = [2 * err * x_i for x_i in x]
    return err_fin

def least_squares_fit(xs, ys, learning_rate = 0.001, num_steps = 1000, batch_size = 1):
    """
    Find the beta that minimizes the sum of squared errors
    assuming the model y = dot(x, beta).
    """
    # Start with a random guess
    guess = [np.random.random() for _ in xs[0]]

    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def logistic_prime(x):
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x, y, beta):
    """The negative log likelihood for one data point""" 
    if y == 1:
        return -np.log(logistic(dot(x, beta)))
    else:
        return -np.log(1 - logistic(dot(x, beta)))
    
def negative_log_likelihood(xs, ys, beta):
    return sum(_negative_log_likelihood(x, y, beta)
               for x, y in zip(xs, ys))

def _negative_log_partial_j(x, y, beta, j):
    """
    The jth partial derivative for one data point.
    Here i is the index of the data point.
    """
    return -(y - logistic(dot(x, beta))) * x[j]

def _negative_log_gradient(x, y, beta):
    """
    The gradient for one data point.
    """
    return [_negative_log_partial_j(x, y, beta, j)
            for j in range(len(beta))]

def negative_log_gradient(xs, ys,beta):
    return vector_sum([_negative_log_gradient(x, y, beta)
                       for x, y in zip(xs, ys)])

def split_data(data, prob):
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

def train_test_split(xs, ys, test_pct):
     # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # this means "infinity" in Python
    return safe_f
                                                                
def negate(f):
    """return a function that for any input x returns -f(x)"""
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def minimize_batch(target_fn, gradient_fn, x,y, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""
    
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(x,y,theta)
    values = []
    
    # set theta to initial value
    # safe version of target_fn
    # value we're minimizing
    while True:
        values.append(value)
        gradient = gradient_fn(x,y, theta)
        next_thetas = [np.array(step(theta, gradient, -step_size))
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(x,y,next_theta)

        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            values.append(next_value)
            break
        else:
            theta, value = next_theta, next_value

    return theta, values
    
    
def maximize_batch(target_fn, gradient_fn, x, y, theta_0, tolerance=0.000001):
    return minimize_batch(target_fn,
                          gradient_fn,
                          x,
                          y,
                          theta_0,
                          tolerance)

import re, math, random # regexes, math functions, random numbers
import matplotlib.pyplot as plt # pyplot
from collections import defaultdict, Counter
from functools import partial, reduce

#
# functions for working with vectors
#

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def magnitude(v):
    return np.sqrt(sum_of_squares(v))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def distance(v, w):
   return np.sqrt(squared_distance(v, w))

#
# functions for working with matrices
#

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A, i):
    return A[i]

def get_column(A, j):
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)]

def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)

#          user 0  1  2  3  4  5  6  7  8  9
#
friendships = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # user 0
               [1, 0, 1, 1, 0, 0, 0, 0, 0, 0], # user 1
               [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], # user 2
               [0, 1, 1, 0, 1, 0, 0, 0, 0, 0], # user 3
               [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], # user 4
               [0, 0, 0, 0, 1, 0, 1, 1, 0, 0], # user 5
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 6
               [0, 0, 0, 0, 0, 1, 0, 0, 1, 0], # user 7
               [0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # user 8
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] # user 9

#####
# DELETE DOWN
#


def matrix_add(A, B):
    if shape(A) != shape(B):
        raise ArithmeticError("cannot add matrices with different shapes")

    num_rows, num_cols = shape(A)
    def entry_fn(i, j): return A[i][j] + B[i][j]

    return make_matrix(num_rows, num_cols, entry_fn)


def make_graph_dot_product_as_vector_projection(plt):

    v = [2, 1]
    w = [np.sqrt(.25), np.sqrt(.75)]
    c = dot(v, w)
    vonw = scalar_multiply(c, w)
    o = [0,0]

    plt.arrow(0, 0, v[0], v[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("v", v, xytext=[v[0] + 0.1, v[1]])
    plt.arrow(0 ,0, w[0], w[1],
              width=0.002, head_width=.1, length_includes_head=True)
    plt.annotate("w", w, xytext=[w[0] - 0.1, w[1]])
    plt.arrow(0, 0, vonw[0], vonw[1], length_includes_head=True)
    plt.annotate(u"(v•w)w", vonw, xytext=[vonw[0] - 0.1, vonw[1] + 0.1])
    plt.arrow(v[0], v[1], vonw[0] - v[0], vonw[1] - v[1],
              linestyle='dotted', length_includes_head=True)
    plt.scatter(*zip(v,w,o),marker='.')
    plt.axis('equal')
    plt.show()
    
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


# In[4]:


def logistic(x):
    return 1.0 / (1 + np.exp(-x))


# In[5]:


z = [zi/50 - 10 for zi in range(1000)]
plt.plot(z, [logistic(zi) for zi in z])
plt.title('logistic function')
plt.show()


# In[6]:


def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i==1:
        return np.log(logistic(dot(x_i, beta)))
    else:
        return np.log(1 - logistic(dot(x_i, beta)))


# In[7]:


def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
        for x_i, y_i in zip(x, y))


# In[8]:


def logistic_log_partial_ij(x_i, y_i, beta, j):
    """here i is the index of the data point, j the index of the derivative"""
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]


# In[9]:


def logistic_log_gradient_i(x_i, y_i, beta):
    """the gradient of the log likelihood corresponding to the ith data point"""
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j,_ in enumerate(beta)]


# In[10]:


def logistic_log_gradient(x, y, beta):
    return reduce(vector_add,
        [logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x,y)])


# In[11]:


np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
X1 = np.insert(X, 0, 1, axis=1)

rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(rescaled_x)


# In[12]:


random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

# want to maximize log likelihood on the training data
fn = partial(logistic_log_likelihood, x_train, y_train)
gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# pick a random starting point
beta_0 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
beta_hat, training_errs = maximize_batch(logistic_log_likelihood, logistic_log_gradient,x_train,y_train, beta_0)

beta_hat
gradient_fn
beta_hat


# In[13]:


def optimize2(x, y,learning_rate,iterations): 
    size = x.shape[0]
#     weight = parameters["weight"] 
#     bias = parameters["bias"]
    w1 = []
#     wg = np.ones(x.shape[1])
    
    for i in np.unique(y):
        y_copy = [1 if c == i else 0 for c in y]
        wg = np.ones(x.shape[1])

        for _ in range(iterations): 
            sigma = sigmoid(np.dot(x, wg))
#             loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
            dW = 1/size * np.dot(x.T, (y_copy - sigma))
#             db = 1/size * np.sum(sigma - y)
            wg += learning_rate * dW
#             bias -= learning_rate * db 
        print(wg)
        w1.append((wg,i))
#     parameters["weight"] = w1
#     parameters["bias"] = bias
    return w1

# Define the train function
# def train(x, y, learning_rate,iterations):
#     parameters_out = optimize2(x, y, learning_rate, iterations ,init_parameters)
#     return parameters_out


# In[14]:


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def predictOne(x,w):
    return max((x.dot(w), c) for w, c in w)[1]

def predict(X,w):
    return np.array([predictOne(i,w) for i in X])

def predict2(X,w):
    output = np.insert(X, 0, 1, axis=1).dot(w)
    return (np.floor(sigmoid(output) + .5)).astype(int)

def score(X, y, w):
    return sum(predict(X,w) == y) / len(y)


# In[15]:


# Initialize the weight and bais
init_parameters = {} 
# init_parameters["weight"] = np.zeros((X1.shape[1],1))
init_parameters["weight"] = np.zeros((X.shape[1],2))
init_parameters["weight"]
init_parameters["bias"] = 0
# init_parameters['weight']


# In[16]:


rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(rescaled_x)
# print(x)
parameters_out = optimize2(x, y, learning_rate = 0.01, iterations = 1000)
parameters_out


# In[17]:


X1 = np.insert(X, 0, 1, axis=1)
# X1 = X
wg = []

for i in np.unique(y):
    y_copy = [1 if c == i else 0 for c in y]
    w = np.ones(X1.shape[1])
    print(w.shape)
    eta = 0.001
    size = X1.shape[0]

    # print('training ', i)
    # counter = 0

    for _ in range(1000):
        output = X1.dot(w)
        errors = y_copy - sigmoid(output)
        w += eta * 1/size * errors.T.dot(X1)
#         diff = -eta * 1/size * errors.T.dot(X1)
        
#         if np.all(np.abs(diff) <= 1e-06):
#             break
#         w += diff

        # counter += 1
        # if counter // 10 == 0:
        #     print(sum(errors**2) / 2.0)
    wg.append((w, i))


# In[18]:


wg


# In[19]:


wg_0 = wg[0][0]
wg_0 = wg_0.reshape(len(wg_0),1)
wg_0.shape
# wg_0


# In[20]:


wg_1 = wg[1][0]
wg_1 = wg_1.reshape(len(wg_1),1)
wg_1.shape
# wg_1


# In[21]:


rescaled_x = rescale(X1)
rescaled_x = np.array(rescaled_x)
score(X1,y,wg)


# In[22]:


random.seed(0)
rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(X1)
# print(x)
parameters_out = optimize2(x, y, learning_rate = 0.01, iterations = 1000)
parameters_out


# In[23]:


# # Y = TypeVar('Y')  # generic type to represent output variables

# # def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
# #     """Split data into fractions [prob, 1 - prob]"""
# #     data = data[:]                    # Make a shallow copy
# #     random.shuffle(data)              # because shuffle modifies the list.
# #     cut = int(len(data) * prob)       # Use prob to find a cutoff
# #     return data[:cut], data[cut:]     # and split the shuffled list there.

# def	split_data(data,	prob):
# 				"""split	data	into	fractions	[prob,	1	-	prob]"""
# 				results	=	[],	[]
# 				for	row	in	data:
# 								results[0	if	random.random()	<	prob	else	1].append(row)
# 				return	results

# # def train_test_split(xs,ys,test_pct):                                                      

# #     # Generate the indices and split them
# #     idxs = [i for i in range(len(xs))]
# #     train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

# #     return ([xs[i] for i in train_idxs],  # x_train 
# #             [xs[i] for i in test_idxs],   # x_test
# #             [ys[i] for i in train_idxs],  # y_train
# #             [ys[i] for i in test_idxs])   # y_test

# def	train_test_split(x,	y,	test_pct):
#     data	=	zip(x,	y)																														#	pair	corresponding	values
#     train,	test	=	split_data(data,	1	-	test_pct)		#	split	the	data	set	of	pairs
#     x_train,	y_train	=	zip(*train)																#	magical	un-zip	trick
#     x_test,	y_test	=	zip(*test)
#     return x_train, x_test, y_train, y_test


# In[24]:


random.seed(0)
rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(X1)
# print(x)
parameters_out = optimize2(x, y, learning_rate = 0.01, iterations = 1000)
parameters_out


# In[25]:


random.seed(0)
# rescaled_x = rescale(X1)

# X_train,X_test,y_train,y_test=train_test_split(X1,breasts.target,0.4)


# In[26]:


# # beta_0 = parameters_out['weight']
# beta_0 = parameters_out[0][0]
# # print(beta_0)
# X_train = np.array(X_train)
# y_train = np.array(y_train)


# In[27]:


def train_test_split(xs, ys, test_pct):
     # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test
                                                                
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """use gradient descent to find theta that minimizes target function"""

    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0                           # set theta to initial value
    target_fn = safe(target_fn)               # safe version of target_fn
    value = target_fn(theta)                  # value we're minimizing

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size)
                       for step_size in step_sizes]

        # choose the one that minimizes the error function
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        # stop if we're "converging"
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value
    
    
def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                          negate_all(gradient_fn),
                          theta_0,
                          tolerance)


# In[28]:


# random.seed(0)
# x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

# # want to maximize log likelihood on the training data
# fn = partial(logistic_log_likelihood, x_train, y_train)
# gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# # pick a random starting point
# # beta_0 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# beta_0 = parameters_out[0][0]
# beta_hat, training_errs = maximize_batch(logistic_log_likelihood, logistic_log_gradient,x_train,y_train, beta_0)

# beta_hat
# gradient_fn
# beta_hat


# In[29]:


import tqdm

random.seed(0)

x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.4)
print(len(y_train))
print(len(x_train))

learning_rate = 0.0000001

# beta_00 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# # print(beta_00)

beta_00 = parameters_out[0][0]

with tqdm.trange(300) as t:
    for epoch in t:
        gradient = negative_log_gradient(x_train, y_train, beta_00)
        beta_00 = gradient_step(beta_00, gradient, -learning_rate)
        loss = negative_log_likelihood(x_train, y_train, beta_00)
        print(sum(gradient))
        t.set_description(f"loss: {loss:.3f} gradient: {gradient}")
        
# print(beta_00)


# In[30]:


import tqdm

random.seed(0)

# x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)
# print(len(y_train))
# print(len(x_train))

learning_rate = 0.0000001

# beta_00 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# # print(beta_00)

beta_1 = parameters_out[1][0]

with tqdm.trange(300) as t:
    for epoch in t:
        gradient = negative_log_gradient(x_train, y_train, beta_1)
        beta_1 = gradient_step(beta_1, gradient, -learning_rate)
        loss = negative_log_likelihood(x_train, y_train, beta_1)
        print(sum(gradient))
        t.set_description(f"loss: {loss:.3f} gradient: {gradient}")
        
# print(beta_1)


# In[31]:


# random.seed(0)
# x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)
# print(len(y_train))
# print(len(x_train))

# # want to maximize log likelihood on the training data
# fn = partial(logistic_log_likelihood, x_train, y_train)
# gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# beta_00 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# print(len(beta_00))

# beta_0 = parameters_out[0][0]
# print(len(beta_0))

# # beta_0 = (beta_0 - np.mean(beta_0, axis = 0))/np.std(beta_0, axis = 0)
# beta_0_1 = []
# min_beta = np.min(beta_0, axis=0)
# max_beta = np.max(beta_0, axis=0)
# mean_beta = np.abs(np.mean(beta_0))
# for i in beta_0:
# #     print(i)
#     beta_0 = (i - min_beta)/(max_beta- min_beta)
#     if beta_0 == 0:
#         beta_0 = mean_beta
#     if beta_0 == 1:
#         beta_0 = mean_beta
#     beta_0_1.append(beta_0)
# print(len(beta_0_1))

# # pick a random starting point
# # beta_0 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# beta_00 = beta_0_1
# beta_hat = maximize_batch(fn, gradient_fn,beta_00)

# beta_hat
# gradient_fn
# beta_hat


# In[32]:


# random.seed(0)
# rescaled_x = rescale(X1)
# x_train, x_test, y_train, y_test = train_test_split(X1, y, 0.4)
# # print(len(y_train))

# # want to maximize log likelihood on the training data
# # fn = logistic_log_likelihood(x_train, y_train, beta_00)
# # gradient_fn = logistic_log_gradient(x_train, y_train, beta_00)

# # pick a random starting point
# beta_00 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# print(beta_00)
# beta_0 = parameters_out[0][0]
# print(list(beta_0))

# # beta_0 = (beta_0 - np.mean(beta_0, axis = 0))/np.std(beta_0, axis = 0)
# beta_0_1 = []
# min_beta = np.min(beta_0, axis=0)
# max_beta = np.max(beta_0, axis=0)
# mean_beta = np.abs(np.mean(beta_0))
# for i in beta_0:
# #     print(i)
#     beta_0 = (i - min_beta)/(max_beta- min_beta)
#     if beta_0 == 0:
#         beta_0 = mean_beta
#     if beta_0 == 1:
#         beta_0 = mean_beta
#     beta_0_1.append(beta_0)
# print(beta_0_1)
# # print(list(fn))
# beta_hat_0, training_errs = maximize_batch(logistic_log_likelihood, logistic_log_gradient, x_train, y_train, beta_0_1, tolerance = 0.000001)

# # beta_hat
# # gradient_fn
# beta_hat_0


# In[33]:


# random.seed(0)
# x_train, x_test, y_train, y_test = train_test_split(x, y, 0.33)
# x_train = np.array(x_train)
# m,n = x_train.shape

# # want to maximize log likelihood on the training data
# fn = partial(logistic_log_likelihood, x_train, y_train)
# gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# # pick a random starting point
# beta_0 = [random.random() for _ in range(n)] # and maximize using gradient descent
# beta_hat, training_errs = maximize_batch(fn, gradient_fn, beta_0)

# # beta_hat
# # gradient_fn
# # beta_hat


# In[34]:


# random.seed(0)
# x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)

# # want to maximize log likelihood on the training data
# fn = partial(logistic_log_likelihood, x_train, y_train)
# gradient_fn = partial(logistic_log_gradient, x_train, y_train)

# # pick a random starting point
# beta_0 = [random.random() for _ in range(len(x_train[0]))] # and maximize using gradient descent
# beta_hat, training_errs = maximize_batch(fn, gradient_fn, beta_0)

# beta_hat
# gradient_fn
# beta_hat


# In[35]:


beta_hat_1 = beta_1
wg_1_lst = []
for i in beta_hat_1:
    wg_1_lst.append(i)
wg_1_lst

wg_1 = np.array([wg_1_lst,wg[0][1]])
wg_1


# In[36]:


beta_hat_0 = beta_00
wg_0_lst = []
for i in beta_hat_0:
    wg_0_lst.append(i)
wg_0_lst

wg_0 = np.array([wg_0_lst,wg[1][1]])
wg_0


# In[37]:


wg_lst = np.array([wg_1,wg_0])


# In[38]:


wg_lst_v0 = np.array(wg_lst[0][0])
wg_lst_0 = [wg_lst_v0,wg_lst[0][1]]
# wg_lst_0 = np.array(wg_lst_0)
# wg_lst_0

wg_lst_v1 = np.array(wg_lst[1][0])
# wg_lst_v1
wg_lst_1 = [wg_lst_v1,wg_lst[1][1]]
wg_lst_1

wg_lst_a = (wg_lst_0,wg_lst_1)
wg_lst_a


# In[39]:


np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
X1 = np.insert(X, 0, 1, axis=1)

rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(rescaled_x)


# In[40]:


print(wg_lst_a)
X1 = np.array(rescaled_x)
y = np.array(y)
wg_lst_a = np.array(wg_lst_a)
score(X1,y,wg_lst_a)


# In[ ]:




