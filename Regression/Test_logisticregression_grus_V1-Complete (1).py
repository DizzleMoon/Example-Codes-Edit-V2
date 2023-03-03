#!/usr/bin/env python
# coding: utf-8

# In[35]:


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
from scipy.special import expit


# In[36]:


def vector_mean(vectors):
    """Computes the element-wise average"""
    n = len(vectors)
    m = np.sum(vectors,axis=0)
    vec_mean = np.multiply(1/n,m)
    return vec_mean

# Standard deviation                        
def standard_deviation(xs):
    """The standard deviation is the square root of the variance"""
    std_dev = np.sqrt(variance(xs)) 
    return std_dev

def variance(xs):
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    vari = sum_of_squares(deviations)/(n-1)
    return vari

def de_mean(xs):
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = np.mean(xs)
    d_mean = [x - x_bar for x in xs]
    return d_mean

def sum_of_squares(v):
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return np.dot(v, v)

def	vector_add(v,	w):
				"""adds	corresponding	elements"""
				return	[v_i	+	w_i
												for	v_i,	w_i	in	zip(v,	w)]
        
def	step(v,	direction,	step_size):
        """move	step_size	in	the	direction	from	v"""
        return	[v_i	+	step_size	*	direction_i
                                        for	v_i,	direction_i	in	zip(v,	direction)]


# In[37]:


def	minimize_batch(target_fn,	gradient_fn,	theta_0,	tolerance=0.000001):
    """use	gradient	descent	to	find	theta	that	minimizes	target	function"""
    step_sizes	=	[10,1,0,	0.1,	0.01,	0.001,	0.0001,	0.00001, 0.000001, 0.0000001]
    #                 step_sizes	=	[100,1,	0.01,	0.0001,	0.00001,	0.0000001,	0.000000001, 0.00000000001, 0.0000000000001]
    theta	=	theta_0																											#	set	theta	to	initial	value
    target_fn	=	safe(target_fn)															#	safe	version	of	target_fn
    value	=	target_fn(theta)																		#	value	we're	minimizing
    while	True:
        gradient	=	gradient_fn(theta)
        next_thetas	=	[step(theta,	gradient,	-step_size) for	step_size	in	step_sizes]
        #	choose	the	one	that	minimizes	the	error	function
        next_theta	=	max(next_thetas,	key=target_fn)
        next_value	=	target_fn(next_theta)
        #	stop	if	we're	"converging"
        if	abs(value	-	next_value)	<	tolerance:
            return	theta
        else:
            theta,	value	=	next_theta,	next_value


# In[38]:


def	negate(f):
				"""return	a	function	that	for	any	input	x	returns	-f(x)"""
				return	lambda	*args,	**kwargs:	-f(*args,	**kwargs)
def	negate_all(f):
				"""the	same	when	f	returns	a	list	of	numbers"""
				return	lambda	*args,	**kwargs:	[-y	for	y	in	f(*args,	**kwargs)]
def	maximize_batch(target_fn,	gradient_fn,	theta_0,	tolerance=0.000001):
				return	minimize_batch(target_fn, gradient_fn, theta_0, tolerance)


# In[39]:


def shape(A):
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0   # number of elements in first row
    return num_rows, num_cols


def scale(data):
    """returns the mean and standard deviation for each position"""
    dim = len(data[0])

    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs

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


# In[40]:


epsilon = 1e-5    


# In[41]:


def logistic(x):
    return 1.0 / (1 + np.exp(-(x + epsilon)))

def logistic_prime(x):
    y = logistic(x)
    return y * (1 - y)


# In[42]:


def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i==1:
        return np.log(expit(np.dot(x_i, beta)) + epsilon)
    else:
        return np.log(1 - expit(np.dot(x_i, beta)) + epsilon)
    
def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
        for x_i, y_i in zip(x, y))

def logistic_log_partial_ij(x_i, y_i, beta, j):
    """here i is the index of the data point, j the index of the derivative"""
    return (y_i - expit(np.dot(x_i, beta))) * x_i[j]

def logistic_log_gradient_i(x_i, y_i, beta):
    """the gradient of the log likelihood corresponding to the ith data point"""
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]

def logistic_log_gradient(x, y, beta):
    return reduce(vector_add,
        [logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x,y)])

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


# In[43]:


def sigmoid(x):
    # use expit as alternative
    return 1.0 / (1 + np.exp(-(x+epsilon)))

def predictOne(x,w):
    return max((x.dot(w), c) for w, c in w)[1]

def predict(X,w):
    return np.array([predictOne(i,w) for i in X])

def predict2(X,w):
    output = np.insert(X, 0, 1, axis=1).dot(w)
    return (np.floor(sigmoid(output) + .5)).astype(int)

def score(X, y, w):
    return sum(predict(X,w) == y) / len(y)


# In[44]:


# np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
# x = [(1) + X]
# x
X1 = np.insert(list(X), 0, 1, axis=1)
X1

x = []
for i in X1:
    x.append(list(i))
    
# x
    
# x = [[1] + list(row) for row in X]
# x


# In[45]:


# data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
# x = [(1,) + row[:2] for row in data]
# y = [row[2] for row in data]
# # xx = [row[:2] for row in data]
# x


# In[46]:



# def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
#     """use gradient descent to find theta that minimizes target function"""
    
#     step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#     theta = theta_0
#     target_fn = safe(target_fn)
#     value = target_fn(theta)
#     values = []
    
#     # set theta to initial value
#     # safe version of target_fn
#     # value we're minimizing
#     while True:
#         values.append(value)
#         gradient = gradient_fn(theta)
#         next_thetas = [np.array(step(theta, gradient, -step_size))
#                        for step_size in step_sizes]

#         # choose the one that minimizes the error function
#         next_theta = min(next_thetas, key=target_fn)
#         next_value = target_fn(next_theta)

#         # stop if we're "converging"
#         if abs(value - next_value) < tolerance:
#             values.append(next_value)
#             break
#         else:
#             theta, value = next_theta, next_value

#     return theta, values
    
    
# def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
#     return minimize_batch(negate(target_fn),
#                           negate_all(gradient_fn),
#                           theta_0,
#                           tolerance)


# In[47]:


def optimize2(x, y,learning_rate,iterations,tolerance): 
    size = x.shape[0]
    size1 = len(y)
    print('size1:\n', size1)
#     weight = parameters["weight"] 
#     bias = parameters["bias"]
    w1 = []
    losses = []
#     wg = np.ones(x.shape[1])
    
    for i in np.unique(y):
        y_copy = [1 if c == i else 0 for c in y]
        wg = np.ones(x.shape[1])
        wg_0 = 0
        loss_0 = np.ones(len(y))
        iteration = 0
        diff = 0.01
        diff_1 = 0.01
        diff_3 = 0.1
        decay = learning_rate/iterations
        decay = decay*learning_rate * 0.1
#         print('wg:\n', wg)
#         print('wg len:\n', len(wg))

        for j in range(iterations):
            iteration += 1
            sigma = expit(np.dot(x, wg))
            loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
            loss[np.isnan(loss)] = 0
            loss[np.isinf(loss)] = 0
#             print('Loss1:\n', loss)
#             if np.all(np.isnan(abs(loss))) or np.all(np.isinf(abs(loss))) or np.any(loss == float('-inf')):
#                 loss = 0
#             print('loss: \n', loss)
#             print('loss len: \n', len(loss))
            dW = 1/size * np.dot(x.T, (y_copy - sigma))
#             db = 1/size * np.sum(sigma - y)
            wg += learning_rate * dW
#             bias -= learning_rate * db 
            diff_0 = np.sum(wg) - wg_0 
            diff_2 = np.abs(diff_0 - diff_1)
            diff = np.abs(diff_2 - diff_3)
            diff_3 = diff_2
            print('Diff: ', diff)
            diff_1 = diff_0
#                 if np.isnan(abs(diff)) or np.isinf(abs(diff)):
#                     diff = 0
            wg_0 = np.sum(wg)
    
#             if diff == 0.013568115566890526:
# #             if diff  0.013568115566890526:
#                 print('Exit')
#                 break

            if diff <= tolerance:
                print('Exit')
                break
            else:
                learning_rate = learning_rate * (1/(1 + decay * iteration))
            

#             for i in range(0,len(loss)):
#                 diff = np.abs(abs(loss_0[i]) - abs(loss[i]))
# #                 if np.isnan(abs(diff)) or np.isinf(abs(diff)):
# #                     diff = 0
#                 loss_0[i] = loss[i]
#                 if np.all(np.abs(diff) <= 1e-4):
#                     break
#                 print('Difference: \n', diff)
#         print(wg)
        w1.append((wg,i))
        losses.append((loss,i))
        print("Iterations: ", iteration)
#     parameters["weight"] = w1
#     parameters["bias"] = bias
    return w1,losses

# Define the train function
# def train(x, y, learning_rate,iterations):
#     parameters_out = optimize2(x, y, learning_rate, iterations ,init_parameters)
#     return parameters_out


# In[48]:


random.seed(0)
rescaled_x = rescale(X1)
x_train, x_test, y_train, y_test = train_test_split(x , y, 0.33)


# In[49]:


# Initialize the weight and bais
init_parameters = {} 
# init_parameters["weight"] = np.zeros((X1.shape[1],1))
init_parameters["weight"] = np.zeros((X.shape[1],2))
init_parameters["weight"]
init_parameters["bias"] = 0
# init_parameters['weight']


# In[50]:


rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(rescaled_x)
# print(x)
parameters_out,losses = optimize2(x, y, learning_rate = 0.00001, iterations = 10000, tolerance = 1e-15)
# parameters_out,losses = optimize2(x, y, learning_rate = 0.00001, iterations = 10000, tolerance = 1e-15)
print('Weights: \n',parameters_out)
# print('Losses: \n', losses[0])


# In[51]:


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


# In[52]:


X1.shape
wg


# In[53]:


# rescaled_x = rescale(X1)
rescaled_x = rescale(x)
rescaled_x = np.array(rescaled_x)
score(X1,y,wg)


# In[54]:


# np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
# x = [(1) + X]
# x
X1 = np.insert(list(X), 0, 1, axis=1)
X1

x = []
for i in X1:
    x.append(list(i))
    


# In[55]:


x_norm = (x - np.min(x))/(np.max(x) - np.min(x))
x_norm = x_norm.tolist()
y_norm = (y - np.min(y))/(np.max(y) - np.min(y))
y_norm = y_norm.tolist()

# x_norm = (x - np.mean(x))/np.std(x)
# y_norm = (y - np.mean(y))/np.std(y)



# In[56]:


random.seed(0)
# rescaled_x = rescale(x_norm)
rescaled_x = rescale(x)
x_train, x_test, y_train, y_test = train_test_split(rescaled_x, y, 0.33)
# type(x_train)


# In[57]:


# def optimize3(x, y,learning_rate,iterations): 
#     size = len(x)
#     size1 = len(y)
#     print('size1:\n', size1)
# #     weight = parameters["weight"] 
# #     bias = parameters["bias"]
#     w1 = []
#     losses = []
# #     wg = np.ones(x.shape[1])
#     x = np.array(x)
#     y = np.array(y)
    
#     for i in np.unique(y):
#         y_copy = [1 if c == i else 0 for c in y]
#         wg = np.ones(len(x))
#         loss_0 = np.ones(len(y))
# #         print('wg:\n', wg)
# #         print('wg len:\n', len(wg))

#         for _ in range(iterations): 
#             sigma = sigmoid(np.dot(x.T, wg))
#             loss = -1/size * np.sum(y * np.log(sigma)) + (1 - y) * np.log(1-sigma)
#             loss[np.isnan(loss)] = 0
#             loss[np.isinf(loss)] = 0
# #             print('Loss1:\n', loss)
# #             if np.all(np.isnan(abs(loss))) or np.all(np.isinf(abs(loss))) or np.any(loss == float('-inf')):
# #                 loss = 0
# #             print('loss: \n', loss)
# #             print('loss len: \n', len(loss))
#             dW = 1/size * np.dot(x.T, (y_copy - sigma))
# #             db = 1/size * np.sum(sigma - y)
#             wg += learning_rate * dW
# #             bias -= learning_rate * db 
#             for i in range(0,len(loss)):
#                 diff = np.abs(abs(loss_0[i]) - abs(loss[i]))
# #                 if np.isnan(abs(diff)) or np.isinf(abs(diff)):
# #                     diff = 0
#                 loss_0[i] = loss[i]
#                 if np.all(np.abs(diff) <= 1e-4):
#                     break
# #                 print('Difference: \n', diff)
# #         print(wg)
#         w1.append((wg,i))
#         losses.append((loss,i))
# #     parameters["weight"] = w1
# #     parameters["bias"] = bias
#     return w1,losses

# # Define the train function
# # def train(x, y, learning_rate,iterations):
# #     parameters_out = optimize2(x, y, learning_rate, iterations ,init_parameters)
# #     return parameters_out


# In[58]:


# # rescaled_x = rescale(X1)

# # Train the model
# y = np.array(y)
# x = np.array(rescaled_x)
# # print(x)
# x_train_1 = np.array(x_train)
# y_train_1 = np.array(y_train)
# parameters_out_1,losses_1 = optimize2(x_train_1, y_train_1, learning_rate = 0.0001, iterations = 10000, tolerance = 1e-12)
# # print('Weights: \n',parameters_out)
# # print('Losses: \n', losses[0])
# # print('Weights: \n',parameters_out)
# print('Weights2: \n',parameters_out_1)
# print('Weights2_0: \n',parameters_out_1[0][0])
# print('Weights2_1: \n',parameters_out_1[1][0])
# print('Weights2_0_len: \n',len(parameters_out_1[0][0]))
# print('Weights2_1_len: \n',len(parameters_out_1[1][0]))


# In[59]:


# # rescaled_x = rescale(X1)

# # Train the model
# y = np.array(y)
# x = np.array(rescaled_x)
# # print(x)
# x_train_1 = np.array(x_train)
# y_train_1 = np.array(y_train)
# parameters_out_1,losses_1 = optimize2(x_train_1, y_train_1, learning_rate = 0.0001, iterations = 10000, tolerance = 1e-11)
# # print('Weights: \n',parameters_out)
# # print('Losses: \n', losses[0])
# # print('Weights: \n',parameters_out)
# print('Weights2: \n',parameters_out_1)
# print('Weights2_0: \n',parameters_out_1[0][0])
# print('Weights2_1: \n',parameters_out_1[1][0])
# print('Weights2_0_len: \n',len(parameters_out_1[0][0]))
# print('Weights2_1_len: \n',len(parameters_out_1[1][0]))


# In[60]:


# rescaled_x = rescale(X1)

# Train the model
y = np.array(y)
x = np.array(rescaled_x)
# print(x)
x_train_1 = np.array(x_train)
y_train_1 = np.array(y_train)
parameters_out_1,losses_1 = optimize2(x_train_1, y_train_1, learning_rate = 0.00001, iterations = 10000, tolerance = 1e-15)
# print('Weights: \n',parameters_out)
# print('Losses: \n', losses[0])
# print('Weights: \n',parameters_out)
print('Weights2: \n',parameters_out_1)
print('Weights2_0: \n',parameters_out_1[0][0])
print('Weights2_1: \n',parameters_out_1[1][0])
print('Weights2_0_len: \n',len(parameters_out_1[0][0]))
print('Weights2_1_len: \n',len(parameters_out_1[1][0]))


# In[61]:


# # rescaled_x = rescale(x)

# # Train the model
# y = np.array(y)
# x = np.array(rescaled_x)
# # print(x)
# parameters_out,losses = optimize2(x, y, learning_rate = 0.01, iterations = 1000)
# # print('Weights: \n',parameters_out)
# print('Losses: \n', losses[0])


# In[62]:


# random.seed(0)
# rescaled_x = rescale(x)
# x_train, x_test, y_train, y_test = train_test_split(x , y, 0.33)
# want to maximize log likelihood on the training data
fn = partial(logistic_log_likelihood, x_train_1, y_train_1)
gradient_fn = partial(logistic_log_gradient, x_train_1, y_train_1)
# pick a random starting point
# beta_0 = [random.random() for _ in range(len(x_train[0]))]
# # and maximize using gradient descent
# beta_hat, values = maximize_batch(fn, gradient_fn, beta_0)
# beta_hat


# In[63]:


# beta_0 = parameters_out['weight']
beta_0 = parameters_out_1[1][0]
print(beta_0)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print(X_train.shape)
# print(y_train)
# beta_0 = [random.random() for _ in range(3)] # and maximize using gradient descent
# list(beta_0)
beta_hat_0 = maximize_batch(fn, gradient_fn, beta_0)
# beta_hat
print(beta_hat_0)
print(len(beta_hat_0))
# print(cnt)


# In[64]:


fn = partial(logistic_log_likelihood, x_train_1, y_train_1)
gradient_fn = partial(logistic_log_gradient, x_train_1, y_train_1)


# In[65]:


# beta_0 = parameters_out['weight']
beta_1 = parameters_out_1[0][0]
print(list(beta_1))
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print(X_train.shape)
# print(y_train)
# beta_0 = [random.random() for _ in range(3)] # and maximize using gradient descent
# list(beta_0)
beta_hat_1 = maximize_batch(fn, gradient_fn, beta_1)
print(beta_hat_1)
print(len(beta_hat_1))
# print(cnt)


# In[66]:


wg_0_lst = []
for i in beta_hat_0:
    wg_0_lst.append(i)
wg_0_lst

wg_0 = np.array([wg_0_lst,wg[1][1]])
print(wg_0)

wg_1_lst = []
for i in beta_hat_1:
    wg_1_lst.append(i)
wg_1_lst

wg_1 = np.array([wg_1_lst,wg[0][1]])
print(wg_1)


# In[67]:


wg_lst = np.array([wg_1,wg_0])

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


# In[68]:


# np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
# x = [(1) + X]
# x
X1 = np.insert(list(X), 0, 1, axis=1)
# X1 = rescale(X1)

x = []
for i in X1:
    x.append(list(i))
x = np.array(x)
    


# x = []
# for i in X1:
#     x.append(list(i))
    
# x
    
# x = [[1] + list(row) for row in X]
# x



print(wg_lst_a)
X1 = np.array(rescale(X1))
y = np.array(y)
wg_lst_a = np.array(wg_lst_a)
score(X1,y,wg_lst_a)


# In[ ]:




