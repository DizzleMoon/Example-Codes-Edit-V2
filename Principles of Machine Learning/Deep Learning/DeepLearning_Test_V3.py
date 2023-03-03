#!/usr/bin/env python
# coding: utf-8

# In[15]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
# import pygal
import sys
import os
import string
import urllib.request
import requests
import curl
import pycurl
import webbrowser
import numpy as np
import math
import pandas as pd
# from IPython import qt
from matplotlib.pyplot import figure
from py.xml import raw
from requests.api import get
from matplotlib import pyplot as plt
# from scratch.working_with_data import rescale
# from scratch.multiple_regression import least_squares_fit, predict
# from scratch.gradient_descent import gradient_step

# from stats import mean, median, de_mean, standard_deviation, correlation
# from gradient_descent import minimize_stochastic, maximize_stochastic, maximize_batch
# from vector import dot, vector_add
# from normal import normal_cdf
# from matrix import make_matrix, get_column, shape, matrix_multiply
# from logistic_regression import *

import math
import os
import random
import sys
from functools import partial, reduce

from scipy.optimize import fmin_tnc

import tqdm

from typing import*

from collections import*
# from scipy import*
from sklearn.metrics import*

from numpy import *

# import mnist
# bltin_sum = np.sum


# In[16]:


# Functions

# def add(a, b): return a + b

Vector = List[float]

Tensor = list

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
    m = sum(vectors,axis=0)
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

#     return np.sum(v_i * w_i for v_i, w_i in zip(v, w))
#     gen = 
    return np.sum(np.fromiter((v_i * w_i for v_i, w_i in zip(v, w)),float))

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
    dim = data.shape[0]
    
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
    dim = data.shape[0]
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
    return 1.0 / (1 + math.exp(-x))

def logistic_prime(x):
    y = logistic(x)
    return y * (1 - y)

def _negative_log_likelihood(x, y, beta):
    """The negative log likelihood for one data point""" 
    if y == 1:
        return -math.log(logistic(dot(x, beta)))
    else:
        return -math.log(1 - logistic(dot(x, beta)))
    
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
                                                                
def step_function(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def sigmoid(t: float) -> float: 
    return 1 / (1 + math.exp(-t))

# Gradient Descent - step
def gradient_step(v: Vector, gradient: Vector, step_size: float):
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

def shape(tensor: Tensor) -> List[int]:
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes

def is_1d(tensor: Tensor) -> bool:
    """
    If tensor[0] is a list, it's a higher-order tensor.
    Otherwise, tensor is 1-dimensional (that is, a vector).
    """
    return not isinstance(tensor[0], list)

def tensor_sum(tensor: Tensor) -> float:
    """Sums up all the values in the tensor"""
    if is_1d(tensor):
        return sum(tensor)  # just a list of floats, use Python sum
    else:
        return sum(tensor_sum(tensor_i)      # Call tensor_sum on each row
                   for tensor_i in tensor)   # and sum up those results.

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """Applies f elementwise""" 
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor] 
    
def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """Applies f to corresponding elements of t1 and t2"""
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]
    
def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
    """Find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0                      # normal_cdf(-10) is (very close to) 0
    hi_z  =  10.0                      # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # Consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the CDF's value there
        if mid_p < p:
            low_z = mid_z              # Midpoint too low, search above it
        else:
            hi_z = mid_z               # Midpoint too high, search below it

    return mid_z

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def softmax(tensor: Tensor) -> Tensor:
    """Softmax along the last dimension"""
    if is_1d(tensor):
        # Subtract largest value for numerical stability.
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]

        sum_of_exps = sum(exps)                 # This is the total "weight."
        return [exp_i / sum_of_exps             # Probability is the fraction
                for exp_i in exps]              # of the total weight.
    else:
        return [softmax(tensor_i) for tensor_i in tensor]


# In[17]:


def softmax(logits):
    #row represents num classes but they may be real numbers
    #So the shape of input is important
    #([[1, 3, 5, 7],
    #  [1,-9, 4, 8]]

    #softmax will be for each of the 2 rows
    #[[2.14400878e-03 1.58422012e-02 1.17058913e-01 8.64954877e-01]
    #[8.94679461e-04 4.06183847e-08 1.79701173e-02 9.81135163e-01]]
    #respectively But if the input is Tranposed clearly the answer
    #will be wrong.

    #That needs to be converted to probability
    #column represents the vocabulary size.

    r, c = logits.shape
    predsl = []
    for row in logits:
        inputs = np.asarray(row)
        #print("inputs:",inputs)
        predsl.append(np.exp(inputs) / float(sum(np.exp(inputs))))
    return np.array(predsl)


# In[18]:


x = np.array([[1, 3, 5, 7],
      [1,-9, 4, 8]])
print("x:",x)
sm=softmax(x)
print("softmax:",sm)
#prints out
#[[2.14400878e-03 1.58422012e-02 1.17058913e-01 8.64954877e-01],
#[8.94679461e-04 4.06183847e-08 1.79701173e-02 9.81135163e-01]]


# In[19]:


def _softmax_grad(sm):
    # Below is the softmax value for [1, 3, 5, 7]
    # [2.14400878e-03 1.58422012e-02 1.17058913e-01 8.64954877e-01]
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(sm)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                print("equal:",i, sm[i],(1-sm[i]))
                #equal: 0 0.002144008783584634 0.9978559912164153
                #equal: 1 0.015842201178506925 0.9841577988214931
                #equal: 2 0.11705891323853292 0.8829410867614671
                #equal: 3 0.8649548767993754 0.13504512320062456
                jacobian_m[i][j] = sm[i] * (1-sm[i])
            else:
                print("not equal:",i,j,sm[i],sm[j])
                #not equal: 0 1 0.002144008783584634 0.015842201178506925
                #not equal: 0 2 0.002144008783584634 0.11705891323853292
                #not equal: 0 3 0.002144008783584634 0.8649548767993754

                #not equal: 1 0 0.015842201178506925 0.002144008783584634
                #not equal: 1 2 0.015842201178506925 0.11705891323853292
                #not equal: 1 3 0.015842201178506925 0.8649548767993754

                #not equal: 2 0 0.11705891323853292 0.002144008783584634
                #not equal: 2 1 0.11705891323853292 0.015842201178506925
                #not equal: 2 3 0.11705891323853292 0.8649548767993754

                #not equal: 3 0 0.8649548767993754 0.002144008783584634
                #not equal: 3 1 0.8649548767993754 0.015842201178506925
                #not equal: 3 2 0.8649548767993754 0.11705891323853292
                jacobian_m[i][j] = -sm[i]*sm[j]

    #finally resulting in
    #[[ 2.13941201e-03 -3.39658185e-05 -2.50975338e-04 -1.85447085e-03]
    #[-3.39658185e-05  1.55912258e-02 -1.85447085e-03 -1.37027892e-02]
    #[-2.50975338e-04 -1.85447085e-03  1.03356124e-01 -1.01250678e-01]
    #[-1.85447085e-03 -1.37027892e-02 -1.01250678e-01  1.16807938e-01]]

    return jacobian_m


# In[20]:


def loss(pred,labels):
  """
  One at a time.
  args:
      pred-(seq=1),input_size
      labels-(seq=1),input_size
  """
  return np.multiply(labels, -np.log(pred)).sum(1)


# In[21]:


def input_one_hot(num,vocab_size):
    #print(num)
    x = np.zeros(vocab_size)
    x[int(num)] = 1
    x=np.reshape(x,[1,-1])
    #print(":",x,x.shape)
    return x;


# In[22]:


x = np.array([[1, 3, 5, 7],
            [1,-9, 4, 8]])
y = np.array([3,1])
sm=softmax(x)

#prints out 0.145
print(loss(sm[0],input_one_hot(y[0],4)))
#prints out 17.01
print(loss(sm[1],input_one_hot(y[1],4)))


# In[23]:


def checkdatadim(data , degree, msg=None):
    if(len(data.shape)!=degree):
        if msg is None:
            raise ValueError('Dimension must be', degree," but is ", len(data.shape),".")
        else:
            raise ValueError(msg)


# In[24]:


def cross_entropy_loss(pred,labels):
    """
    Does an internal softmax before loss calculation.
    args:
        pred- batch,seq,input_size
        labels-batch,seq(has to be transformed before comparision with preds(line-133).)
    """
    checkdatadim(pred,3)
    checkdatadim(labels,2)
    batch,seq,size=pred.shape
    yhat=np.zeros((batch,seq,size))

    for batnum in range(batch):
        for seqnum in range(seq):
            yhat[batnum][seqnum]=softmax(np.reshape(pred[batnum][seqnum],[-1,size]))
    lossesa=np.zeros((batch,seq))
    for batnum in range(batch):
        for seqnum in range(seq):
            lossesa[batnum][seqnum]=loss(np.reshape(yhat[batnum][seqnum],[1,-1]),input_one_hot(labels[batnum][seqnum],size))
    return yhat,lossesa


# In[25]:


x = np.array([[[1, 3, 5, 7],
            [1,-9, 4, 8]]])
y = np.array([[3,1]])

#prints array([[ 0.14507794, 17.01904505]]))
# softmaxed,loss=cross_entropy_loss(x,y)
# print("loss:",loss)


# In[26]:


softmaxed,loss=cross_entropy_loss(x,y)
print("loss:",loss)

batch,seq,size=x.shape
target_one_hot=np.zeros((batch,seq,size))
# Adapting the shape of Y
for batnum in range(batch):
    for i in range(seq):
        target_one_hot[batnum][i]=input_one_hot(y[batnum][i],size)
dy = softmaxed.copy()
dy = dy - target_one_hot
# prints out gradient: [[[ 2.14400878e-03  1.58422012e-02  1.17058913e-01 -1.35045123e-01]
#                        [ 8.94679461e-04 -9.99999959e-01  1.79701173e-02  9.81135163e-01]]]
print("gradient:",dy)


# In[27]:


# x = np.array([[1, 3, 5, 7],
#       [1,-9, 4, 8]])

# y = np.array([3,1])
# print("x:",x.shape)
# sm=softmax(x)
# print("softmax:",sm)
# jacobian=_softmax_grad(sm[0])
# print("jacobian:",jacobian)
# jacobian=_softmax_grad(sm[1])
# print(jacobian)


# In[ ]:




