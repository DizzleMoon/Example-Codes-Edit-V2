#!/usr/bin/env python
# coding: utf-8

# In[75]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
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

import itertools
import random
import tqdm


from typing import*

from collections import*
# from scipy import*
from sklearn.metrics import*

from numpy import *
import random

from itertools import cycle
from sklearn import*
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# In[76]:


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


def num_differences(v1, v2):
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


def cluster_means(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    # clusters[i] contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # if a cluster is empty, just use a random point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]


class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k                      # number of clusters
        self.means = None

    def classify(self, input: Vector) -> int:
        """return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]

        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input) for input in inputs]

                # Check how many assignments changed and if we're done
                num_changed = num_differences(assignments, new_assignments)
                if num_changed == 0:
                    return

                # Otherwise keep the new assignments, and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed} / {len(inputs)}")
                
def squared_clustering_errors(inputs, k):
    """finds the total squared error from k-means clustering the
inputs"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(input) for input in inputs]

    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))

def recolor(pixel):
    cluster = clusterer.classify(pixel)        # index of the closest cluster
    return clusterer.means[cluster]            # mean of the closest cluster

def get_values(cluster):
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]


def cluster_distance(cluster1,
                     cluster2,
                     distance_agg: Callable = min):
    """
    compute all the pairwise distances between cluster1 and cluster2
    and apply the aggregation function _distance_agg_ to the resulting
list
    """
    return distance_agg([distance(v1, v2) for v1 in get_values(cluster1) for v2 in get_values(cluster2)])

def get_merge_order(cluster):
    if isinstance(cluster, Leaf):
        return float('inf')  # was never merged
    else:
        return cluster.order
    
def get_children(cluster):
    if isinstance(cluster, Leaf):
        raise TypeError("Leaf has no children")
    else:
        return cluster.children
    
class Merged(NamedTuple):
    children: tuple
    order: int

def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable = min):
    # Start with all leaves
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair):
        return cluster_distance(pair[0], pair[1], distance_agg)
 
    # as long as we have more than one cluster left...
    while len(clusters) > 1:
        # find the two closest clusters
        c1, c2 = min(((cluster1, cluster2) 
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]),
                      key=pair_distance)

        # remove them from the list of clusters
        clusters = [c for c in clusters if c != c1 and c != c2]

        # merge them, using merge_order = # of clusters left
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # and add their merge
        clusters.append(merged_cluster)

    # when there's only one cluster left, return it
    return clusters[0]

def generate_clusters(base_cluster,
                      num_clusters):
    # start with a list with just the base cluster
    clusters = [base_cluster]

    # as long as we don't have enough clusters yet...
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [c for c in clusters if c != next_cluster]

        # and add its children to the list (i.e., unmerge it)
        clusters.extend(get_children(next_cluster))

    # once we have enough clusters...
    return clusters


# In[77]:


class LogisticRegressionOVR(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = []
        m = X.shape[0]

        for i in np.unique(y):
            y_copy = np.where(y == i, 1, 0)
            w = np.ones(X.shape[1])
#             w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self._sigmoid(output)
                w += self.eta / m * errors.dot(X)
            self.w.append((w, i))
        return self
    
#      def predict(self, X):
#         output = np.insert(X, 0, 1, axis=1).dot(self.w)
#         return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def predict(self, X):
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]
    
    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[78]:


# # Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # Binarize the output
# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = y.shape[1]


# In[79]:


import requests
import csv

# Setup rows array to read iris dataset
rows = []

filename = 'iris.csv'
# reading csv file 
with open(filename, 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
# extracting each data row one by one
    for row in csvreader: 
        rows.append(row) 
        
def parse_iris_row(row):
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    measurements = [float(value) for value in row[:-1]]
    # class is e.g. "Iris-virginica"; we just want "virginica"
    label = [value.split('-')[-1]for value in row]

    return label

# Conver rows list to array
rows = np.array(rows)
label = rows[0][-1].split('-')[-1]
label

# Create iris_data list
iris_data = [parse_iris_row(row) for row in rows]


# Iris dataframe
iris_data_pd = pd.DataFrame(iris_data)
iris_pd = iris_data_pd.iloc[:,0:4]
iris_point = iris_pd.to_numpy().tolist()
# iris_point = iris_pd.to_numpy()
iris_point

iris_target_pd = iris_data_pd.iloc[:,4]
iris_target = iris_target_pd.to_numpy().tolist()

# Convert labels from text to numeral
for i in range(len(iris_target)):
    if iris_target[i] == 'setosa':
        iris_target[i] = 0
    elif iris_target[i] == 'versicolor':
        iris_target[i] = 1
    elif iris_target[i] == 'virginica':
        iris_target[i] = 2

# Subset dataframe into lists/arrays
iris_point_2 = iris_point[:-1]
iris_target_2 = iris_target[:-1]
iris_actual = iris_target_2
print(iris_actual)
# type(iris_point_2)


# In[80]:


# Load dataset
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
dataset = load_breast_cancer()
dataset2 = load_iris()


# In[81]:



def train_test_split2(xs, ys, test_pct):
    # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

x_train, x_test, y_train, y_test = train_test_split2(dataset['data'],
                                                          dataset['target'],
                                                          0.25)
# x_train


# In[82]:


# Load dataset
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import *
# from sklearn.model_selection import train_test_split
dataset = load_breast_cancer()
dataset2 = load_iris()

def logistic_prime(x: float) -> float:
    y = logistic(x)
    return y * (1 - y)

# Split data into train-test pools 
# train, test, train_labels, test_labels = train_test_split2(np.array(iris_point_2),
#                                                           np.array(iris_target_2),
#                                                           0.33)
train1, test1, train1_labels, test1_labels = train_test_split(np.array(iris_point_2),
                                                          np.array(iris_target_2),
                                                          test_size=0.7)

# # Train model 
logregr = LogisticRegression().fit(train1, train1_labels)
print('logregr:', logregr)

# ab_x_tst = []
# ab_y_tst = []
# for i in range(len(train)):
#     alog = logistic(train[i][0])
#     ab_y_tst.append(alog)
#     alog_prime = logistic_prime(train[i][0])
#     ab_x_tst.append(alog_prime)
    
# plt.scatter(ab_x_tst,ab_y_tst)

type(dataset2['data'])
# train1 = train.dtype('u8')
train = train1.astype('float')
test = test1.astype('float')
train_labels = train1_labels.astype('float')
test_labels = test1_labels.astype('float')
# train = np.array(train,dtype('float'))
# test = np.array(test,dtype('float'))
# train_labels = np.array(train_labels,dtype('float'))
# test_labels = np.array(test_labels,dtype('float'))
# test

# Train model 
# logregr = LogisticRegression().fit(train, train_labels)
# print('logregr:', logregr)
# test_labels.tolist()


# In[83]:


logi = LogisticRegressionOVR(n_iter=1000).fit(train, train_labels)
# log_sig = logi._sigmoid(test)
# log_sig
# logi.w

logi.w
log_point = logi.w
log_point = log_point[0][0]
log_point_2 = log_point
log_point_2


# In[84]:


def train_test_split_v2(xs, ys, test_pct):
     # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

random.seed(0)
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split_v2(dataset['data'], dataset['target'],test_size)


# In[85]:



def train_test_split2(xs, ys, test_pct):
    # Generate the indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train 
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])   # y_test

x_train, x_test, y_train, y_test = train_test_split2(dataset['data'],
                                                          dataset['target'],
                                                          0.25)


# In[86]:


# dataset
z = np.dot(test.reshape(84,5),log_point_2.reshape(5,1))


# In[87]:


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
h = sigmoid(z)
scores = h


# In[88]:


h = sigmoid(z)
scores = sorted(h)

print(len(scores))
print(scores)


# In[89]:


# Rename, listify 
# train, test, train_labels, test_labels 
actuals = list(test_labels)


# In[90]:


# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
from csv import reader
from math import sqrt
 
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
# 		print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
# def dataset_minmax(dataset):
# 	minmax = list()
# 	for i in range(len(dataset[0])):
# 		col_values = [row[i] for row in dataset]
# 		value_min = min(col_values)
# 		value_max = max(col_values)
# 		minmax.append([value_min, value_max])
# 	return minmax

# Find the min and max values for each column
def dataset_mean_std(dataset):
	mean_std = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_mean = np.mean(col_values)
		value_std = np.std(col_values)
		mean_std.append([value_mean, value_std])
	return mean_std

# Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, minmax):
# 	for row in dataset:
# 		for i in range(len(row)):
# 			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, mean_std):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - mean_std[i][0]) / (mean_std[i][1])
 
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# Make a prediction with KNN on Iris Dataset
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# define model parameter
num_neighbors = 3
# define a new record
row = [4.7,7.9,3.2,0.3]
row1 = [0.7,1.5,8.4,2.0]
# predict the label
label = predict_classification(dataset, test[0].tolist(), num_neighbors)
# print('Data=%s, Predicted: %s' % (row, label))
# print(dataset)


labels1 = []
for j in range(len(test)):
    labels1.append(predict_classification(dataset, test[j], num_neighbors))
# labels
    


# In[91]:


logi = LogisticRegressionOVR(n_iter=1500).fit(train, train_labels)
log_sig = logi._sigmoid(test)
log_sig
logi.w
# plt.plot(log_sig)


# In[92]:


# Predictions
# Method 1
predictions = list(logregr.predict(test))
# Method 2
predictions2 = list(logi.predict(test))


# In[93]:


# Method 1
print("Accuracy = %.3f" % (sum([p==a for p, a in zip(predictions, actuals)])/len(predictions)))


# In[94]:


# Method 2
print("Accuracy = %.3f" % (sum([p==a for p, a in zip(predictions2, actuals)])/len(predictions2)))


# In[95]:


# Mean Accuracy/Alternative accuracy formula
logi_acc = logi.score(test,test_labels)
logi_acc


# In[96]:


import collections
ConfusionMatrix = collections.namedtuple('conf', ['tp','fp','tn','fn']) 


# In[97]:


def calc_ConfusionMatrix(actuals, scores, threshold=0.22, positive_label=2):
    tp=fp=tn=fn=0
    bool_actuals = [act==positive_label for act in actuals]
    for truth, score in zip(bool_actuals, scores):
        if score > threshold:                      # predicted positive 
            if truth:                              # actually positive 
                tp += 1
            else:                                  # actually negative              
                fp += 1          
        else:                                      # predicted negative 
            if not truth:                          # actually negative 
                tn += 1                          
            else:                                  # actually positive 
                fn += 1

    return ConfusionMatrix(tp, fp, tn, fn)


# In[98]:


def ACC(conf_mtrx):
    return (conf_mtrx.tp + conf_mtrx.tn) / (conf_mtrx.fp + conf_mtrx.tn + conf_mtrx.tp + conf_mtrx.fn)


# In[99]:


def FPR(conf_mtrx):
    return conf_mtrx.fp / (conf_mtrx.fp + conf_mtrx.tn) if (conf_mtrx.fp + conf_mtrx.tn)!=0 else 0
def TPR(conf_mtrx):
    return conf_mtrx.tp / (conf_mtrx.tp + conf_mtrx.fn) if (conf_mtrx.tp + conf_mtrx.fn)!=0 else 0


# In[100]:


def apply(actuals, scores, **fxns):
    # generate thresholds over score domain 
    low = min(scores)
    high = max(scores)
    step = (abs(low) + abs(high)) / 1000
    thresholds = np.arange(low-step, high+step, step)

    # calculate confusion matrices for all thresholds
    confusionMatrices = []
    for threshold in thresholds:
        confusionMatrices.append(calc_ConfusionMatrix(actuals, scores, threshold, positive_label=1))

    # apply functions to confusion matrices 
    results = {fname:list(map(fxn, confusionMatrices)) for fname, fxn in fxns.items()}

    results["THR"] = thresholds
    return results


# In[101]:


acc = apply(actuals,scores, ACC=ACC)


# In[102]:


# ROC curve
def ROC(actuals, scores):
    return apply(sorted(actuals), sorted(scores), FPR=FPR, TPR=TPR)

roc_curve = ROC(sorted(actuals),sorted(scores))
# roc_curve['TPR']


# In[103]:


# Plot ROC curve
plt.plot(roc_curve['FPR'],roc_curve['TPR'])
# plt.plot(np.dot(roc_curve['FPR'],len(test)/6),roc_curve['TPR'])


# In[104]:


# # Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
# from csv import reader
# from math import sqrt
 
# # Load a CSV file
# def load_csv(filename):
# 	dataset = list()
# 	with open(filename, 'r') as file:
# 		csv_reader = reader(file)
# 		for row in csv_reader:
# 			if not row:
# 				continue
# 			dataset.append(row)
# 	return dataset
 
# # Convert string column to float
# def str_column_to_float(dataset, column):
# 	for row in dataset:
# 		row[column] = float(row[column].strip())
 
# # Convert string column to integer
# def str_column_to_int(dataset, column):
# 	class_values = [row[column] for row in dataset]
# 	unique = set(class_values)
# 	lookup = dict()
# 	for i, value in enumerate(unique):
# 		lookup[value] = i
# 		print('[%s] => %d' % (value, i))
# 	for row in dataset:
# 		row[column] = lookup[row[column]]
# 	return lookup
 
# # Find the min and max values for each column
# # def dataset_minmax(dataset):
# # 	minmax = list()
# # 	for i in range(len(dataset[0])):
# # 		col_values = [row[i] for row in dataset]
# # 		value_min = min(col_values)
# # 		value_max = max(col_values)
# # 		minmax.append([value_min, value_max])
# # 	return minmax

# # Find the min and max values for each column
# def dataset_mean_std(dataset):
# 	mean_std = list()
# 	for i in range(len(dataset[0])):
# 		col_values = [row[i] for row in dataset]
# 		value_mean = np.mean(col_values)
# 		value_std = np.std(col_values)
# 		mean_std.append([value_mean, value_std])
# 	return mean_std

# # Rescale dataset columns to the range 0-1
# # def normalize_dataset(dataset, minmax):
# # 	for row in dataset:
# # 		for i in range(len(row)):
# # 			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            
# # Rescale dataset columns to the range 0-1
# def normalize_dataset(dataset, mean_std):
# 	for row in dataset:
# 		for i in range(len(row)):
# 			row[i] = (row[i] - mean_std[i][0]) / (mean_std[i][1])
 
 
# # Calculate the Euclidean distance between two vectors
# def euclidean_distance(row1, row2):
# 	distance = 0.0
# 	for i in range(len(row1)-1):
# 		distance += (row1[i] - row2[i])**2
# 	return sqrt(distance)
 
# # Locate the most similar neighbors
# def get_neighbors(train, test_row, num_neighbors):
# 	distances = list()
# 	for train_row in train:
# 		dist = euclidean_distance(test_row, train_row)
# 		distances.append((train_row, dist))
# 	distances.sort(key=lambda tup: tup[1])
# 	neighbors = list()
# 	for i in range(num_neighbors):
# 		neighbors.append(distances[i][0])
# 	return neighbors
 
# # Make a prediction with neighbors
# def predict_classification(train, test_row, num_neighbors):
# 	neighbors = get_neighbors(train, test_row, num_neighbors)
# 	output_values = [row[-1] for row in neighbors]
# 	prediction = max(set(output_values), key=output_values.count)
# 	return prediction
 
# # Make a prediction with KNN on Iris Dataset
# filename = 'iris.csv'
# dataset = load_csv(filename)
# for i in range(len(dataset[0])-1):
# 	str_column_to_float(dataset, i)
# # convert class column to integers
# str_column_to_int(dataset, len(dataset[0])-1)
# # define model parameter
# num_neighbors = 5
# # define a new record
# row = [4.7,7.9,3.2,0.3]
# row1 = [0.7,1.5,8.4,2.0]
# # predict the label
# label = predict_classification(dataset, row1, num_neighbors)
# print('Data=%s, Predicted: %s' % (row, label))
# # print(dataset)


# In[105]:


plt.plot(sorted(h))

