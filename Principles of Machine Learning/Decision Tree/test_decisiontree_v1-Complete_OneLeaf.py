#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# Functions

# def add(a, b): return a + b

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
                                                                


# In[3]:


x = [i/100.0 for i in range(101) ]
entropy_x = [0] + [-p*math.log(p,2) for p in x if p > 0]
# import matplotlib.pyplot as plt
plt.plot(x, entropy_x)
plt.xlabel('probability')
plt.ylabel('-p*log(p)')
plt.show()


# In[4]:


# Entropy
def entropy(class_probabilities):
    """Given a list of class probabilities, compute the entropy"""
    ent = sum(-p * math.log(p, 2)
               for p in class_probabilities 
               if p > 0)
#     ent_0 = []
#     for p in class_probabilities :
#         if p > 0:
#             ent_0.append(-p * math.log(p,2))
#     ent = sum(ent_0)
            
    return ent

def class_probabilities(labels):
    total_count = len(labels)
    class_prob = [count / total_count for count in Counter(labels).values()]
    return class_prob

def data_entropy(labels):
    data_ent = entropy(class_probabilities(labels))
    return data_ent


# In[5]:


# The Entropy of a Partition

def partition_entropy(subsets):
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)
    part_ent = sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

    return part_ent


# In[6]:


x = np.arange(0,50)
# x = pd.DataFrame({'x':x})

y1 = np.random.uniform(10,15,10)
y2 = np.random.uniform(20,25,10)
y3 = np.random.uniform(0,5,10)
y4 = np.random.uniform(30,32,10)
y5 = np.random.uniform(13,17,10)

y = np.concatenate((y1,y2,y3,y4,y5))
y = y[:,None]

x = x.tolist()
y = y.tolist()


# In[7]:


# Creating a Decision Tree

# Test
# Create class
class Candidate(NamedTuple):
    level: int
    did_well: Optional[bool] = None  # allow unlabeled data
# Data
# inputs = [Candidate('Senior', 'Java',   False, False, False),
#           Candidate('Senior', 'Java',   False, True,  False),
#           Candidate('Mid',    'Python', False, False, True),
#           Candidate('Junior', 'Python', False, False, True),
#           Candidate('Junior', 'R',      True,  False, True),
#           Candidate('Junior', 'R',      True,  True,  False),
#           Candidate('Mid',    'R',      True,  True,  True),
#           Candidate('Senior', 'Python', False, False, False),
#           Candidate('Senior', 'R',      True,  False, True),
#           Candidate('Junior', 'Python', True,  False, True),
#           Candidate('Senior', 'Python', True,  True,  True),
#           Candidate('Mid',    'Python', False, True,  True),
#           Candidate('Mid',    'Java',   True,  False, True),
#           Candidate('Junior', 'Python', False, True,  False)
#          ]


inputs = [Candidate('Senior', False),
          Candidate('Senior', False),
          Candidate('Mid',    True),
          Candidate('Junior', True),
          Candidate('Junior', True),
          Candidate('Junior', False),
          Candidate('Mid',  True),
          Candidate('Senior', False),
          Candidate('Senior', True),
          Candidate('Junior', True),
          Candidate('Senior', True),
          Candidate('Mid',    True),
          Candidate('Mid',    True),
          Candidate('Junior', False)
         ]


# Generic type of inputs
T = TypeVar('T')

def partition_by(inputs, attribute):
    """Partition the inputs into lists based on the specified attribute."""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)  # value of the specified attribute
        partitions[key].append(input)    # add input to the correct partition
    return partitions

# Compute Entropy
def partition_entropy_by(inputs, attribute, label_attribute):
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs 
    partitions = partition_by(inputs, attribute)

    # but partition_entropy needs just the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)

for key in ['level']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))

# partition_entropy_by(inputs, 'level', 'did_well')

# senior_inputs = [input for input in inputs if input.level == 'Senior']

# senior_inputs

# type(senior_inputs)
inputs
# type(inputs)


# In[8]:


# senior_inputs = [input for input in inputs if input.level == x[0]]
# senior_inputs
y


# In[9]:


# Putting It All Together

class Leaf(NamedTuple): 
     value: Any
            
class Split(NamedTuple):
    attribute:Any
    subtrees: dict
    default_value: Any = None

DecisionTree = Union[Leaf, Split]

# Representation
# hiring_tree = Split('level', {   # first, consider "level"
#     'Junior': Split('phd', {     # if level is "Junior", next look at "phd"
#         False: Leaf(True),       #   if "phd" is False, predict True
#         True: Leaf(False)        #   if "phd" is True, predict False
#     }),
#     'Mid': Leaf(True),           # if level is "Mid", just predict True
#     'Senior': Split('tweets', {  # if level is "Senior", look at "tweets"
#         False: Leaf(False),      #   if "tweets" is False, predict False
#         True: Leaf(True)         #   if "tweets" is True, predict True
#     })
# })

hiring_tree = Split('level',{'Junior' : Split('phd', {False:Leaf(True),True: Leaf(False)}),
                             'Mid': Leaf(True), 'Senior': Split('tweets', {False:Leaf(False),True:Leaf(True)})})

def classify(tree, input):
    """classify the input using the given decision tree"""

    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value 

    # Otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values are subtrees to consider next 
    
    subtree_key = getattr(input, tree.attribute)
    print("Tree Attribute: ", tree.attribute)

    if subtree_key not in tree.subtrees:   # If no subtree for key,
        a_trees = tree.default_value 
        print("a_trees",a_trees)
        return a_trees         # return the default value.

    subtree = tree.subtrees[subtree_key]   # Choose the appropriate subtree
    print('Subtree1: ', subtree)
    return classify(subtree, input)        # and use it to classify the input.

def build_tree_id3(inputs, split_attributes, target_attribute):
    print("Target Attribute: ", target_attribute)
    # Count target labels
    label_counts = Counter(getattr(input, target_attribute) for input in inputs)
    most_common_label = label_counts.most_common(1)[0][0]
    print("Label Counts: ", label_counts)
    print("Most Common Label : ", most_common_label)
    print("Split Attributes: ", split_attributes)
    print("Length of Label Counts: ", len(label_counts))
    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)
    
    # Otherwise split by the best attribute
    def split_entropy(attribute):
#     """Helper function for finding the best attribute"""
        return partition_entropy_by(inputs, attribute, target_attribute)
    
    best_attribute = min(split_attributes, key=split_entropy)
    print("Best Attribute:", best_attribute)
    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]
    # Recursively build the subtrees
    subtrees = {attribute_value : build_tree_id3(subset, new_attributes, target_attribute)
               for attribute_value, subset in partitions.items()}
#     print("SubTrees: ", subtrees)
    return Split(best_attribute, subtrees, default_value=most_common_label)

tree = build_tree_id3(inputs, ['level'], 'did_well')
print(tree)

# Tests
# test_a = classify(tree, Candidate("Junior"))
# test_b = classify(tree, Candidate("Junior"))
test_c = classify(tree, Candidate("Intern"))
test_c


# In[10]:


# Random Forests

def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]

def build_forest_id3(inputs, split_candidates=None, num_split_candidates=None):

    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    
    # count Trues and Falses in the inputs
    num_trues = len([label for item, label in inputs if label])
    num_falses = len(inputs) - num_trues
    
    if num_trues == 0:
        return False
    
    if num_falses == 0:
        return True
    
    if not split_candidates:
        return num_trues >= num_falses
    
    # if there's already few enough split candidates, look at all of them
    if num_split_candidates is None or len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates    
    # otherwise pick a random sample
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    # otherwise, split on the best attribute
    best_attribute = min(sampled_split_candidates, key=split_entropy)
    
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    
    # recursively build the subtrees
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.items() }
    
    subtrees[None] = num_trues > num_falses # default case
    
    return (best_attribute, subtrees)

def build_forest(inputs, n=3):
    return [build_tree_id3(inputs, num_split_candidates=3) for i in range(n)]


# In[11]:


# # if there are already few enough split candidates, look at all of them
# if len(split_candidates) <= self.num_split_candidates:
#     sampled_split_candidates = split_candidates
# # otherwise pick a random sample
# # else:
# #     sampled_split_candidates = random.sample(split_candidates, self.num_split_candidates)                                    self.num_split_candidates)

# # now choose the best attribute only from those candidates
# # best_attribute = min(sampled_split_candidates, key=split_entropy)

# # partitions = partition_by(inputs, best_attribute)

