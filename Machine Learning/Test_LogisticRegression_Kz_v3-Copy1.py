#!/usr/bin/env python
# coding: utf-8

# In[21]:


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

import tqdm

from typing import Tuple
# from gradient_descent import maximize_stochastic, maximize_batch


# In[22]:


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
        return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=0)]
    
    def _predict_one(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn import datasets

# np.set_printoptions(precision=3)
iris = datasets.load_iris()
X = iris.data
y = iris.target
logi = LogisticRegressionOVR(n_iter=1000).fit(X, y)
print(logi.w)

iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp =     train_test_split(iris.data, iris.target, test_size=.4)
X_validation, X_test, y_validation, y_test =     train_test_split(X_temp, y_temp, test_size=.5)

logi = LogisticRegressionOVR(n_iter=1000).fit(X_train, y_train)

# print(logi.score(X_train, y_train))
# print(logi.score(X_validation, y_validation))


# In[24]:


logi._sigmoid(y_train)

