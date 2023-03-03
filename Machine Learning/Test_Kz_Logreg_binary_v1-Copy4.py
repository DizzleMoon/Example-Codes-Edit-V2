#!/usr/bin/env python
# coding: utf-8

# In[142]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
import pygal
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

import mnist
# bltin_sum = np.sum

import random

from itertools import cycle
from sklearn import*
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# In[143]:


class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
#         self.w = np.ones(X.shape[1])
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        m = X.shape[0]

        for _ in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - self._sigmoid(output)
            self.w += self.eta / m * errors.dot(X)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[144]:


# import requests
# import csv

# rows = []

# filename = 'iris.csv'
# # reading csv file 
# with open(filename, 'r') as csvfile: 
#     # creating a csv reader object 
#     csvreader = csv.reader(csvfile) 
# #     iris_data = [parse_iris_row(row) for row in csvreader]

# # extracting each data row one by one
#     for row in csvreader: 
#         rows.append(row) 

# MergedLabels = pd.DataFrame(rows)


# In[145]:


# X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
# y = np.array([1,1,0,0])
# logi = LogisticRegression().fit(X, y)
# print(logi.predict(X))

# X = np.array([[-2],[-3],[2],[1]])
# y = np.array([1,1,0,0])
# logi = LogisticRegression().fit(X, y)
# print(logi.predict(X))


# In[146]:


# Create DataFrame

df_loans_full = pd.read_csv('MergedLabeled.csv', header = None, skiprows=1)
df_loans_full


# In[147]:


for i in range(len(df_loans_full[7])):
    label = df_loans_full.iloc[i,7]
    label = label[:2].split(" ")[-1]    
    df_loans_full.iloc[i,7] = label


# In[148]:


# Select columns
# col = [6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,26,27]
# col = [6,7,8,9,15,16,17,18,19]
# col = [6,7,8]
col = [6,7]
df_loans = df_loans_full.iloc[:,col]
loans_df = df_loans
df_loans
# loans_df.squeeze()


# In[149]:


df_isbad = df_loans_full.iloc[:,28]
df_isbad


# In[150]:


loans = df_loans.to_numpy()
pred = df_isbad.to_numpy()
pred = np.asarray(pred,dtype('float'))
loans = np.asarray(loans,dtype('float'))
loans = loans.reshape(len(df_loans),len(col))
pred = pred.reshape(len(df_loans),1)
# loans = loans.tolist()
# pred = pred.tolist()
loans
pred2 = pred.T


# In[151]:


# loans = df_loans.squeeze()
# pred = df_isbad.squeeze()
# pred


# In[152]:


# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(loans, pred, test_size=0.3)

# X_test

logi = LogisticRegression().fit(X_train, y_train.ravel())
print(logi.score(X_test, y_test.ravel()))
pred_X_train = logi.predict(X_test)
pred_X_train[pred_X_train == 1]


# In[153]:


pred = np.array(pred,dtype('float'))
pred


# In[154]:


from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(loans, pred, test_size=0.3)

clf = LogisticRegression().fit(X_train, y_train.ravel())
print(clf.score(X_train,y_train))
clf_pred = clf.predict(X_test)
clf_pred[clf_pred == 1]


# In[155]:


summary_pred = pd.DataFrame([pred[30:60].tolist(),clf_pred[30:60].tolist(),pred_X_train[30:60].tolist()])
summary_pred.T


# In[ ]:




