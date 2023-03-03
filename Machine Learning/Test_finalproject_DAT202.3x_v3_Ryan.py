#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from scipy import*
from sklearn.metrics import*
from numpy import *
import mnist
import random
from itertools import cycle
from sklearn import*
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


# In[ ]:


class LogisticRegressionCustom(object):
    def __init__(self, alpha=0.1, iteration=5000):
        # Learning Rate
        self.alpha = alpha
        self.iteration = iteration

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        # Use normal equation
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        m = X.shape[0]

        for _ in range(self.iteration):
            output = X.dot(self.w)
            errors = y - self.sigmoid(output)
            self.w += self.alpha / m * errors.dot(X)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self.sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# In[ ]:


import requests
import csv

# Create input dataset
df_loans_full = pd.read_csv('MergedLabeled.csv', header = None, skiprows=1)
# df_loans_full = pd.read_csv('MergedLabeled.csv')
df_loans_full


# In[ ]:


# Convert string data column to numeric data column: Term
for i in range(len(df_loans_full[7])):
    label = df_loans_full.iloc[i,7]
    label = label[:2].split(" ")[-1]    
    df_loans_full.iloc[i,7] = label


# In[ ]:


# Select columns
# col = [6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,26,27]
# col = [6,7,8,9,15,16,17,18,19]
# col = [6,7,8]
col = [6,7]
df_loans = df_loans_full.iloc[:,col]
loans_df = df_loans
df_loans
# loans_df.squeeze()


# In[ ]:


# Create output dataset
# Binary vector. Rate of bad loans
df_isbad = df_loans_full.iloc[:,28]
df_isbad


# In[ ]:


# Create loans dataset
loans = df_loans.to_numpy()
pred = df_isbad.to_numpy()
pred = np.asarray(pred,dtype('float'))
loans = np.asarray(loans,dtype('float'))
# Reshape dataset
loans = loans.reshape(len(df_loans),len(col))
pred = pred.reshape(len(df_loans),1)


# In[ ]:


# Custom Logistic Regression

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(loans, pred, test_size=0.3)

# Apply Logistic Regression
logi = LogisticRegressionCustom().fit(X_train, y_train.ravel())
# Accuracy Rate
print(logi.score(X_train, y_train.ravel()))
# Prediction
pred_X_train = logi.predict(X_test) 


# Create prediction vector.
# Create list of indexes 
test_val = 0
test_val_index = []
for i in range(len(pred_X_train)):
#     print(pred_X_train[i])
    if pred_X_train[i] == y_test[i]:
        test_val += 1
        test_val_index.append(i)
# List of Indexes
# print(test_val_index)
# Number of predicted values against orginal values
print(test_val)
# Length of list of split binary vector
len(pred_X_train)

# List of Predicted Values (0 & 1)
# predicted = []
predicted = y_test
for i in range(len(pred_X_train)):
    if i < test_val:
        if i == test_val_index[i]:
            predicted[i] = pred_X_train[i]
    else:
        predicted[i] = int(y_test[i].tolist()[0])
   


# In[ ]:


# Sklearn Logisitc Regression model

# Sklearn Logistic Regression function
clf = LogisticRegression().fit(X_train, y_train.ravel())
# Accuracy
print(clf.score(X_train,y_train))
# Predicted vector
clf_pred = clf.predict(X_test)
# clf_pred[clf_pred == 1]

# Create prediction vector.
# Create list of indexes 
test_val_reg = 0
test_val_index_reg = []
for i in range(len(clf_pred)):
#     print(pred_X_train[i])
    if clf_pred[i] == y_test[i]:
        test_val_reg += 1
        test_val_index_reg.append(i)
# List of Indexes
# print(test_val_index_reg)
# Number of predicted values against orginal values
print(test_val_reg)
# Length of list of split binary vecto
# len(clf_pred)


# In[ ]:



# List of Predicted values (0 or 1)
# predicted = []
predicted_reg = y_test
for i in range(len(pred_X_train)):
    if i < test_val_reg:
        if i == test_val_index_reg[i]:
            predicted_reg[i] = pred_X_train[i]
    else:
        predicted_reg[i] = int(y_test[i].tolist()[0])


# In[ ]:


# Create Dataframe to compare original, sklearn and predicted values (0 or 1)
df_pred_concat = pd.DataFrame([pred[0:len(predicted)].tolist(),predicted,predicted_reg])
df_pred = df_pred_concat.T
df_pred.columns=['Original', 'SkLearn', 'Custom']
df_pred


# In[ ]:




