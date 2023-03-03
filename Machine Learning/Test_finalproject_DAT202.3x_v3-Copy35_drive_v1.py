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
import collections


# In[2]:


class LogisticRegressionCustom(object):
    def __init__(self, eta=0.1, n_iter=5000):
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


# In[3]:


import requests
import csv


df_loans_full = pd.read_csv('MergedLabeled.csv', header = None, skiprows=1)
df_loans_full


# In[4]:


for i in range(len(df_loans_full[7])):
    label = df_loans_full.iloc[i,7]
    label = label[:2].split(" ")[-1]    
    df_loans_full.iloc[i,7] = label


# In[5]:


# Select columns
# col = [6,7,8,9,15,16,17,18,19,20,21,22,23,24,25,26,27]
# col = [6,7,8,9,15,16,17,18,19]
# col = [6,7,8]
col = [6,7]
df_loans = df_loans_full.iloc[:,col]
loans_df = df_loans
df_loans
# loans_df.squeeze()


# In[6]:


# Binary vector. Rate of bad loans
df_isbad = df_loans_full.iloc[:,28]
df_isbad


# In[7]:


loans = df_loans.to_numpy()
pred = df_isbad.to_numpy()
pred = np.asarray(pred,dtype('float'))
loans = np.asarray(loans,dtype('float'))
loans = loans.reshape(len(df_loans),len(col))
pred = pred.reshape(len(df_loans),1)


# In[8]:


# Custom Logistic Regression

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(loans, pred, test_size=0.3)

# Apply Logistic Regression
logi = LogisticRegressionCustom().fit(X_train, y_train.ravel())
# Accuracy Rate
print(logi.score(X_train, y_train.ravel()))
# Prediction
pred_X_train = logi.predict(X_test) 
# pred0 = logi.predict(X_test)

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


# In[9]:


# List of Predicted values (0 or 1)
# predicted = []
# for i in range(len(pred_X_train)):
#     if i < test_val:
#         if i == test_val_index[i]:
#             predicted.append(pred_X_train[i])
#     else:
#         predicted.append(int(y_test[i].tolist()[0]))
# len(predicted)  

# List of Predicted values (0 or 1)
predicted = []
predicted = y_test
for i in range(len(pred_X_train)):
    if i < test_val:
        if i == test_val_index[i]:
            predicted[i] = pred_X_train[i]
    else:
        predicted[i] = int(y_test[i].tolist()[0])
len(predicted)    


# In[10]:


# Sklearn Logisitc Regression model

# for _ in range(10):
# X_train, X_test, y_train, y_test = train_test_split(loans, pred, test_size=0.3)

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
len(clf_pred)


# In[11]:


# List of Predicted values (0 or 1)

# List of Predicted values (0 or 1)
# predicted = []
predicted_reg = y_test
for i in range(len(pred_X_train)):
    if i < test_val_reg:
        if i == test_val_index_reg[i]:
            predicted_reg[i] = pred_X_train[i]
    else:
        predicted_reg[i] = int(y_test[i].tolist()[0])
# print(len(predicted_reg))    
predicted_reg

# predicted_reg = []
# for i in range(len(pred_X_train)):
#     if i < test_val_reg:
#         if i == test_val_index_reg[i]:
#             predicted_reg.append(pred_X_train[i])
#     else:
#         predicted_reg.append(int(y_test[i].tolist()[0]))
# predicted_reg  


# In[12]:


# Create Dataframe to compare original and predicted values (0 or 1)
df_pred_concat = pd.DataFrame([pred[0:len(predicted)].tolist(),predicted,predicted_reg])
df_pred = df_pred_concat.T
df_pred


# In[13]:


wg_0 = logi.w
wg = (logi.w - np.mean(logi.w))/np.std(logi.w)
wg 


# In[14]:


# Actual
actuals = list(y_test)
# Scored values
scored = logi._sigmoid(y_test)
# Threshold
threshold = np.max(np.unique(scored))
threshold


# In[15]:


X_test_0 = X_test
X_test = (X_test - np.mean(X_test))/np.std(X_test)

# Size of array
X_test_shape = X_test.shape
# Total number of observations
X_obs = X_test_shape[0] * X_test_shape[1]
X_obs


# In[16]:


# Coefficients
# Reshape input array
x_0 = int(X_obs/len(wg))
z = np.dot(X_test.reshape(x_0,len(wg)),wg)


# In[17]:


# Apply sigmoid function
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
    
h = sigmoid(z)
h1 = h[0:len(z)]

# Plot sigmoid function
plt.plot(sorted(z),sorted(h))
plt.grid()


# In[18]:


# Apply random threshold

# Threshold value
threshold_rand = 0.2
# Create Threshold to seperate values
h2 = h[h >= threshold_rand]
print(len(h2))
# Find corresponding set of array indices
h2_index = np.where(h == np.min(h2))
print(h2_index)

# z: X-axis
zz = h[len(h)-len(h2):]
len(zz)
plt.plot(sorted(zz),sorted(h2))
plt.grid()


# In[19]:


# Create Confusion Matrix
ConfMat = collections.namedtuple('conf', ['tp','fp','tn','fn']) 


# In[20]:


def ConfusionMatrix(actuals, scores, threshold, positive_label=1):
    tp=fp=tn=fn=0
    bool_actuals = [act==positive_label for act in actuals]
    FPR = TPR = []
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
        if (fp + tn) != 0:
            FPR.append(fp/(fp+tn))
        else:
            FPR.append(0)
        
        if (tp+fn) != 0 :
            TPR.append(tp/(tp+fn))
        else:
            TPR.append(0)

    return ConfMat(tp, fp, tn, fn), FPR,TPR


# In[25]:


# Create Threshold to seperate values
h3 = h[h >= threshold]

con_mat, FPR, TPR = ConfusionMatrix(actuals, h3, threshold, positive_label=1)
plt.plot(sorted(FPR),sorted(TPR))
plt.grid(True, 'major', color='k')
plt.minorticks_on()
plt.grid(True, 'minor', 'y')
plt.show()


# In[22]:


# Setup Confusion Matrix
# tp = 0
# fp = 1
# tn = 2
# fn = 3

FPR = con_mat[1]/(con_mat[1] + con_mat[2])
TPR = con_mat[0]/(con_mat[0] + con_mat[3])
FPR


# In[ ]:




