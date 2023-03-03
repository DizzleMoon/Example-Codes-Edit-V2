#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from matplotlib.pyplot import figure
from py.xml import raw
from requests.api import get


# In[3]:


# Sample size
Ts = 0.001

# Number of samples
N = 1000
# N2 = 1/50

# Sample range
sample_range = np.linspace(0,Ts*N,N)
sample_range2 = np.linspace(0,Ts*N,N)
# sample_range2


# In[4]:


# Initializze state matrices
X = np.array([[100], [0]])
# P0 = np.diag((0.01, 0.01, 0.01, 0.01))
P = np.diag((10,0.01))
# A = np.array([[1, 0, Ts , 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]])
# Q = np.eye(X.shape[0])
Q = np.zeros((2,2))
# Q = np.array([[0,0],[1,0]])
# B = np.eye(X.shape[0])
# U = np.zeros((X.shape[0],1)) 

# Measurement matrices
Y = np.array([[X[0,0] + abs(0.1*np.random.randn(1)[0])], [X[1,0] + abs(0.1*np.random.randn(1)[0])]])
H = np.array([[1, 0], [0, 1]])
R = np.eye(Y.shape[0])
# R = np.array([[1,0],[0,0]])


# In[5]:


# # System matrix - state
A = np.array([[1, Ts], [0,1]])

# # System matrix - input
B= np.array([[-0.5*(Ts**2)], [-Ts]])
# # G = [[-0.5*(Ts**2)], [-Ts]]

# # Input vector
# u = 9.80665
# U = np.array([[0],[9.80665]])
U = 9.80665

# # Observation matrix
# H = np.array([1,0])

# # Sigma - Standard Deviations
# Q = np.array([[0,0],[0,0]])


# In[6]:


# Apply Kalman Filter
xx = []
for i in range(0,N):
    
    # Prediction
    X = np.dot(A,X) + np.dot(B.T,U)
    P = np.dot(A,np.dot(P, A.T)) + Q
    xx.append(X)
    plt.plot((X[0,0]), (X[1,0]), 'o', color='red')
    
    # Update
    nu = np.dot(H,X)
    S = np.dot(H, np.dot(P,H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    X = X + np.dot(K, Y-nu)
    P = P - np.dot(K, np.dot(S,K.T))
    plt.plot(X[0,0], X[1,0], 'o', color='black')
    
    Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] + abs(0.1 * np.random.randn(1)[0])]]) 
    plt.plot(Y[0,0], Y[1,0], color='green')
    
# Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] + abs(0.1 * np.random.randn(1)[0])]]) 
# plt.plot(Y[0,:], Y[1,:], color='blue')


# In[7]:


x_t = np.zeros((2,N))
x_t[:,0] = [100,0]
for k in range(1,N):
    x_t[:,k] = np.dot(A,x_t[:,k-1]) + np.dot(B.T,U)
R1 = 4
# Measurement noise
v = np.sqrt(R1) + np.random.randn(N)
# Noisy measurement
z = np.matmul(H,x_t) + v

# z
plt.plot(sample_range2,z[0])

# plt.plot(Y[0,0], Y[1,0], color='green')
# plt.plot(Y)


# In[8]:


# Apply Kalman Filter
# Prediction
pred_x = []
pred_y = []
# Update
update_x = []
update_y = []
# Measurement
measure_x = []
measure_y = []

# X = np.array([[105],[0]])
for i in range(0,N):
    
    # Prediction
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A,np.dot(P, A.T)) + Q
    pred_x.append(X[0,0])
    pred_y.append(X[1,0]) 
    plt.plot(X[0,0], X[1,0], 'o', color='red')
    
    # Update
    nu = np.dot(H,X)
    S = np.dot(H, np.dot(P,H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    X = X + np.dot(K, Y-nu)
    P = P - np.dot(K, np.dot(S,K.T))
    update_x.append(X[0,0])
    update_y.append(X[1,0]) 
    plt.plot(X[0,0], X[1,0], 'o', color='black')
    
    Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] + abs(0.1 * np.random.randn(1)[0])]]) 
    measure_x.append(Y[0,0])
    measure_y.append(Y[1,0]) 
    plt.plot(Y[0,:], Y[1,:], color='blue')
# 
Y


# In[9]:


# Plot

plot1 = plt.figure(1)
plt.plot(pred_x,pred_y, color='green')
# plt.scatter(pred_x,pred_y, color='red')
plt.plot(update_x,update_y)
# plt.scatter(update_x,update_y, color='blue')
plt.plot(measure_x,measure_y, color='blue')
plt.scatter(measure_x,measure_y, color = 'orange')
plt.grid()


# In[10]:


plot2 = plt.figure(2)
# plt.plot(pred_x,pred_y, color='green')
# plt.scatter(pred_x,pred_y, color='red')
plt.plot(sample_range2,update_x,sample_range2,measure_x)
# plt.plot(sample_range2,z[0])
# plt.yticks([85,90,95,100,105])
# plt.xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
# plt.scatter(update_x,update_y, color='blue')
# plt.plot(measure_x,measure_y, color='blue')
# plt.scatter(measure_x,measure_y, color = 'orange')
plt.grid()


# In[11]:


plot3 = plt.figure(3)
# plt.plot(sample_range2,z[1])
# plt.plot(pred_x,pred_y, color='green')
# plt.scatter(pred_x,pred_y, color='red')
plt.plot(sample_range2,update_y)
# plt.scatter(update_x,update_y, color='blue')
# plt.plot(measure_x,measure_y, color='blue')
# plt.scatter(measure_x,measure_y, color = 'orange')
plt.scatter(sample_range2,measure_y, color = 'orange')
plt.grid()


# In[12]:


plot4 = plt.figure(4)
plt.plot(sample_range2,np.abs(np.array(measure_y) - np.array(update_y)))
# plt.plot(sample_range2,np.abs(np.array(z[1]) - np.array(update_y)))
plot5 = plt.figure(5)
plt.plot(sample_range2,np.abs(np.array(measure_x) - np.array(update_x)))
# plt.plot(sample_range2,np.abs(np.array(z[0]) - np.array(update_x)))


# In[ ]:




